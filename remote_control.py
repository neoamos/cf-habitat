import pygame
import time
import logging
import argparse
import threading
from datetime import datetime
from PIL import Image
import numpy as np
import quaternion
import signal
import traceback
import json
import os

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GdkPixbuf, GLib

from matplotlib.backends.backend_gtk3agg import (
    FigureCanvasGTK3Agg as FigureCanvas)
from matplotlib.figure import Figure
from matplotlib.backends.backend_gtk3 import (
    NavigationToolbar2GTK3 as NavigationToolbar)

from matplotlib.patches import Rectangle

from curses import wrapper

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander

from agents import PointGoalAgent, compute_pointgoal_cf
from viewer import ImgThread


logging.basicConfig(level=logging.ERROR)


class ControlThread(threading.Thread):
  DEFAULT_VELOCITY = 0.3 # m/s
  DEFAULT_YAW_RATE = 20 # deg/s

  def __init__(self, experiment=None, agent_config=None, eps_id=None, setpoint_mode="velocity"):
    threading.Thread.__init__(self, daemon=False)
    self.running = True

    self.URI = 'radio://0/80/2M/E7E7E7E7E0'
    self.DEFAULT_HEIGHT = 0.4 # m
    self.DEADZONE = 0.1

    self.BUTTON_RB = 5
    self.BUTTON_LB = 4
    self.BUTTON_A = 0
    self.BUTTON_B = 1
    self.BUTTON_X = 2
    self.BUTTON_Y = 3
    self.BUTTON_START = 7
    self.BUTTON_SELECT = 6

    self.setpoint_mode=setpoint_mode

    self.is_deck_attached = False
    self.mc = None # Motion commander
    self.hlc = None # High level commander

    self.goal = np.array([0,0,0])
    self.start_yaw = None
    self.position_estimate = np.zeros((3))
    self.orientation_estimate = np.zeros((3))
    self.quaternions = quaternion.from_euler_angles(
      self.orientation_estimate/180*np.pi)
    self.range_estimate = np.zeros((5))
    self.battery_volts = 0.0
    self.flying = False

    self.last_image = None
    self.image_count = 0
    self.last_image_at = None
    self.recording = []
    self.ranger_points = []
    self.recording_period_ms = 300

    pygame.init()
    self.joysticks = []
    self.clock = pygame.time.Clock()

    for i in range(0, pygame.joystick.get_count()):
      self.joysticks.append(pygame.joystick.Joystick(i))
      self.joysticks[-1].init()
      print("Detected joystick " + self.joysticks[-1].get_name())

    if agent_config:
      # self.pointgoal_agent = PointGoalAgent(agent_config)
      self.pointgoal_agent = PointGoalAgent(
        agent_config,
        weights="data/pretrained-models/ddppo-grayscale-99.pth"
      )

    self.ui = Gtk.Builder()
    self.ui.add_from_file("ui.glade")
    self.window = self.ui.get_object("window")
    self.window.connect("destroy", app_exit)
    self.window.show_all()

    vbox = Gtk.VBox()
    self.ui.get_object("map_frame").add(vbox)

    self.fig = Figure(figsize=(5,4), dpi=100)
    self.ax = self.fig.add_subplot(111)
    self.ax.set_xlim([-3, 7])
    self.ax.set_ylim([-3, 7])

    self.drone_path, = self.ax.plot([])
    self.drone_position, = self.ax.plot(0, 0, 
      marker=(3, 0, 0), linestyle='None', color="green")
    self.ranger_points_scatter = self.ax.scatter([], [])

    self.chart_canvas = FigureCanvas(self.fig)  # a Gtk.DrawingArea
    vbox.pack_start(self.chart_canvas, True, True, 0)
    toolbar = NavigationToolbar(self.chart_canvas, self.window)
    vbox.pack_start(toolbar, False, False, 0)

    self.set_label("control_rl_action", "-")

    if experiment:
      with open(experiment, 'r') as f:
        self.experiment_cfg = json.load(f)
        b = self.experiment_cfg["room_boundaries"]
        area = Rectangle((b[0],b[2]),b[1]-b[0],b[3]-b[2],linewidth=1,edgecolor='r',facecolor='none')
        self.ax.add_patch(area)
        for b in self.experiment_cfg["box_obstacles"]:
          box = Rectangle((b[0],b[1]),b[2],-b[3],linewidth=1,edgecolor='r',facecolor='none')
          self.ax.add_patch(box)

        name_store = Gtk.ListStore(int, str) 
        for id, e in enumerate(self.experiment_cfg["episodes"]):
          name = "[{:.2f},{:.2f}]->[{:.2f},{:.2f}]".format(
            e["start"][0], e["start"][1],
            e["end"][0], e["end"][1],
            )
          name_store.append([id, name])
          print(name)

        self.ui.get_object("episode_picker").set_model(name_store)
        self.ui.get_object("episode_picker").set_active(0)

        if eps_id != None:
          self.ui.get_object("episode_picker").set_active(eps_id)
          episode = self.experiment_cfg["episodes"][eps_id]
          print(episode)
          start = episode["start"]
          end = episode["end"]
          self.goal = np.array([end[0], end[1], 0])
          print(self.goal)
          self.start_yaw = episode["start_yaw"]
          self.ax.plot(self.goal[0], self.goal[1], 
            marker="*", linestyle='None', color="red")
          self.ax.plot(start[0], start[1], 
            marker="D", linestyle='None', color="blue")
          self.fig.canvas.draw_idle()

    recording_file_name = "recordings/{}-{}-{}.csv".format(datetime.now().timestamp(), os.path.basename(agent_config), eps_id)
    self.recording_file = open(recording_file_name, "a")
    self.recording_file.write("time,x,y,z,yaw,pitch,roll,action\n")
    self.recording_file.flush()

    self.window.show_all()

  def control_loop(self):
    max_vel = 0.3
    max_yaw_rate = 40
    x_vel = 0.0
    y_vel = 0.0
    yaw_rate = 0.0
    control_mode = "joystick"

    next_action_after = time.time()
    action_count = 0
    current_target_pos = None
    current_target_yaw = 0.0

    axes = [0, 0, 0, 0, 0, 0]
    while self.running:
      self.clock.tick(30)
      if self.flying:
        self.record()
      else:
        self.drone_position.remove()
        self.drone_position, = self.ax.plot(
          self.position_estimate[0], self.position_estimate[1], 
          marker=(3, 0, self.orientation_estimate[0]), linestyle='None', color="green")
        self.fig.canvas.draw_idle()
      
      for event in pygame.event.get():
        if event.type == pygame.JOYBUTTONDOWN:
          #Change control mode
          if event.button == self.BUTTON_RB:
            if self.setpoint_mode == "velocity":
              control_mode = "pointgoal"
              self.pointgoal_agent.reset_state()
            elif self.setpoint_mode == "position":
              current_target_pos = np.copy(self.position_estimate)
              current_target_yaw = self.orientation_estimate[0]
              self.go_to(current_target_pos, current_target_yaw)

          #Start/Stop flight
          elif event.button == self.BUTTON_START:
            if self.hlc != None and self.flying:
              print("Landing")
              self.land()
            elif self.hlc and not self.flying:
              print("Taking off")
              self.takeoff()

          #Reset kalman filter
          elif event.button == self.BUTTON_SELECT:
            self.scf.cf.param.set_value('kalman.resetEstimation', '1')

        elif event.type == pygame.JOYBUTTONUP:
          if event.button == self.BUTTON_RB:
            control_mode = "joystick"
            current_target_pos = None

        elif event.type == pygame.JOYAXISMOTION:
          axes[event.axis] = event.value

          if abs(axes[4]) > self.DEADZONE:
            x_vel = max_vel * axes[4] * -1
          else:
            x_vel = 0.0

          if abs(axes[3]) > self.DEADZONE:
            y_vel = max_vel * axes[3] * -1
          else:
            y_vel = 0.0

          if abs(axes[0]) > self.DEADZONE:
            yaw_rate = max_yaw_rate * axes[0]
          else:
            yaw_rate = 0.0

      self.update_ui()

      self.set_label("control_x_vel", "{:0.2f}".format(x_vel))
      self.set_label("control_y_vel", "{:0.2f}".format(y_vel))
      self.set_label("control_yaw", "{:0.2f}".format(yaw_rate))

      self.set_label("control_mode", control_mode)

      distance, angle = compute_pointgoal_cf(self.position_estimate, self.orientation_estimate[0], self.goal)
      self.set_label("control_goal_distance", "{:0.2f}".format(distance))
      self.set_label("control_goal_angle", "{:0.2f}".format(angle*180/np.pi))
      self.set_label("control_goal", "{:0.1f}, {:0.1f}".format(self.goal[0], self.goal[1]))

      if self.cf != None and self.flying:
        if control_mode == "joystick":
          # self.linear_motion(x_vel, y_vel, yaw_rate=yaw_rate)
          self.mc.start_linear_motion(x_vel, y_vel, 0, rate_yaw=yaw_rate)
        elif control_mode == "pointgoal":
          # distance = calculate_distance(current_target_pos, self.position_estimate)
          # yaw_diff = np.abs(self.orientation_estimate[0] - current_target_yaw)
          # Only take next action if within 5cm and 1 degree of target
          if time.time() > next_action_after: #(distance < 0.05 and yaw_diff < 3) or time.time() < next_action_after:
            # print("reached target ", distance, yaw_diff)
            # self.cf.commander.send_hover_setpoint(0,0,0, self.DEFAULT_HEIGHT)

            # Only take an action if we have a recent image
            if self.last_image != None and time.time() - self.last_image_at < 0.4:
              image = self.last_image.resize((256,256))
              image = np.array(image)
              image = np.stack((image, image, image), axis=-1)
              yaw = self.orientation_estimate[0]

              action = self.pointgoal_agent.act(
                image, self.position_estimate, yaw, self.goal)

              action_count += 1
              self.set_label("control_rl_action", "{} ({})".format(action, action_count))

              self.recording_file.write("{},,,,,,,{}\n".format(datetime.now().timestamp(), action))

              if self.setpoint_mode == "velocity":
                if action == "MOVE_FORWARD":
                  if self.range_estimate[0] > 0.25 or True:
                    self.mc.start_linear_motion(0.25, 0.0, 0.0)
                    next_action_after = time.time() + 1.0
                  else:
                    self.mc.stop()
                    next_action_after = time.time() + 0.5
                elif action == "TURN_LEFT":
                  self.mc.start_linear_motion(0.0, 0.0, 0.0, rate_yaw=-20.0)
                  next_action_after = time.time() + 0.5
                elif action == "TURN_RIGHT":
                  self.mc.start_linear_motion(0.0, 0.0, 0.0, rate_yaw=20.0)
                  next_action_after = time.time() + 0.5
                elif action == "STOP":
                  self.mc.stop()
                  return
              elif self.setpoint_mode == "position":
                if action == "MOVE_FORWARD":
                  if True: # or self.range_estimate[0] > 0.25:
                    current_target_pos = forward(current_target_pos, current_target_yaw, 0.25)
                    self.go_to(current_target_pos, current_target_yaw)
                  else:
                    self.stop()
                    next_action_after = time.time() + 0.5
                elif action == "TURN_LEFT":
                  current_target_yaw = turn(current_target_yaw, 10)
                  self.go_to(current_target_pos, current_target_yaw)
                elif action == "TURN_RIGHT":
                  current_target_yaw = turn(current_target_yaw, -10)
                  self.go_to(current_target_pos, current_target_yaw)
                elif action == "STOP":
                  self.stop()
                  return
                next_action_after = time.time() + 1.5
            else: #stop if we need to take and action but have no recent image
              self.stop()
          else:
            if self.setpoint_mode == "position":
              self.go_to(current_target_pos, current_target_yaw)

  def update_ui(self):
    self.set_label("drone_x", "{:0.2f}".format(self.position_estimate[0]))
    self.set_label("drone_y", "{:0.2f}".format(self.position_estimate[1]))
    self.set_label("drone_z", "{:0.2f}".format(self.position_estimate[2]))

    self.set_label("drone_yaw", "{:0.2f}".format(self.orientation_estimate[0]))
    self.set_label("drone_roll", "{:0.2f}".format(self.orientation_estimate[1]))
    self.set_label("drone_pitch", "{:0.2f}".format(self.orientation_estimate[2]))

    self.set_label("drone_front", "{}".format(self.range_estimate[0]))
    self.set_label("drone_left", "{}".format(self.range_estimate[1]))
    self.set_label("drone_right", "{}".format(self.range_estimate[2]))
    self.set_label("drone_back", "{}".format(self.range_estimate[3]))
    self.set_label("drone_top", "{}".format(self.range_estimate[4]))

    self.set_label("drone_battery", "{:0.2f}".format(self.battery_volts))

  def record(self):
    if len(self.recording) == 0 or time.time() - self.recording[-1][0] > self.recording_period_ms/1000:
      self.recording_file.write("{},{},{},{},{},{},{},\n".format(
        datetime.now().timestamp(), 
        self.position_estimate[0],
        self.position_estimate[1],
        self.position_estimate[2],
        self.orientation_estimate[0],
        self.orientation_estimate[1],
        self.orientation_estimate[2],
      ))
      self.recording.append([
        time.time(),
        self.position_estimate.copy(),
        self.orientation_estimate.copy(),
        self.range_estimate.copy()
      ])

      self.ranger_points.append(self.drone_to_world(
        [self.range_estimate[0], 0, 0]
      ))
      self.ranger_points.append(self.drone_to_world(
        [-self.range_estimate[3], 0, 0]
      ))
      self.ranger_points.append(self.drone_to_world(
        [0, self.range_estimate[1], 0]
      ))
      self.ranger_points.append(self.drone_to_world(
        [0, -self.range_estimate[2], 0]
      ))

      range_x = [x[0] for x in self.ranger_points]
      range_y = [y[1] for y in self.ranger_points]
      self.ranger_points_scatter.remove()
      self.ranger_points_scatter = self.ax.scatter(range_x, range_y, color="red", s=1)

      x = [x[1][0] for x in self.recording]
      y = [y[1][1] for y in self.recording]
      self.drone_path.remove()
      self.drone_path, = self.ax.plot(x, y, color="blue")

      self.drone_position.remove()
      self.drone_position, = self.ax.plot(
        self.position_estimate[0], self.position_estimate[1], 
        marker=(3, 0, self.orientation_estimate[0]), linestyle='None', color="green")
      self.fig.canvas.draw_idle()

  def drone_to_world(self, vector):
    return quaternion.rotate_vectors(self.quaternions,
      vector
    ) + self.position_estimate

  def set_label(self, label_id, text):
    GLib.idle_add(self.ui.get_object(label_id).set_text, text)

  def log_pos_callback(self, timestamp, data, logconf):
    self.position_estimate[0] = data['stateEstimate.x']
    self.position_estimate[1] = data['stateEstimate.y']
    self.position_estimate[2] = data['stateEstimate.z']
    self.orientation_estimate[0] = data['stateEstimate.yaw']
    self.orientation_estimate[1] = data['stateEstimate.roll']
    self.orientation_estimate[2] = data['stateEstimate.pitch']
    self.quaternions = quaternion.from_euler_angles(self.orientation_estimate/180*np.pi)

    # self.drone_path.set_xdata(self.position_estimate[0])
    # self.drone_path.set_ydata(self.position_estimate[1])


  def log_range_callback(self, timestamp, data, logconf):
    self.range_estimate[0] = data['range.front']/1000
    self.range_estimate[1] = data['range.left']/1000
    self.range_estimate[2] = data['range.right']/1000
    self.range_estimate[3] = data['range.back']/1000
    self.range_estimate[4] = data['range.up']/1000

  def on_battery(self, timestamp, data, logconf):
    self.battery_volts = data["pm.vbat"]

  def on_image(self, img, imgdata):
    if self.last_image_at != None:
      fps = 1 / (time.time() - self.last_image_at)
      self.set_label("camera_status", "{:.1f} fps / {:.1f} kb ({})".format(fps, len(imgdata)/1000, self.image_count))
    self.last_image_at = time.time()
    self.image_count = self.image_count + 1
    self.last_image = img
    img.save("recordings/images/{}-{}.jpg".format(datetime.now().timestamp(), self.image_count))

    # Try to decode JPEG from the data sent from the stream
    try:
        img_loader = GdkPixbuf.PixbufLoader()
        img_loader.write(imgdata)
        img_loader.close()
        pix = img_loader.get_pixbuf()
        # GLib.idle_add(self._update_image, pix)
        GLib.idle_add(self.ui.get_object("camera").set_from_pixbuf, pix)
    except gi.repository.GLib.Error:
        print("Could not set image!")
  
  def param_deck_positioning(self, name, value_str):
    value = int(value_str)
    if value:
      self.is_deck_attached = True
      print(name, 'deck is attached!')
    else:
      is_deck_attached = False
      print(name, 'deck is NOT attached!')

  def run(self):
    cflib.crtp.init_drivers()

    with SyncCrazyflie(self.URI, cf=Crazyflie(rw_cache='./cache')) as scf:
      scf.cf.param.add_update_callback(group='deck', name='bcFlow2', cb=self.param_deck_positioning)
      scf.cf.param.add_update_callback(group='deck', name='bcLighthouse4', cb=self.param_deck_positioning)
      self.scf = scf
      self.cf = scf.cf
      self.hlc = scf.cf.high_level_commander
      time.sleep(1)

      pos_logger = LogConfig(name='Position', period_in_ms=100)
      pos_logger.add_variable('stateEstimate.x', 'float')
      pos_logger.add_variable('stateEstimate.y', 'float')
      pos_logger.add_variable('stateEstimate.z', 'float')
      pos_logger.add_variable('stateEstimate.pitch', 'float')
      pos_logger.add_variable('stateEstimate.roll', 'float')
      pos_logger.add_variable('stateEstimate.yaw', 'float')
      scf.cf.log.add_config(pos_logger)
      pos_logger.data_received_cb.add_callback(self.log_pos_callback)

      range_logger = LogConfig(name="Range", period_in_ms=100)
      range_logger.add_variable('range.back', 'uint16_t')
      range_logger.add_variable('range.left', 'uint16_t')
      range_logger.add_variable('range.right', 'uint16_t')
      range_logger.add_variable('range.front', 'uint16_t')
      range_logger.add_variable('range.up', 'uint16_t')
      scf.cf.log.add_config(range_logger)
      range_logger.data_received_cb.add_callback(self.log_range_callback)
      
      battery_logger = LogConfig(name="Battery", period_in_ms=300)
      battery_logger.add_variable('pm.vbat', 'float')
      scf.cf.log.add_config(battery_logger)
      battery_logger.data_received_cb.add_callback(self.on_battery)

      if self.is_deck_attached:
        pos_logger.start()
        range_logger.start()
        battery_logger.start()
        self.mc = MotionCommander(scf, default_height=self.DEFAULT_HEIGHT)
        try:
          self.control_loop()
        except Exception as e:
          print("Control loop exited with an exception:")
          print(traceback.format_exc())
          print(e)

        print("landing")
        self.land()
        pos_logger.stop()
        range_logger.stop()
        battery_logger.stop()
        self.recording_file.flush()

  def land(self):
    if self.setpoint_mode == "position":
      if self.flying:
        self.flying = False
        if self.position_estimate[2] > 0.1:
          self.cf.commander.send_hover_setpoint(0, 0, 0, 0.1)
          time.sleep(1)
        self.cf.commander.send_stop_setpoint()
      self.hlc.land(0.0, 
        2.0, 
        yaw=None)
    elif self.setpoint_mode == "velocity":
      self.mc.land()
    self.flying = False

  def takeoff(self):
    self.flying = True
    if self.setpoint_mode == "position":
      self.hlc.takeoff(self.DEFAULT_HEIGHT, 
        self.duration(self.DEFAULT_HEIGHT, 0.0),
        yaw=None)
    elif self.setpoint_mode == "velocity":
      self.mc.take_off()
  
  def go_to(self, xyz, yaw, velocity=DEFAULT_VELOCITY, relative=False):
    print("Go to ", xyz, " yaw ", yaw)
    if self.flying:
      print("Executing go to")
      distance = calculate_distance(xyz, self.position_estimate)
      yaw_change = np.abs(yaw-self.orientation_estimate[0])
      # self.hlc.go_to(xyz[0], xyz[1], xyz[2], 
      #   np.deg2rad(yaw), 
      #   self.duration(distance, yaw_change, velocity=velocity),
      #   relative=relative)
      self.cf.commander.send_position_setpoint(xyz[0], xyz[1], xyz[2], yaw)

  def linear_motion(self, x_vel, y_vel, yaw_rate=0.0, height=None):
    if self.flying:
      height = height or self.DEFAULT_HEIGHT
      self.cf.commander.send_hover_setpoint(x_vel, y_vel, yaw_rate, height)
  
  def stop(self):
    if self.setpoint_mode == "position":
      self.linear_motion(0.0, 0.0)
    elif self.setpoint_mode == "velocity":
      self.mc.stop()

  def duration(self, distance, yaw_change, 
    velocity=DEFAULT_VELOCITY, yaw_rate=DEFAULT_YAW_RATE):
    print("duration: ", distance, yaw_change)
    d = max(distance / velocity, yaw_change / yaw_rate, 0.2)
    print("duration2: ", d)
    return max(distance / velocity, yaw_change / yaw_rate, 0.2)

def forward(xyz, yaw, distance):
  print("forward ", xyz, yaw, distance)
  rads = np.deg2rad(yaw)
  delta = np.array([np.cos(rads), np.sin(rads), 0]) * distance
  return xyz + delta

def turn(yaw, change):
  print("turn ", yaw, change)
  return ((yaw+180+change)%360)-180

def calculate_distance(a, b):
  return np.linalg.norm(a-b)


control_thread = None

def app_exit(*args):
  print("Exiting app")
  if control_thread:
    control_thread.running = False
  Gtk.main_quit()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Remote control a Crazyflie with a flow deck and AI deck camera')
  parser.add_argument("-n",  default="192.168.4.1", metavar="ip", help="AI-deck IP")
  parser.add_argument("-p", type=int, default='5000', metavar="port", help="AI-deck port")
  parser.add_argument("-e", default=None, metavar="experiment", help="Experiment config file")
  parser.add_argument("--eps_id", type=int, default=None, metavar="episode_id", help="Experiment config file")
  parser.add_argument("-a", default=None, metavar="agent_config", help="Agent config file")
  parser.add_argument("-c", default="velocity", metavar="setpoint_mode", help="Setpoint mode, either velocity or position")
  args = parser.parse_args()

  control_thread = ControlThread(experiment=args.e, agent_config=args.a, eps_id = args.eps_id, setpoint_mode=args.c)
  img_thread = ImgThread(args.n, args.p, control_thread.on_image, viewer=False)
  img_thread.start()
  control_thread.start()

  GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGINT, app_exit)

  Gtk.main()

