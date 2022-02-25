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

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GdkPixbuf, GLib

from matplotlib.backends.backend_gtk3agg import (
    FigureCanvasGTK3Agg as FigureCanvas)
from matplotlib.figure import Figure
from matplotlib.backends.backend_gtk3 import (
    NavigationToolbar2GTK3 as NavigationToolbar)

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
  def __init__(self):
    threading.Thread.__init__(self, daemon=False)
    self.running = True

    self.URI = 'radio://0/80/2M/E7E7E7E7E0'
    self.DEFAULT_HEIGHT = 0.3
    self.DEADZONE = 0.1

    self.BUTTON_RB = 5
    self.BUTTON_LB = 4
    self.BUTTON_A = 0
    self.BUTTON_B = 1
    self.BUTTON_X = 2
    self.BUTTON_Y = 3
    self.BUTTON_START = 7
    self.BUTTON_SELECT = 6

    self.is_deck_attached = False
    self.mc = None # Motion commander

    self.goal = np.array([0,0,0])
    self.position_estimate = np.zeros((3))
    self.orientation_estimate = np.zeros((3))
    self.quaternions = quaternion.from_euler_angles(
      self.orientation_estimate/180*np.pi)
    self.range_estimate = np.zeros((5))
    self.battery_volts = 0.0

    self.last_image = None
    self.last_image_at = None
    self.recording = []
    self.ranger_points = []
    self.recording_period_ms = 500

    pygame.init()
    self.joysticks = []
    self.clock = pygame.time.Clock()

    for i in range(0, pygame.joystick.get_count()):
      self.joysticks.append(pygame.joystick.Joystick(i))
      self.joysticks[-1].init()
      print("Detected joystick " + self.joysticks[-1].get_name())

    self.pointgoal_agent = PointGoalAgent(
      "configs/experiments/crazyflie_baseline_rgb.yaml",
      "data/pretrained-models/grayscale_5m_95.pth"
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
    self.ax.set_xlim([-5, 5])
    self.ax.set_ylim([-5, 5])

    self.drone_path, = self.ax.plot([])
    self.drone_position, = self.ax.plot(0, 0, 
      marker=(3, 0, 0), linestyle='None', color="green")
    self.ranger_points_scatter = self.ax.scatter([], [])

    self.chart_canvas = FigureCanvas(self.fig)  # a Gtk.DrawingArea
    vbox.pack_start(self.chart_canvas, True, True, 0)
    toolbar = NavigationToolbar(self.chart_canvas, self.window)
    vbox.pack_start(toolbar, False, False, 0)

    self.window.show_all()

  def control_loop(self):
    flying = False
    max_vel = 0.3
    max_yaw_rate = 40
    x_vel = 0.0
    y_vel = 0.0
    yaw_rate = 0.0
    control_mode = "joystick"
    next_action_after = time.time()
    action_count = 0

    axes = [0, 0, 0, 0, 0, 0]
    while self.running:
      self.clock.tick(30)
      if flying:
        self.record()
      
      for event in pygame.event.get():
        if event.type == pygame.JOYBUTTONDOWN:
          #Change control mode
          if event.button == self.BUTTON_RB:
            control_mode = "pointgoal"
            self.pointgoal_agent.reset_state()

          #Start/Stop flight
          elif event.button == self.BUTTON_START:
            if self.mc != None and flying:
              print("Landing")
              self.mc.land()
              flying = False
            elif self.mc and not flying:
              print("Taking off")
              self.mc.take_off()
              flying = True

          #Reset kalman filter
          elif event.button == self.BUTTON_SELECT:
            self.scf.cf.param.set_value('kalman.resetEstimation', '1')

        elif event.type == pygame.JOYBUTTONUP:
          if event.button == self.BUTTON_RB:
            control_mode = "joystick"

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

      if self.mc != None and flying:
        if control_mode == "joystick":
          self.mc.start_linear_motion(x_vel, y_vel, 0, rate_yaw=yaw_rate)
        elif control_mode == "pointgoal":
          # Only take the next axtion after some time
          if time.time() > next_action_after:

            # Only take an action if we have a recent image
            if self.last_image != None and time.time() - self.last_image_at < 1.0:
              image = self.last_image.resize((256,256))
              image = np.array(image)
              image = np.stack((image, image, image), axis=-1)
              yaw = self.orientation_estimate[0]

              action = self.pointgoal_agent.act(
                image, self.position_estimate, yaw, self.goal)

              action_count += 1
              self.set_label("control_rl_action", "{} ({})".format(action, action_count))

              if action == "MOVE_FORWARD":
                if self.range_estimate[0] > 0.25:
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
            else:
              self.mc.stop()
          else:
            if action == "MOVE_FORWARD" and self.range_estimate[0] < 0.25:
              self.mc.stop()

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
      self.set_label("camera_status", "{:.1f} fps / {:.1f} kb".format(fps, len(imgdata)/1000))
    self.last_image_at = time.time()
    self.last_image = img

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
  
  def param_deck_flow(self, name, value_str):
    value = int(value_str)
    if value:
      self.is_deck_attached = True
      print('Deck is attached!')
    else:
      is_deck_attached = False
      print('Deck is NOT attached!')

  def run(self):
    cflib.crtp.init_drivers()

    with SyncCrazyflie(self.URI, cf=Crazyflie(rw_cache='./cache')) as scf:
      scf.cf.param.add_update_callback(group='deck', name='bcFlow2', cb=self.param_deck_flow)
      self.scf = scf
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
        if self.mc:
          self.mc.land()
        pos_logger.stop()
        range_logger.stop()
        battery_logger.stop()


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
  parser.add_argument("-s", default=None, metavar="save_dir", help="Folder to save images")
  args = parser.parse_args()

  control_thread = ControlThread()
  img_thread = ImgThread(args.n, args.p, control_thread.on_image, viewer=False)
  img_thread.start()
  control_thread.start()

  GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGINT, app_exit)

  Gtk.main()

