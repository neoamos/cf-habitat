from turtle import pos
from unittest.mock import DEFAULT
import pygame
import time
import logging

from curses import wrapper

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander

URI = 'radio://0/80/2M/E7E7E7E7E0'
DEFAULT_HEIGHT = 0.3
BOX_LIMIT = 0.3
DEADZONE = 0.1

is_deck_attached = False

position_estimate = [0, 0]

logging.basicConfig(level=logging.ERROR)

pygame.init()
joysticks = []
clock = pygame.time.Clock()

for i in range(0, pygame.joystick.get_count()):
  joysticks.append(pygame.joystick.Joystick(i))
  joysticks[-1].init()
  print("Detected joystick " + joysticks[-1].get_name())


def joystick_control(stdscr, mc):
  max_vel = 0.3
  max_yaw_rate = 40
  x_vel = 0.0
  y_vel = 0.0
  yaw_rate = 0.0

  axes = [0, 0, 0, 0, 0, 0]
  running = True
  while running:
    clock.tick(30)

    
    for event in pygame.event.get():
      if event.type == pygame.JOYBUTTONDOWN:
        running = False
        mc.stop()
        return
      elif event.type == pygame.JOYAXISMOTION:
        axes[event.axis] = event.value

        if abs(axes[4]) > DEADZONE:
          x_vel = max_vel * axes[4] * -1
        else:
          x_vel = 0.0

        if abs(axes[3]) > DEADZONE:
          y_vel = max_vel * axes[3] * -1
        else:
          y_vel = 0.0

        if abs(axes[0]) > DEADZONE:
          yaw_rate = max_yaw_rate * axes[0]
        else:
          yaw_rate = 0.0

    stdscr.clear()
    stdscr.addstr(1, 1, "x_vel: {:0.2f}\n y_vel: {:0.2f}\n yaw_rate: {:0.2f}".format(x_vel, y_vel, yaw_rate))
    stdscr.addstr(5, 1, "X: {:0.2f}".format(position_estimate[0]))
    stdscr.addstr(6, 1, "Y: {:0.2f}".format(position_estimate[1]))

    stdscr.refresh()
    # print("x_vel: {}, y_vel: {}, yaw_rate: {}".format(x_vel, y_vel, yaw_rate))
    # mc.start_linear_motion(x_vel, y_vel, 0.0, rate_yaw=yaw_rate)

  
    # print(str(axes), end="\r")

def move_linear_simple(scf):
  with MotionCommander(scf, default_height=DEFAULT_HEIGHT) as mc:
    time.sleep(1)
    mc.forward(0.5)
    time.sleep(1)
    mc.turn_left(180)
    time.sleep(1)
    mc.forward(0.5)
    time.sleep(1)

def move_box_limit(scf):
    with MotionCommander(scf, default_height=DEFAULT_HEIGHT) as mc:
        body_x_cmd = 0.2
        body_y_cmd = 0.1
        max_vel = 0.2

        while (1):
            #if position_estimate[0] > BOX_LIMIT:
            #    mc.start_back()
            #elif position_estimate[0] < -BOX_LIMIT:
            #    mc.start_forward()

            if position_estimate[0] > BOX_LIMIT:
                body_x_cmd = -max_vel
            elif position_estimate[0] < -BOX_LIMIT:
                body_x_cmd = max_vel
            if position_estimate[1] > BOX_LIMIT:
                body_y_cmd = -max_vel
            elif position_estimate[1] < -BOX_LIMIT:
                body_y_cmd = max_vel

            mc.start_linear_motion(body_x_cmd, body_y_cmd, 0)

            time.sleep(0.1)


def take_off_simple(scf):
  with MotionCommander(scf) as mc:
    time.sleep(3)


def param_deck_flow(name, value_str):
  value = int(value_str)
  print(value)
  global is_deck_attached
  if value:
    is_deck_attached = True
    print('Deck is attached!')
  else:
    is_deck_attached = False
    print('Deck is NOT attached!')

def log_pos_callback(timestamp, data, logconf):
  print(data)
  global position_estimate
  position_estimate[0] = data['stateEstimate.x']
  position_estimate[1] = data['stateEstimate.y']


def main(stdscr):
  joystick_control(stdscr, 0)
  cflib.crtp.init_drivers()

  with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
    scf.cf.param.add_update_callback(group='deck', name='bcFlow2', cb=param_deck_flow)

    time.sleep(1)

    logconf = LogConfig(name='Position', period_in_ms=100)
    logconf.add_variable('stateEstimate.x', 'float')
    logconf.add_variable('stateEstimate.y', 'float')
    scf.cf.log.add_config(logconf)
    print(logconf.valid)
    logconf.data_received_cb.add_callback(log_pos_callback)


    if is_deck_attached:
      logconf.start()
      with MotionCommander(scf, default_height=DEFAULT_HEIGHT) as mc:
        joystick_control(mc)
      logconf.stop()

if __name__ == '__main__':
  wrapper(main)