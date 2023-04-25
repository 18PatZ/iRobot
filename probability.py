# -*- coding: utf-8 -*-

import socket
import math
import serial
from threading import Thread
import time
import json
import random
from socketUtil import *

port = '/dev/ttyUSB0'
baud = 57600

ser = serial.Serial(port = port, baudrate = baud)

command_queue = []


def clamp(val, low, high):
    if val > high:
        return high
    if val < low:
        return low
    return val


def speedToBytes(speed_val):
    return int16ToBytes(int(clamp(speed_val, -500, 500)))
def radiusToBytes(radius_val):
    return int16ToBytes(int(clamp(radius_val, -2000, 2000)))


def int16ToBytes(val):
    
    first_byte = 0
    second_byte = 0

    if val < 0:
        val = 2**16 + val
    #    first_byte = 255 # FF
    #    second_byte = speed_val - (255 << 8)
    #else: 
    first_byte = val >> 8
    second_byte = val - (first_byte << 8)
    
    return first_byte, second_byte


def start_mode():
    # Opcode 128: Start
    print("Start")
    command_queue.append([128])

def passive_mode():
    # Opcode 128: Start, changes any mode to passive

    # Upon sending the Start command or any one of the demo
    # commands (which also starts the specific demo, e.g., Spot
    # Cover, Cover, Cover and Dock, or Demo), the OI enters
    # into Passive mode. When the OI is in Passive mode, you
    # can request and receive sensor data using any of the
    # sensors commands, but you cannot change the current
    # command parameters for the actuators (motors, speaker,
    # lights, low side drivers, digital outputs) to something else.
    # To change how one of the actuators operates, you must
    # switch from Passive mode to Full mode or Safe mode.
    # While in Passive mode, you can read Create’s sensors,
    # watch Create perform any one of its ten built-in demos,
    # and charge the battery.
    print("Passive")
    command_queue.append([128])

def safe_mode():
    # Opcode 131: Safe
    # This command puts the OI into Safe mode, enabling user
    # control of Create. It turns off all LEDs. The OI can be in
    # Passive, Safe, or Full mode to accept this command.

    # When you send a Safe command to the OI, Create enters into
    # Safe mode. Safe mode gives you full control of Create, with
    # the exception of the following safety-related conditions:
    # • Detection of a cliff while moving forward (or moving
    # backward with a small turning radius, less than one robot
    # radius).
    # • Detection of a wheel drop (on any wheel).
    # • Charger plugged in and powered.
    # Should one of the above safety-related conditions occur
    # while the OI is in Safe mode, Create stops all motors and
    # reverts to the Passive mode.
    # If no commands are sent to the OI when in Safe mode, Create
    # waits with all motors and LEDs off and does not respond to
    # Play or Advance button presses or other sensor input.
    # Note that charging terminates when you enter Safe Mode.
    print("Safe")
    command_queue.append([131])

def full_mode():
    # Opcode 132: Full
    # This command gives you complete control over Create
    # by putting the OI into Full mode, and turning off the cliff,
    # wheel-drop and internal charger safety features. That is, in
    # Full mode, Create executes any command that you send
    # it, even if the internal charger is plugged in, or the robot
    # senses a cliff or wheel drop.

    # When you send a Full command to the OI, Create enters
    # into Full mode. Full mode gives you complete control over
    # Create, all of its actuators, and all of the safety-related
    # conditions that are restricted when the OI is in Safe mode,
    # as Full mode shuts off the cliff, wheel-drop and internal
    # charger safety features. To put the OI back into Safe mode,
    # you must send the Safe command.
    # If no commands are sent to the OI when in Full mode, Create
    # waits with all motors and LEDs off and does not respond to
    # Play or Advance button presses or other sensor input.
    # Note that charging terminates when you enter Full Mode.
    command_queue.append([132])

def drive(speed = 100, turn_radius = None):
    print("Drive " + str(speed) + ("" if turn_radius is None else (" " + str(turn_radius))))
    # Opcode 137: Drive
    velocity_high_byte, velocity_low_byte = speedToBytes(speed)
    if turn_radius is None:
        # radius bytes 0x8000 indicates straight motion
        command_queue.append([137, velocity_high_byte, velocity_low_byte, 128, 0])
    else:
        # positive radius goes left for some reason, flip to right
        radius_high_byte, radius_low_byte = radiusToBytes(-turn_radius)
        command_queue.append([137, velocity_high_byte, velocity_low_byte, radius_high_byte, radius_low_byte])

def stop_bot():
    print("Stop")
    command_queue.append([137, 0, 0, 0, 0])

def turn(speed = 100):
    print("Turn " + str(speed))
    # Opcode 137: Drive
    # Turn in place clockwise = 0xFFFF
    # Turn in place counter-clockwise = 0x0001
    # Works with negative speed as well

    velocity_high_byte, velocity_low_byte = speedToBytes(speed)
    radius_high_byte, radius_low_byte = (255, 255)
    command_queue.append([137, velocity_high_byte, velocity_low_byte, radius_high_byte, radius_low_byte])

def register_beeps():
    print("Register beeps")
    # Opcode 140: Song
    command_queue.append([140, 0, 4, 60, 12, 64, 12, 67, 12, 72, 36])
    #command_queue.append([140, 1, 2, 72, 12, 72, 36])
    command_queue.append([140, 1, 2, 55, 12, 55, 36])
    command_queue.append([140, 2, 2, 60, 12, 60, 36])
    command_queue.append([140, 3, 2, 64, 12, 64, 36])
    command_queue.append([140, 4, 2, 67, 12, 67, 36])
    command_queue.append([140, 5, 2, 72, 12, 72, 36])
    command_queue.append([140, 6, 5, 60, 12, 64, 12, 67, 12, 72, 12, 72, 36])
    #command_queue.append([140 0 4 62 12 66 12 69 12 74 36])

def beep(num):
    print("Beep " + str(num))
    # Opcode 141: Play Song
    # Need to register with Opcode 140 first!
    command_queue.append([141, num])

def beep2():
    print("Beep 2")
    command_queue.append([141, 1])

def beep3():
    print("Beep 3")
    command_queue.append([141, 2])

def send_commands():
    global command_queue
    for command in command_queue:
        ser.write(bytearray(command))
    del command_queue[:]

def plan(action, duration):
    global last_action_time
    global stages
    stages.append({'time': last_action_time, 'action': action})
    last_action_time += duration




stage = -1
spd = 300
last_action_time = 0
stages = []

action_headings = {
    "NORTH": 0,
    "WEST": 270, 
    "EAST": 90,
    "SOUTH": 180
}

time_step = 5 # seconds
travel_distance = 330
current_heading = 0

start_mode()
safe_mode()
register_beeps()
beep(0)
send_commands()

time.sleep(2)



for i in range(20):
    temp = raw_input("Press enter to move.")
    
    rand_radius = int(300, 800)
    sign = randint(0, 1)
    if(sign == 1):
        rand_radius = -rand_radius
    arclength = 2*math.PI*travel_distance * (rand_radius / 360)

    drive(speed = 200, turn_radius = 300)
    send_commands()
    time.sleep(2)
    stop_bot()
    send_commands()

    time.sleep(5)
    drive(speed = 200, turn_radius = 300)
    send_commands()
    time.sleep(2)

    stop_bot()
    send_commands()
    """
    if(temp == 'q' or temp == 'quit' or temp == 'e' or temp == 'exit'):
        break

    # Measured about 187deg/s when turning at full speed
    # Calculated 220deg/s when turning at full speed
    # May need to change the 190 below for more/less uncertainty
    #
    #  degrees      1      degrees   seconds
    #          X ------- =         X ------- = seconds
    #            deg/sec             degrees
    time_to_turn = 90.0 / 187
    print('Turning for ' + str(time_to_turn) + 's')
    turn_speed = 500


    beep(2)
    turn(speed = turn_speed)
    send_commands()

    time.sleep(time_to_turn)
    
    time_to_drive = time_step - time_to_turn # Driving takes rest of timestep
    drive_speed = int(travel_distance / time_to_drive) # Set speed to reach goal at end of timestep, assumes high accel (d=st)
    
    drive(speed = drive_speed, turn_radius = random.randint(-rand_radius, rand_radius))
    send_commands()

    time.sleep(time_to_drive)

    stop_bot()
    send_commands()
    """

ser.close()
