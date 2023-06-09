# -*- coding: utf-8 -*-

import socket
import serial
import random
from threading import Thread
import time
import math
import json
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
    command_queue.append([140, 3, 1, 64, 36])
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





#
# Connect to planner and fetch plan
#
PlannerIp = "10.224.123.203"
PlannerPort = 6667
planner = socket.socket()

print("Connecting to planner at " + PlannerIp + ":" + str(PlannerPort) + "...")
planner.connect((PlannerIp, PlannerPort))
planner.send("SUBS".encode()) # Request plan
planJSON = receiveMessage(planner)
plan = json.loads(planJSON) # Decode plan
planner.close()

print("Received plan: \n")



#
# Get current status from observer
#
ObserverIP = "10.224.112.177" # Standard loopback interface address (localhost)
ObserverPort = 6666
observer = socket.socket()
print("Connecting to observer at " + ObserverIP + ":" + str(ObserverPort) + "...")
observer.connect((ObserverIP, ObserverPort))
print("Connected!")
def getUpdate():
    global observer

    # Get update from camera (returns time interval and current x,  y, and heading)
    data = receiveMessage(observer)

    if data == "exit":
        return (None, None, None, None)
    splitData = data.split(" ")

    # Respond to camera computer with ACK
    observer.send( ("ACK").encode() )
    
    # time_int, x, y, heading
    return int(splitData[0]), int(splitData[1]), int(splitData[2]), int(splitData[3])



stage = -1
spd = 300
last_action_time = 0
stages = []


"""
# plan((lambda: start_mode() or safe_mode() or register_beeps() or beep(0)), 2)
# d = 2
# dm = 3
# dt = 1.0/(3.0/2.0)
# tr = 300
# plan(lambda: beep(1) or drive(speed = spd, turn_radius = None), d)
# plan(lambda: drive(speed = -spd, turn_radius = None), d)
# plan(lambda: turn(speed = spd), d)
# plan(lambda: turn(speed = -spd), d)
# plan(lambda: drive(speed = spd, turn_radius = tr), d)
# plan(lambda: drive(speed = -spd, turn_radius = tr), d)
# plan(lambda: drive(speed = spd, turn_radius = -tr), d)
# plan(lambda: drive(speed = -spd, turn_radius = -tr), d)

# for i in range(4):
#     if i == 0:
#         plan(lambda: beep(1), 0.1)
        
#     plan(lambda: drive(speed = spd, turn_radius = None), dm)
#     plan(lambda j=i: beep(2+j), 0.1)
#     plan(lambda: turn(speed = spd), dt)

# plan(lambda: beep(6), 3)
# plan(lambda: stop_bot(), 3)
# plan(lambda: passive_mode(), 1)

"""

action_headings = {
    "NORTH": 0,
    "WEST": 270, 
    "EAST": 90,
    "SOUTH": 180
}

time_step = 2 # seconds
travel_distance = 245
current_heading = 0

start_mode()
safe_mode()
register_beeps()
beep(0)
send_commands()

time.sleep(2)

# 'o' indicates NOP
#plan = [{(1,0): ["e", "w"]},
#        {(1,0): ["n", "s", "s"], (0,0): ["o", "s", "s"], (2,0): ["o", "o", "o"]},
#        {(2,0): ["n", "e"], (1, 0): ["o", "e"]},
#        {(1,1): ["n", "w", "s", "e"]}]

while True:
    # Fetch current state action map from plan
    #stride_map, plan = plan[0], plan[1:]

    # Get current state from camera
    interval, xPos, yPos, reported_heading = getUpdate()
    current_heading = reported_heading
    print("Received status:", interval, xPos, yPos, current_heading)

    if interval is None:
        print("Exiting.")
        observer.close()
        break

    if current_heading < 0:
        current_heading += 360
    

    # Fetch current policy we'll take
    if interval >= len(plan['Schedule'])-1:
        interval = len(plan['Schedule'])-1
    policy = plan['Policy'][interval]
    
    state = str(xPos) + '-' + str(yPos)
    if state in policy:
        action_sequence = policy[str(xPos) + '-' + str(yPos)]
    else:
        action_sequence = ["NORTH"]
    print("Executing sequence:", action_sequence)

    beep(1)
    send_commands()

    firstMove = True
    # Perform each movement in the selected policy
    while action_sequence != []:
        # Advance through the policy
        current_move, action_sequence = action_sequence[0], action_sequence[1:]

        if not firstMove:
            beep(3)
            send_commands()
        else:
            firstMove = False

        # Skip movement if no-op action
        if(current_move == "NO-OP"):

            time.sleep(time_step)
            continue

        # Get amount to turn
        new_heading = action_headings[current_move]
        degreesToTurn = new_heading - current_heading

        # Correct >=270 degrees in one direction to <=90 degrees in other dir
        if(degreesToTurn >= 180):
            degreesToTurn = -360 + degreesToTurn
        elif(degreesToTurn <= -180):
            degreesToTurn = 360 + degreesToTurn

        # Measured about 187deg/s when turning at full speed
        # Calculated 220deg/s when turning at full speed
        # May need to change the 190 below for more/less uncertainty
        #
        #  degrees      1      degrees   seconds
        #          X ------- =         X ------- = seconds
        #            deg/sec             degrees
        turn_speed = 300 * (-1 if(degreesToTurn < 0) else 1) # need to make negative if turning other way
        #degreesPerSecond = turn_speed / 154 * 180/math.pi
        #time_to_turn = abs(degreesToTurn / degreesPerSecond)
        time_to_turn = abs(degreesToTurn / 120.0)
        print("Turning from " + str(current_heading) + " to " + str(new_heading) + ". (" + str(degreesToTurn) + ")degrees. " + str(time_to_turn) + "s turn.")


        turn(speed = turn_speed)
        current_heading = new_heading
        send_commands()

        time.sleep(time_to_turn)
        
        time_to_drive = time_step - time_to_turn # Driving takes rest of timestep
        drive_speed = int(travel_distance / time_to_drive) # Set speed to reach goal at end of timestep, assumes high accel (d=st)
        
        #drive(speed = drive_speed, turn_radius = random.randint(1000, 2000))
        drive(speed = drive_speed, turn_radius = None)
        send_commands()

        time.sleep(time_to_drive)

        stop_bot()

        send_commands()




# start = time.time()
# while True:
#     t = time.time() - start


#     if stage < len(stages)-1:
#         next_stage = stages[stage + 1]
#         print(next_stage['time'],'vs',t)
#         if next_stage['time'] <= t:
#             next_stage['action']()
#             stage += 1

#     for command in command_queue:
#         ser.write(bytearray(command))
#     del command_queue[:]
    

#     line = raw_input("Enter action: ")

#     action = line.lower()
#     if action in action_headings:
#         new_heading = action_headings[action]

#         time_to_turn = 2 #replace
#         turn_speed = 500 #need to make negative if turning other way
        
#         last_action_time = t
#         plan(lambda: beep(2) or turn(speed = turn_speed), time_to_turn)
        
#         time_to_drive = 2 #replace
#         drive_speed = 100 # change this based on time to turn, so that it reaches the end after full timestep
#         plan(lambda: drive(speed = drive_speed, turn_radius = None), time_to_drive)

#         current_heading = new_heading #update belief

#     else:
#         print("Not found.")

#     time.sleep(0.010)
ser.close()
