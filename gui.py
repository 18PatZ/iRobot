from tkinter import *
from tkinter import ttk
import os
import re

import time

import pygame

from irobot_interface import iRobotInterface, NOTES

w = None

pressed = set()
robot = None

DEADZONE = 0.1

KEY_NOTE_MAP = {
    "q": "C",
    "w": "D",
    "e": "E",
    "r": "F",
    "t": "G",
    "y": "A",
    "u": "B",
    "i": "C_1",
    "o": "D_1",
    "p": "E_1",
    "bracketleft": "F_1",
    "bracketright": "G_1"
}

pygame.init()
joy = pygame.joystick.Joystick(0)
joy.init()

def key_pressed(event):
    if event.keysym == 'Escape':
        exit()
    print("PRESSED",event.keysym)
    # w.config(text="Key Pressed:"+event.char)
    pressed.add(event.keysym.lower())
    # print()
    update()


def key_released(event):
    # w.config(text = "Key Released:"+event.char)
    print("RELEASED",event.keysym)
    pressed.remove(event.keysym.lower())
    update()


def init_robot():
    global robot

    robot = iRobotInterface(use_serial = False, use_TCP = True, 
        tcp_ip = '10.228.106.75',#'127.0.0.1', 
        tcp_port = 6666)

    robot.start_mode()
    robot.safe_mode()
    robot.register_beeps()
    # robot.beep(3)


def init():
    global w

    root = Tk()

    frame = ttk.Frame(root, padding=200)
    frame.grid()

    ttk.Label(frame, text="WASD to control robot. Escape to quit.").grid(column=0, row=0)
    ttk.Button(frame, text="Quit", command=root.destroy).grid(column=1, row=0)

    w = Label(root, text="")
    w.place(x=70,y=90)

    root.bind("<KeyPress>", key_pressed)
    root.bind("<KeyRelease>", key_released)

    init_robot()

    # while True:
    #     update()
    #     time.sleep(0.1)

    return root


def update():
    t = ""
    for c in pressed:
        t += ("" if len(t) == 0 else ", ") + c
    w.config(text="Keys pressed: " + t)
    
    for event in pygame.event.get():
        pass

    sp = 500
    # if "shift_l" in pressed:
    #     sp = 500

    ra = 250
    if "shift_l" in pressed:
        ra = 100

    duration = 12
    if "shift_l" in pressed:
        duration = 36

    turn_radius = ra if "right" in pressed else (-ra if "left" in pressed else None)

    if "up" in pressed:
        robot.drive(speed = sp, turn_radius = turn_radius)
    elif "down" in pressed:
        robot.drive(speed = -sp, turn_radius = turn_radius)
    elif "right" in pressed:
        robot.turn(speed = sp)
    elif "left" in pressed:
        robot.turn(speed = -sp)
    else:

        turning = abs(joy.get_axis(2)) >= DEADZONE

        turn_radius = (250 * joy.get_axis(2)) if turning else None

        # if joy.get_axis(5) >= -1 + DEADZONE:
        #     robot.drive(speed = (joy.get_axis(5) + 1) / 2 * 5, turn_radius = turn_radius)
        # elif joy.get_axis(4) >= -1 + DEADZONE:
        #     robot.drive(speed = - (joy.get_axis(4) + 1) / 2 * 5, turn_radius = turn_radius)
        if abs(joy.get_axis(1)) >= DEADZONE:
            robot.drive(speed = -joy.get_axis(1) * 500, turn_radius = turn_radius)
        elif turning:
            robot.turn(speed = 500 * joy.get_axis(2))
        else:
            robot.stop()

        if(joy.get_button(0)):
            robot.beep(7)

        print("Axes", joy.get_axis(0), joy.get_axis(1), joy.get_axis(2), joy.get_axis(3), joy.get_axis(4), joy.get_axis(5))

        

    for p in pressed:
        if re.search("^[0-9]$", p) is not None:
            robot.beep(int(p))
            break

    # if "space" in pressed:
    #     robot.beep(8)

    for note in KEY_NOTE_MAP:
        if note in pressed:
            robot.register_beep(10, [(NOTES[KEY_NOTE_MAP[note]], duration)])
            robot.beep(10)

# disable X's autorepeat behaviour - holding down key causes continuous presses and releases
# os.system('xset r off')

root = init()

root.mainloop()

# os.system('xset r on')
