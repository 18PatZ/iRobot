from tkinter import *
from tkinter import ttk
import os

from irobot_interface import iRobotInterface

w = None

pressed = set()
robot = None

def key_pressed(event):
    if event.keysym == 'Escape':
        exit()
    print("PRESSED",event.keysym)
    # w.config(text="Key Pressed:"+event.char)
    pressed.add(event.keysym)
    # print()
    update()


def key_released(event):
    # w.config(text = "Key Released:"+event.char)
    print("RELEASED",event.keysym)
    pressed.remove(event.keysym)
    update()


def init_robot():
    global robot

    robot = iRobotInterface(use_serial = False, use_TCP = True, 
        tcp_ip = '127.0.0.1', 
        tcp_port = 6666)

    robot.start_mode()
    robot.safe_mode()
    robot.register_beeps()
    robot.beep(0)


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

    return root


def update():
    t = ""
    for c in pressed:
        t += ("" if len(t) == 0 else ", ") + c
    w.config(text="Keys pressed: " + t)

    if "w" in pressed:
        robot.drive(speed = 200)
    else:
        robot.stop()

# disable X's autorepeat behaviour - holding down key causes continuous presses and releases
# os.system('xset r off')

root = init()

root.mainloop()

# os.system('xset r on')
