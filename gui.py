from tkinter import *
from tkinter import ttk

w = None

pressed = set()


def key_pressed(event):
    if event.keysym == 'Escape':
        exit()
    # w.config(text="Key Pressed:"+event.char)
    pressed.add(event.keysym)
    # print()
    update()


def key_released(event):
    # w.config(text = "Key Released:"+event.char)
    pressed.remove(event.keysym)
    update()


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

    return root


def update():
    t = ""
    for c in pressed:
        t += ("" if len(t) == 0 else ", ") + c
    w.config(text="Keys pressed: " + t)


root = init()


root.mainloop()
