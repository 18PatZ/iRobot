import pygame
from pygame.locals import *

pygame.init()

joy = pygame.joystick.Joystick(0)

while True:
    for event in pygame.event.get():
        pass

    print("Axes", joy.get_axis(0), joy.get_axis(1), joy.get_axis(2), joy.get_axis(3), joy.get_axis(4), joy.get_axis(5))