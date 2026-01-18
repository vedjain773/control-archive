import math

import control as ctrl
import numpy as np
import pygame

MAX_TORQUE = 5


def rotPoint(x_0, y_0, length, theta):
    x = x_0 + length * math.sin(theta)
    y = y_0 + (length - length * math.cos(theta))

    return (x, y)


mass = 1
length = 1
g = 9.8

# pygame setup
pygame.init()
screen = pygame.display.set_mode((600, 600))
clock = pygame.time.Clock()
running = True

scr_height = screen.get_height()
scr_width = screen.get_width()

dt_sim = 0.01
dt = 0

pivot_pos = [scr_width / 2, scr_height / 2]
tip_pos = [scr_width / 2, scr_height / 2 - length]

accumulator = 0

x_0 = np.array([[0.5], [0]])
x_t = np.array([[0.0], [0.0]])

ang_acc = 0

A = np.array([[1, dt_sim], [g * dt_sim / length, 1]])
B = np.array(
    [
        [dt_sim * dt_sim / (2 * mass * length * length)],
        [dt_sim / (mass * length * length)],
    ]
)

Q = np.array([[1, 0], [0, 1]])
R = 1

K, E, S = ctrl.dlqr(A, B, Q, R)

while running:
    dt = clock.tick(60) / 1000

    # poll for events
    # pygame.QUIT event means the user clicked X to close your window

    accumulator += dt

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # fill the screen with a color to wipe away anything from last frame
    screen.fill("black")

    if accumulator > dt_sim:
        # RENDER YOUR GAME HERE
        U = (-K @ x_0)[0, 0]

        U = np.clip(U, -MAX_TORQUE, MAX_TORQUE)

        ang_acc = g * math.sin(x_0[0, 0]) / length + U / (mass * length**2)

        x_t[1, 0] = x_0[1, 0] + ang_acc * dt_sim
        x_t[0, 0] = x_0[0, 0] + x_t[1, 0] * dt_sim

        x_0[1, 0] = x_t[1, 0]
        x_0[0, 0] = x_t[0, 0]

        pygame.draw.line(screen, "white", pivot_pos, tip_pos)
        tip_pos = rotPoint(300, 200, 100, x_0[0, 0])

        font = pygame.font.Font("../assets/IBM_Plex_Mono/IBMPlexMono-Regular.ttf", 15)
        text = font.render(f"angle: {x_0[0, 0]}\nangle_vel: {x_0[1, 0]}", True, "white")
        textpos = text.get_rect(centerx=screen.get_width() / 2, y=10)
        screen.blit(text, textpos)

        accumulator -= dt_sim

    # flip() the display to put your work on screen
    pygame.display.flip()

pygame.quit()
