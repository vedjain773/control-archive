import math

import control as ctrl
import numpy as np
import pygame


def scale(length):
    return length * 100


def rotPoint(x_0, y_0, length, theta):
    x = x_0 + scale(length * math.sin(theta))
    y = y_0 + scale(length - length * math.cos(theta))

    return (x, y)


mass_p = 1
length_p = 1
mass_c = 1
length_c = 1
width_c = 2
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

x_0 = np.array([[0], [0], [0.7], [0]])
x_t = np.array([[0.0], [0.0], [0.0], [0.0]])

pivot_pos = [scr_width / 2, scr_height / 2]
tip_pos = [scr_width / 2, scr_height / 2 - scale(length_p)]

accumulator = 0

ang_acc = 0
acc = 0

box = pygame.Rect(
    pivot_pos[0] - scale((width_c / 2)), pivot_pos[1], scale(width_c), scale(length_c)
)

A = np.array(
    [
        [1, (dt_sim**2) / 2, 0, 0],
        [0, 1, -mass_p * g * (dt_sim**2) / (2 * mass_c), 0],
        [0, 0, 1, (dt_sim**2) / 2],
        [0, 0, (mass_p + mass_c) * g * (dt_sim**2) / (mass_c * 2 * length_p), 1],
    ]
)

B = np.array(
    [
        [(dt_sim**2) / (2 * mass_c)],
        [dt_sim / mass_c],
        [-(dt_sim**2) / (mass_c * 2 * length_p)],
        [-dt_sim / (length_p * mass_c)],
    ]
)

Q = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 100, 0], [0, 0, 0, 10]])

R = 0.1

K, S, E = ctrl.dlqr(A, B, Q, R)

while running:
    dt = clock.tick(60) / 1000

    # poll for events
    # pygame.QUIT event means the user clicked X to close your window

    accumulator += dt

    U = -(K @ x_0)[0, 0]

    aa_num = (
        0 * math.cos(x_0[2, 0])
        + mass_p
        * length_p
        * (x_0[3, 0] ** 2)
        * math.sin(x_0[2, 0])
        * math.cos(x_0[2, 0])
        - (mass_p + mass_c) * g * math.sin(x_0[2, 0])
    )

    aa_den = (
        mass_p * length_p * (math.cos(x_0[2, 0]) ** 2) - (mass_c + mass_p) * length_p
    )

    ang_acc = aa_num / aa_den

    a_num = 0 + mass_p * length_p * (
        (x_0[3, 0] ** 2) * math.sin(x_0[2, 0]) - ang_acc * math.cos(x_0[2, 0])
    )

    a_den = mass_p + mass_c

    acc = a_num / a_den

    x_t[1, 0] = x_0[1, 0] + acc * dt_sim
    x_t[0, 0] = x_0[0, 0] + x_t[1, 0] * dt_sim

    x_t[3, 0] = x_0[3, 0] + ang_acc * dt_sim
    x_t[2, 0] = x_0[2, 0] + x_t[3, 0] * dt_sim

    x_0 = x_t.copy()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # fill the screen with a color to wipe away anything from last frame
    screen.fill("black")

    if accumulator > dt_sim:
        # RENDER YOUR GAME HERE

        pivot_pos[0] = scr_width / 2 + scale(x_0[0, 0])

        box.update(
            pivot_pos[0] - scale((width_c / 2)),
            pivot_pos[1],
            scale(width_c),
            scale(length_c),
        )

        pygame.draw.rect(screen, "white", box, 1)
        pygame.draw.line(screen, "white", pivot_pos, tip_pos)
        tip_pos = rotPoint(
            pivot_pos[0], scr_height / 2 - scale(length_p), length_p, x_0[2, 0]
        )

        font = pygame.font.Font("../assets/IBM_Plex_Mono/IBMPlexMono-Regular.ttf", 13)
        text = font.render(f"angle: {x_0[0, 0]}\nForce: {U}", True, "white")
        textpos = text.get_rect(centerx=screen.get_width() / 2, y=10)
        screen.blit(text, textpos)

        accumulator -= dt_sim

    # flip() the display to put your work on screen
    pygame.display.flip()

pygame.quit()
