import control as ct
import matplotlib.pyplot as plt
import numpy as np
import pygame

# pygame setup
pygame.init()
screen = pygame.display.set_mode((600, 600))
clock = pygame.time.Clock()
running = True

dt_sim = 0.01
dt = 0

box_pos = pygame.Vector2(0, screen.get_height() / 2)

x_0 = np.array([[0.0], [200.0]])
x_t1 = np.array([[0.0], [0.0]])
box = pygame.Rect(x_0[0, 0], (screen.get_height() / 2) - 10, 30, 30)

A = np.array([[1, dt_sim], [0, 1]])
B = np.array([[0.5 * (dt_sim**2)], [dt_sim]])
Q = np.array([[1, 0], [0, 1]])
R = 0.1

K, S, E = ct.dlqr(A, B, Q, R)

accumulator = 0

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
        U = -(K @ x_0)
        acc = U[0, 0]

        x_t1[1, 0] = x_0[1, 0] + acc * dt_sim
        x_t1[0, 0] = x_0[0, 0] + x_t1[1, 0] * dt_sim

        x_0 = x_t1.copy()

        # RENDER YOUR GAME HERE
        pygame.draw.rect(screen, "white", box, width=1)
        box.update(x_0[0, 0], (screen.get_height() / 2) - 10, 30, 30)

        font = pygame.font.Font("../assets/IBM_Plex_Mono/IBMPlexMono-Regular.ttf", 15)
        text = font.render(f"x: {x_0[0, 0]}\nv: {x_0[1, 0]}", True, "white")
        textpos = text.get_rect(centerx=screen.get_width() / 2, y=10)
        screen.blit(text, textpos)

        accumulator -= dt_sim

    # flip() the display to put your work on screen
    pygame.display.flip()

pygame.quit()
