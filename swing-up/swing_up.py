import math

import control as ctrl
import numpy as np
import pygame


# Simulation Mechanics
def scale(length):
    return length * 100


def rotPoint(x_0, y_0, length, theta):
    x = x_0 + scale(length * math.sin(theta))
    y = y_0 + scale(length - length * math.cos(theta))

    return (x, y)


def dynamics(x, u, dt):
    x_next = x.copy()

    aa_num = (
        u * math.cos(x[2, 0])
        + mass_p * length_p * (x[3, 0] ** 2) * math.sin(x[2, 0]) * math.cos(x[2, 0])
        - (mass_p + mass_c) * g * math.sin(x[2, 0])
    )

    aa_den = mass_p * length_p * (math.cos(x[2, 0]) ** 2) - (mass_c + mass_p) * length_p

    ang_acc = aa_num / aa_den

    a_num = u + mass_p * length_p * (
        (x[3, 0] ** 2) * math.sin(x[2, 0]) - ang_acc * math.cos(x[2, 0])
    )

    a_den = mass_p + mass_c

    acc = a_num / a_den

    x_next[1, 0] = x[1, 0] + acc * dt
    x_next[0, 0] = x[0, 0] + x_next[1, 0] * dt

    x_next[3, 0] = x[3, 0] + ang_acc * dt
    x_next[2, 0] = x[2, 0] + x_next[3, 0] * dt

    return x_next


# Derivatives
def linearize_dynamics(x, u, dt):
    A = np.zeros((4, 4))
    B = np.zeros((4, 1))
    epsilon = 1e-4

    x_next = dynamics(x, u, dt)

    for i in range(0, 4):
        x_perturbed = x.copy()
        x_perturbed[i, 0] = x_perturbed[i, 0] + epsilon
        derivative = (dynamics(x_perturbed, u, dt) - x_next) / epsilon

        A[:, i] = derivative.flatten()

    B = (dynamics(x, u + epsilon, dt) - dynamics(x, u, dt)) / epsilon

    return (A, B)


# Cost
def calculate_cost(x, x_target, u):
    state_diff = x - x_target
    state_cost = np.transpose(state_diff) @ Q @ state_diff

    action_cost = np.transpose(u) @ R @ u

    total_cost = state_cost + action_cost

    return total_cost


# Backpass
def backPass(x_traj, u_traj, list_A, list_B, x_target, N, Q_f):
    diff = x_traj[N - 1] - x_target
    Vx = 2 @ Q_f @ diff
    Vxx = 2 @ Q_f

    list_k = []
    list_K = []

    for i in range(N - 1, -1, -1):
        diff = x_traj[i] - x_target
        lx = 2 @ Q @ diff
        lu = 2 @ R @ u_traj[i]
        lxx = 2 @ Q
        luu = 2 @ R

        Qx = lx + np.transpose(list_A[i]) @ Vx
        Qu = lu + np.transpose(list_B[i]) @ Vx

        Qxx = lxx + np.transpose(list_A[i]) @ Vxx @ list_A[i]
        Quu = luu + np.transpose(list_B[i]) @ Vxx @ list_B[i]
        Qux = np.transpose(list_B[i]) @ Vxx @ list_A[i]

        Quu_i = np.linalg.inv(Quu)

        k = -Quu_i @ Qu
        K = -Quu_i @ Qux

        list_k.append(k)
        list_K.append(K)

        Vx = Qx - np.transpose(Qu) @ Quu_i @ Qux
        Vxx = Qxx - np.transpose(Qux) @ Quu_i @ Qux

    return (list_k, list_K)


# Forward Pass
def forward_pass(x_0, x_traj, u_traj, list_k, list_K, alpha, N, dt):
    x_traj_new = [x_0]
    u_traj_new = []

    for i in range(0, N - 1):
        x_old = x_traj[i]
        x_new = x_traj_new[i]
        dx = x_new - x_old

        if dx[2, 0] > 3.14:
            dx[2, 0] = dx[2, 0] - 6.28
        elif dx[2, 0] < -3.14:
            dx[2, 0] = dx[2, 0] + 6.28

        u_new = u_traj[i] + alpha * list_k[i] + list_K[i] @ dx
        x_next = dynamics(x_new, u_new, dt)

        x_traj_new.append(x_next)
        u_traj_new.append(u_new)

    return (x_traj_new, u_traj_new)


N = 100
x_target = np.array([[0], [0], [0], [0]])

x_0 = np.array([[0], [0], [3.14], [0]])
x_t = np.array([[0.0], [0.0], [0.0], [0.0]])

Q = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 100, 0], [0, 0, 0, 10]])
R = np.array([[0.1]])

Q_f = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 200, 0], [0, 0, 0, 50]])
x_traj = [x_0]
u_traj = [np.zeros((1, 1)) for _ in range(N)]


# Training loop
def main_loop(numIter):
    global x_traj
    global u_traj
    cost = 0
    alpha = 1

    # Initial Rollout
    for i in range(1, N):
        x_next = dynamics(x_traj[i - 1], u_traj[i - 1], dt_sim)
        x_traj.append(x_next)

    # Initial Cost
    for x, u in zip(x_traj, u_traj):
        cost += calculate_cost(x, x_target, u)

    cost += (x_traj[-1] - x_target).T @ Q_f @ (x_traj[-1] - x_target)

    for i in range(0, numIter):
        alpha = 1
        list_A = []
        list_B = []

        for x, u in zip(x_traj, u_traj):
            a_curr, b_curr = linearize_dynamics(x, u, dt_sim)
            list_A.append(a_curr)
            list_B.append(b_curr)

        list_k, list_K = backPass(x_traj, u_traj, list_A, list_B, x_target, N, Q_f)

        list_k.reverse()
        list_K.reverse()

        x_traj_new = []
        u_traj_new = []

        x_traj_new, u_traj_new = forward_pass(
            x_0, x_traj, u_traj, list_k, list_K, alpha, N, dt_sim
        )

        cost_new = 0
        for x, u in zip(x_traj_new, u_traj_new):
            cost_new += calculate_cost(x, x_target, u)

        cost_new += (x_traj[-1] - x_target).T @ Q_f @ (x_traj[-1] - x_target)

        while cost_new > cost:
            cost_new = 0
            alpha *= 0.5
            x_traj_new, u_traj_new = forward_pass(
                x_0, x_traj, u_traj, list_k, list_K, alpha, N, dt_sim
            )

            for x, u in zip(x_traj_new, u_traj_new):
                cost_new += calculate_cost(x, x_target, u)

            cost_new += (x_traj[-1] - x_target).T @ Q_f @ (x_traj[-1] - x_target)

        x_traj = x_traj_new
        u_traj = u_traj_new

        cost = cost_new


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

pivot_pos = [scr_width / 2, scr_height / 2]
tip_pos = [scr_width / 2, scr_height / 2 - scale(length_p)]

accumulator = 0

ang_acc = 0
acc = 0

box = pygame.Rect(
    pivot_pos[0] - scale((width_c / 2)), pivot_pos[1], scale(width_c), scale(length_c)
)

main_loop(50)

# while running:
#     dt = clock.tick(60) / 1000

#     # poll for events
#     # pygame.QUIT event means the user clicked X to close your window

#     accumulator += dt

#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False

#     # fill the screen with a color to wipe away anything from last frame
#     screen.fill("black")

#     if accumulator > dt_sim:
#         # RENDER YOUR GAME HERE

#         x_next = dynamics(x_0, 0, dt_sim)
#         x_0 = x_next.copy()

#         pivot_pos[0] = scr_width / 2 + scale(x_0[0, 0])

#         box.update(
#             pivot_pos[0] - scale((width_c / 2)),
#             pivot_pos[1],
#             scale(width_c),
#             scale(length_c),
#         )

#         tip_pos = rotPoint(
#             pivot_pos[0], scr_height / 2 - scale(length_p), length_p, x_0[2, 0]
#         )

#         pygame.draw.rect(screen, "white", box, 1)
#         pygame.draw.line(screen, "white", pivot_pos, tip_pos)

#         font = pygame.font.Font("./IBM_Plex_Mono/IBMPlexMono-Regular.ttf", 13)
#         text = font.render(f"angle: {x_0[0, 0]}\nForce: {0}", True, "white")
#         textpos = text.get_rect(centerx=screen.get_width() / 2, y=10)
#         screen.blit(text, textpos)

#         accumulator -= dt_sim

#     # flip() the display to put your work on screen
#     pygame.display.flip()

# pygame.quit()
