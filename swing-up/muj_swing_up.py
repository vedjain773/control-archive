import copy

import control as ctrl
import mujoco
import numpy as np
from dm_control import suite, viewer

env = suite.load(domain_name="cartpole", task_name="swingup")
time_step = env.reset()
physics_act = env.physics

physics_copy = copy.deepcopy(physics_act)

model = physics_act.model

cart_joint_id = 0
x_min, x_max = model.jnt_range[cart_joint_id]

rail_margin = 0.9
x_safe_min = rail_margin * x_min
x_safe_max = rail_margin * x_max

W_RAIL = 200.0

Q = np.array([[75, 0, 0, 0], [0, 100, 0, 0], [0, 0, 1, 0], [0, 0, 0, 10]])
R = np.array([[1]])

x_0 = np.array([[0], [3.14], [0], [0]])
x_target = np.array([[0], [0], [0], [0]])


def get_state(physics):
    x = np.vstack([*physics.data.qpos, *physics.data.qvel])
    return x


def dynamics(x, u, physics):
    physics.set_state(x.flatten())
    physics.after_reset()

    physics.set_control(u)
    physics.step()

    return get_state(physics)


def linearize_dynamics(x, physics):
    A = np.zeros((4, 4))
    B = np.zeros((4, 1))

    physics.set_state(x.flatten())
    physics.after_reset()

    physics.model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER

    mujoco.mjd_transitionFD(
        physics.model.ptr, physics.data.ptr, 1e-6, True, A, B, None, None
    )

    return (A, B)


def calculate_cost(x, x_target, u):
    state_diff = x - x_target
    state_cost = np.transpose(state_diff) @ Q @ state_diff

    action_cost = np.transpose(u) @ R @ u

    total_cost = state_cost + action_cost

    cart_pos = x[0, 0]

    if cart_pos > x_safe_max:
        total_cost += W_RAIL * (cart_pos - x_safe_max) ** 2
    elif cart_pos < x_safe_min:
        total_cost += W_RAIL * (cart_pos - x_safe_min) ** 2

    return total_cost


def backPass(x_traj, u_traj, list_A, list_B, x_target, N, Q_f):
    diff = x_traj[N - 1] - x_target
    Vx = 2 * (Q_f @ diff)
    Vxx = 2 * (Q_f)

    list_k = []
    list_K = []

    for i in range(N - 2, -1, -1):
        diff = x_traj[i] - x_target
        lx = 2 * (Q @ diff)
        lu = 2 * (R @ u_traj[i])
        lxx = 2 * (Q)
        luu = 2 * (R)

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

        Vx = Qx - np.transpose(Qux) @ Quu_i @ Qu
        Vxx = Qxx - np.transpose(Qux) @ Quu_i @ Qux

    return (list_k, list_K)


def forward_pass(x_0, x_traj, u_traj, list_k, list_K, alpha, N, physics):
    x_traj_new = [x_0]
    u_traj_new = []

    for i in range(0, N - 1):
        x_old = x_traj[i]
        x_new = x_traj_new[i]
        dx = x_new - x_old

        if dx[1, 0] > 3.14:
            dx[1, 0] = dx[1, 0] - 6.28
        elif dx[1, 0] < -3.14:
            dx[1, 0] = dx[1, 0] + 6.28

        u_new = u_traj[i] + alpha * list_k[i] + list_K[i] @ dx
        x_next = dynamics(x_new, u_new, physics)

        x_traj_new.append(x_next)
        u_traj_new.append(u_new)

    return (x_traj_new, u_traj_new)


N = 50
x_traj = [x_0]
u_traj = [np.zeros((1, 1)) for _ in range(N - 1)]

Q_f = np.array([[100, 0, 0, 0], [0, 0, 300, 0], [0, 0, 0, 0], [0, 0, 0, 100]])


def main_loop(numIter, physics):
    global x_traj
    global u_traj
    cost = 0
    alpha = 1

    # Initial Rollout
    for i in range(1, N):
        x_next = dynamics(x_traj[i - 1], u_traj[i - 1], physics)
        x_traj.append(x_next)

    # Initial Cost
    for x, u in zip(x_traj, u_traj):
        cost += calculate_cost(x, x_target, u)

    cost += (x_traj[-1] - x_target).T @ Q_f @ (x_traj[-1] - x_target)

    for i in range(0, numIter):
        # print(
        #     f"\r Iteration {i}: x_len={len(x_traj)}, u_len={len(u_traj)}",
        #     end=" ",
        #     flush=True,
        # )
        alpha = 1
        list_A = []
        list_B = []

        for x, u in zip(x_traj, u_traj):
            a_curr, b_curr = linearize_dynamics(x, physics)
            list_A.append(a_curr)
            list_B.append(b_curr)

        list_k, list_K = backPass(x_traj, u_traj, list_A, list_B, x_target, N, Q_f)

        list_k.reverse()
        list_K.reverse()

        x_traj_new = []
        u_traj_new = []

        x_traj_new, u_traj_new = forward_pass(
            x_0, x_traj, u_traj, list_k, list_K, alpha, N, physics
        )

        cost_new = 0
        for x, u in zip(x_traj_new, u_traj_new):
            cost_new += calculate_cost(x, x_target, u)

        cost_new += (x_traj[-1] - x_target).T @ Q_f @ (x_traj[-1] - x_target)

        while cost_new > cost:
            cost_new = 0
            alpha *= 0.5

            if alpha < 1e-5:
                x_traj_new, u_traj_new = x_traj, u_traj
                cost_new = cost
                break

            x_traj_new, u_traj_new = forward_pass(
                x_0, x_traj, u_traj, list_k, list_K, alpha, N, physics
            )

            for x, u in zip(x_traj_new, u_traj_new):
                cost_new += calculate_cost(x, x_target, u)

            cost_new += (x_traj[-1] - x_target).T @ Q_f @ (x_traj[-1] - x_target)

        x_traj = x_traj_new
        u_traj = u_traj_new

        cost = cost_new


# main_loop(150, physics_copy)

env.reset()
physics_act.set_state(x_0.flatten())
physics_act.after_reset()

act_u = []


def mpc(num):
    global last_u, x_0, x_traj, u_traj
    i = 0
    while i < num:
        print(f"\r {i}", end=" ", flush=True)
        x_0 = get_state(physics_copy)

        x_traj = [x_0]
        u_traj = [np.zeros((1, 1)) for _ in range(N - 1)]

        main_loop(numIter=5, physics=physics_copy)
        last_u = u_traj[0]
        act_u.append(last_u)
        i += 1


mpc(1000)

u_iter = iter(u_traj)


def policy(timestep):
    try:
        return next(u_iter)
    except StopIteration:
        return np.zeros_like(u_traj[0])


viewer.launch(env, policy=policy)
