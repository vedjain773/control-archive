import control as ctrl
import mujoco
import numpy as np
from dm_control import suite, viewer

env = suite.load(domain_name="pendulum", task_name="swingup")
time_step = env.reset()
physics = env.physics

with env.physics.reset_context():
    initial_angle = np.radians(30)
    env.physics.data.qpos[0] = initial_angle
    env.physics.data.qvel[0] = 0.0

nstate = physics.model.nq + physics.model.nv
nctrl = physics.model.nu

A = np.zeros((nstate, nstate))
B = np.zeros((nstate, nctrl))

mujoco.mjd_transitionFD(
    physics.model.ptr, physics.data.ptr, 1e-6, True, A, B, None, None
)

Q = np.array([[1, 0], [0, 1]])
R = 0.1

K, E, S = ctrl.dlqr(A, B, Q, R)


def balance(time_step):
    x = np.concatenate([env.physics.data.qpos, env.physics.data.qvel])

    x_goal = np.zeros(2)

    u = -K @ (x - x_goal)

    return u


viewer.launch(env, policy=balance)
