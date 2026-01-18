# Control Archive

A collection of optimal control implementations focusing on the Linear Quadratic Regulator (LQR) and its variants. This repository documents the transition from controlling simple linear systems to complex nonlinear robotics via trajectory optimization.
Implemented Systems

## Double Integrator

The double integrator serves as the fundamental baseline for linear optimal control. This implementation demonstrates the application of LQR on a second-order system where the control input directly influences acceleration. It focuses on the trade-offs between state error and control effort by tuning the Q and R cost matrices to achieve precise position and velocity tracking.

## Inverted Pendulum

The inverted pendulum represents the first step into nonlinear control. By linearizing the dynamics around the unstable upright equilibrium, this implementation applies LQR to maintain balance within a local basin of attraction. It illustrates the limitations of linear controllers when applied to nonlinear systems and the importance of the Taylor expansion in state-space modeling.

## Cart-Pole

The cart-pole system introduces the challenge of underactuation. In this model, the controller must stabilize a free-spinning pole by applying horizontal forces to the cart base. This implementation handles multi-input multi-output (MIMO) dynamics to balance the pole while simultaneously regulating the cart's position, highlighting the power of LQR in managing coupled degrees of freedom.
