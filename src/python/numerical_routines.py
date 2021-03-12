from math import *

def compute_double_pendulum_step_euler(m1, m2, g, length1, length2, initialAngle1, initialAngle2, w1, w2, timeStep):
    d2theta1 = compute_angular_velocity_derivative_point1(m1, m2, g, length1, length2, w1, w2, initialAngle1, initialAngle2)
    d2theta2 = compute_angular_velocity_derivative_point2(m1, m2, g, length1, length2, w1, w2, initialAngle1, initialAngle2)

    point1NewAngularVelocity = d2theta1*timeStep + w1
    point2NewAngularVelocity = d2theta2*timeStep + w2

    point1NewAngle = point1NewAngularVelocity*timeStep + initialAngle1
    point2NewAngle = point2NewAngularVelocity*timeStep + initialAngle2

    return point1NewAngularVelocity, point2NewAngularVelocity, point1NewAngle, point2NewAngle


def compute_double_pendulum_step_rk4(m1, m2, g, length1, length2, initialAngle1, initialAngle2, w1, w2, timeStep):
    # Compute the angular velocities and angular accelerations at the four Runge-Kutta points.
    k1 = compute_k_step(m1, m2, g, length1, length2, initialAngle1, initialAngle2, w1, w2, (0, 0, 0, 0), timeStep/2)
    k2 = compute_k_step(m1, m2, g, length1, length2, initialAngle1, initialAngle2, w1, w2, k1, timeStep/2)
    k3 = compute_k_step(m1, m2, g, length1, length2, initialAngle1, initialAngle2, w1, w2, k2, timeStep/2)
    k4 = compute_k_step(m1, m2, g, length1, length2, initialAngle1, initialAngle2, w1, w2, k3, timeStep)

    # Combine the results of the Runge-Kutta steps.
    velocity1 = (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])/6
    velocity2 = (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])/6
    acceleration1 = (k1[2] + 2*k2[2] + 2*k3[2] + k4[2])/6
    acceleration2 = (k1[3] + 2*k2[3] + 2*k3[3] + k4[3])/6

    # Compute the new state of the pendulum.
    point1NewAngle = initialAngle1 + velocity1*timeStep
    point2NewAngle = initialAngle2 + velocity2*timeStep
    point1NewAngularVelocity = w1 + acceleration1*timeStep
    point2NewAngularVelocity = w2 + acceleration2*timeStep

    return point1NewAngle, point2NewAngle, point1NewAngularVelocity, point2NewAngularVelocity,


def compute_k_step(m1, m2, g, length1, length2, initialAngle1, initialAngle2, w1, w2, previousK, timeStep):
    # Compute the new pendulum state using Forward Euler.
    newAngle1 = initialAngle1 + timeStep*previousK[0]
    newAngle2 = initialAngle2 + timeStep*previousK[1]
    newAngularVelocity1 = w1 + timeStep*previousK[2]
    newAngularVelocity2 = w2 + timeStep*previousK[3]

    # Compute the accelerations at the new pendulum state.
    angularAcceleration1 = compute_angular_velocity_derivative_point1(m1, m2, g, length1, length2, newAngle1, newAngle2, newAngularVelocity1, newAngularVelocity2)
    angularAcceleration2 = compute_angular_velocity_derivative_point2(m1, m2, g, length1, length2, newAngle1, newAngle2, newAngularVelocity1, newAngularVelocity2)

    return newAngularVelocity1, newAngularVelocity2, angularAcceleration1, angularAcceleration2


def compute_angular_velocity_derivative_point1(m1, m2, g, length1, length2, angle1, angle2, w1, w2):
    u = 1 + m1/m2
    delta = angle1 - angle2
    return (g*(sin(angle2)*cos(delta) - u*sin(angle1)) - (length2*w2**2 + length1*w1**2*cos(delta))*sin(delta)) / (length1*(u - cos(delta)**2))


def compute_angular_velocity_derivative_point2(m1, m2, g, length1, length2, angle1, angle2, w1, w2):
    u = 1 + m1/m2
    delta = angle1 - angle2
    return (g*u*(sin(angle1)*cos(delta) - sin(angle2)) + (u*length1*w1**2 + length2*w2**2*cos(delta))*sin(delta)) / (length2*(u - cos(delta)**2))
