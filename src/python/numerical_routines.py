from math import *

def compute_double_pendulum_step_euler(m1, m2, g, length1, length2, initialAngle1, initialAngle2, w1, w2, timestep):

    d2theta1 = compute_angular_velocity_derivative_point1(m1, m2, g, length1, length2, w1, w2, initialAngle1, initialAngle2)
    d2theta2 = compute_angular_velocity_derivative_point2(m1, m2, g, length1, length2, w1, w2, initialAngle1, initialAngle2)

    point1NewAngularVelocity = d2theta1*timestep + w1
    point2NewAngularVelocity = d2theta2*timestep + w2

    point1NewAngle = point1NewAngularVelocity*timestep + initialAngle1
    point2NewAngle = point2NewAngularVelocity*timestep + initialAngle2

    return point1NewAngularVelocity, point2NewAngularVelocity, point1NewAngle, point2NewAngle


def compute_double_pendulum_step_rk4(m1, m2, g, length1, length2, initialAngle1, initialAngle2, w1, w2, timestep):

    point1K1, point2K1, point1AngleK1, point2AngleK1 = compute_k_step(m1, m2, g, length1, length2, initialAngle1, initialAngle2, w1, w2, 0, 0, 0, 0, timestep/2)
    point1K2, point2K2, point1AngleK2, point2AngleK2 = compute_k_step(m1, m2, g, length1, length2, initialAngle1, initialAngle2, w1, w2, point1K1, point2K1, point1AngleK1, point2AngleK1, timestep/2)
    point1K3, point2K3, point1AngleK3, point2AngleK3 = compute_k_step(m1, m2, g, length1, length2, initialAngle1, initialAngle2, w1, w2, point1K2, point2K2, point1AngleK2, point2AngleK2, timestep/2)
    point1K4, point2K4, point1AngleK4, point2AngleK4 = compute_k_step(m1, m2, g, length1, length2, initialAngle1, initialAngle2, w1, w2, point1K3, point2K3, point1AngleK3, point2AngleK3, timestep)

    point1K = (point1K1 + 2*point1K2 + 2*point1K3 + point1K4)/6
    point2K = (point2K1 + 2*point2K2 + 2*point2K3 + point2K4)/6
    point1AngleK = (point1AngleK1 + 2*point1AngleK2 + 2*point1AngleK3 + point1AngleK4)/6
    point2AngleK = (point2AngleK1 + 2*point2AngleK2 + 2*point2AngleK3 + point2AngleK4)/6

    point1NewAngularVelocity = point1K*timestep + w1
    point2NewAngularVelocity = point2K*timestep + w2

    point1NewAngle = point1AngleK*timestep + initialAngle1
    point2NewAngle = point2AngleK*timestep + initialAngle2
    
    return point1NewAngularVelocity, point2NewAngularVelocity, point1NewAngle, point2NewAngle


def compute_k_step(m1, m2, g, length1, length2, initialAngle1, initialAngle2, w1, w2, previousPoint1K, previousPoint2K, previousPoint1AngleK, previousPoint2AngleK, timestep):
    newAngle1 = initialAngle1 + timestep*previousPoint1AngleK
    newAngle2 = initialAngle2 + timestep*previousPoint2AngleK

    newAngularVelocity1 = w1 + timestep*previousPoint1K
    newAngularVelocity2 = w2 + timestep*previousPoint2K

    point1K = compute_angular_velocity_derivative_point1(m1, m2, g, length1, length2, newAngle1, newAngle2, newAngularVelocity1, newAngularVelocity2)
    point2K = compute_angular_velocity_derivative_point2(m1, m2, g, length1, length2, newAngle1, newAngle2, newAngularVelocity1, newAngularVelocity2)

    point1AngleK = w1 + timestep*previousPoint1K
    point2AngleK = w2 + timestep*previousPoint2K

    return point1K, point2K, point1AngleK, point2AngleK


def compute_angular_velocity_derivative_point1(m1, m2, g, length1, length2, angle1, angle2, w1, w2):
    u = 1 + m1/m2
    delta = angle1 - angle2
    return (g*(sin(angle2)*cos(delta) - u*sin(angle1)) - (length2*w2**2 + length1*w1**2*cos(delta))*sin(delta)) / (length1*(u - cos(delta)**2))


def compute_angular_velocity_derivative_point2(m1, m2, g, length1, length2, angle1, angle2, w1, w2):
    u = 1 + m1/m2
    delta = angle1 - angle2
    return (g*u*(sin(angle1)*cos(delta) - sin(angle2)) + (u*length1*w1**2 + length2*w2**2*cos(delta))*sin(delta)) / (length2*(u - cos(delta)**2))
