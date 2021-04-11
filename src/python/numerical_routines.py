import enum
import math

from mpmath import *


class SimulationAlgorithm(enum.Enum):
   RK4 = 1
   RKF45 = 2

kScalesList = mp.matrix([[0, 0, 0, 0, 0],
                         [1/4, 0, 0, 0, 0],
                         [3/32, 9/32, 0, 0, 0],
                         [1932/2197, -7200/2197, 7296/2197, 0, 0],
                         [439/216, -8, 3680/513, -845/4104, 0],
                         [-8/27, 2, -3544/2565, 1859/4104, -11/40]]).tolist()


def compute_double_pendulum_step_rk4(m1, m2, g, length1, length2, initialAngle1, initialAngle2, w1, w2, timeStep):
    # Compute the angular velocities and angular accelerations at the four Runge-Kutta points.
    k1 = compute_k_step(m1, m2, g, length1, length2, initialAngle1, initialAngle2, w1, w2, (0, 0, 0, 0), 0)
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

    return point1NewAngle, point2NewAngle, point1NewAngularVelocity, point2NewAngularVelocity


def compute_double_pendulum_step_rkf45(m1, m2, g, length1, length2, initialAngle1, initialAngle2, w1, w2, timeStep, errorTolerance):
    # Compute K values.
    kList = []
    k1 = compute_rkf_step(m1, m2, g, length1, length2, initialAngle1, initialAngle2, w1, w2, kList, kScalesList[0], timeStep)
    kList.append(k1)
    k2 = compute_rkf_step(m1, m2, g, length1, length2, initialAngle1, initialAngle2, w1, w2, kList, kScalesList[1], timeStep)
    kList.append(k2)
    k3 = compute_rkf_step(m1, m2, g, length1, length2, initialAngle1, initialAngle2, w1, w2, kList, kScalesList[2], timeStep)
    kList.append(k3)
    k4 = compute_rkf_step(m1, m2, g, length1, length2, initialAngle1, initialAngle2, w1, w2, kList, kScalesList[3], timeStep)
    kList.append(k4)
    k5 = compute_rkf_step(m1, m2, g, length1, length2, initialAngle1, initialAngle2, w1, w2, kList, kScalesList[4], timeStep)
    kList.append(k5)
    k6 = compute_rkf_step(m1, m2, g, length1, length2, initialAngle1, initialAngle2, w1, w2, kList, kScalesList[5], timeStep)

    # Compute the new state of the pendulum with 4th and 5th order methods, and compute what the new time step should be.
    initialPendulumState = [initialAngle1, initialAngle2, w1, w2]
    newPendulumState = [0]*4
    stepNeedsToBeRecalculated = False
    timeStepToUseInRecalculation = 2*timeStep
    timeStepToUseInNextStep = 2*timeStep
    for i in range(4):
        # Compute the value of the variable after one step with 4th and 5th order methods.
        cur4thOrderResult = initialPendulumState[i] + (25/216*k1[i] + 1408/2565*k3[i] + 2197/4104*k4[i] - 1/5*k5[i])*timeStep
        cur5thOrderResult = initialPendulumState[i] + (16/135*k1[i] + 6656/12825*k3[i] + 28561/56430*k4[i] - 9/50*k5[i] + 2/55*k6[i])*timeStep
        newPendulumState[i] = cur4thOrderResult

        # Compute what the new time step should be. The smallest new time step computed for the four pendulum state variables is used.
        if cur4thOrderResult != cur5thOrderResult:
            R = abs(cur4thOrderResult - cur5thOrderResult) / timeStep
            delta = .84*pow(errorTolerance/R, 1/4)
            curTimeStepToUseInNextStep = delta*timeStep
            timeStepToUseInNextStep = min(timeStepToUseInNextStep, curTimeStepToUseInNextStep)

            # If R is greater than the error tolerance then recompute the step with a smaller step size.
            if R > errorTolerance:
                stepNeedsToBeRecalculated = True
                timeStepToUseInRecalculation = min(timeStepToUseInRecalculation, curTimeStepToUseInNextStep)

    # If the tolerance was not met for one of the variables, then recursively recalculate the step with the
    # smallest time step found above.
    if stepNeedsToBeRecalculated:
        return compute_double_pendulum_step_rkf45(m1, m2, g, length1, length2, initialAngle1, initialAngle2, w1, w2, timeStepToUseInRecalculation, errorTolerance)

    # Create the return value.
    returnValue = list(newPendulumState)
    returnValue.append(timeStep)
    returnValue.append(timeStepToUseInNextStep)
    return tuple(returnValue)


def compute_rkf_step(m1, m2, g, length1, length2, initialAngle1, initialAngle2, w1, w2, kList, kScales, timeStep):
    # Compute the new pendulum state using Forward Euler using every k element.
    kSums = [0,0,0,0]
    for i in range(len(kList)):
        for j in range(len(kList[i])):
            kSums[j] += kList[i][j]*kScales[i]

    newAngle1 = initialAngle1 + timeStep*kSums[0]
    newAngle2 = initialAngle2 + timeStep*kSums[1]
    newAngularVelocity1 = w1 + timeStep*kSums[2]
    newAngularVelocity2 = w2 + timeStep*kSums[3]

    # Compute the accelerations at the new pendulum state.
    angularAcceleration1 = compute_angular_velocity_derivative_point1(m1, m2, g, length1, length2, newAngle1, newAngle2, newAngularVelocity1, newAngularVelocity2)
    angularAcceleration2 = compute_angular_velocity_derivative_point2(m1, m2, g, length1, length2, newAngle1, newAngle2, newAngularVelocity1, newAngularVelocity2)

    return newAngularVelocity1, newAngularVelocity2, angularAcceleration1, angularAcceleration2


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


def get_point_position(origin, angle, pendulumLength):
    x = math.sin(angle)*pendulumLength + origin[0]
    y = -math.cos(angle)*pendulumLength + origin[1]
    return [x, y]


def get_point_position_mp(origin, angle, pendulumLength):
    x = sin(angle)*pendulumLength + origin[0]
    y = -cos(angle)*pendulumLength + origin[1]
    return [x, y]


def get_total_energy_of_pendulum(origin,
                                 point1Angle, point2Angle,
                                 point1AngularVelocity, point2AngularVelocity,
                                 pendulum1Length, pendulum2Length,
                                 point1Mass, point2Mass,
                                 gravity):

    point1Position = get_point_position(origin, point1Angle, pendulum1Length)
    point2Position = get_point_position(point1Position, point2Angle, pendulum2Length)

    # Compute the potential energy of the masses.
    potentialEnergy1 = point1Position[1]*point1Mass*gravity
    potentialEnergy2 = point2Position[1]*point2Mass*gravity

    # Compute the kinetic energy of the first mass.
    point1Velocity = pendulum1Length*point1AngularVelocity
    kineticEnergyPoint1 = .5*point1Mass*point1Velocity**2

    # Compute the kinetic energy of the second mass.
    point1VelocityX = math.cos(point1Angle)*point1Velocity
    point1VelocityY = math.sin(point1Angle)*point1Velocity
    point2LocalVelocity = pendulum2Length*point2AngularVelocity
    point2VelocityX = math.cos(point2Angle)*point2LocalVelocity + point1VelocityX
    point2VelocityY = math.sin(point2Angle)*point2LocalVelocity + point1VelocityY
    point2Velocity = math.sqrt(point2VelocityX**2 + point2VelocityY**2)
    kineticEnergyPoint2 = .5*point2Mass*point2Velocity**2

    totalEnergy = potentialEnergy1 + potentialEnergy2 + kineticEnergyPoint1 + kineticEnergyPoint2

    return totalEnergy


def get_total_energy_of_pendulum_mp(origin,
                                    point1Angle, point2Angle,
                                    point1AngularVelocity, point2AngularVelocity,
                                    pendulum1Length, pendulum2Length,
                                    point1Mass, point2Mass,
                                    gravity):

    point1Position = get_point_position_mp(origin, point1Angle, pendulum1Length)
    point2Position = get_point_position_mp(point1Position, point2Angle, pendulum2Length)

    # Compute the potential energy of the masses.
    potentialEnergy1 = point1Position[1]*point1Mass*gravity
    potentialEnergy2 = point2Position[1]*point2Mass*gravity

    # Compute the kinetic energy of the first mass.
    point1Velocity = pendulum1Length*point1AngularVelocity
    kineticEnergyPoint1 = .5*point1Mass*point1Velocity**2

    # Compute the kinetic energy of the second mass.
    point1VelocityX = cos(point1Angle)*point1Velocity
    point1VelocityY = sin(point1Angle)*point1Velocity
    point2LocalVelocity = pendulum2Length*point2AngularVelocity
    point2VelocityX = cos(point2Angle)*point2LocalVelocity + point1VelocityX
    point2VelocityY = sin(point2Angle)*point2LocalVelocity + point1VelocityY
    point2Velocity = sqrt(point2VelocityX**2 + point2VelocityY**2)
    kineticEnergyPoint2 = .5*point2Mass*point2Velocity**2

    totalEnergy = potentialEnergy1 + potentialEnergy2 + kineticEnergyPoint1 + kineticEnergyPoint2

    return totalEnergy