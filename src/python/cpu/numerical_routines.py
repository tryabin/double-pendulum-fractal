import enum
import math

from mpmath import *


class SimulationAlgorithm(enum.Enum):
   RK_4 = 1
   RKF_45 = 2
   CASH_KARP_45 = 3
   DORMAND_PRINCE_54 = 4
   VERNER_65 = 5

ADAPTIVE_TIME_STEP_METHODS = [SimulationAlgorithm.RKF_45,
                              SimulationAlgorithm.CASH_KARP_45,
                              SimulationAlgorithm.DORMAND_PRINCE_54,
                              SimulationAlgorithm.VERNER_65]

# Runge-Kutta-Felhberg 4(5) Butcher Tableau
rkf45ButcherTableau = mp.matrix([[0, 0, 0, 0, 0],
                                 [1/4, 0, 0, 0, 0],
                                 [3/32, 9/32, 0, 0, 0],
                                 [1932/2197, -7200/2197, 7296/2197, 0, 0],
                                 [439/216, -8, 3680/513, -845/4104, 0],
                                 [-8/27, 2, -3544/2565, 1859/4104, -11/40]]).tolist()
rkf45FourthOrderConstants = mp.matrix([25/216, 1408/2565, 2197/4104, -1/5])
rkf45FifthOrderConstants = mp.matrix([16/135, 6656/12825, 28561/56430, -9/50, 2/55])

# Cash-Karp 4(5) Butcher Tableau
cashKarp45ButcherTableau = mp.matrix([[0, 0, 0, 0, 0],
                                      [1/5, 0, 0, 0, 0],
                                      [3/40, 9/40, 0, 0, 0],
                                      [3/10, -9/10, 6/5, 0, 0],
                                      [-11/54, 5/2, -70/27, 35/27, 0],
                                      [1631/55296, 175/512, 575/13824, 44275/110592, 253/4096]]).tolist()
cashKarp45FourthOrderConstants = mp.matrix([2825/27648, 18575/48384, 13525/55296, 277/14336, 1/4])
cashKarp45FifthOrderConstants = mp.matrix([37/378, 250/621, 125/594, 512/1771])

# Dormand-Prince 5(4) Butcher Tableau
dormandPrince54ButcherTableau = mp.matrix([[0, 0, 0, 0, 0, 0],
                                           [1/5, 0, 0, 0, 0, 0],
                                           [3/40, 9/40, 0, 0, 0, 0],
                                           [44/45, -56/15, 32/9, 0, 0, 0],
                                           [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0],
                                           [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0],
                                           [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]]).tolist()
dormandPrince54FourthOrderConstants = mp.matrix([5179/57600, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])
dormandPrince54FifthOrderConstants = mp.matrix([35/384, 500/1113, 125/192, -2187/6784, 11/84])

# Verner 6(5) Butcher Tableau
verner65ButcherTableau = mp.matrix([[0, 0, 0, 0, 0, 0, 0, 0],
                                           [1/6, 0, 0, 0, 0, 0, 0, 0],
                                           [4/75, 16/75, 0, 0, 0, 0, 0, 0],
                                           [5/6, -8/3, 5/2, 0, 0, 0, 0, 0],
                                           [-165/64, 55/6, -425/64, 85/96, 0, 0, 0, 0],
                                           [12/5, -8, 4015/612, -11/36, 88/255, 0, 0, 0],
                                           [-8263/15000, 124/75, -643/680, -81/250, 2484/10625, 0, 0, 0],
                                           [3501/1720, -300/43, 297275/52632, -319/2322, 24068/84065, 0, 3850/26703, 0]]).tolist()
verner65FifthOrderConstants = mp.matrix([13/160, 2375/5984, 5/16, 12/85, 3/44])
verner65SixthOrderConstants = mp.matrix([3/40, 875/2244, 23/72, 264/1955, 125/11592, 43/616])

# Mappings for the Butcher Tableaus
butcherTableauMap = {SimulationAlgorithm.RKF_45 : rkf45ButcherTableau,
                     SimulationAlgorithm.CASH_KARP_45 : cashKarp45ButcherTableau,
                     SimulationAlgorithm.DORMAND_PRINCE_54 : dormandPrince54ButcherTableau,
                     SimulationAlgorithm.VERNER_65 : verner65ButcherTableau}

lowerOrderConstantsMap = {SimulationAlgorithm.RKF_45 : rkf45FourthOrderConstants,
                          SimulationAlgorithm.CASH_KARP_45 : cashKarp45FourthOrderConstants,
                          SimulationAlgorithm.DORMAND_PRINCE_54 : dormandPrince54FourthOrderConstants,
                          SimulationAlgorithm.VERNER_65 : verner65FifthOrderConstants}

higherOrderConstantsMap = {SimulationAlgorithm.RKF_45 : rkf45FifthOrderConstants,
                           SimulationAlgorithm.CASH_KARP_45 : cashKarp45FifthOrderConstants,
                           SimulationAlgorithm.DORMAND_PRINCE_54 : dormandPrince54FifthOrderConstants,
                           SimulationAlgorithm.VERNER_65 : verner65SixthOrderConstants}


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


def compute_double_pendulum_step_with_adaptive_step_size_method(m1, m2, g, length1, length2, initialAngle1, initialAngle2, w1, w2, timeStep, errorTolerance, algorithm):

    # Use the right Butcher Tableau for the chosen method.
    butcherTableau = butcherTableauMap.get(algorithm)
    lowerOrderConstants = lowerOrderConstantsMap.get(algorithm)
    higherOrderConstants = higherOrderConstantsMap.get(algorithm)

    # for row in butcherTableau:
    #     sum = 0
    #     for value in row:
    #         sum += value
    #     print(sum)

    # Compute K values.
    kList = []
    for row in butcherTableau:
        kValue = compute_adaptive_step_size_step(m1, m2, g, length1, length2, initialAngle1, initialAngle2, w1, w2, kList, row, timeStep)
        kList.append(kValue)

    # Compute the new state of the pendulum with 4th and 5th order methods, and compute what the new time step should be.
    initialPendulumState = [initialAngle1, initialAngle2, w1, w2]
    newPendulumState = [0]*4
    stepNeedsToBeRecalculated = False
    timeStepToUseInRecalculation = 2*timeStep
    timeStepToUseInNextStep = 2*timeStep
    for i in range(4):
        # Compute the value of the variable after one step with 4th and 5th order methods.
        curLowerOrderResult = curHigherOrderResult = None
        if algorithm is SimulationAlgorithm.RKF_45:
            curLowerOrderResult = initialPendulumState[i] + (lowerOrderConstants[0]*kList[0][i] + lowerOrderConstants[1]*kList[2][i] + lowerOrderConstants[2]*kList[3][i] + lowerOrderConstants[3]*kList[4][i])*timeStep
            curHigherOrderResult = initialPendulumState[i] + (higherOrderConstants[0]*kList[0][i] + higherOrderConstants[1]*kList[2][i] + higherOrderConstants[2]*kList[3][i] + higherOrderConstants[3]*kList[4][i] + higherOrderConstants[4]*kList[5][i])*timeStep
            newPendulumState[i] = curLowerOrderResult
        elif algorithm is SimulationAlgorithm.CASH_KARP_45:
            curLowerOrderResult = initialPendulumState[i] + (lowerOrderConstants[0]*kList[0][i] + lowerOrderConstants[1]*kList[2][i] + lowerOrderConstants[2]*kList[3][i] + lowerOrderConstants[3]*kList[4][i] + lowerOrderConstants[4]*kList[5][i])*timeStep
            curHigherOrderResult = initialPendulumState[i] + (higherOrderConstants[0]*kList[0][i] + higherOrderConstants[1]*kList[2][i] + higherOrderConstants[2]*kList[3][i] + higherOrderConstants[3]*kList[5][i])*timeStep
            newPendulumState[i] = curLowerOrderResult
        elif algorithm in {SimulationAlgorithm.DORMAND_PRINCE_54}:
            curLowerOrderResult = initialPendulumState[i] + (lowerOrderConstants[0]*kList[0][i] + lowerOrderConstants[1]*kList[2][i] + lowerOrderConstants[2]*kList[3][i] + lowerOrderConstants[3]*kList[4][i] + lowerOrderConstants[4]*kList[5][i] + lowerOrderConstants[5]*kList[6][i])*timeStep
            if i < 2:
                curHigherOrderResult = kList[6][i + 4]
            else:
                curHigherOrderResult = kList[6][i-2]
            newPendulumState[i] = curHigherOrderResult
        elif algorithm is SimulationAlgorithm.VERNER_65:
            curLowerOrderResult = initialPendulumState[i] + (lowerOrderConstants[0]*kList[0][i] + lowerOrderConstants[1]*kList[2][i] + lowerOrderConstants[2]*kList[3][i] + lowerOrderConstants[3]*kList[4][i] + lowerOrderConstants[4]*kList[5][i])*timeStep
            curHigherOrderResult = initialPendulumState[i] + (higherOrderConstants[0]*kList[0][i] + higherOrderConstants[1]*kList[2][i] + higherOrderConstants[2]*kList[3][i] + higherOrderConstants[3]*kList[4][i] + higherOrderConstants[4]*kList[6][i] + higherOrderConstants[5]*kList[7][i])*timeStep
            newPendulumState[i] = curHigherOrderResult

        # Compute what the new time step should be. The smallest new time step computed for the four pendulum state variables is used.
        if curLowerOrderResult != curHigherOrderResult:
            R = abs(curLowerOrderResult - curHigherOrderResult) / timeStep
            delta = None
            if algorithm in {SimulationAlgorithm.RKF_45, SimulationAlgorithm.CASH_KARP_45}:
                delta = pow(errorTolerance/(2*R), 1/4)
            elif algorithm in {SimulationAlgorithm.DORMAND_PRINCE_54}:
                delta = pow(errorTolerance/(2*R), 1/5)
            elif algorithm in {SimulationAlgorithm.VERNER_65}:
                delta = pow(errorTolerance/(2*R), 1/6)
            curTimeStepToUseInNextStep = delta*timeStep
            timeStepToUseInNextStep = min(timeStepToUseInNextStep, curTimeStepToUseInNextStep)

            # If R is greater than the error tolerance then recompute the step with a smaller step size.
            if R > errorTolerance:
                stepNeedsToBeRecalculated = True
                timeStepToUseInRecalculation = min(timeStepToUseInRecalculation, curTimeStepToUseInNextStep)

    # If the tolerance was not met for one of the variables, then recursively recalculate the step with the
    # smallest time step found above.
    if stepNeedsToBeRecalculated:
        return compute_double_pendulum_step_with_adaptive_step_size_method(m1, m2, g, length1, length2, initialAngle1, initialAngle2, w1, w2, timeStepToUseInRecalculation, errorTolerance, algorithm)

    # Create the return value.
    returnValue = list(newPendulumState)
    returnValue.append(timeStep)
    returnValue.append(timeStepToUseInNextStep)
    return tuple(returnValue)


def compute_adaptive_step_size_step(m1, m2, g, length1, length2, initialAngle1, initialAngle2, w1, w2, kList, butcherTableauRow, timeStep):
    # Compute the new pendulum state using Forward Euler using every k element.
    kSums = [0,0,0,0]
    for i in range(len(kList)):
        for j in range(4):
            kSums[j] += kList[i][j]*butcherTableauRow[i]

    newAngle1 = initialAngle1 + timeStep*kSums[0]
    newAngle2 = initialAngle2 + timeStep*kSums[1]
    newAngularVelocity1 = w1 + timeStep*kSums[2]
    newAngularVelocity2 = w2 + timeStep*kSums[3]

    # Compute the accelerations at the new pendulum state.
    angularAcceleration1 = compute_angular_velocity_derivative_point1(m1, m2, g, length1, length2, newAngle1, newAngle2, newAngularVelocity1, newAngularVelocity2)
    angularAcceleration2 = compute_angular_velocity_derivative_point2(m1, m2, g, length1, length2, newAngle1, newAngle2, newAngularVelocity1, newAngularVelocity2)

    return newAngularVelocity1, newAngularVelocity2, angularAcceleration1, angularAcceleration2, newAngle1, newAngle2


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