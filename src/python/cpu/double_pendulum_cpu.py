import time

from mpmath import *

from numerical_routines import compute_double_pendulum_step_rk4, compute_double_pendulum_step_with_adaptive_step_size_method, get_point_position_mp, get_total_energy_of_pendulum_mp, SimulationAlgorithm, ADAPTIVE_TIME_STEP_METHODS

# Pendulum parameters
gravity = mpf(1)
point1Mass = mpf(1)
point2Mass = mpf(1)
pendulum1Length = mpf(1)
pendulum2Length = mpf(1)
point1AngularVelocity = mpf(0)
point2AngularVelocity = mpf(0)

# Simulation parameters
# mp.prec = 24
mp.prec = 53
# mp.prec = 113

point1Angle = mpf((-3.371910665006095 - -3.396454357612266) / 2 + -3.396454357612266) # center of stable area
point2Angle = mpf((1.925992646191392 - 1.901448953585222) / 2 + 1.901448953585222) # center of stable area
# point1Angle = mpf((-3.39507991082632 - -3.3828080645232346) / 2 + -3.3828080645232346) # edge of stable area
# point2Angle = mpf((1.907830313662826 - 1.9201021599659112) / 2 + 1.9201021599659112) # edge of stable area
# point1Angle = mpf(-3.264360)
# point2Angle = mpf(1.446579)

origin = [mpf(0), mpf(0)]
timeStep = mpf(.01/2**2)
printCurrentStateEveryNSeconds = 2**10
maxTimeToSeeIfPendulumFlips = mpf(2**6)
errorTolerance = mpf(9e-15)
# simulationAlgorithm = SimulationAlgorithm.RK4
# simulationAlgorithm = SimulationAlgorithm.RKF45
# simulationAlgorithm = SimulationAlgorithm.CASH_KARP
# simulationAlgorithm = SimulationAlgorithm.DORMAND_PRINCE_54
simulationAlgorithm = SimulationAlgorithm.VERNER_65


# Print simulation info.
print('algorithm = ' + str(simulationAlgorithm.name))
print('binary precision = ' + str(mp.prec))

if simulationAlgorithm in ADAPTIVE_TIME_STEP_METHODS:
    print('error tolerance = ' + str(errorTolerance))

elif simulationAlgorithm is SimulationAlgorithm.RK_4:
    print('time step = ' + str(timeStep))


# Initialization
initialTotalEnergy = get_total_energy_of_pendulum_mp(origin,
                                                     point1Angle, point2Angle,
                                                     point1AngularVelocity, point2AngularVelocity,
                                                     pendulum1Length, pendulum2Length,
                                                     point1Mass, point2Mass,
                                                     gravity)

point1OriginalPosition = get_point_position_mp(origin, point1Angle, pendulum1Length)
elapsedTime = mpf(0)

# Simulate the pendulum.
start = time.time()
timeSinceLastReport = time.time()
while elapsedTime < maxTimeToSeeIfPendulumFlips:
    # RK4
    if simulationAlgorithm is SimulationAlgorithm.RK_4:
        point1Angle, \
        point2Angle, \
        point1AngularVelocity, \
        point2AngularVelocity = compute_double_pendulum_step_rk4(point1Mass, point2Mass,
                                                                 gravity,
                                                                 pendulum1Length, pendulum2Length,
                                                                 point1Angle, point2Angle,
                                                                 point1AngularVelocity, point2AngularVelocity,
                                                                 timeStep)
        elapsedTime += timeStep

    # RKF45
    elif simulationAlgorithm in ADAPTIVE_TIME_STEP_METHODS:
        point1Angle, \
        point2Angle, \
        point1AngularVelocity, \
        point2AngularVelocity, \
        timeStepUsedInCalculation, \
        newTimeStep = compute_double_pendulum_step_with_adaptive_step_size_method(point1Mass, point2Mass,
                                                                                  gravity,
                                                                                  pendulum1Length, pendulum2Length,
                                                                                  point1Angle, point2Angle,
                                                                                  point1AngularVelocity, point2AngularVelocity,
                                                                                  timeStep,
                                                                                  errorTolerance,
                                                                                  simulationAlgorithm)

        elapsedTime += timeStepUsedInCalculation
        timeStep = newTimeStep

    # Stop if the pendulum flipped.
    point1CurrentPosition = get_point_position_mp(origin, point1Angle, pendulum1Length)
    if point1CurrentPosition[0]*point1OriginalPosition[0] < 0 and point1CurrentPosition[1] > 0:
        print('the pendulum flipped')
        break
    point1OriginalPosition = point1CurrentPosition

    # Print the current pendulum state at regular intervals.
    if time.time() - timeSinceLastReport > printCurrentStateEveryNSeconds:
        print('simulation time elapsed seconds = ' + str(elapsedTime))
        print('wall time elapsed seconds = ' + str(time.time() - start))
        print('point1Angle = ' + str(point1Angle))
        print('point2Angle = ' + str(point2Angle))
        print('point1AngularVelocity = ' + str(point1AngularVelocity))
        print('point2AngularVelocity = ' + str(point2AngularVelocity))
        print('initial total energy = ' + str(initialTotalEnergy))
        print('current time step = ' + str(timeStep))
        currentTotalEnergy = get_total_energy_of_pendulum_mp(origin,
                                                             point1Angle, point2Angle,
                                                             point1AngularVelocity, point2AngularVelocity,
                                                             pendulum1Length, pendulum2Length,
                                                             point1Mass, point2Mass,
                                                             gravity)

        print('current total energy = ' + str(currentTotalEnergy))
        print('energy error = ' + str(abs(currentTotalEnergy - initialTotalEnergy)))
        print('')
        timeSinceLastReport = time.time()



# Print the results.
finalTotalEnergy = get_total_energy_of_pendulum_mp(origin,
                                                   point1Angle, point2Angle,
                                                   point1AngularVelocity, point2AngularVelocity,
                                                   pendulum1Length, pendulum2Length,
                                                   point1Mass, point2Mass,
                                                   gravity)


print('final energy error = ' + str(abs(initialTotalEnergy - finalTotalEnergy)))
print('elapsed simulation time = ' + str(elapsedTime))
print('computation time seconds = ' + str(time.time() - start))

