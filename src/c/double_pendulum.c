#define _GNU_SOURCE
#include <stdio.h>
#include <float_type.h>
#include <math.h>
#include <point.h>
#include <util.h>
#include <time.h>
#include <stdbool.h>

#define TAU (2*M_PI)

typedef struct PendulumState {
    FloatType angle1;
    FloatType angle2;
    FloatType angularVelocity1;
    FloatType angularVelocity2;
} PendulumState;


typedef struct RungeKuttaStepResults {
    FloatType velocity1;
    FloatType velocity2;
    FloatType acceleration1;
    FloatType acceleration2;
} RungeKuttaStepResults;


typedef struct AccelerationResults {
    FloatType acceleration1;
    FloatType acceleration2;
} AccelerationResults;


AccelerationResults compute_accelerations(PendulumState pendulumState,
                                          FloatType u,
                                          FloatType length1, FloatType length2,
                                          FloatType g) {

    // Store calculations done multiple times in variables for performance.
    FloatType delta = pendulumState.angle1 - pendulumState.angle2;
    FloatType sinDelta;
    FloatType cosDelta;
    sincos(delta, &sinDelta, &cosDelta);
    FloatType sinAngle1 = sin(pendulumState.angle1);
    FloatType sinAngle2 = sin(pendulumState.angle2);
    FloatType angularVelocity1Squared = pendulumState.angularVelocity1*pendulumState.angularVelocity1;
    FloatType angularVelocity2Squared = pendulumState.angularVelocity2*pendulumState.angularVelocity2;
    FloatType uMinusCosDeltaSquared = u - (cosDelta*cosDelta);

    // Compute the accelerations.
    AccelerationResults results;
    results.acceleration1 = (g*(sinAngle2*cosDelta - u*sinAngle1) - (length2*angularVelocity2Squared + length1*angularVelocity1Squared*cosDelta)*sinDelta) / (length1*uMinusCosDeltaSquared);
    results.acceleration2 = (g*u*(sinAngle1*cosDelta - sinAngle2) + (u*length1*angularVelocity1Squared + length2*angularVelocity2Squared*cosDelta)*sinDelta) / (length2*uMinusCosDeltaSquared);

    return results;
}


RungeKuttaStepResults compute_rk_step(PendulumState pendulumState,
                                      RungeKuttaStepResults previousRungeKuttaStepResults,
                                      FloatType u,
                                      FloatType length1, FloatType length2,
                                      FloatType g,
                                      FloatType timeStep) {

    // Compute the new pendulum state using Forward Euler.
    PendulumState newPendulumState;
    newPendulumState.angle1 = pendulumState.angle1 + timeStep*previousRungeKuttaStepResults.velocity1;
    newPendulumState.angle2 = pendulumState.angle2 + timeStep*previousRungeKuttaStepResults.velocity2;
    newPendulumState.angularVelocity1 = pendulumState.angularVelocity1 + timeStep*previousRungeKuttaStepResults.acceleration1;
    newPendulumState.angularVelocity2 = pendulumState.angularVelocity2 + timeStep*previousRungeKuttaStepResults.acceleration2;

    // Compute the accelerations at the new pendulum state.
    AccelerationResults accelerationResults = compute_accelerations(newPendulumState, u, length1, length2, g);

    // Return the computed derivatives of position and velocity.
    RungeKuttaStepResults newRungeKuttaStepResults;
    newRungeKuttaStepResults.velocity1 = newPendulumState.angularVelocity1;
    newRungeKuttaStepResults.velocity2 = newPendulumState.angularVelocity2;
    newRungeKuttaStepResults.acceleration1 = accelerationResults.acceleration1;
    newRungeKuttaStepResults.acceleration2 = accelerationResults.acceleration2;

    return newRungeKuttaStepResults;
}


PendulumState compute_double_pendulum_step_rk4(PendulumState pendulumState,
                                               FloatType u,
                                               FloatType length1, FloatType length2,
                                               FloatType g,
                                               FloatType timeStep) {

    // Compute the four steps of the classical Runge-Kutta 4th order algorithm.
    RungeKuttaStepResults k1 = compute_rk_step(pendulumState, (RungeKuttaStepResults){0, 0, 0, 0}, u, length1, length2, g, timeStep/2);
    RungeKuttaStepResults k2 = compute_rk_step(pendulumState, k1, u, length1, length2, g, timeStep/2);
    RungeKuttaStepResults k3 = compute_rk_step(pendulumState, k2, u, length1, length2, g, timeStep/2);
    RungeKuttaStepResults k4 = compute_rk_step(pendulumState, k3, u, length1, length2, g, timeStep);

    // Combine the results of the Runge-Kutta steps.
    FloatType velocity1 = (k1.velocity1 + 2*k2.velocity1 + 2*k3.velocity1 + k4.velocity1)/6;
    FloatType velocity2 = (k1.velocity2 + 2*k2.velocity2 + 2*k3.velocity2 + k4.velocity2)/6;
    FloatType acceleration1 = (k1.acceleration1 + 2*k2.acceleration1 + 2*k3.acceleration1 + k4.acceleration1)/6;
    FloatType acceleration2 = (k1.acceleration2 + 2*k2.acceleration2 + 2*k3.acceleration2 + k4.acceleration2)/6;

    // Compute the new state of the pendulum.
    PendulumState newPendulumState;
    newPendulumState.angle1 = velocity1*timeStep + pendulumState.angle1;
    newPendulumState.angle2 = velocity2*timeStep + pendulumState.angle2;
    newPendulumState.angularVelocity1 = acceleration1*timeStep + pendulumState.angularVelocity1;
    newPendulumState.angularVelocity2 = acceleration2*timeStep + pendulumState.angularVelocity2;

    return newPendulumState;
}


int main() {
    // Configuration
    FloatType m1 = 1;
    FloatType m2 = 1;
    FloatType length1 = 1;
    FloatType length2 = 1;
    FloatType g = 1;
    FloatType timeStep = .01 / pow(2, 3);
    FloatType maxTimeToSeeIfPendulumFlips = pow(2, 19);
    long maxNumberOfTimeStepsToSeeIfPendulumFlips = ceil(maxTimeToSeeIfPendulumFlips / timeStep);
    PendulumState pendulumState;
    pendulumState.angle1 = (-3.39507991082632 - -3.3828080645232346) / 2 + -3.3828080645232346;
    pendulumState.angle2 = (1.907830313662826 - 1.9201021599659112) / 2 + 1.9201021599659112;
    pendulumState.angularVelocity1 = 0;
    pendulumState.angularVelocity2 = 0;
    
    printf("time step = %f\n", timeStep);
    printf("max time to see if pendulum flips in seconds = %f\n", maxTimeToSeeIfPendulumFlips);
    
    // Pre-compute a commonly used value.
    FloatType u = 1 + m1/m2;

    // Simulate the pendulum until it flips or time runs out.
    FloatType originalAngle1 = pendulumState.angle1;
    int numberOfTimeStepsExecuted = 0;
    bool pendulumFlipped = false;
    clock_t start = clock();
    while (numberOfTimeStepsExecuted < maxNumberOfTimeStepsToSeeIfPendulumFlips) {
        // Compute one time step of the pendulum simulation.
        pendulumState = compute_double_pendulum_step_rk4(pendulumState, u, length1, length2, g, timeStep);
        numberOfTimeStepsExecuted++;
        
        // Check to see if the first mass flipped.
        if (floor((pendulumState.angle1 - M_PI) / TAU) != floor((originalAngle1 - M_PI) / TAU)) {
            pendulumFlipped = true;
            break;
        }
        originalAngle1 = pendulumState.angle1;
    }

    // Print the results.
    FloatType totalWallTime = (FloatType)(clock() - start) / (FloatType)CLOCKS_PER_SEC;
    FloatType totalSimulationTime = numberOfTimeStepsExecuted * timeStep;
    
    printf("the pendulum flipped = %s\n", pendulumFlipped ? "true" : "false");
    printf("total simulation time seconds = %f\n", totalSimulationTime);
    printf("total wall time seconds = %f", totalWallTime);
}