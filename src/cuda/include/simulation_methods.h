#include <stdio.h>
#include <float_type.h>
#include <math.h>
#include <math_constants.h>
#include <point.h>
#include <util.h>


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


enum PendulumFlipStatus {NotEnoughEnergyToFlip = -1, DidNotFlip = -2};


__device__ AccelerationResults compute_accelerations(PendulumState pendulumState,
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