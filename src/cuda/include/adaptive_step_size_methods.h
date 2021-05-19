#include <simulation_methods.h>
#include <butcher_tableaus.h>


typedef struct AdaptiveStepSizeResult {
    PendulumState pendulumState;
    FloatType timeStepUsedInCalculation;
    FloatType newTimeStep;
} AdaptiveStepSizeResult;


__device__ void compute_step(PendulumState pendulumState,
                             FloatType u,
                             FloatType length1, FloatType length2,
                             FloatType g,
                             FloatType kList[7][4],
                             int butcherTableauRow,
                             FloatType timeStep) {

    // Compute the new pendulum state using Forward Euler using every k element.
    FloatType kSums[] = {0,0,0,0};
    int startingButcherTableauIndex = butcherTableauRow*(butcherTableauRow - 1)/2;
    for (int i = 0; i < butcherTableauRow; i++) {
        for (int j = 0; j < 4; j++) {
            kSums[j] += kList[i][j]*butcherTableau[startingButcherTableauIndex + i];
        }
    }

    PendulumState newPendulumState;
    newPendulumState.angle1 = pendulumState.angle1 + timeStep*kSums[0];
    newPendulumState.angle2 = pendulumState.angle2 + timeStep*kSums[1];
    newPendulumState.angularVelocity1 = pendulumState.angularVelocity1 + timeStep*kSums[2];
    newPendulumState.angularVelocity2 = pendulumState.angularVelocity2 + timeStep*kSums[3];

    // Compute the accelerations at the new pendulum state.
    AccelerationResults accelerationResults = compute_accelerations(newPendulumState, u, length1, length2, g);

    // Return the computed derivatives of position and velocity.
    kList[butcherTableauRow][0] = newPendulumState.angularVelocity1;
    kList[butcherTableauRow][1] = newPendulumState.angularVelocity2;
    kList[butcherTableauRow][2] = accelerationResults.acceleration1;
    kList[butcherTableauRow][3] = accelerationResults.acceleration2;
}


__device__ void compute_all_steps_fehlberg_87(PendulumState pendulumState,
                                              FloatType u,
                                              FloatType length1, FloatType length2,
                                              FloatType g,
                                              FloatType kList[13][4],
                                              FloatType timeStep) {

    // 1st k
    AccelerationResults accelerationResults = compute_accelerations(pendulumState, u, length1, length2, g);
    kList[0][0] = pendulumState.angularVelocity1;
    kList[0][1] = pendulumState.angularVelocity2;
    kList[0][2] = accelerationResults.acceleration1;
    kList[0][3] = accelerationResults.acceleration2;

    // 2nd k
    FloatType kSums[4];
    for (int i = 0; i < 4; i++) {
        kSums[i] = kList[0][i]*butcherTableau[0];
    }
    PendulumState newPendulumState;
    newPendulumState.angle1 = pendulumState.angle1 + timeStep*kSums[0];
    newPendulumState.angle2 = pendulumState.angle2 + timeStep*kSums[1];
    newPendulumState.angularVelocity1 = pendulumState.angularVelocity1 + timeStep*kSums[2];
    newPendulumState.angularVelocity2 = pendulumState.angularVelocity2 + timeStep*kSums[3];
    accelerationResults = compute_accelerations(newPendulumState, u, length1, length2, g);
    kList[1][0] = newPendulumState.angularVelocity1;
    kList[1][1] = newPendulumState.angularVelocity2;
    kList[1][2] = accelerationResults.acceleration1;
    kList[1][3] = accelerationResults.acceleration2;

    // 3rd k
    for (int i = 0; i < 4; i++) {
        kSums[i] = kList[0][i]*butcherTableau[1] + kList[1][i]*butcherTableau[2];
    }
    newPendulumState.angle1 = pendulumState.angle1 + timeStep*kSums[0];
    newPendulumState.angle2 = pendulumState.angle2 + timeStep*kSums[1];
    newPendulumState.angularVelocity1 = pendulumState.angularVelocity1 + timeStep*kSums[2];
    newPendulumState.angularVelocity2 = pendulumState.angularVelocity2 + timeStep*kSums[3];
    accelerationResults = compute_accelerations(newPendulumState, u, length1, length2, g);
    kList[2][0] = newPendulumState.angularVelocity1;
    kList[2][1] = newPendulumState.angularVelocity2;
    kList[2][2] = accelerationResults.acceleration1;
    kList[2][3] = accelerationResults.acceleration2;

    // 4th k
    for (int i = 0; i < 4; i++) {
        kSums[i] = kList[0][i]*butcherTableau[3] + kList[2][i]*butcherTableau[5];
    }
    newPendulumState.angle1 = pendulumState.angle1 + timeStep*kSums[0];
    newPendulumState.angle2 = pendulumState.angle2 + timeStep*kSums[1];
    newPendulumState.angularVelocity1 = pendulumState.angularVelocity1 + timeStep*kSums[2];
    newPendulumState.angularVelocity2 = pendulumState.angularVelocity2 + timeStep*kSums[3];
    accelerationResults = compute_accelerations(newPendulumState, u, length1, length2, g);
    kList[3][0] = newPendulumState.angularVelocity1;
    kList[3][1] = newPendulumState.angularVelocity2;
    kList[3][2] = accelerationResults.acceleration1;
    kList[3][3] = accelerationResults.acceleration2;

    // 5th k
    for (int i = 0; i < 4; i++) {
        kSums[i] = kList[0][i]*butcherTableau[6] + kList[2][i]*butcherTableau[8] + kList[3][i]*butcherTableau[9];
    }
    newPendulumState.angle1 = pendulumState.angle1 + timeStep*kSums[0];
    newPendulumState.angle2 = pendulumState.angle2 + timeStep*kSums[1];
    newPendulumState.angularVelocity1 = pendulumState.angularVelocity1 + timeStep*kSums[2];
    newPendulumState.angularVelocity2 = pendulumState.angularVelocity2 + timeStep*kSums[3];
    accelerationResults = compute_accelerations(newPendulumState, u, length1, length2, g);
    kList[4][0] = newPendulumState.angularVelocity1;
    kList[4][1] = newPendulumState.angularVelocity2;
    kList[4][2] = accelerationResults.acceleration1;
    kList[4][3] = accelerationResults.acceleration2;

    // 6th k
    for (int i = 0; i < 4; i++) {
        kSums[i] = kList[0][i]*butcherTableau[10] + kList[3][i]*butcherTableau[13] + kList[4][i]*butcherTableau[14];
    }
    newPendulumState.angle1 = pendulumState.angle1 + timeStep*kSums[0];
    newPendulumState.angle2 = pendulumState.angle2 + timeStep*kSums[1];
    newPendulumState.angularVelocity1 = pendulumState.angularVelocity1 + timeStep*kSums[2];
    newPendulumState.angularVelocity2 = pendulumState.angularVelocity2 + timeStep*kSums[3];
    accelerationResults = compute_accelerations(newPendulumState, u, length1, length2, g);
    kList[5][0] = newPendulumState.angularVelocity1;
    kList[5][1] = newPendulumState.angularVelocity2;
    kList[5][2] = accelerationResults.acceleration1;
    kList[5][3] = accelerationResults.acceleration2;

    // 7th k
    for (int i = 0; i < 4; i++) {
        kSums[i] = kList[0][i]*butcherTableau[15] + kList[3][i]*butcherTableau[18] + kList[4][i]*butcherTableau[19] + kList[5][i]*butcherTableau[20];
    }
    newPendulumState.angle1 = pendulumState.angle1 + timeStep*kSums[0];
    newPendulumState.angle2 = pendulumState.angle2 + timeStep*kSums[1];
    newPendulumState.angularVelocity1 = pendulumState.angularVelocity1 + timeStep*kSums[2];
    newPendulumState.angularVelocity2 = pendulumState.angularVelocity2 + timeStep*kSums[3];
    accelerationResults = compute_accelerations(newPendulumState, u, length1, length2, g);
    kList[6][0] = newPendulumState.angularVelocity1;
    kList[6][1] = newPendulumState.angularVelocity2;
    kList[6][2] = accelerationResults.acceleration1;
    kList[6][3] = accelerationResults.acceleration2;

    // 8th k
    for (int i = 0; i < 4; i++) {
        kSums[i] = kList[0][i]*butcherTableau[21] + kList[4][i]*butcherTableau[25] + kList[5][i]*butcherTableau[26] + kList[6][i]*butcherTableau[27];
    }
    newPendulumState.angle1 = pendulumState.angle1 + timeStep*kSums[0];
    newPendulumState.angle2 = pendulumState.angle2 + timeStep*kSums[1];
    newPendulumState.angularVelocity1 = pendulumState.angularVelocity1 + timeStep*kSums[2];
    newPendulumState.angularVelocity2 = pendulumState.angularVelocity2 + timeStep*kSums[3];
    accelerationResults = compute_accelerations(newPendulumState, u, length1, length2, g);
    kList[7][0] = newPendulumState.angularVelocity1;
    kList[7][1] = newPendulumState.angularVelocity2;
    kList[7][2] = accelerationResults.acceleration1;
    kList[7][3] = accelerationResults.acceleration2;

    // 9th k
    for (int i = 0; i < 4; i++) {
        kSums[i] = kList[0][i]*butcherTableau[28] + kList[3][i]*butcherTableau[31] + kList[4][i]*butcherTableau[32] + kList[5][i]*butcherTableau[33] + kList[6][i]*butcherTableau[34] + kList[7][i]*butcherTableau[35];
    }
    newPendulumState.angle1 = pendulumState.angle1 + timeStep*kSums[0];
    newPendulumState.angle2 = pendulumState.angle2 + timeStep*kSums[1];
    newPendulumState.angularVelocity1 = pendulumState.angularVelocity1 + timeStep*kSums[2];
    newPendulumState.angularVelocity2 = pendulumState.angularVelocity2 + timeStep*kSums[3];
    accelerationResults = compute_accelerations(newPendulumState, u, length1, length2, g);
    kList[8][0] = newPendulumState.angularVelocity1;
    kList[8][1] = newPendulumState.angularVelocity2;
    kList[8][2] = accelerationResults.acceleration1;
    kList[8][3] = accelerationResults.acceleration2;

    // 10th k
    for (int i = 0; i < 4; i++) {
        kSums[i] = kList[0][i]*butcherTableau[36] + kList[3][i]*butcherTableau[39] + kList[4][i]*butcherTableau[40] + kList[5][i]*butcherTableau[41] + kList[6][i]*butcherTableau[42] + kList[7][i]*butcherTableau[43] + kList[8][i]*butcherTableau[44];
    }
    newPendulumState.angle1 = pendulumState.angle1 + timeStep*kSums[0];
    newPendulumState.angle2 = pendulumState.angle2 + timeStep*kSums[1];
    newPendulumState.angularVelocity1 = pendulumState.angularVelocity1 + timeStep*kSums[2];
    newPendulumState.angularVelocity2 = pendulumState.angularVelocity2 + timeStep*kSums[3];
    accelerationResults = compute_accelerations(newPendulumState, u, length1, length2, g);
    kList[9][0] = newPendulumState.angularVelocity1;
    kList[9][1] = newPendulumState.angularVelocity2;
    kList[9][2] = accelerationResults.acceleration1;
    kList[9][3] = accelerationResults.acceleration2;

    // 11th k
    for (int i = 0; i < 4; i++) {
        kSums[i] = kList[0][i]*butcherTableau[45] + kList[3][i]*butcherTableau[48] + kList[4][i]*butcherTableau[49] + kList[5][i]*butcherTableau[50] + kList[6][i]*butcherTableau[51] + kList[7][i]*butcherTableau[52] + kList[8][i]*butcherTableau[53] + kList[9][i]*butcherTableau[54];
    }
    newPendulumState.angle1 = pendulumState.angle1 + timeStep*kSums[0];
    newPendulumState.angle2 = pendulumState.angle2 + timeStep*kSums[1];
    newPendulumState.angularVelocity1 = pendulumState.angularVelocity1 + timeStep*kSums[2];
    newPendulumState.angularVelocity2 = pendulumState.angularVelocity2 + timeStep*kSums[3];
    accelerationResults = compute_accelerations(newPendulumState, u, length1, length2, g);
    kList[10][0] = newPendulumState.angularVelocity1;
    kList[10][1] = newPendulumState.angularVelocity2;
    kList[10][2] = accelerationResults.acceleration1;
    kList[10][3] = accelerationResults.acceleration2;

    // 12th k
    for (int i = 0; i < 4; i++) {
        kSums[i] = kList[0][i]*butcherTableau[55] + kList[5][i]*butcherTableau[60] + kList[6][i]*butcherTableau[61] + kList[7][i]*butcherTableau[62] + kList[8][i]*butcherTableau[63] + kList[9][i]*butcherTableau[64];
    }
    newPendulumState.angle1 = pendulumState.angle1 + timeStep*kSums[0];
    newPendulumState.angle2 = pendulumState.angle2 + timeStep*kSums[1];
    newPendulumState.angularVelocity1 = pendulumState.angularVelocity1 + timeStep*kSums[2];
    newPendulumState.angularVelocity2 = pendulumState.angularVelocity2 + timeStep*kSums[3];
    accelerationResults = compute_accelerations(newPendulumState, u, length1, length2, g);
    kList[11][0] = newPendulumState.angularVelocity1;
    kList[11][1] = newPendulumState.angularVelocity2;
    kList[11][2] = accelerationResults.acceleration1;
    kList[11][3] = accelerationResults.acceleration2;

    // 13th k
    for (int i = 0; i < 4; i++) {
        kSums[i] = kList[0][i]*butcherTableau[66] + kList[3][i]*butcherTableau[69] + kList[4][i]*butcherTableau[70] + kList[5][i]*butcherTableau[71] + kList[6][i]*butcherTableau[72] + kList[7][i]*butcherTableau[73] + kList[8][i]*butcherTableau[74] + kList[9][i]*butcherTableau[75] + kList[11][i]*butcherTableau[77];
    }
    newPendulumState.angle1 = pendulumState.angle1 + timeStep*kSums[0];
    newPendulumState.angle2 = pendulumState.angle2 + timeStep*kSums[1];
    newPendulumState.angularVelocity1 = pendulumState.angularVelocity1 + timeStep*kSums[2];
    newPendulumState.angularVelocity2 = pendulumState.angularVelocity2 + timeStep*kSums[3];
    accelerationResults = compute_accelerations(newPendulumState, u, length1, length2, g);
    kList[12][0] = newPendulumState.angularVelocity1;
    kList[12][1] = newPendulumState.angularVelocity2;
    kList[12][2] = accelerationResults.acceleration1;
    kList[12][3] = accelerationResults.acceleration2;
}


__device__ AdaptiveStepSizeResult compute_double_pendulum_step_with_adaptive_step_size_method(PendulumState pendulumState,
                                                                                              FloatType u,
                                                                                              FloatType length1, FloatType length2,
                                                                                              FloatType g,
                                                                                              FloatType timeStep, FloatType errorTolerance) {

    // Keep recalculating the step with a smaller time step until the given error tolerance is reached.
    FloatType kList[13][4];
    while(1) {
//      Compute K values.
        #ifdef FEHLBERG_87
        compute_all_steps_fehlberg_87(pendulumState, u, length1, length2, g, kList, timeStep);
        #else
        #if DORMAND_PRINCE_54
        for (int i = 0; i <= 6; i++) {
        #else
        for (int i = 0; i <= 5; i++) {
        #endif
            compute_step(pendulumState, u, length1, length2, g, kList, i, timeStep);
        }
        #endif

        // Compute the new state of the pendulum with 4th and 5th order methods, and compute what the new time step should be.
        PendulumState newPendulumState;
        FloatType* pendulumStateValues = &(pendulumState.angle1);
        FloatType* newPendulumStateValues = &(newPendulumState.angle1);
        bool stepNeedsToBeRecalculated = false;
        FloatType timeStepToUseInRecalculation = 2*timeStep;
        FloatType timeStepToUseInNextStep = 2*timeStep;
        for (int i = 0; i < 4; i++) {
            // Compute the value of the variable after one step using the
            // adaptive step size method specified at compile time.
            #ifdef RKF_45
                FloatType curLowerOrderResult = pendulumStateValues[i] + (rkLowerOrderConstants[0]*kList[0][i] + rkLowerOrderConstants[1]*kList[2][i] + rkLowerOrderConstants[2]*kList[3][i] + rkLowerOrderConstants[3]*kList[4][i])*timeStep;
                FloatType curHigherOrderResult = pendulumStateValues[i] + (rkHigherOrderConstants[0]*kList[0][i] + rkHigherOrderConstants[1]*kList[2][i] + rkHigherOrderConstants[2]*kList[3][i] + rkHigherOrderConstants[3]*kList[4][i] + rkHigherOrderConstants[4]*kList[5][i])*timeStep;
                newPendulumStateValues[i] = curLowerOrderResult;
            #elif CASH_KARP_45
                FloatType curLowerOrderResult = pendulumStateValues[i] + (rkLowerOrderConstants[0]*kList[0][i] + rkLowerOrderConstants[1]*kList[2][i] + rkLowerOrderConstants[2]*kList[3][i] + rkLowerOrderConstants[3]*kList[4][i] + rkLowerOrderConstants[4]*kList[5][i])*timeStep;
                FloatType curHigherOrderResult = pendulumStateValues[i] + (rkHigherOrderConstants[0]*kList[0][i] + rkHigherOrderConstants[1]*kList[2][i] + rkHigherOrderConstants[2]*kList[3][i] + rkHigherOrderConstants[3]*kList[5][i])*timeStep;
                newPendulumStateValues[i] = curLowerOrderResult;
            #elif DORMAND_PRINCE_54
                FloatType curLowerOrderResult = pendulumStateValues[i] + (rkLowerOrderConstants[0]*kList[0][i] + rkLowerOrderConstants[1]*kList[2][i] + rkLowerOrderConstants[2]*kList[3][i] + rkLowerOrderConstants[3]*kList[4][i] + rkLowerOrderConstants[4]*kList[5][i] + rkLowerOrderConstants[5]*kList[6][i])*timeStep;
                FloatType curHigherOrderResult = pendulumStateValues[i] + (rkHigherOrderConstants[0]*kList[0][i] + rkHigherOrderConstants[1]*kList[2][i] + rkHigherOrderConstants[2]*kList[3][i] + rkHigherOrderConstants[3]*kList[4][i] + rkHigherOrderConstants[4]*kList[5][i])*timeStep;
                newPendulumStateValues[i] = curHigherOrderResult;
            #elif FEHLBERG_87
                FloatType curLowerOrderResult = pendulumStateValues[i] + (rkLowerOrderConstants[0]*kList[0][i] + rkLowerOrderConstants[1]*kList[5][i] + rkLowerOrderConstants[2]*kList[6][i] + rkLowerOrderConstants[3]*kList[7][i] + rkLowerOrderConstants[4]*kList[8][i] + rkLowerOrderConstants[5]*kList[9][i] + rkLowerOrderConstants[6]*kList[10][i])*timeStep;
                FloatType curHigherOrderResult = pendulumStateValues[i] + (rkHigherOrderConstants[0]*kList[5][i] + rkHigherOrderConstants[1]*kList[6][i] + rkHigherOrderConstants[2]*kList[7][i] + rkHigherOrderConstants[3]*kList[8][i] + rkHigherOrderConstants[4]*kList[9][i] + rkHigherOrderConstants[5]*kList[11][i] + rkHigherOrderConstants[6]*kList[12][i])*timeStep;
                newPendulumStateValues[i] = curHigherOrderResult;
            #endif

            // Compute what the new time step should be. The smallest new time step computed for the four pendulum state variables is used.
            if (curLowerOrderResult != curHigherOrderResult) {
                FloatType R = abs(curLowerOrderResult - curHigherOrderResult) / timeStep;
                #ifdef DORMAND_PRINCE_54
                    #ifdef FLOAT_32
                    FloatType delta = powf(errorTolerance/(2*R), 1.0/5.0);
                    #else
                    FloatType delta = pow(errorTolerance/(2*R), 1.0/5.0);
                    #endif
                #elif FEHLBERG_87
                    #ifdef FLOAT_32
                    FloatType delta = powf(errorTolerance/(2*R), 1.0/8.0);
                    #else
                    FloatType delta = pow(errorTolerance/(2*R), 1.0/8.0);
                    #endif
                #else
                    #ifdef FLOAT_32
                    FloatType delta = sqrtf(sqrtf(errorTolerance/(2*R)));
                    #else
                    FloatType delta = sqrt(sqrt(errorTolerance/(2*R)));
                    #endif
                #endif

                FloatType curTimeStepToUseInNextStep = delta*timeStep;
                timeStepToUseInNextStep = min(timeStepToUseInNextStep, curTimeStepToUseInNextStep);

                // If R is greater than the error tolerance then recompute the step with a smaller step size.
                if (R > errorTolerance) {
                    stepNeedsToBeRecalculated = true;
                    timeStepToUseInRecalculation = min(timeStepToUseInRecalculation, curTimeStepToUseInNextStep);
                }
            }
        }

        // If the tolerance was met for all of the variables then return the result.
        if (!stepNeedsToBeRecalculated) {
            AdaptiveStepSizeResult result;
            result.pendulumState = newPendulumState;
            result.timeStepUsedInCalculation = timeStep;
            result.newTimeStep = timeStepToUseInNextStep;
            return result;
        }

        timeStep = timeStepToUseInNextStep;
    }
}



