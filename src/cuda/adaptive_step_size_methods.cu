#include <simulation_methods.h>

// Runge-Kutta-Fehlberg Butcher tableau constants
#ifdef RKF45
    __constant__ FloatType butcherTableau[6][5] = {{0,0,0,0,0},
                                                      {1.0/4.0,0,0,0,0},
                                                      {3.0/32.0,9.0/32.0,0,0,0},
                                                      {1932.0/2197.0,-7200.0/2197.0,7296.0/2197.0,0,0},
                                                      {439.0/216.0,-8.0,3680.0/513.0,-845.0/4104.0,0},
                                                      {-8.0/27.0,2.0,-3544.0/2565.0,1859.0/4104.0,-11.0/40.0}};
    __constant__ FloatType rkFourthOrderConstants[4] = {25.0/216.0, 1408.0/2565.0, 2197.0/4104.0, -1.0/5.0};
    __constant__ FloatType rkFifthOrderConstants[5] = {16.0/135.0, 6656.0/12825.0, 28561.0/56430.0, -9.0/50.0, 2.0/55.0};

// Cash-Karp Butcher tableau constants
#elif CASH_KARP
    __constant__ FloatType butcherTableau[6][5] = {{0,0,0,0,0},
                                                   {1.0/5.0,0,0,0,0},
                                                   {3.0/40.0,9.0/40.0,0,0,0},
                                                   {3.0/10.0,-9.0/10.0,6.0/5.0,0,0},
                                                   {-11.0/54.0,5.0/2.0,-70.0/27.0,35.0/27.0,0},
                                                   {1631.0/55296.0,175.0/512,575.0/13824.0,44275.0/110592.0,253.0/4096.0}};
    __constant__ FloatType rkFourthOrderConstants[5] = {2825.0/27648.0, 18575.0/48384.0, 13525.0/55296.0, 277.0/14336.0, 1.0/4.0};
    __constant__ FloatType rkFifthOrderConstants[4] = {37.0/378.0, 250.0/621.0, 125.0/594.0, 512.0/1771.0};
#else
    #error Adaptive step-size method not provided
#endif


typedef struct AdaptiveStepSizeResult {
    PendulumState pendulumState;
    FloatType timeStepUsedInCalculation;
    FloatType newTimeStep;
} AdaptiveStepSizeResult;


__device__ void compute_step(PendulumState pendulumState,
                             FloatType u,
                             FloatType length1, FloatType length2,
                             FloatType g,
                             FloatType kList[6][4], int kListSize, FloatType* kScales,
                             FloatType timeStep) {

    // Compute the new pendulum state using Forward Euler using every k element.
    FloatType kSums[4] = {0,0,0,0};
    for (int i = 0; i < kListSize; i++) {
        for (int j = 0; j < 4; j++) {
            kSums[j] += kList[i][j]*kScales[i];
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
    kList[kListSize][0] = newPendulumState.angularVelocity1;
    kList[kListSize][1] = newPendulumState.angularVelocity2;
    kList[kListSize][2] = accelerationResults.acceleration1;
    kList[kListSize][3] = accelerationResults.acceleration2;
}


__device__ AdaptiveStepSizeResult compute_double_pendulum_adaptive_step_size_method(PendulumState pendulumState,
                                                                                    FloatType u,
                                                                                    FloatType length1, FloatType length2,
                                                                                    FloatType g,
                                                                                    FloatType timeStep, FloatType errorTolerance) {

    FloatType kList[6][4] = {{0,0,0,0},
                             {0,0,0,0},
                             {0,0,0,0},
                             {0,0,0,0},
                             {0,0,0,0},
                             {0,0,0,0}};

    // Keep recalculating the step with a smaller time step until the given error tolerance is reached.
    while(1) {
        // Compute K values.
        for (int i = 0; i < 6; i++) {
            compute_step(pendulumState, u, length1, length2, g, kList, i, butcherTableau[i], timeStep);
        }

        // Compute the new state of the pendulum with 4th and 5th order methods, and compute what the new time step should be.
        PendulumState newPendulumState;
        FloatType* pendulumStateValues = &(pendulumState.angle1);
        FloatType* newPendulumStateValues = &(newPendulumState.angle1);
        bool stepNeedsToBeRecalculated = false;
        FloatType timeStepToUseInRecalculation = 2*timeStep;
        FloatType timeStepToUseInNextStep = 2*timeStep;
        for (int i = 0; i < 4; i++) {
            // Compute the value of the variable after one step with 4th and 5th order methods.
            // Different methods are compiled depending on the chosen algorithm.
            #ifdef RKF45
                FloatType cur4thOrderResult = pendulumStateValues[i] + (rkFourthOrderConstants[0]*kList[0][i] + rkFourthOrderConstants[1]*kList[2][i] + rkFourthOrderConstants[2]*kList[3][i] + rkFourthOrderConstants[3]*kList[4][i])*timeStep;
                FloatType cur5thOrderResult = pendulumStateValues[i] + (rkFifthOrderConstants[0]*kList[0][i] + rkFifthOrderConstants[1]*kList[2][i] + rkFifthOrderConstants[2]*kList[3][i] + rkFifthOrderConstants[3]*kList[4][i] + rkFifthOrderConstants[4]*kList[5][i])*timeStep;
                newPendulumStateValues[i] = cur4thOrderResult;
            #elif CASH_KARP
                FloatType cur4thOrderResult = pendulumStateValues[i] + (rkFourthOrderConstants[0]*kList[0][i] + rkFourthOrderConstants[1]*kList[2][i] + rkFourthOrderConstants[2]*kList[3][i] + rkFourthOrderConstants[3]*kList[4][i] + rkFourthOrderConstants[4]*kList[5][i])*timeStep;
                FloatType cur5thOrderResult = pendulumStateValues[i] + (rkFifthOrderConstants[0]*kList[0][i] + rkFifthOrderConstants[1]*kList[2][i] + rkFifthOrderConstants[2]*kList[3][i] + rkFifthOrderConstants[3]*kList[5][i])*timeStep;
                newPendulumStateValues[i] = cur4thOrderResult;
            #endif

            // Compute what the new time step should be. The smallest new time step computed for the four pendulum state variables is used.
            if (cur4thOrderResult != cur5thOrderResult) {
                FloatType R = abs(cur4thOrderResult - cur5thOrderResult) / timeStep;
                FloatType delta = .84*sqrt(sqrt(errorTolerance/R));
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


__global__ void compute_double_pendulum_fractal_time_till_flip_from_initial_states(FloatType m1, FloatType m2,
                                                                                   FloatType length1, FloatType length2,
                                                                                   FloatType g,
                                                                                   FloatType angle1Min, FloatType angle1Max,
                                                                                   FloatType angle2Min, FloatType angle2Max,
                                                                                   PendulumState *pendulumStates,
                                                                                   bool startFromDefaultState,
                                                                                   FloatType amountOfTimeAlreadyExecuted,
                                                                                   int totalNumberOfAnglesToTestX, int totalNumberOfAnglesToTestY,
                                                                                   FloatType timeStep,
                                                                                   FloatType errorTolerance,
                                                                                   FloatType maxTimeToSeeIfPendulumFlips,
                                                                                   FloatType *timeTillFlip) {

    int stepX = gridDim.x*blockDim.x;
    int stepY =  gridDim.y*blockDim.y;

    int startX = threadIdx.x + blockDim.x*blockIdx.x;
    int startY = threadIdx.y + blockDim.y*blockIdx.y;

    // Pre-compute a commonly used value.
    FloatType u = 1 + m1/m2;

    // Simulate the double pendulums.
    for (int x = startX; x < totalNumberOfAnglesToTestX; x += stepX) {
        for (int y = startY; y < totalNumberOfAnglesToTestY; y += stepY) {
            int pixelIndex = (totalNumberOfAnglesToTestY - y - 1)*totalNumberOfAnglesToTestX + x;

            // Set the initial state of the pendulum for the current pixel.
            PendulumState initialPendulumState;
            if (startFromDefaultState) {
                initialPendulumState.angle1 = angle1Min + FloatType(x)*(angle1Max - angle1Min)/FloatType(totalNumberOfAnglesToTestX - 1);
                initialPendulumState.angle2 = angle2Min + FloatType(y)*(angle2Max - angle2Min)/FloatType(totalNumberOfAnglesToTestY - 1);
                initialPendulumState.angularVelocity1 = 0;
                initialPendulumState.angularVelocity2 = 0;
            }
            else {
                initialPendulumState.angle1 = pendulumStates[pixelIndex].angle1;
                initialPendulumState.angle2 = pendulumStates[pixelIndex].angle2;
                initialPendulumState.angularVelocity1 = pendulumStates[pixelIndex].angularVelocity1;
                initialPendulumState.angularVelocity2 = pendulumStates[pixelIndex].angularVelocity2;
            }

            // If not given initial states, skip the current pendulum if it doesn't have enough initial energy to
            // flip the first mass.
            if (startFromDefaultState) {
                Point point1Position = get_point_position({0,0}, initialPendulumState.angle1, length1);
                Point point2Position = get_point_position(point1Position, initialPendulumState.angle2, length2);
                FloatType potentialEnergy1 = point1Position.y*m1*g;
                FloatType potentialEnergy2 = point2Position.y*m2*g;
                FloatType totalPotentialEnergy = potentialEnergy1 + potentialEnergy2;
                FloatType minimumEnergyNeededForFlip = m1*length1*g + m2*(length1 - length2)*g;
                if (totalPotentialEnergy < minimumEnergyNeededForFlip) {
                    timeTillFlip[pixelIndex] = NotEnoughEnergyToFlip;
                    continue;
                }
            }

            // Otherwise skip the pendulum if the number of current time steps at the current pendulum is -1, indicating
            // it originally didn't have enough energy to flip, or the pendulum already flipped.
            else if (timeTillFlip[pixelIndex] == NotEnoughEnergyToFlip ||
                     timeTillFlip[pixelIndex] != DidNotFlip) {
                continue;
            }

            // Simulate the pendulum until it flips or time runs out.
            PendulumState pendulumState = initialPendulumState;
            FloatType originalAngle1 = pendulumState.angle1;
            FloatType totalTimeExecuted = amountOfTimeAlreadyExecuted;
            bool pendulumFlipped = false;
            while (totalTimeExecuted < maxTimeToSeeIfPendulumFlips) {
                // Compute one time step of the pendulum simulation.
                AdaptiveStepSizeResult result = compute_double_pendulum_adaptive_step_size_method(pendulumState, u, length1, length2, g, timeStep, errorTolerance);
                pendulumState = result.pendulumState;
                totalTimeExecuted += result.timeStepUsedInCalculation;
                timeStep = result.newTimeStep;

                // Check to see if the first mass flipped.
                if (floor((pendulumState.angle1 - PI) / TAU) != floor((originalAngle1 - PI) / TAU)) {
                    pendulumFlipped = true;
                    break;
                }
                originalAngle1 = pendulumState.angle1;
            }

            // Set the new number of time steps for the pendulum to flip, and the new pendulum state.
            // Set the number of time steps to -2 if it didn't flip.
            timeTillFlip[pixelIndex] = pendulumFlipped ? totalTimeExecuted : DidNotFlip;
            pendulumStates[pixelIndex] = pendulumState;
        }
    }
}


__global__ void compute_colors_from_time_till_flip(FloatType *timeTillFlip,
                                                   char *colors,
                                                   int totalNumberOfAnglesToTestX,
                                                   int totalNumberOfAnglesToTestY,
                                                   FloatType redScale,
                                                   FloatType greenScale,
                                                   FloatType blueScale,
                                                   FloatType shift) {

    int stepX = gridDim.x*blockDim.x;
    int stepY =  gridDim.y*blockDim.y;

    int startX = threadIdx.x + blockDim.x*blockIdx.x;
    int startY = threadIdx.y + blockDim.y*blockIdx.y;

    int area = totalNumberOfAnglesToTestX*totalNumberOfAnglesToTestY;
    FloatType colorScales[] = {redScale, greenScale, blueScale};

    // Compute the color of each pixel.
    for (int x = startX; x < totalNumberOfAnglesToTestX; x += stepX) {
        for (int y = startY; y < totalNumberOfAnglesToTestY; y += stepY) {
            int pixelIndex = (totalNumberOfAnglesToTestY - y - 1)*totalNumberOfAnglesToTestX + x;
            FloatType curTimeTillFlip = timeTillFlip[pixelIndex];

            // Compute the color of the sample. Color it black if the pendulum did not flip.
            FloatType timeTillFlipMs = curTimeTillFlip*1000.0;
            if (curTimeTillFlip == NotEnoughEnergyToFlip || curTimeTillFlip == DidNotFlip) {
                timeTillFlipMs = 0;
            }
            for (int i = 0; i < 3; i++) {
                colors[pixelIndex + i*area] = lroundf(abs(sin(1.0/255 * PI * timeTillFlipMs * colorScales[i] * shift)) * 255);
            }
        }
    }
}



