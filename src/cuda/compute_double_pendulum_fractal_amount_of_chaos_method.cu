#include <adaptive_step_size_methods.h>

#define CHAOTIC (0)

__global__ void compute_double_pendulum_fractal_amount_of_chaos_method(FloatType m1, FloatType m2,
                                                                       FloatType length1, FloatType length2,
                                                                       FloatType g,
                                                                       FloatType angle1Min, FloatType angle1Max,
                                                                       FloatType angle2Min, FloatType angle2Max,
                                                                       PendulumState* pendulumStates,
                                                                       FloatType* amountOfChaos,
                                                                       bool startFromDefaultState,
                                                                       FloatType amountOfTimeAlreadyExecuted,
                                                                       int totalNumberOfAnglesToTestX, int totalNumberOfAnglesToTestY,
                                                                       FloatType timeStep,
                                                                       FloatType errorTolerance,
                                                                       FloatType totalExecutionTime) {

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

            // Skip the current pendulum if it has already been determined to be chaotic.
            if (amountOfChaos[pixelIndex] == CHAOTIC) {
                continue;
            }

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

            // Simulate the pendulum until it the time limit is reached.
            PendulumState pendulumState = initialPendulumState;
            FloatType totalTimeExecuted = amountOfTimeAlreadyExecuted;
            while (totalTimeExecuted < totalExecutionTime) {
                // If the time to simulate will be reached in the next time step, change the time step to simulate
                // exactly the total amount of time to simulate.
                if (totalTimeExecuted + timeStep > totalExecutionTime) {
                    timeStep = totalExecutionTime - totalTimeExecuted;
                }

                // Compute one time step of the pendulum simulation.
                AdaptiveStepSizeResult result = compute_double_pendulum_step_with_adaptive_step_size_method(pendulumState, u, length1, length2, g, timeStep, errorTolerance);
                pendulumState = result.pendulumState;
                totalTimeExecuted += result.timeStepUsedInCalculation;
                timeStep = result.newTimeStep;
            }

            // Set the new pendulum state.
            pendulumStates[pixelIndex] = pendulumState;
        }
    }
}


__global__ void compute_amount_of_chaos(PendulumState* pendulumStates,
                                        FloatType* amountOfChaos,
                                        int totalNumberOfAnglesToTestX,
                                        int totalNumberOfAnglesToTestY,
                                        FloatType differenceCutoff) {

    int stepX = gridDim.x*blockDim.x;
    int stepY =  gridDim.y*blockDim.y;

    int startX = threadIdx.x + blockDim.x*blockIdx.x;
    int startY = threadIdx.y + blockDim.y*blockIdx.y;

    // Compute the color of each pixel.
    for (int x = startX; x < totalNumberOfAnglesToTestX; x += stepX) {
        for (int y = startY; y < totalNumberOfAnglesToTestY; y += stepY) {
            // Skip pixels on the border.
            if (x == 0 || y == 0 || x == totalNumberOfAnglesToTestX - 1 || y == totalNumberOfAnglesToTestY - 1) {
                continue;
            }

            // Skip pixels that have already been determined to be chaotic.
            int pixelIndex = (totalNumberOfAnglesToTestY - y - 1)*totalNumberOfAnglesToTestX + x;
            if (amountOfChaos[pixelIndex] == CHAOTIC) {
                continue;
            }

            // Compute the average absolute difference for angle1 between the current pixel and the
            // four surrounding pixels, up, down, left, right. Skip neighbors that have become chaotic.
            FloatType centralAngle1 = pendulumStates[pixelIndex].angle1;
            FloatType centralAngle2 = pendulumStates[pixelIndex].angle2;
            FloatType averageAbsoluteDifference = 0;

            // Up
            PendulumState curSidePendulum;
            int numberOfChaoticNeighbors = 0;
            int neighborPixelIndex = (totalNumberOfAnglesToTestY - y - 2)*totalNumberOfAnglesToTestX + x;
            if (amountOfChaos[neighborPixelIndex] != CHAOTIC) {
                curSidePendulum = pendulumStates[neighborPixelIndex];
                averageAbsoluteDifference += abs(centralAngle1 - curSidePendulum.angle1) + abs(centralAngle2 - curSidePendulum.angle2);
            }
            else {
                numberOfChaoticNeighbors++;
            }

            // Down
            neighborPixelIndex = (totalNumberOfAnglesToTestY - y)*totalNumberOfAnglesToTestX + x;
            if (amountOfChaos[neighborPixelIndex] != CHAOTIC) {
                curSidePendulum = pendulumStates[neighborPixelIndex];
                averageAbsoluteDifference += abs(centralAngle1 - curSidePendulum.angle1) + abs(centralAngle2 - curSidePendulum.angle2);
            }
            else {
                numberOfChaoticNeighbors++;
            }

            // Left
            neighborPixelIndex = pixelIndex - 1;
            if (amountOfChaos[neighborPixelIndex] != CHAOTIC) {
                curSidePendulum = pendulumStates[neighborPixelIndex];
                averageAbsoluteDifference += abs(centralAngle1 - curSidePendulum.angle1) + abs(centralAngle2 - curSidePendulum.angle2);
            }
            else {
                numberOfChaoticNeighbors++;
            }

            // Right
            neighborPixelIndex = pixelIndex + 1;
            if (amountOfChaos[neighborPixelIndex] != CHAOTIC) {
                curSidePendulum = pendulumStates[neighborPixelIndex];
                averageAbsoluteDifference += abs(centralAngle1 - curSidePendulum.angle1) + abs(centralAngle2 - curSidePendulum.angle2);
            }
            else {
                numberOfChaoticNeighbors++;
            }

            // Compute a quantitative value for the chaos at the current pixel from the average absolute difference.
            // If all the neighbors are chaotic, then set the current pixel as chaotic also.
            if (numberOfChaoticNeighbors == 4) {
                amountOfChaos[pixelIndex] = CHAOTIC;
            }
            else {
                averageAbsoluteDifference /= 4.0;
                FloatType currentChaosValue = averageAbsoluteDifference > differenceCutoff ? CHAOTIC : -log(averageAbsoluteDifference)*10;
                amountOfChaos[pixelIndex] = currentChaosValue;
            }
        }
    }
}



