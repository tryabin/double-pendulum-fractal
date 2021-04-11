#include <simulation_methods.h>



__device__ RungeKuttaStepResults compute_rk_step(PendulumState pendulumState,
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
    RungeKuttaStepResults rungeKuttaStepResults;
    rungeKuttaStepResults.velocity1 = newPendulumState.angularVelocity1;
    rungeKuttaStepResults.velocity2 = newPendulumState.angularVelocity2;
    rungeKuttaStepResults.acceleration1 = accelerationResults.acceleration1;
    rungeKuttaStepResults.acceleration2 = accelerationResults.acceleration2;

    return rungeKuttaStepResults;
}


__device__ PendulumState compute_double_pendulum_step_rk4(PendulumState pendulumState,
                                                          FloatType u,
                                                          FloatType length1, FloatType length2,
                                                          FloatType g,
                                                          FloatType timeStep) {

    // Compute the four steps of the classical Runge-Kutta 4th order algorithm.
    RungeKuttaStepResults k1 = compute_rk_step(pendulumState, {0, 0, 0, 0}, u, length1, length2, g, timeStep/2);
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


__global__ void compute_double_pendulum_fractal_steps_till_flip_from_initial_states(FloatType m1, FloatType m2,
                                                                                    FloatType length1, FloatType length2,
                                                                                    FloatType g,
                                                                                    FloatType angle1Min, FloatType angle1Max,
                                                                                    FloatType angle2Min, FloatType angle2Max,
                                                                                    PendulumState *pendulumStates,
                                                                                    bool startFromDefaultState,
                                                                                    int numberOfTimeStepsAlreadyExecuted,
                                                                                    int totalNumberOfAnglesToTestX, int totalNumberOfAnglesToTestY,
                                                                                    FloatType timeStep,
                                                                                    int maxNumberOfTimeStepsToSeeIfPendulumFlips,
                                                                                    int *numTimeStepsTillFlip) {

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
                    numTimeStepsTillFlip[pixelIndex] = NotEnoughEnergyToFlip;
                    continue;
                }
            }

            // Otherwise skip the pendulum if the number of current time steps at the current pendulum is -1, indicating
            // it originally didn't have enough energy to flip, or the pendulum already flipped.
            else if (numTimeStepsTillFlip[pixelIndex] == NotEnoughEnergyToFlip ||
                     numTimeStepsTillFlip[pixelIndex] != DidNotFlip) {
                continue;
            }

            // Simulate the pendulum until it flips or time runs out.
            PendulumState pendulumState = initialPendulumState;
            FloatType originalAngle1 = pendulumState.angle1;
            int numberOfTimeStepsExecuted = numberOfTimeStepsAlreadyExecuted;
            bool pendulumFlipped = false;
            while (numberOfTimeStepsExecuted < maxNumberOfTimeStepsToSeeIfPendulumFlips) {
                // Compute one time step of the pendulum simulation.
                pendulumState = compute_double_pendulum_step_rk4(pendulumState, u, length1, length2, g, timeStep);
                numberOfTimeStepsExecuted++;

                // Check to see if the first mass flipped.
                if (floor((pendulumState.angle1 - PI) / TAU) != floor((originalAngle1 - PI) / TAU)) {
                    pendulumFlipped = true;
                    break;
                }
                originalAngle1 = pendulumState.angle1;
            }

            // Set the new number of time steps for the pendulum to flip, and the new pendulum state.
            // Set the number of time steps to -2 if it didn't flip.
            numTimeStepsTillFlip[pixelIndex] = pendulumFlipped ? numberOfTimeStepsExecuted : DidNotFlip;
            pendulumStates[pixelIndex] = pendulumState;
        }
    }
}


__global__ void compute_colors_from_steps_till_flip(int *numTimeStepsTillFlip,
                                                    char *colors,
                                                    int totalNumberOfAnglesToTestX,
                                                    int totalNumberOfAnglesToTestY,
                                                    FloatType timeStep,
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
            int timeStepsTillFlip = numTimeStepsTillFlip[pixelIndex];

            // Compute the color of the sample. Color it black if the pendulum did not flip.
            FloatType timeTillFlipMs = FloatType(timeStepsTillFlip)*timeStep*1000.0;
            if (timeStepsTillFlip == NotEnoughEnergyToFlip || timeStepsTillFlip == DidNotFlip) {
                timeTillFlipMs = 0;
            }
            for (int i = 0; i < 3; i++) {
                colors[pixelIndex + i*area] = lroundf(abs(sin(1.0/255 * PI * timeTillFlipMs * colorScales[i] * shift)) * 255);
            }
        }
    }
}