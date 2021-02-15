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
    FloatType acceleration1;
    FloatType acceleration2;
    FloatType velocity1;
    FloatType velocity2;
} RungeKuttaStepResults;


typedef struct AccelerationResults {
    FloatType acceleration1;
    FloatType acceleration2;
} AccelerationResults;


enum PendulumFlipStatus {NotEnoughEnergy = -1, DidNotFlip = -2};


__device__ AccelerationResults compute_accelerations(FloatType m1, FloatType m2,
                                                     FloatType length1, FloatType length2,
                                                     FloatType angle1, FloatType angle2,
                                                     FloatType w1, FloatType w2,
                                                     FloatType g) {
                                                                
    FloatType u = 1 + m1/m2;
    FloatType delta = angle1 - angle2;

    AccelerationResults results;
    results.acceleration1 = (g*(sin(angle2)*cos(delta) - u*sin(angle1)) - (length2*pow(w2, 2) + length1*pow(w1, 2)*cos(delta))*sin(delta)) / (length1*(u - pow(cos(delta), 2)));
    results.acceleration2 = (g*u*(sin(angle1)*cos(delta) - sin(angle2)) + (u*length1*pow(w1, 2) + length2*pow(w2, 2)*cos(delta))*sin(delta)) / (length2*(u - pow(cos(delta), 2)));

    return results;
}
 
 
__device__ RungeKuttaStepResults compute_rk_step(FloatType m1, FloatType m2,
                                                 FloatType length1, FloatType length2,
                                                 FloatType angle1, FloatType angle2,
                                                 FloatType w1, FloatType w2,
                                                 RungeKuttaStepResults previousRungeKuttaStepResults,
                                                 FloatType g,
                                                 FloatType timeStep) {
                                                     
    FloatType newAngle1 = angle1 + timeStep*previousRungeKuttaStepResults.velocity1;
    FloatType newAngle2 = angle2 + timeStep*previousRungeKuttaStepResults.velocity2;

    FloatType newAngularVelocity1 = w1 + timeStep*previousRungeKuttaStepResults.acceleration1;
    FloatType newAngularVelocity2 = w2 + timeStep*previousRungeKuttaStepResults.acceleration2;

    AccelerationResults accelerationResults = compute_accelerations(m1, m2, length1, length2, newAngle1, newAngle2, newAngularVelocity1, newAngularVelocity2, g);
    FloatType newVelocity1 = w1 + timeStep*previousRungeKuttaStepResults.acceleration1;
    FloatType newVelocity2 = w2 + timeStep*previousRungeKuttaStepResults.acceleration2;

    RungeKuttaStepResults newRungeKuttaStepResults;
    newRungeKuttaStepResults.acceleration1 = accelerationResults.acceleration1;
    newRungeKuttaStepResults.acceleration2 = accelerationResults.acceleration2;
    newRungeKuttaStepResults.velocity1 = newVelocity1;
    newRungeKuttaStepResults.velocity2 = newVelocity2;
    
    return newRungeKuttaStepResults;
}        


__device__ PendulumState compute_double_pendulum_step_rk4(FloatType m1, FloatType m2,
                                                          FloatType length1, FloatType length2,
                                                          FloatType angle1, FloatType angle2,
                                                          FloatType w1, FloatType w2,
                                                          FloatType g,
                                                          FloatType timeStep) {
                                                              
    // Compute the four steps of the classical Runge-Kutta 4th order algorithm.
    RungeKuttaStepResults k1 = compute_rk_step(m1, m2, length1, length2, angle1, angle2, w1, w2, {0, 0, 0, 0}, g, timeStep/2);
    RungeKuttaStepResults k2 = compute_rk_step(m1, m2, length1, length2, angle1, angle2, w1, w2, k1, g, timeStep/2);
    RungeKuttaStepResults k3 = compute_rk_step(m1, m2, length1, length2, angle1, angle2, w1, w2, k2, g, timeStep/2);
    RungeKuttaStepResults k4 = compute_rk_step(m1, m2, length1, length2, angle1, angle2, w1, w2, k3, g, timeStep);
    
    // Combine the results of the Runge-Kutta steps.
    FloatType velocity1 = (k1.velocity1 + 2*k2.velocity1 + 2*k3.velocity1 + k4.velocity1)/6;
    FloatType velocity2 = (k1.velocity2 + 2*k2.velocity2 + 2*k3.velocity2 + k4.velocity2)/6;
    FloatType acceleration1 = (k1.acceleration1 + 2*k2.acceleration1 + 2*k3.acceleration1 + k4.acceleration1)/6;
    FloatType acceleration2 = (k1.acceleration2 + 2*k2.acceleration2 + 2*k3.acceleration2 + k4.acceleration2)/6;
    
    // Compute the new state of the pendulum.
    FloatType point1NewAngularVelocity = acceleration1*timeStep + w1;
    FloatType point2NewAngularVelocity = acceleration2*timeStep + w2;
    FloatType point1NewAngle = velocity1*timeStep + angle1;
    FloatType point2NewAngle = velocity2*timeStep + angle2;
    
    // Return the new state of the pendulum.
    PendulumState newPendulumState;
    newPendulumState.angle1 = point1NewAngle;
    newPendulumState.angle2 = point2NewAngle;
    newPendulumState.angularVelocity1 = point1NewAngularVelocity;
    newPendulumState.angularVelocity2 = point2NewAngularVelocity;
    
    return newPendulumState;
}


__global__ void compute_double_pendulum_fractal_steps_till_flip_from_initial_states(FloatType point1Mass, FloatType point2Mass,
                                                                                    FloatType pendulum1Length, FloatType pendulum2Length,
                                                                                    FloatType gravity,
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
                Point point1Position = get_point_position({0,0}, initialPendulumState.angle1, pendulum1Length);
                Point point2Position = get_point_position(point1Position, initialPendulumState.angle2, pendulum2Length);
                FloatType potentialEnergy1 = point1Position.y*point1Mass*gravity;
                FloatType potentialEnergy2 = point2Position.y*point2Mass*gravity;
                FloatType totalPotentialEnergy = potentialEnergy1 + potentialEnergy2;

                FloatType minimumEnergyNeededForFlip = point1Mass*pendulum1Length*gravity + point2Mass*(pendulum1Length - pendulum2Length)*gravity;
                if (totalPotentialEnergy < minimumEnergyNeededForFlip) {
                    numTimeStepsTillFlip[pixelIndex] = NotEnoughEnergy;
                    continue;
                }
            }

            // Otherwise skip the pendulum if the number of current time steps at the current pendulum is -1, indicating
            // it originally didn't have enough energy to flip, or the pendulum already flipped.
            else if (numTimeStepsTillFlip[pixelIndex] == NotEnoughEnergy ||
                     numTimeStepsTillFlip[pixelIndex] != DidNotFlip) {
                continue;
            }

            // Simulate the pendulum until it flips or time runs out.
            PendulumState pendulumState = initialPendulumState;
            Point point1OriginalPosition = get_point_position({0,0}, pendulumState.angle1, pendulum1Length);
            int numberOfTimeStepsExecuted = numberOfTimeStepsAlreadyExecuted;
            bool pendulumFlipped = false;
            while (numberOfTimeStepsExecuted < maxNumberOfTimeStepsToSeeIfPendulumFlips) {
                pendulumState = compute_double_pendulum_step_rk4(point1Mass, point2Mass,
                                                                 pendulum1Length, pendulum2Length,
                                                                 pendulumState.angle1, pendulumState.angle2,
                                                                 pendulumState.angularVelocity1, pendulumState.angularVelocity2,
                                                                 gravity,
                                                                 timeStep);
                numberOfTimeStepsExecuted++;

                // Check to see if the first mass flipped.
                Point point1CurrentPosition = get_point_position({0,0}, pendulumState.angle1, pendulum1Length);
                if (point1CurrentPosition.x*point1OriginalPosition.x < 0 && point1CurrentPosition.y > 0) {
                    pendulumFlipped = true;
                    break;
                }
                point1OriginalPosition = point1CurrentPosition;
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
            if (timeStepsTillFlip == NotEnoughEnergy || timeStepsTillFlip == DidNotFlip) {
                timeTillFlipMs = 0;
            }
            for (int i = 0; i < 3; i++) {
                colors[pixelIndex + i*area] = lroundf(abs(sin(1.0/255 * CUDART_PI_F * timeTillFlipMs * colorScales[i] * shift)) * 255);
            }
        }
    }
}