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
                                                 FloatType timestep) {
                                                     
    FloatType newAngle1 = angle1 + timestep*previousRungeKuttaStepResults.velocity1;
    FloatType newAngle2 = angle2 + timestep*previousRungeKuttaStepResults.velocity2;

    FloatType newAngularVelocity1 = w1 + timestep*previousRungeKuttaStepResults.acceleration1;
    FloatType newAngularVelocity2 = w2 + timestep*previousRungeKuttaStepResults.acceleration2;

    AccelerationResults accelerationResults = compute_accelerations(m1, m2, length1, length2, newAngle1, newAngle2, newAngularVelocity1, newAngularVelocity2, g);
    FloatType newVelocity1 = w1 + timestep*previousRungeKuttaStepResults.acceleration1;
    FloatType newVelocity2 = w2 + timestep*previousRungeKuttaStepResults.acceleration2;

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
                                                          FloatType timestep) {
                                                              
    // Compute the four steps of the classical Runge-Kutta 4th order algorithm.
    RungeKuttaStepResults k1 = compute_rk_step(m1, m2, length1, length2, angle1, angle2, w1, w2, {0, 0, 0, 0}, g, timestep/2);
    RungeKuttaStepResults k2 = compute_rk_step(m1, m2, length1, length2, angle1, angle2, w1, w2, k1, g, timestep/2);
    RungeKuttaStepResults k3 = compute_rk_step(m1, m2, length1, length2, angle1, angle2, w1, w2, k2, g, timestep/2);
    RungeKuttaStepResults k4 = compute_rk_step(m1, m2, length1, length2, angle1, angle2, w1, w2, k3, g, timestep);
    
    // Combine the results of the Runge-Kutta steps.
    FloatType velocity1 = (k1.velocity1 + 2*k2.velocity1 + 2*k3.velocity1 + k4.velocity1)/6;
    FloatType velocity2 = (k1.velocity2 + 2*k2.velocity2 + 2*k3.velocity2 + k4.velocity2)/6;
    FloatType acceleration1 = (k1.acceleration1 + 2*k2.acceleration1 + 2*k3.acceleration1 + k4.acceleration1)/6;
    FloatType acceleration2 = (k1.acceleration2 + 2*k2.acceleration2 + 2*k3.acceleration2 + k4.acceleration2)/6;
    
    // Compute the new state of the pendulum.
    FloatType point1NewAngularVelocity = acceleration1*timestep + w1;
    FloatType point2NewAngularVelocity = acceleration2*timestep + w2;
    FloatType point1NewAngle = velocity1*timestep + angle1;
    FloatType point2NewAngle = velocity2*timestep + angle2;
    
    // Return the new state of the pendulum.
    PendulumState newPendulumState;
    newPendulumState.angle1 = point1NewAngle;
    newPendulumState.angle2 = point2NewAngle;
    newPendulumState.angularVelocity1 = point1NewAngularVelocity;
    newPendulumState.angularVelocity2 = point2NewAngularVelocity;
    
    return newPendulumState;
}


__global__ void compute_double_pendulum_fractal_image(FloatType point1Mass, FloatType point2Mass,
                                                      FloatType pendulum1Length, FloatType pendulum2Length,
                                                      FloatType gravity,
                                                      FloatType angle1Min, FloatType angle1Max,
                                                      FloatType angle2Min, FloatType angle2Max,
                                                      int numberOfAnglesToTestPerKernelCallRatio,
                                                      int curKernelStartX, int curKernelStartY,
                                                      int totalNumberOfAnglesToTestX, int totalNumberOfAnglesToTestY,
                                                      FloatType timestep,
                                                      FloatType maxTimeToSeeIfPendulumFlips,
                                                      int antiAliasingGridWidth,
                                                      char *colors) {

    int stepX = gridDim.x*blockDim.x;
    int stepY =  gridDim.y*blockDim.y;

    int startX = threadIdx.x + blockDim.x*blockIdx.x;
    int startY = threadIdx.y + blockDim.y*blockIdx.y;
    
    int realStepX = stepX*numberOfAnglesToTestPerKernelCallRatio;
    int realStepY = stepY*numberOfAnglesToTestPerKernelCallRatio;
    int realStartX = startX*numberOfAnglesToTestPerKernelCallRatio + curKernelStartX;
    int realStartY = startY*numberOfAnglesToTestPerKernelCallRatio + curKernelStartY;

    // Pre-compute reused values.
    FloatType distanceBetweenSamples = ((angle1Max - angle1Min) / totalNumberOfAnglesToTestX)/antiAliasingGridWidth;
    FloatType pixelWidth = (angle1Max - angle1Min)/totalNumberOfAnglesToTestX;
    int maxNumberOfTimestepsToSeeIfPendulumFlips = lroundf(maxTimeToSeeIfPendulumFlips / timestep);
    
    // Simulate the double pendulums.
    for (int x = realStartX; x < totalNumberOfAnglesToTestX; x += realStepX) {        
        for (int y = realStartY; y < totalNumberOfAnglesToTestY; y += realStepY) {    
            FloatType angle1 = angle1Min + FloatType(x)*(angle1Max - angle1Min)/FloatType(totalNumberOfAnglesToTestX - 1);
            FloatType angle2 = angle2Min + FloatType(y)*(angle2Max - angle2Min)/FloatType(totalNumberOfAnglesToTestY - 1);
            
            // Perform grid-based supersampling anti-aliasing.
            FloatType colorValues[] = {0, 0, 0};
            for (int i = 0; i < antiAliasingGridWidth; i++) {
                for (int j = 0; j < antiAliasingGridWidth; j++) {
                    FloatType angle1Sample = distanceBetweenSamples*(.5 + FloatType(i)) + angle1 - pixelWidth/2;
                    FloatType angle2Sample = distanceBetweenSamples*(.5 + FloatType(j)) + angle2 - pixelWidth/2;    
                     
                    // Skip the current pendulum if it doesn't have enough initial energy to flip the first mass.
                    Point point1Position = get_point_position({0,0}, angle1Sample, pendulum1Length);
                    Point point2Position = get_point_position(point1Position, angle2Sample, pendulum2Length);
                    FloatType potentialEnergy1 = point1Position.y*point1Mass*gravity;
                    FloatType potentialEnergy2 = point2Position.y*point2Mass*gravity;
                    FloatType totalPotentialEnergy = potentialEnergy1 + potentialEnergy2;

                    FloatType minimumEnergyNeededForFlip = point1Mass*pendulum1Length*gravity + point2Mass*(pendulum1Length - pendulum2Length)*gravity;
                    if (totalPotentialEnergy < minimumEnergyNeededForFlip) {
                        continue;
                    }

                    // Simulate the pendulum until it flips or time runs out.
                    PendulumState pendulumState;
                    pendulumState.angle1 = angle1Sample;
                    pendulumState.angle2 = angle2Sample;
                    pendulumState.angularVelocity1 = 0;
                    pendulumState.angularVelocity2 = 0;
 
                    Point point1OriginalPosition = get_point_position({0,0}, pendulumState.angle1, pendulum1Length);
                    int numberOfTimestepsExecuted = 0;  
                    while (numberOfTimestepsExecuted < maxNumberOfTimestepsToSeeIfPendulumFlips) {                      
                    
                        pendulumState = compute_double_pendulum_step_rk4(point1Mass, point2Mass,
                                                                         pendulum1Length, pendulum2Length,
                                                                         pendulumState.angle1, pendulumState.angle2,
                                                                         pendulumState.angularVelocity1, pendulumState.angularVelocity2,
                                                                         gravity,
                                                                         timestep);
                        
                        // Check to see if the first mass flipped. 
                        Point point1CurrentPosition = get_point_position({0,0}, pendulumState.angle1, pendulum1Length);                
                        if (point1CurrentPosition.x*point1OriginalPosition.x < 0 && point1CurrentPosition.y > 0) {
                            break;
                        }
                        point1OriginalPosition = point1CurrentPosition;
        
                        numberOfTimestepsExecuted++;
                    }
                    
                    // Compute the color of the sample. Color it black if the pendulum did not flip.
                    FloatType timeTillFlipMs = numberOfTimestepsExecuted < maxNumberOfTimestepsToSeeIfPendulumFlips ? FloatType(numberOfTimestepsExecuted)*timestep*1000 : 0;
                    FloatType shift = .11;
                    FloatType colorScales[] = {1.0, 4.0, 7.2};
                    for (int k = 0; k < 3; k++) {
                        colorValues[k] += abs(sin(1.0/255 * CUDART_PI_F * timeTillFlipMs * colorScales[k] * shift)) * 255;
                    }
                }
            }
            
            // Set the color of the pixel to be the average color of the samples.
            int area = totalNumberOfAnglesToTestX*totalNumberOfAnglesToTestY;
            int pixelIndex = (totalNumberOfAnglesToTestY - y - 1)*totalNumberOfAnglesToTestX + x;            
            for (int i = 0; i < 3; i++) {
                colors[pixelIndex + i*area] = lroundf(colorValues[i] / pow_fast(antiAliasingGridWidth, 2));
            }
        }
    }
}