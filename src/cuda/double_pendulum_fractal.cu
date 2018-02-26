#include <stdio.h>
#include <math.h>
#include <math_constants.h>
#include <point.h>
#include <util.h>


typedef struct PendulumState {
    float angle1;
    float angle2;
    float angularVelocity1;
    float angularVelocity2;
} PendulumState;


typedef struct RungeKuttaStepResults {
    float acceleration1;
    float acceleration2;
    float velocity1;
    float velocity2;
} RungeKuttaStepResults;


typedef struct AccelerationResults {
    float acceleration1;
    float acceleration2;
} AccelerationResults;



__device__ AccelerationResults compute_accelerations(float m1, float m2,
                                                     float length1, float length2,
                                                     float angle1, float angle2,
                                                     float w1, float w2,
                                                     float g) {
                                                                
    float u = 1 + m1/m2;
    float delta = angle1 - angle2;

    AccelerationResults results;
    results.acceleration1 = (g*(sin(angle2)*cos(delta) - u*sin(angle1)) - (length2*pow(w2, 2) + length1*pow(w1, 2)*cos(delta))*sin(delta)) / (length1*(u - pow(cos(delta), 2)));
    results.acceleration2 = (g*u*(sin(angle1)*cos(delta) - sin(angle2)) + (u*length1*pow(w1, 2) + length2*pow(w2, 2)*cos(delta))*sin(delta)) / (length2*(u - pow(cos(delta), 2)));

    return results;
}
 
 
__device__ RungeKuttaStepResults compute_rk_step(float m1, float m2,
                                                 float length1, float length2,
                                                 float angle1, float angle2,
                                                 float w1, float w2,
                                                 RungeKuttaStepResults previousRungeKuttaStepResults,
                                                 float g,
                                                 float timestep) {
                                                     
                                                     
    float newAngle1 = angle1 + timestep*previousRungeKuttaStepResults.velocity1;
    float newAngle2 = angle2 + timestep*previousRungeKuttaStepResults.velocity2;

    float newAngularVelocity1 = w1 + timestep*previousRungeKuttaStepResults.acceleration1;
    float newAngularVelocity2 = w2 + timestep*previousRungeKuttaStepResults.acceleration2;

    AccelerationResults accelerationResults = compute_accelerations(m1, m2, length1, length2, newAngle1, newAngle2, newAngularVelocity1, newAngularVelocity2, g);
    float newVelocity1 = w1 + timestep*previousRungeKuttaStepResults.acceleration1;
    float newVelocity2 = w2 + timestep*previousRungeKuttaStepResults.acceleration2;

    RungeKuttaStepResults newRungeKuttaStepResults;
    newRungeKuttaStepResults.acceleration1 = accelerationResults.acceleration1;
    newRungeKuttaStepResults.acceleration2 = accelerationResults.acceleration2;
    newRungeKuttaStepResults.velocity1 = newVelocity1;
    newRungeKuttaStepResults.velocity2 = newVelocity2;
    
    return newRungeKuttaStepResults;
}        


__device__ PendulumState compute_double_pendulum_step_rk4(float m1, float m2,
                                                          float length1, float length2,
                                                          float angle1, float angle2,
                                                          float w1, float w2,
                                                          float g,
                                                          float timestep) {
                                                              
                                                              
    // Compute the four steps of the classical Runge-Kutta 4th order algorithm.
    RungeKuttaStepResults k1 = compute_rk_step(m1, m2, length1, length2, angle1, angle2, w1, w2, {0, 0, 0, 0}, g, timestep/2);
    RungeKuttaStepResults k2 = compute_rk_step(m1, m2, length1, length2, angle1, angle2, w1, w2, k1, g, timestep/2);
    RungeKuttaStepResults k3 = compute_rk_step(m1, m2, length1, length2, angle1, angle2, w1, w2, k2, g, timestep/2);
    RungeKuttaStepResults k4 = compute_rk_step(m1, m2, length1, length2, angle1, angle2, w1, w2, k3, g, timestep);
    
    // Combine the results of the Runge-Kutta steps.
    float acceleration1 = (k1.acceleration1 + 2*k2.acceleration1 + 2*k3.acceleration1 + k4.acceleration1)/6;
    float acceleration2 = (k1.acceleration2 + 2*k2.acceleration2 + 2*k3.acceleration2 + k4.acceleration2)/6;
    float velocity1 = (k1.velocity1 + 2*k2.velocity1 + 2*k2.velocity1 + k2.velocity1)/6;
    float velocity2 = (k1.velocity2 + 2*k2.velocity2 + 2*k2.velocity2 + k2.velocity2)/6;

    // Compute the new state of the pendulum.
    float point1NewAngularVelocity = acceleration1*timestep + w1;
    float point2NewAngularVelocity = acceleration2*timestep + w2;
    float point1NewAngle = velocity1*timestep + angle1;
    float point2NewAngle = velocity2*timestep + angle2;
    
    // Return the new state of the pendulum.
    PendulumState newPendulumState;
    newPendulumState.angle1 = point1NewAngle;
    newPendulumState.angle2 = point2NewAngle;
    newPendulumState.angularVelocity1 = point1NewAngularVelocity;
    newPendulumState.angularVelocity2 = point2NewAngularVelocity;
    
    return newPendulumState;
}


__global__ void compute_double_pendulum_fractal_image(float point1Mass, float point2Mass,
                                                      float pendulum1Length, float pendulum2Length,
                                                      float gravity,
                                                      float angle1Min, float angle1Max,
                                                      float angle2Min, float angle2Max,
                                                      int numberOfAnglesToTestPerKernalCallRatio,
                                                      int curKernelStartX, int curKernelStartY,
                                                      int totalNumberOfAnglesToTestX, int totalNumberOfAnglesToTestY,
                                                      float timestep,
                                                      float maxTimeToSeeIfPendulumFlips,
                                                      char *colors) {

    int stepX = gridDim.x*blockDim.x;
    int stepY =  gridDim.y*blockDim.y;

    int startX = threadIdx.x + blockDim.x*blockIdx.x;
    int startY = threadIdx.y + blockDim.y*blockIdx.y;
    
    int realStepX = stepX*numberOfAnglesToTestPerKernalCallRatio;
    int realStepY = stepY*numberOfAnglesToTestPerKernalCallRatio;
    int realStartX = startX*numberOfAnglesToTestPerKernalCallRatio + curKernelStartX;
    int realStartY = startY*numberOfAnglesToTestPerKernalCallRatio + curKernelStartY;
    
    for (int x = realStartX; x < totalNumberOfAnglesToTestX; x += realStepX) {        
        float angle1 = angle1Min + float(x)*(angle1Max - angle1Min)/float(totalNumberOfAnglesToTestX);
        
        for (int y = realStartY; y < totalNumberOfAnglesToTestY; y += realStepY) {    
            float angle2 = angle2Min + float(y)*(angle2Max - angle2Min)/float(totalNumberOfAnglesToTestY);
            
            // Skip the current pendulum if it doesn't have enough initial energy to flip the first mass.
            Point point1Position = get_point_position({0,0}, angle1, pendulum1Length);
            Point point2Position = get_point_position(point1Position, angle2, pendulum2Length);
            float potentialEnergy1 = (point1Position.y + pendulum1Length)*point1Mass*gravity;
            float potentialEnergy2 = (point2Position.y + pendulum1Length + pendulum2Length)*point2Mass*gravity;
            float totalPotentialEnergy = potentialEnergy1 + potentialEnergy2;
            
            float minimumEnergyNeededForFlip = point1Mass*2*pendulum1Length*gravity;
            if (totalPotentialEnergy < minimumEnergyNeededForFlip) {
                continue;
            }

            PendulumState pendulumState;
            pendulumState.angle1 = angle1;
            pendulumState.angle2 = angle2;
            pendulumState.angularVelocity1 = 0;
            pendulumState.angularVelocity2 = 0;
            
            float curTime = 0;
            while (curTime < maxTimeToSeeIfPendulumFlips) {
                
                Point point1OriginalPosition = get_point_position({0,0}, pendulumState.angle1, pendulum1Length);
                      
                pendulumState = compute_double_pendulum_step_rk4(point1Mass, point2Mass,
                                                                 pendulum1Length, pendulum2Length,
                                                                 pendulumState.angle1, pendulumState.angle2,
                                                                 pendulumState.angularVelocity1, pendulumState.angularVelocity2,
                                                                 gravity,
                                                                 timestep);
                
                Point point1CurrentPosition = get_point_position({0,0}, pendulumState.angle1, pendulum1Length);
                
                // Check to see if the first mass flipped.
                curTime += timestep;
                if (point1CurrentPosition.x*point1OriginalPosition.x < 0 && point1CurrentPosition.y > 0) {
                    break;
                }
            }

            // Color the pixel.
            float curTimeMs = curTime*1000;
            int area = totalNumberOfAnglesToTestX*totalNumberOfAnglesToTestY;
            int pixelIndex = (totalNumberOfAnglesToTestY - y - 1)*totalNumberOfAnglesToTestX + x;
            
            float shift = .11;
            float r = 1.0;
            float g = 4.0;
            float b = 7.2;
            
            colors[pixelIndex] = abs(sin(1.0/255 * CUDART_PI_F * curTimeMs * r * shift)) * 255;
            colors[pixelIndex+area] = abs(sin(1.0/255 * CUDART_PI_F * curTimeMs * g * shift)) * 255;
            colors[pixelIndex+2*area] = abs(sin(1.0/255 * CUDART_PI_F * curTimeMs * b * shift)) * 255;
        }
    }
}