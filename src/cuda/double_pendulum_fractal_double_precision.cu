#include <stdio.h>
#include <math.h>
#include <math_constants.h>
#include <point_double_precision.h>
#include <util_double_precision.h>


typedef struct PendulumState {
    double angle1;
    double angle2;
    double angularVelocity1;
    double angularVelocity2;
} PendulumState;


typedef struct RungeKuttaStepResults {
    double acceleration1;
    double acceleration2;
    double velocity1;
    double velocity2;
} RungeKuttaStepResults;


typedef struct AccelerationResults {
    double acceleration1;
    double acceleration2;
} AccelerationResults;



__device__ AccelerationResults compute_accelerations(double m1, double m2,
                                                     double length1, double length2,
                                                     double angle1, double angle2,
                                                     double w1, double w2,
                                                     double g) {
                                                                
    double u = 1 + m1/m2;
    double delta = angle1 - angle2;

    AccelerationResults results;
    results.acceleration1 = (g*(sin(angle2)*cos(delta) - u*sin(angle1)) - (length2*pow(w2, 2) + length1*pow(w1, 2)*cos(delta))*sin(delta)) / (length1*(u - pow(cos(delta), 2)));
    results.acceleration2 = (g*u*(sin(angle1)*cos(delta) - sin(angle2)) + (u*length1*pow(w1, 2) + length2*pow(w2, 2)*cos(delta))*sin(delta)) / (length2*(u - pow(cos(delta), 2)));

    return results;
}
 
 
__device__ RungeKuttaStepResults compute_rk_step(double m1, double m2,
                                                 double length1, double length2,
                                                 double angle1, double angle2,
                                                 double w1, double w2,
                                                 RungeKuttaStepResults previousRungeKuttaStepResults,
                                                 double g,
                                                 double timestep) {
                                                     
                                                     
    double newAngle1 = angle1 + timestep*previousRungeKuttaStepResults.velocity1;
    double newAngle2 = angle2 + timestep*previousRungeKuttaStepResults.velocity2;

    double newAngularVelocity1 = w1 + timestep*previousRungeKuttaStepResults.acceleration1;
    double newAngularVelocity2 = w2 + timestep*previousRungeKuttaStepResults.acceleration2;

    AccelerationResults accelerationResults = compute_accelerations(m1, m2, length1, length2, newAngle1, newAngle2, newAngularVelocity1, newAngularVelocity2, g);
    double newVelocity1 = w1 + timestep*previousRungeKuttaStepResults.acceleration1;
    double newVelocity2 = w2 + timestep*previousRungeKuttaStepResults.acceleration2;

    RungeKuttaStepResults newRungeKuttaStepResults;
    newRungeKuttaStepResults.acceleration1 = accelerationResults.acceleration1;
    newRungeKuttaStepResults.acceleration2 = accelerationResults.acceleration2;
    newRungeKuttaStepResults.velocity1 = newVelocity1;
    newRungeKuttaStepResults.velocity2 = newVelocity2;
    
    return newRungeKuttaStepResults;
}        


__device__ PendulumState compute_double_pendulum_step_rk4(double m1, double m2,
                                                          double length1, double length2,
                                                          double angle1, double angle2,
                                                          double w1, double w2,
                                                          double g,
                                                          double timestep) {
                                                              
                                                              
    // Compute the four steps of the classical Runge-Kutta 4th order algorithm.
    RungeKuttaStepResults k1 = compute_rk_step(m1, m2, length1, length2, angle1, angle2, w1, w2, {0, 0, 0, 0}, g, timestep/2);
    RungeKuttaStepResults k2 = compute_rk_step(m1, m2, length1, length2, angle1, angle2, w1, w2, k1, g, timestep/2);
    RungeKuttaStepResults k3 = compute_rk_step(m1, m2, length1, length2, angle1, angle2, w1, w2, k2, g, timestep/2);
    RungeKuttaStepResults k4 = compute_rk_step(m1, m2, length1, length2, angle1, angle2, w1, w2, k3, g, timestep);
    
    // Combine the results of the Runge-Kutta steps.
    double velocity1 = (k1.velocity1 + 2*k2.velocity1 + 2*k3.velocity1 + k4.velocity1)/6;
    double velocity2 = (k1.velocity2 + 2*k2.velocity2 + 2*k3.velocity2 + k4.velocity2)/6;
    double acceleration1 = (k1.acceleration1 + 2*k2.acceleration1 + 2*k3.acceleration1 + k4.acceleration1)/6;
    double acceleration2 = (k1.acceleration2 + 2*k2.acceleration2 + 2*k3.acceleration2 + k4.acceleration2)/6;
    
    // Compute the new state of the pendulum.
    double point1NewAngularVelocity = acceleration1*timestep + w1;
    double point2NewAngularVelocity = acceleration2*timestep + w2;
    double point1NewAngle = velocity1*timestep + angle1;
    double point2NewAngle = velocity2*timestep + angle2;
    
    // Return the new state of the pendulum.
    PendulumState newPendulumState;
    newPendulumState.angle1 = point1NewAngle;
    newPendulumState.angle2 = point2NewAngle;
    newPendulumState.angularVelocity1 = point1NewAngularVelocity;
    newPendulumState.angularVelocity2 = point2NewAngularVelocity;
    
    return newPendulumState;
}


__global__ void compute_double_pendulum_fractal_image(double point1Mass, double point2Mass,
                                                      double pendulum1Length, double pendulum2Length,
                                                      double gravity,
                                                      double angle1Min, double angle1Max,
                                                      double angle2Min, double angle2Max,
                                                      int numberOfAnglesToTestPerKernelCallRatio,
                                                      int curKernelStartX, int curKernelStartY,
                                                      int totalNumberOfAnglesToTestX, int totalNumberOfAnglesToTestY,
                                                      double timestep,
                                                      double maxTimeToSeeIfPendulumFlips,
                                                      char *colors) {

    int stepX = gridDim.x*blockDim.x;
    int stepY =  gridDim.y*blockDim.y;

    int startX = threadIdx.x + blockDim.x*blockIdx.x;
    int startY = threadIdx.y + blockDim.y*blockIdx.y;
    
    int realStepX = stepX*numberOfAnglesToTestPerKernelCallRatio;
    int realStepY = stepY*numberOfAnglesToTestPerKernelCallRatio;
    int realStartX = startX*numberOfAnglesToTestPerKernelCallRatio + curKernelStartX;
    int realStartY = startY*numberOfAnglesToTestPerKernelCallRatio + curKernelStartY;

    for (int x = realStartX; x < totalNumberOfAnglesToTestX; x += realStepX) {        
        double angle1 = angle1Min + double(x)*(angle1Max - angle1Min)/double(totalNumberOfAnglesToTestX);
        
        for (int y = realStartY; y < totalNumberOfAnglesToTestY; y += realStepY) {    
            double angle2 = angle2Min + double(y)*(angle2Max - angle2Min)/double(totalNumberOfAnglesToTestY);
            
            // Skip the current pendulum if it doesn't have enough initial energy to flip the first mass.
            Point point1Position = get_point_position({0,0}, angle1, pendulum1Length);
            Point point2Position = get_point_position(point1Position, angle2, pendulum2Length);
            double potentialEnergy1 = point1Position.y*point1Mass*gravity;
            double potentialEnergy2 = point2Position.y*point2Mass*gravity;
            double totalPotentialEnergy = potentialEnergy1 + potentialEnergy2;

            double minimumEnergyNeededForFlip = point1Mass*pendulum1Length*gravity + point2Mass*(pendulum1Length - pendulum2Length)*gravity;
            if (totalPotentialEnergy < minimumEnergyNeededForFlip) {
                continue;
            }

            // Simulate the pendulum.
            PendulumState pendulumState;
            pendulumState.angle1 = angle1;
            pendulumState.angle2 = angle2;
            pendulumState.angularVelocity1 = 0;
            pendulumState.angularVelocity2 = 0;
            
            double curTime = 0;
            Point point1OriginalPosition = get_point_position({0,0}, pendulumState.angle1, pendulum1Length);
            while (curTime < maxTimeToSeeIfPendulumFlips) {               
            
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
                
                curTime += timestep;
            }

            // Color the pixel.
            double curTimeMs = curTime*1000;
            int area = totalNumberOfAnglesToTestX*totalNumberOfAnglesToTestY;
            int pixelIndex = (totalNumberOfAnglesToTestY - y - 1)*totalNumberOfAnglesToTestX + x;
            
            double shift = .11;
            double r = 1.0;
            double g = 4.0;
            double b = 7.2;
            
            colors[pixelIndex] = abs(sin(1.0/255 * CUDART_PI_F * curTimeMs * r * shift)) * 255;
            colors[pixelIndex+area] = abs(sin(1.0/255 * CUDART_PI_F * curTimeMs * g * shift)) * 255;
            colors[pixelIndex+2*area] = abs(sin(1.0/255 * CUDART_PI_F * curTimeMs * b * shift)) * 255;
        }
    }
}