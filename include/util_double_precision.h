#include <math.h>
#include <point_double_precision.h>


__device__ Point get_point_position(Point origin, double angle, double pendulumLength) {
    double x = sin(angle)*pendulumLength + origin.x;
    double y = -cos(angle)*pendulumLength + origin.y;
    
    Point p = {x, y};
    return p;
}


__device__ double pow_fast(double x, int n) {
    double value = 1;
    for (int i = 0; i < n; i++) {
        value *= x;
    }
    
    return value;
}