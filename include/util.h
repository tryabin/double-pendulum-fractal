#include <math.h>
#include <point.h>


__device__ Point get_point_position(Point origin, float angle, float pendulumLength) {
    float x = sin(angle)*pendulumLength + origin.x;
    float y = -cos(angle)*pendulumLength + origin.y;
    
    Point p = {x, y};
    return p;
}


__device__ float pow_fast(float x, int n) {
    float value = 1;
    for (int i = 0; i < n; i++) {
        value *= x;
    }
    
    return value;
}