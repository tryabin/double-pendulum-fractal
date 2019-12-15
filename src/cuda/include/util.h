#include <math.h>
#include <point.h>
#include <float_type.h>


__device__ Point get_point_position(Point origin, FloatType angle, FloatType pendulumLength) {
    FloatType x = sin(angle)*pendulumLength + origin.x;
    FloatType y = -cos(angle)*pendulumLength + origin.y;
    
    Point p = {x, y};
    return p;
}


__device__ FloatType pow_fast(FloatType x, int n) {
    FloatType value = 1;
    for (int i = 0; i < n; i++) {
        value *= x;
    }
    
    return value;
}

