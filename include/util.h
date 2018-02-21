#include <math.h>
#include <point.h>


__device__ Point get_point_position(Point origin, float angle, float pendulumLength) {
    float x = sin(angle)*pendulumLength + origin.x;
    float y = -cos(angle)*pendulumLength + origin.y;
    
    Point p = {x, y};
    return p;
}
