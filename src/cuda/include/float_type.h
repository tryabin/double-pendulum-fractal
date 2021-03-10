#ifdef FLOAT_32
   typedef float FloatType;
   #define PI CUDART_PI_F
#else
   typedef double FloatType;
   #define PI CUDART_PI
#endif

#define TAU (2*PI)
