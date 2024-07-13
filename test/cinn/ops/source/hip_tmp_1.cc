
    #include <cstdint>
    
	#define CINN_WITH_ROCM
	#include "float16.h"
    
    
	using namespace cinn;
	using cinn::common::float16;
    #include "cinn_hip_runtime_source.h"

extern "C" {

__global__
void __launch_bounds__(1024) fn_broadcast_to_0_elementwise_add_1_5_kernel(const double* __restrict__ y, const double* __restrict__ x, double* __restrict__ var_1)
{
  if (((int)threadIdx.x < 1024)) {
    var_1[(((int)threadIdx.x & 1) + ((((((int)threadIdx.x / 2) / 4) / 8) * 64) + ((2 * (((int)threadIdx.x / 2) & 3)) + (8 * ((((int)threadIdx.x / 2) / 4) & 7)))))] = (x[(((int)threadIdx.x & 1) + ((((((int)threadIdx.x / 2) / 4) / 8) * 64) + ((2 * (((int)threadIdx.x / 2) & 3)) + (8 * ((((int)threadIdx.x / 2) / 4) & 7)))))] + y[0]);
  };
}

}