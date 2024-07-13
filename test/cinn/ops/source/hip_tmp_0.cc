
    #include <cstdint>
    
	#define CINN_WITH_ROCM
	#include "float16.h"
    
    
	using namespace cinn;
	using cinn::common::float16;
    #include "cinn_hip_runtime_source.h"

extern "C" {

__global__
void __launch_bounds__(25) fn_cast_0_1_kernel(const double* __restrict__ y, float* __restrict__ var_1)
{
  if (((int)threadIdx.x < 25)) {
    var_1[(int)threadIdx.x] = ((float)(y[(int)threadIdx.x]));
  };
}

}