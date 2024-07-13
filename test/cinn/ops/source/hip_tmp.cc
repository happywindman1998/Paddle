
    #include <cstdint>
    
	#define CINN_WITH_ROCM
	#include "float16.h"
    
    
	using namespace cinn;
	using cinn::common::float16;
    #include "cinn_hip_runtime_source.h"

extern "C" {

__global__
void __launch_bounds__(1024) fn_tan_0_kernel(const double* __restrict__ x, double* __restrict__ var_0)
{
  if (((int)blockIdx.x < 2)) {
    if (((int)threadIdx.x < 1024)) {
      var_0[((int)threadIdx.x + (1024 * (int)blockIdx.x))] = cinn_hip_tan_fp64(x[((int)threadIdx.x + (1024 * (int)blockIdx.x))]);
    };
  };
}

}