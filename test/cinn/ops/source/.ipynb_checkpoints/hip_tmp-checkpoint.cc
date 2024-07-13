#include "cinn_hip_runtime_source.h"
	#define CINN_WITH_ROCM
	#include "float16.h"
	using namespace cinn;
	using cinn::common::float16;

extern "C" {

__global__
void __launch_bounds__(10) fn_ceil_0_kernel(const float* __restrict__ x, float* __restrict__ var_0)
{
  if (((int)threadIdx.x < 10)) {
    var_0[(int)threadIdx.x] = cinn_hip_ceil_fp32(x[(int)threadIdx.x]);
  };
}

}