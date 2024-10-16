#include <sycl/sycl.hpp>
#include "cinn_sycl_runtime_source.h"
typedef sycl::half float16;
#ifdef __cplusplus
extern "C" {
#endif
// CodeGenSYCL: NOTE: Auto-generated packed function
void fn_elementwise_add_0_2_kernel(sycl::queue &Q, sycl::range<3> dimGrid, sycl::range<3> dimBlock, void** void_args) {
  const float*  x = (float* )(*(void **)(void_args[0]));
  const float*  y = (float* )(*(void **)(void_args[1]));
  float*  var_1 = (float* )(*(void **)(void_args[2]));
  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class space36_fn_elementwise_add_0_2_kernel>(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item) [[intel::kernel_args_restrict]][[intel::max_work_group_size(1, 1, 1)]]
    {
      var_1[0] = (x[0] + y[0]);
    });
  });
}

#ifdef __cplusplus
}
#endif
