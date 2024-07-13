#include <sycl/sycl.hpp>
#include "cinn_sycl_runtime_source.h"
typedef sycl::half float16;
#ifdef __cplusplus
extern "C" {
#endif
// CodeGenSYCL: NOTE: Auto-generated packed function
void fn_tan_0_kernel(sycl::queue &Q, sycl::range<3> dimGrid, sycl::range<3> dimBlock, void** void_args) {
  const float*  x = (float* )(*(void **)(void_args[0]));
  float*  var_0 = (float* )(*(void **)(void_args[1]));
  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class space5_fn_tan_0_kernel>(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item) [[intel::kernel_args_restrict]][[intel::max_work_group_size(1024, 1, 1)]]
    {
      if (((int)item.get_local_id(2) < 1024)) {
        var_0[(((int)item.get_local_id(2) & 1) + ((((((int)item.get_local_id(2) / 2) / 4) / 8) * 64) + ((2 * (((int)item.get_local_id(2) / 2) & 3)) + (8 * ((((int)item.get_local_id(2) / 2) / 4) & 7)))))] = cinn_sycl_tan_fp32(x[(((int)item.get_local_id(2) & 1) + ((((((int)item.get_local_id(2) / 2) / 4) / 8) * 64) + ((2 * (((int)item.get_local_id(2) / 2) & 3)) + (8 * ((((int)item.get_local_id(2) / 2) / 4) & 7)))))]);
      };
    });
  });
}

#ifdef __cplusplus
}
#endif
