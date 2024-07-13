#include <sycl/sycl.hpp>
#include "cinn_sycl_runtime_source.h"
typedef sycl::half float16;
#ifdef __cplusplus
extern "C" {
#endif
// CodeGenSYCL: NOTE: Auto-generated packed function
void fn_broadcast_to_0_equal_1_0_kernel(sycl::queue &Q, sycl::range<3> dimGrid, sycl::range<3> dimBlock, void** void_args) {
  const float*  x = (float* )(*(void **)(void_args[0]));
  const float*  y = (float* )(*(void **)(void_args[1]));
  bool*  var_1 = (bool* )(*(void **)(void_args[2]));
  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class space258_fn_broadcast_to_0_equal_1_0_kernel>(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item) [[intel::kernel_args_restrict]][[intel::max_work_group_size(1024, 1, 1)]]
    {
      if (((int)item.get_group(2) < 1024)) {
        if (((int)item.get_local_id(2) < 1024)) {
          var_1[(((int)item.get_local_id(2) & 31) + (((((((int)item.get_local_id(2) / 32) / 16) + (2 * (int)item.get_group(2))) / 32) * 16384) + ((32 * (((int)item.get_local_id(2) / 32) & 15)) + (512 * (((((int)item.get_local_id(2) / 32) / 16) + (2 * (int)item.get_group(2))) & 31)))))] = (x[0] == y[(((int)item.get_local_id(2) & 31) + (((((((int)item.get_local_id(2) / 32) / 16) + (2 * (int)item.get_group(2))) / 32) * 16384) + ((32 * (((int)item.get_local_id(2) / 32) & 15)) + (512 * (((((int)item.get_local_id(2) / 32) / 16) + (2 * (int)item.get_group(2))) & 31)))))]);
        };
      };
    });
  });
}

#ifdef __cplusplus
}
#endif
