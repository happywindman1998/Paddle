#include <sycl/sycl.hpp>
#include "cinn_sycl_runtime_source.h"
typedef sycl::half float16;
#ifdef __cplusplus
extern "C" {
#endif
// CodeGenSYCL: NOTE: Auto-generated packed function
void fn_elementwise_add_0_2_kernel(sycl::queue &Q, sycl::range<3> dimGrid, sycl::range<3> dimBlock, void** void_args) {
  const int64_t*  x = (int64_t* )(*(void **)(void_args[0]));
  const int64_t*  y = (int64_t* )(*(void **)(void_args[1]));
  int64_t*  var_1 = (int64_t* )(*(void **)(void_args[2]));
  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class space32_fn_elementwise_add_0_2_kernel>(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item) [[intel::kernel_args_restrict]][[intel::max_work_group_size(1024, 1, 1)]]
    {
      if (((int)item.get_group(2) < 256)) {
        if (((int)item.get_local_id(2) < 1024)) {
          var_1[(((int)item.get_local_id(2) & 31) + ((((((int)item.get_local_id(2) / 32) + (32 * (int)item.get_group(2))) / 64) * 2048) + (32 * ((((int)item.get_local_id(2) / 32) + (32 * (int)item.get_group(2))) & 63))))] = (x[(((int)item.get_local_id(2) & 31) + ((((((int)item.get_local_id(2) / 32) + (32 * (int)item.get_group(2))) / 64) * 2048) + (32 * ((((int)item.get_local_id(2) / 32) + (32 * (int)item.get_group(2))) & 63))))] + y[(((int)item.get_local_id(2) & 31) + ((((((int)item.get_local_id(2) / 32) + (32 * (int)item.get_group(2))) / 64) * 2048) + (32 * ((((int)item.get_local_id(2) / 32) + (32 * (int)item.get_group(2))) & 63))))]);
        };
      };
    });
  });
}

#ifdef __cplusplus
}
#endif
