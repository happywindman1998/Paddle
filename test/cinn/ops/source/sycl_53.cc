#include <sycl/sycl.hpp>
#include "cinn_sycl_runtime_source.h"
typedef sycl::half float16;
#ifdef __cplusplus
extern "C" {
#endif
// CodeGenSYCL: NOTE: Auto-generated packed function
void fn_reverse_0_kernel(sycl::queue &Q, sycl::range<3> dimGrid, sycl::range<3> dimBlock, void** void_args) {
  const bool*  x = (bool* )(*(void **)(void_args[0]));
  bool*  var_0 = (bool* )(*(void **)(void_args[1]));
  Q.submit([&](sycl::handler &h) {
    h.parallel_for<class space53_fn_reverse_0_kernel>(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<3> item) [[intel::kernel_args_restrict]][[intel::max_work_group_size(1024, 1, 1)]]
    {
      if (((int)item.get_group(2) < 110)) {
        if (((int)item.get_local_id(2) < 1024)) {
          if ((((1024 * (int)item.get_group(2)) + (int)item.get_local_id(2)) < 112000)) {
            var_0[((((1024 * (int)item.get_group(2)) + (int)item.get_local_id(2)) % 7) + (((((((1024 * (int)item.get_group(2)) + (int)item.get_local_id(2)) / 7) / 5) / 40) * 1400) + ((7 * ((((1024 * (int)item.get_group(2)) + (int)item.get_local_id(2)) / 7) % 5)) + (35 * (((((1024 * (int)item.get_group(2)) + (int)item.get_local_id(2)) / 7) / 5) % 40)))))] = x[(110600 + ((((1024 * (int)item.get_group(2)) + (int)item.get_local_id(2)) % 7) + ((-1400 * (((((1024 * (int)item.get_group(2)) + (int)item.get_local_id(2)) / 7) / 5) / 40)) + ((7 * ((((1024 * (int)item.get_group(2)) + (int)item.get_local_id(2)) / 7) % 5)) + (35 * (((((1024 * (int)item.get_group(2)) + (int)item.get_local_id(2)) / 7) / 5) % 40))))))];
          };
        };
      };
    });
  });
}

#ifdef __cplusplus
}
#endif
