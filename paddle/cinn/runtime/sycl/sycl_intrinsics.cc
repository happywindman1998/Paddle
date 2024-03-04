#include "paddle/cinn/backends/cuda_util.h"
#include "paddle/cinn/backends/extern_func_jit_register.h"
#include "paddle/cinn/backends/function_prototype.h"
#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/runtime/custom_function.h"

#include "paddle/cinn/runtime/sycl/sycl_module.h"
using cinn::runtime::Sycl::cinn_call_sycl_kernel;
#include "paddle/cinn/backends/llvm/runtime_symbol_registry.h"
using cinn::backends::GlobalSymbolRegistry;
#include "paddle/cinn/runtime/sycl/sycl_backend_api.h"
using cinn::runtime::Sycl::SYCLBackendAPI;

#include "paddle/cinn/runtime/sycl/onednn_util.h"
using cinn::runtime::onednn::cinn_call_onednn;

CINN_REGISTER_HELPER(cinn_sycl_host_api) {
  REGISTER_EXTERN_FUNC_HELPER(cinn_call_sycl_kernel,
                              cinn::common::DefaultHostTarget())
      .SetRetType<void>()
      .AddInputType<void *>()  // kernel_fn
      .AddInputType<void *>()  // args
      .AddInputType<int>()     // num_args
      .AddInputType<int>()     // grid_x
      .AddInputType<int>()     // grid_y
      .AddInputType<int>()     // grid_z
      .AddInputType<int>()     // block_x
      .AddInputType<int>()     // block_y
      .AddInputType<int>()     // block_z
      .AddInputType<void *>()  // stream
      .End();
  GlobalSymbolRegistry::Global().RegisterFn("backend_api.sycl", reinterpret_cast<void*>(SYCLBackendAPI::Global()));
  
  REGISTER_EXTERN_FUNC_HELPER(cinn_call_onednn,
                              cinn::common::DefaultHostTarget())
      .SetRetType<void>()
      .AddInputType<void *>()  // v_args
      .AddInputType<int>()     // num_args
      .AddInputType<bool>()    // trans_a
      .AddInputType<bool>()    // trans_b
      .AddInputType<float>()   // alpha
      .AddInputType<float>()   // beta
      .AddInputType<int>()     // a1
      .AddInputType<int>()     // a2
      .AddInputType<int>()     // a3
      .AddInputType<int>()     // a4
      .AddInputType<int>()     // b1
      .AddInputType<int>()     // b2
      .AddInputType<int>()     // b3
      .AddInputType<int>()     // b4
      .AddInputType<void *>()  // stream
      .End();
  
  return true;
}