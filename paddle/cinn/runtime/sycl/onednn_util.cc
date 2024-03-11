#include <glog/logging.h>
#include <vector>
#include <unordered_map>

#include <algorithm>
#include <string>

#include <iostream>

#include "paddle/cinn/backends/extern_func_jit_register.h"
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/runtime/sycl/onednn_util.h"
#include "paddle/cinn/runtime/custom_function.h"
#include "paddle/cinn/runtime/flags.h"
#include "paddle/cinn/utils/profiler.h"
#include "paddle/cinn/utils/timer.h"

#include <sycl/sycl.hpp>
#include "paddle/cinn/runtime/sycl/sycl_backend_api.h"
using cinn::runtime::Sycl::SYCLBackendAPI;

#include "dnnl.hpp"
#include "dnnl_sycl.hpp"

using namespace dnnl;

using tag = memory::format_tag;
using dt = memory::data_type;

namespace cinn {
namespace runtime {
namespace Sycl {

class OneDNNHandle {
public:
  OneDNNHandle(const OneDNNHandle &) = delete;
  OneDNNHandle &operator=(const OneDNNHandle &) = delete;
  static OneDNNHandle &GetInstance() {
    static OneDNNHandle instance;
    return instance;
  }

  dnnl::engine GetOneDNNEngine() { return onednn_engine; }
  dnnl::stream GetOneDNNStream() { return onednn_stream; }

private:
  OneDNNHandle() {
    // Create execution dnnl::engine.
    sycl::context *sycl_context = SYCLBackendAPI::Global()->get_default_context();
    sycl::device sycl_device = SYCLBackendAPI::Global()->get_default_device();
    onednn_engine = sycl_interop::make_engine(sycl_device, *sycl_context);
    sycl::queue interop_queue(*sycl_context, sycl_device);
    onednn_stream = sycl_interop::make_stream(onednn_engine, interop_queue);
  }

  dnnl::engine onednn_engine;
  dnnl::stream onednn_stream;
};

void cinn_gpu_onednn_matmul(const std::vector<int> &attrs,
                          cinn_buffer_t *lhs,
                          cinn_buffer_t *rhs,
                          cinn_buffer_t *bias,
                          cinn_buffer_t *output,
                          void* vqueue) {
  
  std::cout<<"============= call onednn matmul ==============="<<std::endl;
  VLOG(3) << "call cinn_gpu_onednn_matmul";
  dnnl::engine onednn_engine = OneDNNHandle::GetInstance().GetOneDNNEngine();
  dnnl::stream onednn_stream = OneDNNHandle::GetInstance().GetOneDNNStream();
  
  //float *x_data = reinterpret_cast<float *>(input1->memory);
  //float *y_data = reinterpret_cast<float *>(input2->memory);
  //float *out_data = reinterpret_cast<float *>(output->memory);
  int M = 1;
  CHECK_GE(attrs.size(), 6);
  for (int i = 0; i < attrs[attrs.size() - 2]; i++) {
    M *= attrs[i];
  }
  int N = attrs[attrs.size() - 3];
  int K = attrs[attrs.size() - 4];
  float alpha = 1.f;
  float beta = 0.f;
  
  auto type_code = lhs->type.code;
  memory::data_type onednn_dtype;
  bool is_float = type_code == cinn_type_float;
  bool is_bfloat16 = type_code == cinn_type_bfloat;
  int bytes = lhs->type.bits / CHAR_BIT;
  if (is_float && bytes == sizeof(common::float16)) {
    onednn_dtype = memory::data_type::f16;
  } else if (is_float && bytes == sizeof(float)) {
    onednn_dtype = memory::data_type::f32;
  } else if (is_float && bytes == sizeof(double)) {
    onednn_dtype = memory::data_type::f64;
  } else if (is_bfloat16) {
    onednn_dtype = memory::data_type::bf16;
  } else {
    LOG(FATAL) << "unsupported cublas data type: "
               << static_cast<int>(type_code) << ", bytes = " << bytes;
  }

  void *A = lhs->memory;
  void *B = rhs->memory;
  void *C = output->memory;

  // Source (A), weights (B), and destination (C) matrix dimensions.
  memory::dims a_dims = {M, K};
  memory::dims b_dims = {K, N};
  memory::dims c_dims = {M, N};

  auto a_md = memory::desc(a_dims, onednn_dtype, tag::ab);
  auto b_md = memory::desc(b_dims, onednn_dtype, tag::ab);
  auto c_md = memory::desc(c_dims, onednn_dtype, tag::ab);
  
  auto a_mem = dnnl::memory(a_md, onednn_engine, A);
  auto b_mem = dnnl::memory(b_md, onednn_engine, B);
  auto c_mem = dnnl::memory(c_md, onednn_engine, C);
  // Create primitive descriptor.
  auto matmul_pd = matmul::primitive_desc(onednn_engine, a_md, b_md, c_md);

  // Create the primitive.
  auto matmul_prim = matmul(matmul_pd);

  // Primitive arguments.
  std::unordered_map<int, memory> matmul_args;
  matmul_args.insert({DNNL_ARG_SRC, a_mem});
  matmul_args.insert({DNNL_ARG_WEIGHTS, b_mem});
  matmul_args.insert({DNNL_ARG_DST, c_mem});

  // Execution.
  matmul_prim.execute(onednn_stream, matmul_args);
  onednn_stream.wait();
}

void cinn_call_onednn(void *v_args,
                      int num_args,
                      bool trans_a,
                      bool trans_b,
                      bool trans_o,
                      float alpha,
                      float beta,
                      int a1,
                      int a2,
                      int a3,
                      int a4,
                      int b1,
                      int b2,
                      int b3,
                      int b4,
                      void *vqueue) {
  cinn::utils::RecordEvent record_run("cinn_call_onednn",
                                      cinn::utils::EventType::kInstruction);

  std::cout<<"============= call onednn ==============="<<std::endl;
  CHECK_EQ(num_args, 3);
  
  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);

  dnnl::engine onednn_engine = OneDNNHandle::GetInstance().GetOneDNNEngine();
  dnnl::stream onednn_stream = OneDNNHandle::GetInstance().GetOneDNNStream();

  VLOG(3) << "a1 ~ a4: " << a1 << " " << a2 << " " << a3 << " " << a4;
  VLOG(3) << "b1 ~ b4: " << b1 << " " << b2 << " " << b3 << " " << b4;
  VLOG(3) << "trans_a: " << trans_a << ", trans_b: " << trans_b
          << ", trans_o: " << trans_o;

  void *A = args[0].operator cinn_buffer_t *()->memory;
  void *B = args[1].operator cinn_buffer_t *()->memory;
  void *C = args[2].operator cinn_buffer_t *()->memory;

  int m = trans_o ? (trans_a ? a4 : a3) : (trans_b ? b3 : b4);
  int n = trans_o ? (trans_b ? b3 : b4) : (trans_a ? a4 : a3);
  int k = trans_a ? a3 : a4;

  VLOG(3) << "m: " << m << ", n: " << n << ", k: " << k;

  memory::data_type onednn_dtype;
  auto type_code = args[0].operator cinn_buffer_t *()->type.code;
  bool is_float = type_code == cinn_type_float;
  bool is_bfloat16 = type_code == cinn_type_bfloat;
  int bytes = args[0].operator cinn_buffer_t *()->type.bits / CHAR_BIT;
  if (is_float && bytes == sizeof(common::float16)) {
    onednn_dtype = memory::data_type::f16;
  } else if (is_float && bytes == sizeof(float)) {
    onednn_dtype = memory::data_type::f32;
  } else if (is_float && bytes == sizeof(double)) {
    onednn_dtype = memory::data_type::f64;
  } else if (is_bfloat16) {
    onednn_dtype = memory::data_type::bf16;
  } else {
    LOG(FATAL) << "unsupported cublas data type: "
               << static_cast<int>(type_code) << ", bytes = " << bytes;
  }

  memory::dims a_dims = {m, k};
  memory::dims b_dims = {k, n};
  memory::dims c_dims = {m, n};

  auto a_md = memory::desc(a_dims, onednn_dtype, tag::ab);
  auto b_md = memory::desc(b_dims, onednn_dtype, tag::ab);
  auto c_md = memory::desc(c_dims, onednn_dtype, tag::ab);
  
  auto a_mem = dnnl::memory(a_md, onednn_engine, A);
  auto b_mem = dnnl::memory(b_md, onednn_engine, B);
  auto c_mem = dnnl::memory(c_md, onednn_engine, C);

  // Create primitive descriptor.
  auto matmul_pd = matmul::primitive_desc(onednn_engine, a_md, b_md, c_md);

  // Create the primitive.
  auto matmul_prim = matmul(matmul_pd);

  // Primitive arguments.
  std::unordered_map<int, memory> matmul_args;
  matmul_args.insert({DNNL_ARG_SRC, a_mem});
  matmul_args.insert({DNNL_ARG_WEIGHTS, b_mem});
  matmul_args.insert({DNNL_ARG_DST, c_mem});

  // Execution.
  matmul_prim.execute(onednn_stream, matmul_args);
    
  onednn_stream.wait();
}

}
}
}