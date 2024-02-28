// Copyright (c) 2021 CINN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/cinn/runtime/onednn/onednn_util.h"

#include <absl/container/flat_hash_map.h>
#include <glog/logging.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <string>


#include "paddle/cinn/backends/cuda_util.h"
#include "paddle/cinn/backends/extern_func_jit_register.h"
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/runtime/custom_function.h"
#include "paddle/cinn/runtime/flags.h"
#include "paddle/cinn/utils/profiler.h"
#include "paddle/cinn/utils/timer.h"

namespace cinn {
namespace runtime {
namespace cuda {

class OneDNNHandle {
 public:
  OneDNNHandle(const OneDNNHandle &) = delete;
  OneDNNHandle &operator=(const OneDNNHandle &) = delete;
  ~OneDNNHandle() {
  }
  static OneDNNHandle &GetInstance() {
    static OneDNNHandle instance;
    return instance;
  }
  cudaStream_t GetOneDNNStream() { return onednn_stream; }

 private:
  OneDNNHandle() {
  }
  // Create execution dnnl::engine.
  dnnl::engine engine(dnnl_gpu, 0);

  // Create dnnl::stream.
  dnnl::stream onednn_stream(engine);
};

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
                      void *stream) {
  cinn::utils::RecordEvent record_run("cinn_call_onednn",
                                      cinn::utils::EventType::kInstruction);

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

  int ldl = trans_op_l ? m : k;  // trans_o ? (trans_a ? k : m) : (trans_b ? k : m);
  int ldr = trans_op_r ? k : n;  // trans_o ? (trans_b ? n : k) : (trans_a ? n : k);
  int ldc = m;

  void *lhs = trans_o ? A : B;
  void *rhs = trans_o ? B : A;

  dt onednn_dtype;
  auto type_code = args[0].operator cinn_buffer_t *()->type.code;
  bool is_float = type_code == cinn_type_float;
  bool is_bfloat16 = type_code == cinn_type_bfloat;
  int bytes = args[0].operator cinn_buffer_t *()->type.bits / CHAR_BIT;
  if (is_float && bytes == sizeof(common::float16)) {
    onednn_dtype = dt::f16;
  } else if (is_float && bytes == sizeof(float)) {
    onednn_dtype = dt::f32;
  } else if (is_float && bytes == sizeof(double)) {
    onednn_dtype = dt::f64;
  } else if (is_bfloat16) {
    onednn_dtype = dt::bf16;
  } else {
    LOG(FATAL) << "unsupported cublas data type: "
               << static_cast<int>(type_code) << ", bytes = " << bytes;
  }

  // Create primitive descriptor.
  auto matmul_pd = matmul::primitive_desc(engine, a_md, b_md, c_md);
  // Create the primitive.
  auto matmul_prim = matmul(matmul_pd);
  // Primitive arguments.
  std::unordered_map<int, memory> matmul_args;
  matmul_args.insert({DNNL_ARG_SRC, a_mem});
  matmul_args.insert({DNNL_ARG_WEIGHTS, b_mem});
  matmul_args.insert({DNNL_ARG_DST, c_mem});
  // Execution.
  matmul_prim.execute(engine_stream, matmul_args);
  engine_stream.wait();
  
}

void cinn_gpu_cublas_mul(const std::vector<int> &attrs,
                         cinn_buffer_t *input1,
                         cinn_buffer_t *input2,
                         cinn_buffer_t *output,
                         cudaStream_t stream) {
  cublasHandle_t &handle = CublasHandle::GetInstance().GetCublasHandle();
  CHECK_EQ(input1->type.code, cinn_type_code_t::cinn_type_float);
  cudaStream_t custream = static_cast<cudaStream_t>(stream);
  CUBLAS_CALL(cublasSetStream(handle, custream));
  float *x_data = reinterpret_cast<float *>(input1->memory);
  float *y_data = reinterpret_cast<float *>(input2->memory);
  float *out_data = reinterpret_cast<float *>(output->memory);
  int M = 1;
  CHECK_GE(attrs.size(), 6);
  for (int i = 0; i < attrs[attrs.size() - 2]; i++) {
    M *= attrs[i];
  }
  int N = attrs[attrs.size() - 3];
  int K = attrs[attrs.size() - 4];
  float alpha = 1.f;
  float beta = 0.f;
  // M,N * N,K
  cublasSgemm(handle,
              CUBLAS_OP_N,
              CUBLAS_OP_N,
              K,
              M,
              N,
              &alpha,
              y_data,
              K,
              x_data,
              N,
              &beta,
              out_data,
              K);
}

void cinn_gpu_cublas_gemm(const std::vector<int> &attrs,
                          cinn_buffer_t *lhs,
                          cinn_buffer_t *rhs,
                          cinn_buffer_t *bias,
                          cinn_buffer_t *output,
                          cudaStream_t stream) {
  cublasHandle_t &handle = CublasHandle::GetInstance().GetCublasHandle();
  cudaStream_t custream = static_cast<cudaStream_t>(stream);
  CUBLAS_CALL(cublasSetStream(handle, custream));

  CHECK_EQ(lhs->type.code, cinn_type_code_t::cinn_type_float);
  const float *lhs_data = reinterpret_cast<const float *>(lhs->memory);
  const float *rhs_data = reinterpret_cast<const float *>(rhs->memory);
  const float *bias_data =
      bias ? reinterpret_cast<const float *>(bias->memory) : nullptr;
  float *output_data = reinterpret_cast<float *>(output->memory);

  CHECK_GE(attrs.size(), 13);
  int lhs_dim_size = attrs[attrs.size() - 7];
  int rhs_dim_size = attrs[attrs.size() - 6];
  int out_dim_size = attrs[attrs.size() - 5];
  bool lhs_trans = static_cast<bool>(attrs[attrs.size() - 4]);
  bool rhs_trans = static_cast<bool>(attrs[attrs.size() - 3]);
  bool out_trans = static_cast<bool>(attrs[attrs.size() - 2]);
  // 1）C = A^T * B    -->  C^T = B^T * A
  // 2）C = A * B^T    -->  C^T = B * A^T
  // 3）C = A^T * B^T  -->  C^T = B * A
  // 4）C = A * B      -->  C^T = B^T * A^T
  if (out_trans) {
    lhs_trans = static_cast<bool>(attrs[attrs.size() - 3]) ^ out_trans;
    rhs_trans = static_cast<bool>(attrs[attrs.size() - 4]) ^ out_trans;
  }
  const float alpha =
      *reinterpret_cast<const float *>(&attrs[attrs.size() - 1]);
  const float beta = bias ? 1.f : 0.f;
  VLOG(4) << "The lhs_trans value used by cinn_gpu_cublas_gemm: " << lhs_trans;
  VLOG(4) << "The rhs_trans value used by cinn_gpu_cublas_gemm: " << rhs_trans;
  VLOG(4) << "The out_trans value used by cinn_gpu_cublas_gemm: " << out_trans;
  VLOG(4) << "The alpha value used by cinn_gpu_cublas_gemm: " << alpha;
  VLOG(4) << "The beta value used by cinn_gpu_cublas_gemm: " << beta;
  CHECK_EQ(lhs_dim_size, rhs_dim_size);
  CHECK_EQ(lhs_dim_size, out_dim_size);
  CHECK((lhs_dim_size == 2 || lhs_dim_size == 3));

  if (lhs_dim_size == 2) {
    // [row, col]
    std::vector<int> lhs_shape{attrs[0], attrs[1]};
    std::vector<int> rhs_shape{attrs[2], attrs[3]};
    std::vector<int> output_shape{attrs[4], attrs[5]};
    if (out_trans) {
      std::swap(lhs_shape, rhs_shape);
      std::swap(lhs_data, rhs_data);
    }
    details::Gemm(handle,
                  lhs_trans,
                  rhs_trans,
                  alpha,
                  lhs_data,
                  lhs_shape,
                  rhs_data,
                  rhs_shape,
                  bias_data,
                  beta,
                  output_data,
                  output_shape,
                  stream);
  } 
}


}  // namespace cuda
}  // namespace runtime
}  // namespace cinn
