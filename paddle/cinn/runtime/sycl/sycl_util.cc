// Copyright (c) 2024 CINN Authors. All Rights Reserved.
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

#include <dlfcn.h>
#include <glog/logging.h>
#include <glog/raw_logging.h>

#include "paddle/cinn/runtime/flags.h"
#include "paddle/cinn/runtime/sycl/sycl_backend_api.h"
using cinn::runtime::sycl::SYCLBackendAPI;
#include "paddle/cinn/runtime/sycl/sycl_util.h"
#include "paddle/cinn/runtime/cinn_runtime.h"
#include "paddle/cinn/utils/profiler.h"
#include "paddle/common/enforce.h"

#ifdef CINN_WITH_ONDNN
#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_sycl.hpp"
// #include "dnnl.hpp"
// #include "dnnl_sycl.hpp"
#endif

using namespace dnnl;

using tag = memory::format_tag;
using dt = memory::data_type;

namespace cinn {
namespace runtime {
namespace sycl {

void cinn_call_sycl_kernel(void* kernel_fn,
                           void* v_args,
                           int num_args,
                           int grid_x,
                           int grid_y,
                           int grid_z,
                           int block_x,
                           int block_y,
                           int block_z) {
  VLOG(3) << "cinn_call_sycl_kernel, grid_dim={" << grid_x << ", " << grid_y
          << ", " << grid_z << "}, block_dim={" << block_x << ", " << block_y
          << ", " << block_z << "}, num_args=" << num_args;

  std::vector<void*> kernel_args;
  {
    cinn::utils::RecordEvent record_run("prepare_args",
                                        cinn::utils::EventType::kInstruction);
    kernel_args.reserve(num_args);
    cinn_pod_value_t* args = static_cast<cinn_pod_value_t*>(v_args);
    for (int idx = 0; idx < num_args; ++idx) {
      if (args[idx].type_code() == ::cinn_type_code<cinn_buffer_t*>()) {
        auto &addr = args[idx].operator cinn_buffer_t *()->memory;
        VLOG(4) << "cinn_call_sycl_kernel: arg[" << idx << "]=" << (void *)addr;
        kernel_args.emplace_back(&addr);
      } else {
        kernel_args.emplace_back((args[idx].data_addr()));
      }
    }
  }

  {
    cinn::utils::RecordEvent record_run("syclLaunchKernel",
                                        cinn::utils::EventType::kInstruction);
    void (*kernel_func)(::sycl::queue & Q,
                        ::sycl::range<3> k0_dimGrid,
                        ::sycl::range<3> k0_dimBlock,
                        void** void_args) =
        (void (*)(::sycl::queue & Q,
                  ::sycl::range<3> k0_dimGrid,
                  ::sycl::range<3> k0_dimBlock,
                  void** void_args))(kernel_fn);
    ::sycl::queue* Queue = SYCLBackendAPI::Global()->get_now_queue();
    ::sycl::range<3> Grid(grid_z, grid_y, grid_x);
    ::sycl::range<3> Block(block_z, block_y, block_x);
    kernel_func(*Queue, Grid, Block, kernel_args.data());
    Queue->wait_and_throw();
  }
}

void cinn_call_sycl_memset(
    void *v_args, int num_args, int value, size_t count) {
  PADDLE_ENFORCE_EQ(num_args,
                    1,
                    phi::errors::PreconditionNotMet(
                        "The cinn_call_sycl_memset only accept a output."));
  VLOG(4) << "call cinn_call_sycl_memset with value=" << value
          << ", count=" << count;

  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);
  void *output = args[0].operator cinn_buffer_t *()->memory;

  VLOG(4) << "cinn_call_sycl_memset: output=" << output;

  auto Queue = SYCLBackendAPI::Global()->get_now_queue();

  SYCL_CALL(Queue->memset(output, value, count));
}

void cinn_call_sycl_memcpy(void *v_args,
                           int num_args,
                           size_t count) {
  PADDLE_ENFORCE_EQ(
      num_args,
      2,
      phi::errors::PreconditionNotMet(
          "The cinn_call_sycl_memcpy only accept a input and a output."));
  VLOG(4) << "call cinn_call_sycl_memcpy with count=" << count;

  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);
  void *input = args[0].operator cinn_buffer_t *()->memory;
  void *output = args[1].operator cinn_buffer_t *()->memory;

  if (input == output) {
    VLOG(4) << "Skip memcpy as input and output are the same.";
    return;
  }

  VLOG(4) << "cinn_call_sycl_memcpy: input=" << input << ", output=" << output;

  auto Queue = SYCLBackendAPI::Global()->get_now_queue();

  Queue->memcpy(output, input, count);
}

// #ifdef CINN_WITH_CNNL
#ifdef CINN_WITH_ONEDNN

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
    ::sycl::context *sycl_context = SYCLBackendAPI::Global()->get_default_context();
    ::sycl::device sycl_device = SYCLBackendAPI::Global()->get_default_device();
    ::sycl::queue *sycl_queue = SYCLBackendAPI::Global()->get_now_queue();
    
    onednn_engine = sycl_interop::make_engine(sycl_device, *sycl_context);
    onednn_stream = sycl_interop::make_stream(onednn_engine, *sycl_queue);
  }

  dnnl::engine onednn_engine;
  dnnl::stream onednn_stream;
};

memory::data_type convert_to_onednn_dtype(void *v_args, int num_args) {
  CHECK_GT(num_args, 0) << "the number of arguments must larger than zero";
  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);
  auto type_code = args[0].operator cinn_buffer_t *()->type.code;
  int bits = args[0].operator cinn_buffer_t *()->type.bits;
  for (int i = 1; i < num_args; ++i) {
    auto t = args[i].operator cinn_buffer_t *()->type.code;
    int b = args[0].operator cinn_buffer_t *()->type.bits;
    if (t != type_code || bits != b) {
      LOG(FATAL) << "The types of all arguments need to be consistent.";
    }
  }
  memory::data_type onednn_dtype;
  bool is_float = type_code == cinn_type_float;
  bool is_bfloat16 = type_code == cinn_type_bfloat;
  if (is_float && bits == 16) {
    onednn_dtype = memory::data_type::f16;
  } else if (is_float && bits == 32) {
    onednn_dtype = memory::data_type::f32;
  } else if (is_bfloat16) {
    onednn_dtype = memory::data_type::bf16;
  } else if (is_float && bits == 64) {
    onednn_dtype = memory::data_type::f64;
  } else {
    LOG(FATAL) << "unsupported onednn data type: " << static_cast<int>(type_code)
               << ", bits = " << bits;
  }
  return onednn_dtype;
}

memory::data_type convert_to_onednn_dtype(cinn_buffer_t *input) {
  CHECK(input) << "the pointer of input is null";
  auto type_code = input->type.code;
  int bits = input->type.bits;
  memory::data_type onednn_dtype;
  bool is_float = type_code == cinn_type_float;
  bool is_bfloat16 = type_code == cinn_type_bfloat;
  if (is_float && bits == 16) {
    onednn_dtype = memory::data_type::f16;
  } else if (is_float && bits == 32) {
    onednn_dtype = memory::data_type::f32;
  } else if (is_bfloat16) {
    onednn_dtype = memory::data_type::bf16;
  } else if (is_float && bits == 64) {
    onednn_dtype = memory::data_type::f64;
  } else {
    LOG(FATAL) << "unsupported onednn data type: " << static_cast<int>(type_code)
               << ", bits = " << bits;
  }
  return onednn_dtype;
}


/*
class CnnlRandGenerator {
 public:
  CnnlRandGenerator() {
    CNNL_CALL(cnnlRandCreateGenerator(&generator_, CNNL_RAND_RNG_FAST));
  }

  explicit CnnlRandGenerator(cnnlRandRngType_t rng_type) {
    CNNL_CALL(cnnlRandCreateGenerator(&generator_, rng_type));
  }

  ~CnnlRandGenerator() { CNNL_CALL(cnnlRandDestroyGenerator(generator_)); }

  cnnlRandGenerator_t &GetGenerator() { return generator_; }

  CnnlRandGenerator &SetSeed(uint64_t seed = 0ULL) {
    // set global seed if seed is zero
    auto rand_seed = (seed == 0ULL) ? RandomSeed::GetOrSet() : seed;
    if (rand_seed != 0ULL && rand_seed != seed_) {
      CNNL_CALL(cnnlRandSetPhiloxSeed(generator_, rand_seed));
      VLOG(4) << "Change curand random seed from: " << seed_
              << " to: " << rand_seed;
      seed_ = rand_seed;
    }
    return *this;
  }

 private:
  cnnlRandGenerator_t generator_;
  uint64_t seed_ = 0ULL;
};

cnnlDataType_t convert_to_cnnl_dtype(void *v_args, int num_args) {
  PADDLE_ENFORCE_GT(num_args,
                    0,
                    phi::errors::PreconditionNotMet(
                        "the number of arguments must larger than zero"));
  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);
  auto type_code = args[0].operator cinn_buffer_t *()->type.code;
  int bits = args[0].operator cinn_buffer_t *()->type.bits;
  for (int i = 1; i < num_args; ++i) {
    auto t = args[i].operator cinn_buffer_t *()->type.code;
    int b = args[0].operator cinn_buffer_t *()->type.bits;
    if (t != type_code || bits != b) {
      PADDLE_THROW(phi::errors::InvalidArgument(
          "The types of all arguments need to be consistent."));
    }
  }
  cnnlDataType_t data_type;
  bool is_float = type_code == cinn_type_float;
  bool is_bfloat16 = type_code == cinn_type_bfloat;
  if (is_float && bits == 16) {
    data_type = CNNL_DTYPE_HALF;
  } else if (is_float && bits == 32) {
    data_type = CNNL_DTYPE_FLOAT;
  } else if (is_bfloat16) {
    data_type = CNNL_DTYPE_BFLOAT16;
  } else {
    std::stringstream ss;
    ss << "unsupported cudnn data type: " << static_cast<int>(type_code)
       << ", bits = " << bits;
    PADDLE_THROW(phi::errors::InvalidArgument(ss.str()));
  }
  return data_type;
}

std::string debug_cnnl_tensor_format(cnnlTensorLayout_t tensor_format) {
  switch (tensor_format) {
    case CNNL_LAYOUT_NCHW:
      return "NCHW";
    case CNNL_LAYOUT_NHWC:
      return "NHWC";
    default:
      PADDLE_THROW(phi::errors::InvalidArgument(
          "Only support NCHW and NHWC data layout\n"));
  }
  return "";
}

std::string debug_cnnl_tensor_dtype(cnnlDataType_t tensor_dtype) {
  switch (tensor_dtype) {
    case CNNL_DTYPE_FLOAT:
      return "float32";
    case CNNL_DTYPE_HALF:
      return "float16";
    case CNNL_DTYPE_BFLOAT16:
      return "bfloat16";
    default:
      PADDLE_THROW(phi::errors::InvalidArgument(
          "Only support float16/bfloat16/float32 now!"));
  }
  return "";
}

std::string debug_cnnl_pool_mode(cnnlPoolingMode_t pool_mode) {
  switch (pool_mode) {
    case CNNL_POOLING_MAX:
      return "max";
    case CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING:
      return "avg_include_padding";
    case CNNL_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING:
      return "avg_exclude_padding";
    case CNNL_POOLING_FIXED:
      return "fixed";
    default:
      PADDLE_THROW(
          phi::errors::InvalidArgument("Pool only support max and avg now!"));
  }
  return "";
}

class CnnlRandGeneratorFactory {
 public:
  enum class CnnlRandGeneratorType {
    GENERATOR_DEFAULT,
    GENERATOR_GAUSSIAN,
    GENERATOR_UNIFORM,
    GENERATOR_RANDINT,
  };

  static CnnlRandGenerator &Get(CnnlRandGeneratorType type) {
    switch (type) {
      case CnnlRandGeneratorType::GENERATOR_GAUSSIAN:
        static CnnlRandGenerator gaussian_generator(CNNL_RAND_RNG_PHILOX);
        return gaussian_generator;
      case CnnlRandGeneratorType::GENERATOR_UNIFORM:
        static CnnlRandGenerator uniform_generator(CNNL_RAND_RNG_PHILOX);
        return uniform_generator;
      case CnnlRandGeneratorType::GENERATOR_RANDINT:
        static CnnlRandGenerator randint_generator(CNNL_RAND_RNG_PHILOX);
        return randint_generator;
      default:
        static CnnlRandGenerator default_generator;
        return default_generator;
    }
  }
};

*/

void cinn_call_onednn_gaussian_random(
    void *v_args, int num_args, float mean, float std, int seed, void* stream) {
  /*
  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);
  cinn_buffer_t *output = args[0].operator cinn_buffer_t *();
  cinn_type_t dtype = output->type;
  size_t numel = output->num_elements();

  auto Queue = SYCLBackendAPI::Global()->get_now_queue();
  CNdev device = Queue->get_device().get_native<::sycl::backend::ext_oneapi_cnrt>();
  CNRT_CALL(cnrtSetDevice(device));
  cnnlHandle_t handle = CnnlHandle::GetInstance().GetCnnlHandle();

  // CNqueue queue = Queue->get_native<::sycl::backend::ext_oneapi_cnrt>();
  cnrtQueue_t queue;
  CNRT_CALL(cnrtQueueCreate(&queue));
  CNNL_CALL(cnnlSetQueue(handle, queue));

  cnnlRandGenerator_t generator =
      CnnlRandGeneratorFactory::Get(
          CnnlRandGeneratorFactory::CnnlRandGeneratorType::GENERATOR_GAUSSIAN)
          .SetSeed(seed)
          .GetGenerator();

  VLOG(4) << "cinn_call_onednn_gaussian_random: output_size=" << numel
          << ", mean=" << mean << ", std=" << std << ", seed=" << seed;

  if (dtype == cinn_float32_t()) {
    float *ptr = reinterpret_cast<float *>(output->memory);
    CNNL_CALL(cnnlRandGenerateNormal(handle, generator, CNNL_DTYPE_FLOAT, NULL, numel, mean, std, ptr));
    CNRT_CALL(cnrtQueueSync(queue));
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "gaussian_random_sycl only support float32! Please check."));
  }
  CNRT_CALL(cnrtQueueDestroy(queue));
  */
 CINN_RUNTIME_NOT_IMPLEMENTED
}

void cinn_call_onednn_uniform_random(
    void *v_args, int num_args, float min, float max, int seed, void* stream) {
  // cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);
  // cinn_buffer_t *output = args[0].operator cinn_buffer_t *();
  // cinn_type_t dtype = output->type;
  // size_t numel = output->num_elements();

  // auto Queue = SYCLBackendAPI::Global()->get_now_queue();
  // CNdev device = Queue->get_device().get_native<::sycl::backend::ext_oneapi_cnrt>();
  // CNRT_CALL(cnrtSetDevice(device));
  // cnnlHandle_t handle = CnnlHandle::GetInstance().GetCnnlHandle();

  // cnrtQueue_t queue;
  // CNRT_CALL(cnrtQueueCreate(&queue));
  // CNNL_CALL(cnnlSetQueue(handle, queue));

  // cnnlRandGenerator_t generator =
  //     CnnlRandGeneratorFactory::Get(
  //         CnnlRandGeneratorFactory::CnnlRandGeneratorType::GENERATOR_UNIFORM)
  //         .SetSeed(seed)
  //         .GetGenerator();

  // VLOG(4) << "cinn_call_onednn_uniform_random: output_size=" << numel
  //         << ", min=" << min << ", max=" << max << ", seed=" << seed;

  // if (dtype == cinn_float32_t()) {
  //   float *ptr = reinterpret_cast<float *>(output->memory);
  //   CNNL_CALL(cnnlRandGenerateUniform(handle, generator, CNNL_DTYPE_FLOAT, NULL, numel, 0.0f, 1.0f, ptr));
  //   CNRT_CALL(cnrtQueueSync(queue));
  // } else {
  //   PADDLE_THROW(phi::errors::InvalidArgument(
  //       "uniform_random_sycl only support float32! Please check."));
  // }
  // CNRT_CALL(cnrtQueueDestroy(queue));
  CINN_RUNTIME_NOT_IMPLEMENTED
}

void cinn_call_onednn_randint(void *v_args, int num_args, int seed, void* stream) {
  // cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);
  // cinn_buffer_t *output = args[0].operator cinn_buffer_t *();
  // cinn_type_t dtype = output->type;
  // size_t numel = output->num_elements();

  // auto Queue = SYCLBackendAPI::Global()->get_now_queue();
  // CNdev device = Queue->get_device().get_native<::sycl::backend::ext_oneapi_cnrt>();
  // CNRT_CALL(cnrtSetDevice(device));
  // cnnlHandle_t handle = CnnlHandle::GetInstance().GetCnnlHandle();

  // cnrtQueue_t queue;
  // CNRT_CALL(cnrtQueueCreate(&queue));
  // CNNL_CALL(cnnlSetQueue(handle, queue));

  // VLOG(4) << "cinn_call_onednn_randint: output_size=" << numel << ", seed=" << seed;

  // cnnlRandGenerator_t generator =
  //     CnnlRandGeneratorFactory::Get(
  //         CnnlRandGeneratorFactory::CnnlRandGeneratorType::GENERATOR_RANDINT)
  //         .SetSeed(seed)
  //         .GetGenerator();

  // if (dtype == cinn_int32_t()) {
  //   unsigned int *ptr = reinterpret_cast<unsigned int *>(output->memory);
  //   // TODO: fix range
  //   CNNL_CALL(cnnlRandGenerateDescreteUniform(handle, generator, CNNL_DTYPE_INT32, NULL, numel, 0, 1 << 23, ptr));
  //   CNRT_CALL(cnrtQueueSync(queue));
  // } else {
  //   PADDLE_THROW(phi::errors::InvalidArgument(
  //       "randint only support int32! Please check."));
  // }
  // CNRT_CALL(cnrtQueueDestroy(queue));
  CINN_RUNTIME_NOT_IMPLEMENTED
}

void cinn_call_onednn_cholesky(void *v_args,
                              int num_args,
                              int batch_size,
                              int m,
                              bool upper,
                              void* stream) {
  CINN_RUNTIME_NOT_IMPLEMENTED
}

void cinn_call_onednn_triangular_solve(void *v_args,
                                      int num_args,
                                      int batch_size,
                                      int m,
                                      int k,
                                      bool left_side,
                                      bool upper,
                                      bool transpose_a,
                                      bool unit_diagonal,
                                      void* stream) {
  CINN_RUNTIME_NOT_IMPLEMENTED
}
void cinn_call_onednn_matmul(void *v_args,
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
  std::cout<<"============= cinn call onednn matmul ==============="<<std::endl; 
  cinn::utils::RecordEvent record_run("cinn_call_onednn_matmul",
                                      cinn::utils::EventType::kInstruction);
  PADDLE_ENFORCE_EQ(
      num_args,
      3,
      phi::errors::InvalidArgument(
          "Expected number of arguments is 3, but received %d.", num_args));
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

  int m = trans_o ? (trans_b ? b3 : b4) : (trans_a ? a4 : a3);
  int n = trans_o ? (trans_a ? a4 : a3) : (trans_b ? b3 : b4);
  int k = trans_a ? a3 : a4;

  VLOG(3) << "m: " << m << ", n: " << n << ", k: " << k;
  std::cout<< "m: " << m << ", n: " << n << ", k: " << k <<std::endl;

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

  auto a_onednn_tag = trans_a ? tag::ba : tag::ab;
  auto b_onednn_tag = trans_b ? tag::ba : tag::ab;
  auto o_onednn_tag = trans_o ? tag::ba : tag::ab;

  memory::dims a_dims = {m, k};
  memory::dims b_dims = {k, n};
  memory::dims c_dims = {m, n};

  auto a_md = memory::desc(a_dims, onednn_dtype, a_onednn_tag);
  auto b_md = memory::desc(b_dims, onednn_dtype, b_onednn_tag);
  auto c_md = memory::desc(c_dims, onednn_dtype, o_onednn_tag);
  
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
}


void cinn_call_onednn_conv2d_forward(void *v_args,
                                    int num_args,
                                    int format,
                                    float alpha,
                                    float beta,
                                    int input_n,
                                    int input_c,
                                    int input_h,
                                    int input_w,
                                    int filter_n,
                                    int filter_c,
                                    int filter_h,
                                    int filter_w,
                                    int pad_h,
                                    int pad_w,
                                    int stride_h,
                                    int stride_w,
                                    int dilation_h,
                                    int dilation_w,
                                    int groups,
                                    int output_n,
                                    int output_c,
                                    int output_h,
                                    int output_w,
                                    void* stream) {
  std::cout<<"============= cinn call onednn conv2d forward ==============="<<std::endl;                                      
  PADDLE_ENFORCE_EQ(
    num_args,
    3,
    phi::errors::InvalidArgument(
        "Expected number of argruments is 3, but recived %d.", num_args));
  cinn::utils::RecordEvent record_run("cinn_call_onednn_conv2d_forward",
                                      cinn::utils::EventType::kInstruction);
  

  dnnl::engine onednn_engine = OneDNNHandle::GetInstance().GetOneDNNEngine();
  dnnl::stream onednn_stream = OneDNNHandle::GetInstance().GetOneDNNStream();

  // Get tensor data layout
  memory::format_tag tensor_format;
  if (format == static_cast<int>(memory::format_tag::nchw)) {
    tensor_format = memory::format_tag::nchw;
  } else if (format == static_cast<int>(memory::format_tag::nhwc)) {
    tensor_format = memory::format_tag::nhwc;
  } else {
    //tensor_format = memory::format_tag::nchw;
    //std::cout<<"common::layout is: "<<format<<std::endl;
    CINN_RUNTIME_NOT_IMPLEMENTED
  }

  // Tensor dimensions.
  const memory::dim N = input_n, // batch size
          IC = input_c, // input channels
          IH = input_h, // input height
          IW = input_w, // input width
          OC = output_c, // output channels
          KH = filter_h, // weights height
          KW = filter_w, // weights width
          PH_L = pad_h, // height padding: left
          PH_R = pad_h, // height padding: right
          PW_L = pad_w, // width padding: left
          PW_R = pad_w, // width padding: right
          SH = stride_h, // height-wise stride
          SW = stride_w, // width-wise stride
          //DH = dilation_h,
          //DW = dilation_w,
          OH = output_h, // output height
          OW = output_w; // output width

  // Source (src), weights, bias, and destination (dst) tensors
  // dimensions.
  memory::dims src_dims = {N, IC, IH, IW};
  memory::dims weights_dims = {OC, IC, KH, KW};
  memory::dims bias_dims = {OC};
  memory::dims dst_dims = {N, OC, OH, OW};

  // Strides, padding dimensions.
  memory::dims strides_dims = {SH, SW};
  memory::dims padding_dims_l = {PH_L, PW_L};
  memory::dims padding_dims_r = {PH_R, PW_R};
  //memory::dims dilation_dims = {DH, DW};

  // TODO: Get data type
  auto data_type = convert_to_onednn_dtype(v_args, num_args);

  // Get input and output memory handle
  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);
  void *_x = args[0].operator cinn_buffer_t *()->memory;
  void *_w = args[1].operator cinn_buffer_t *()->memory;
  void *_y = args[2].operator cinn_buffer_t *()->memory;

  // Create memory objects for tensor data (src, weights, dst). In this
  // example, NCHW layout is assumed for src and dst, and OIHW for weights.
  auto conv_src_mem = dnnl::memory({src_dims, data_type, tensor_format}, onednn_engine, _x);
  auto conv_weights_mem = dnnl::memory({weights_dims, data_type, tag::oihw}, onednn_engine, _w);
  auto conv_dst_mem = dnnl::memory({dst_dims, data_type, tensor_format}, onednn_engine, _y);

  // Create memory descriptors with format_tag::any for the primitive. This
  // enables the convolution primitive to choose memory layouts for an
  // optimized primitive implementation, and these layouts may differ from the
  // ones provided by the user.
  auto conv_src_md = dnnl::memory::desc(src_dims, data_type, tensor_format);
  auto conv_weights_md = dnnl::memory::desc(weights_dims, data_type, tensor_format);
  auto conv_dst_md = dnnl::memory::desc(dst_dims, data_type, tensor_format);


  // Create primitive post-ops (ReLU).
  /*
  const float alpha = 0.f;
  const float beta = 0.f;
  post_ops conv_ops;
  conv_ops.append_eltwise(algorithm::eltwise_relu, alpha, beta);
  primitive_attr conv_attr;
  conv_attr.set_post_ops(conv_ops);
  */

  // Create primitive descriptor.
  auto conv_pd = convolution_forward::primitive_desc(onednn_engine,
          prop_kind::forward_inference, algorithm::convolution_direct,
          conv_src_md, conv_weights_md, conv_dst_md,
          strides_dims, 
          //dilation_dims,
          padding_dims_l, padding_dims_r);
  
  // Create the primitive.
  auto conv_prim = convolution_forward(conv_pd);

  // Primitive arguments.
  std::unordered_map<int, memory> conv_args;
  conv_args.insert({DNNL_ARG_SRC, conv_src_mem});
  conv_args.insert({DNNL_ARG_WEIGHTS, conv_weights_mem});
  //conv_args.insert({DNNL_ARG_BIAS, user_bias_mem});
  conv_args.insert({DNNL_ARG_DST, conv_dst_mem});

  // Primitive execution: convolution with ReLU.
  conv_prim.execute(onednn_stream, conv_args);
}

void cinn_call_onednn_conv2d_backward_data(void *v_args,
                                          int num_args,
                                          int format,
                                          float alpha,
                                          float beta,
                                          int input_n,
                                          int input_c,
                                          int input_h,
                                          int input_w,
                                          int filter_n,
                                          int filter_c,
                                          int filter_h,
                                          int filter_w,
                                          int pad_h,
                                          int pad_w,
                                          int stride_h,
                                          int stride_w,
                                          int dilation_h,
                                          int dilation_w,
                                          int groups,
                                          int output_n,
                                          int output_c,
                                          int output_h,
                                          int output_w,
                                          void* stream) {
  PADDLE_ENFORCE_EQ(
      num_args,
      3,
      phi::errors::InvalidArgument(
          "Expected number of argruments is 3, but recived %d.", num_args));
  // TODO reuse onednn_common(...)
  CINN_RUNTIME_NOT_IMPLEMENTED
}

void cinn_call_onednn_conv2d_backward_filter(void *v_args,
                                            int num_args,
                                            int format,
                                            float alpha,
                                            float beta,
                                            int input_n,
                                            int input_c,
                                            int input_h,
                                            int input_w,
                                            int filter_n,
                                            int filter_c,
                                            int filter_h,
                                            int filter_w,
                                            int pad_h,
                                            int pad_w,
                                            int stride_h,
                                            int stride_w,
                                            int dilation_h,
                                            int dilation_w,
                                            int groups,
                                            int output_n,
                                            int output_c,
                                            int output_h,
                                            int output_w,
                                            void* stream) {
  PADDLE_ENFORCE_EQ(
      num_args,
      3,
      phi::errors::InvalidArgument(
          "Expected number of argruments is 3, but recived %d.", num_args));
  // TODO
  CINN_RUNTIME_NOT_IMPLEMENTED
}

void cinn_call_onednn_pool2d_forward(void *v_args,
                                    int num_args,
                                    int mode,
                                    int format,
                                    float alpha,
                                    float beta,
                                    int input_n,
                                    int input_c,
                                    int input_h,
                                    int input_w,
                                    int kernel_h,
                                    int kernel_w,
                                    int pad_h,
                                    int pad_w,
                                    int stride_h,
                                    int stride_w,
                                    int output_n,
                                    int output_c,
                                    int output_h,
                                    int output_w,
                                    void* stream) {
  std::cout<<"============= cinn call onednn pool2d forward ==============="<<std::endl;                                      
  PADDLE_ENFORCE_EQ(
      num_args,
      2,
      phi::errors::InvalidArgument(
          "Expected number of argruments is 2, but recived %d.", num_args));
  cinn::utils::RecordEvent record_run("cinn_call_onednn_pool2d_forward",
                                      cinn::utils::EventType::kInstruction);


  dnnl::engine onednn_engine = OneDNNHandle::GetInstance().GetOneDNNEngine();
  dnnl::stream onednn_stream = OneDNNHandle::GetInstance().GetOneDNNStream();

  // Get pool mode
  dnnl::algorithm pool_mode;
  if (mode == static_cast<int>(dnnl::algorithm::pooling_max)) {
    pool_mode = dnnl::algorithm::pooling_max;
  } else if (mode == static_cast<int>(dnnl::algorithm::pooling_avg_exclude_padding)) {
    pool_mode = dnnl::algorithm::pooling_avg_exclude_padding;
  } else if (mode == static_cast<int>(dnnl::algorithm::pooling_avg_include_padding)) {
    pool_mode = dnnl::algorithm::pooling_avg_include_padding;
  } else {
    pool_mode = dnnl::algorithm::pooling_max;
    //CINN_RUNTIME_NOT_IMPLEMENTED
  }
  std::cout<<"pooling is: "<<mode<<" pool mode is: "<<static_cast<int>(pool_mode)<<std::endl;

  // Get tensor data layout
  memory::format_tag tensor_format;
  if (format == static_cast<int>(memory::format_tag::nchw)) {
    tensor_format = memory::format_tag::nchw;
  } else if (format == static_cast<int>(memory::format_tag::nhwc)) {
    tensor_format = memory::format_tag::nhwc;
  } else {
    tensor_format = memory::format_tag::nchw; 
    //CINN_RUNTIME_NOT_IMPLEMENTED
  }
  std::cout<<"common::layout is: "<<format<<" tensor format: "<<static_cast<int>(tensor_format)<<std::endl;

  // TODO: Get data type
  auto data_type = convert_to_onednn_dtype(v_args, num_args);

  // Tensor dimensions.
  const memory::dim N = input_n, // batch size
          IC = input_c, // input channels
          IH = input_h, // input height
          IW = input_w, // input width
          KH = kernel_h, // weights height
          KW = kernel_w, // weights width
          PH_L = pad_h, // height padding: left
          PH_R = pad_h, // height padding: right
          PW_L = pad_w, // width padding: left
          PW_R = pad_w, // width padding: right
          SH = stride_h, // height-wise stride
          SW = stride_w, // width-wise stride
          DH = 0, // height-wise dilation
          DW = 0, // width-wise dilation
          OH = output_h, // output height
          OW = output_w; // output width
    
  // Source (src) and destination (dst) tensors dimensions.
  memory::dims src_dims = {N, IC, IH, IW};
  memory::dims dst_dims = {N, IC, OH, OW};

  // Kernel dimensions.
  memory::dims kernel_dims = {KH, KW};

  // Strides, padding dimensions.
  memory::dims strides_dims = {SH, SW};
  memory::dims padding_dims_l = {PH_L, PW_L};
  memory::dims padding_dims_r = {PH_R, PW_R};
  memory::dims dilation = {DH, DW};

  // Get input and output memory handle
  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);
  void *_x = args[0].operator cinn_buffer_t *()->memory;
  void *_y = args[1].operator cinn_buffer_t *()->memory;


  // Create memory descriptors and memory objects for src and dst.
  auto src_md = dnnl::memory::desc(src_dims, data_type, tensor_format);
  auto dst_md = dnnl::memory::desc(dst_dims, data_type, tensor_format);

  // Create memory objects for tensor data (src, weights, dst). In this
  // example, NCHW layout is assumed for src and dst, and OIHW for weights.
  auto src_mem = dnnl::memory(src_md, onednn_engine, _x);
  auto dst_mem = dnnl::memory(dst_md, onednn_engine, _y);

 
  // Create primitive descriptor.
  auto pooling_pd = pooling_forward::primitive_desc(onednn_engine,
          prop_kind::forward_inference, pool_mode, src_md, dst_md,
          strides_dims, kernel_dims, dilation, padding_dims_l,
          padding_dims_r);

  // Create the primitive.
  auto pooling_prim = pooling_forward(pooling_pd);

  // Create workspace memory objects using memory descriptor created by the
  // primitive descriptor.
  // NOTE: Here, the workspace is required to save the indices where maximum
  // was found, and is used in backward pooling to perform upsampling.
  auto workspace_mem = dnnl::memory(pooling_pd.workspace_desc(), onednn_engine);
  
  // Primitive arguments. Set up in-place execution by assigning src as DST.
  std::unordered_map<int, memory> pooling_args;
  pooling_args.insert({DNNL_ARG_SRC, src_mem});
  pooling_args.insert({DNNL_ARG_DST, dst_mem});
  pooling_args.insert({DNNL_ARG_WORKSPACE, workspace_mem});

  // Primitive execution: pooling.
  pooling_prim.execute(onednn_stream, pooling_args);
}

void cinn_call_onednn_pool2d_backward(void *v_args,
                                     int num_args,
                                     int mode,
                                     int format,
                                     float alpha,
                                     float beta,
                                     int input_n,
                                     int input_c,
                                     int input_h,
                                     int input_w,
                                     int kernel_h,
                                     int kernel_w,
                                     int pad_h,
                                     int pad_w,
                                     int stride_h,
                                     int stride_w,
                                     int output_n,
                                     int output_c,
                                     int output_h,
                                     int output_w,
                                     void* stream) {
  PADDLE_ENFORCE_EQ(
      num_args,
      4,
      phi::errors::InvalidArgument(
          "Expected number of argruments is 4, but recived %d.", num_args));
  CINN_RUNTIME_NOT_IMPLEMENTED
}

#endif // CINN_WITH_ONEDNN

}  // namespace sycl
}  // namespace runtime
}  // namespace cinn
