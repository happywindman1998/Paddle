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
#include "paddle/cinn/runtime/sycl/sycl_util.h"
#include "paddle/cinn/runtime/cinn_runtime.h"
#include "paddle/cinn/utils/profiler.h"
#include "paddle/common/enforce.h"

#ifdef CINN_WITH_DNNL
#include <dnnl.hpp>
#include <dnnl_sycl.hpp>

using namespace dnnl;
using tag = memory::format_tag;
using dt = memory::data_type;
#endif

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

#ifdef CINN_WITH_DNNL

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
    ::sycl::queue *queue = SYCLBackendAPI::Global()->get_now_queue();
    ::sycl::device device = queue->get_device();
    ::sycl::context context = queue->get_context();

    onednn_engine = sycl_interop::make_engine(device, context);
    onednn_stream = sycl_interop::make_stream(onednn_engine, *queue);
  }

  dnnl::engine onednn_engine;
  dnnl::stream onednn_stream;
};

memory::data_type convert_to_onednn_dtype(void *v_args, int num_args) {
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
    std::stringstream ss;
    ss << "unsupported cudnn data type: " << static_cast<int>(type_code)
       << ", bits = " << bits;
    PADDLE_THROW(phi::errors::InvalidArgument(ss.str()));
  }
  return onednn_dtype;
}

memory::data_type convert_to_onednn_dtype(cinn_buffer_t *input) {
  PADDLE_ENFORCE_NOT_NULL(
      input, phi::errors::NotFound("the pointer of input is null"));
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
    std::stringstream ss;
    ss << "unsupported cudnn data type: " << static_cast<int>(type_code)
       << ", bits = " << bits;
    PADDLE_THROW(phi::errors::InvalidArgument(ss.str()));
  }
  return onednn_dtype;
}

std::string debug_onednn_tensor_format(memory::format_tag tensor_format) {
  switch (tensor_format) {
    case tag::nchw:
      return "NCHW";
    case tag::nhwc:
      return "NHWC";
    default:
      PADDLE_THROW(phi::errors::InvalidArgument(
          "Only support NCHW and NHWC data layout\n"));
  }
  return "";
}

std::string debug_onednn_tensor_dtype(memory::data_type tensor_dtype) {
  switch (tensor_dtype) {
    case dt::f32:
      return "float32";
    case dt::f16:
      return "float16";
    case dt::bf16:
      return "bfloat16";
    case dt::f64:
      return "float64";
    default:
      PADDLE_THROW(phi::errors::InvalidArgument(
          "Only support float16/bfloat16/float32/float64 now!"));
  }
  return "";
}

std::string debug_onednn_pool_mode(algorithm pool_mode) {
  switch (pool_mode) {
    case algorithm::pooling_max:
      return "max";
    case algorithm::pooling_avg_include_padding:
      return "avg_include_padding";
    case algorithm::pooling_avg_exclude_padding:
      return "avg_exclude_padding";
    default:
      PADDLE_THROW(
          phi::errors::InvalidArgument("Pool only support max and avg now!"));
  }
  return "";
}

void cinn_call_onednn_gaussian_random(
    void *v_args, int num_args, float mean, float std, int seed) {
 CINN_RUNTIME_NOT_IMPLEMENTED
}

void cinn_call_onednn_uniform_random(
    void *v_args, int num_args, float min, float max, int seed) {
  CINN_RUNTIME_NOT_IMPLEMENTED
}

void cinn_call_onednn_randint(void *v_args, int num_args, int seed) {
  CINN_RUNTIME_NOT_IMPLEMENTED
}

void cinn_call_onednn_cholesky(void *v_args,
                              int num_args,
                              int batch_size,
                              int m,
                              bool upper) {
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
                                      bool unit_diagonal) {
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
                          int b4) {
  cinn::utils::RecordEvent record_run("cinn_call_onednn_matmul",
                                      cinn::utils::EventType::kInstruction);
  PADDLE_ENFORCE_EQ(
      num_args,
      3,
      phi::errors::InvalidArgument(
          "Expected number of arguments is 3, but received %d.", num_args));
  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);

  dnnl::engine engine = OneDNNHandle::GetInstance().GetOneDNNEngine();
  dnnl::stream stream = OneDNNHandle::GetInstance().GetOneDNNStream();
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

  int trans_op_l = trans_o ? !trans_b: trans_a;
  int trans_op_r = trans_o ? !trans_a: trans_b;

  void *lhs = trans_o ? B : A;
  void *rhs = trans_o ? A : B;

  dt a_type = convert_to_onednn_dtype(args[0].operator cinn_buffer_t *());
  dt b_type = convert_to_onednn_dtype(args[1].operator cinn_buffer_t *());
  dt o_type = convert_to_onednn_dtype(args[2].operator cinn_buffer_t *());
  dt l_type = trans_o ? b_type : a_type;
  dt r_type = trans_o ? a_type : b_type;

  tag l_tag = trans_op_l ? tag::abdc : tag::abcd;
  tag r_tag = trans_op_r ? tag::abdc : tag::abcd;

  memory::dims l_dims = {trans_o ? b1 : a1, trans_o ? b2 : a2, m, k};
  memory::dims r_dims = {trans_o ? a1 : b1, trans_o ? a2 : b2, k, n};
  memory::dims o_dims = {std::max(a1, b1), std::max(a2, b2), m, n};

  auto src_md = memory::desc(l_dims, l_type, l_tag);
  auto weight_md = memory::desc(r_dims, r_type, r_tag);
  auto dst_md = memory::desc(o_dims, o_type, tag::abcd);

  auto src_mem = dnnl::memory(src_md, engine, lhs);
  auto weight_mem = dnnl::memory(weight_md, engine, rhs);
  auto dst_mem = dnnl::memory(dst_md, engine, C);

  primitive_attr matmul_attr;
  if (beta != 0.0f) {
    post_ops matmul_post_ops;
    matmul_post_ops.append_sum(beta);
    matmul_attr.set_post_ops(matmul_post_ops);
  }

  dnnl::memory src_scale_mem;
  if (alpha != 1.0f) {
    auto q = sycl_interop::get_queue(stream);
    src_scale_mem = dnnl::memory({{1}, dt::f32, tag::x}, engine);
    q.memcpy(src_scale_mem.get_data_handle(), &alpha, sizeof(float));
    matmul_attr.set_scales_mask(DNNL_ARG_SRC, 0);
  }

  auto matmul_pd = matmul::primitive_desc(engine, src_md, weight_md, dst_md, matmul_attr);
  auto matmul_prim = matmul(matmul_pd);

  std::unordered_map<int, memory> matmul_args;
  matmul_args.insert({DNNL_ARG_SRC, src_mem});
  matmul_args.insert({DNNL_ARG_WEIGHTS, weight_mem});
  matmul_args.insert({DNNL_ARG_DST, dst_mem});
  if (alpha != 1.0f) {
    matmul_args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, src_scale_mem});
  }

  matmul_prim.execute(stream, matmul_args);
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
                                    int output_w) {
  PADDLE_ENFORCE_GE(
    num_args,
    3,
    phi::errors::InvalidArgument(
        "Expected number of argruments >= 3, but recived %d.", num_args));

  dnnl::engine engine = OneDNNHandle::GetInstance().GetOneDNNEngine();
  dnnl::stream stream = OneDNNHandle::GetInstance().GetOneDNNStream();
  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);
  void *_x = args[0].operator cinn_buffer_t *()->memory;
  void *_w = args[1].operator cinn_buffer_t *()->memory;
  void *_y = args[2].operator cinn_buffer_t *()->memory;

  tag tensor_format = static_cast<tag>(format);
  dt data_type = convert_to_onednn_dtype(v_args, num_args);
  memory::dims input_dims = {input_n, input_c, input_h, input_w};
  memory::dims output_dims = {output_n, output_c, output_h, output_w};
  memory::dims weights_dims = groups > 1
      ? memory::dims{groups, filter_n / groups, filter_c, filter_h, filter_w}
      : memory::dims{filter_n, filter_c, filter_h, filter_w};
  tag weights_format = groups > 1
      ? tensor_format == tag::nchw ? tag::goihw : tag::gohwi
      : tensor_format == tag::nchw ? tag::oihw : tag::ohwi;

  std::string hash_key =
      "conv2d forward, layout=" + debug_onednn_tensor_format(tensor_format) +
      ", dtype=" + debug_onednn_tensor_dtype(data_type) + ", input_nchw={" +
      std::to_string(input_n) + "," + std::to_string(input_c) + "," +
      std::to_string(input_h) + "," + std::to_string(input_w) +
      "}, filter_nchw={" + std::to_string(filter_n) + "," +
      std::to_string(filter_c) + "," + std::to_string(filter_h) + "," +
      std::to_string(filter_w) + "}, output_nchw={" + std::to_string(output_n) +
      "," + std::to_string(output_c) + "," + std::to_string(output_h) + "," +
      std::to_string(output_w) + "}";
  VLOG(4) << hash_key;

  // only nchw is supported when using cnnl
  auto user_src_mem = dnnl::memory({input_dims, data_type, tensor_format}, engine, _x);
  auto user_weights_mem = dnnl::memory({weights_dims, data_type, weights_format}, engine, _w);
  auto user_dst_mem = dnnl::memory({output_dims, data_type, tensor_format}, engine, _y);

  auto src_md = memory::desc(input_dims, data_type, tag::any);
  auto weights_md = memory::desc(weights_dims, data_type, tag::any);
  auto dst_md = memory::desc(output_dims, data_type, tag::any);

  memory::dims strides = {stride_h, stride_w};
  memory::dims dilation = {dilation_h - 1, dilation_w - 1};
  memory::dims padding_l = {pad_h, pad_w};
  memory::dims padding_r = {pad_h, pad_w};

  primitive_attr conv_attr;
  if (beta != 0.0f) {
    post_ops conv_post_ops;
    conv_post_ops.append_sum(beta);
    conv_attr.set_post_ops(conv_post_ops);
  }

  auto conv_pd = convolution_forward::primitive_desc(engine,
      prop_kind::forward_inference, algorithm::convolution_direct,
      src_md, weights_md, dst_md, strides, dilation,
      padding_l, padding_r, conv_attr);

  // convert the data layout if needed
  memory src_mem = user_src_mem, weights_mem = user_weights_mem, dst_mem = user_dst_mem;
  if (conv_pd.src_desc() != user_src_mem.get_desc()) {
    src_mem = dnnl::memory(conv_pd.src_desc(), engine);
    reorder(user_src_mem, src_mem).execute(stream, user_src_mem, src_mem);
  }
  if (conv_pd.weights_desc() != user_weights_mem.get_desc()) {
    weights_mem = dnnl::memory(conv_pd.weights_desc(), engine);
    reorder(user_weights_mem, weights_mem).execute(stream, user_weights_mem, weights_mem);
  }
  if (conv_pd.dst_desc() != user_dst_mem.get_desc()) {
    dst_mem = dnnl::memory(conv_pd.dst_desc(), engine);
  }

  dnnl::memory src_scale_mem;
  if (alpha != 1.0f) {
    auto q = sycl_interop::get_queue(stream);
    src_scale_mem = dnnl::memory({{1}, dt::f32, tag::x}, engine);
    q.memcpy(src_scale_mem.get_data_handle(), &alpha, sizeof(float));
    conv_attr.set_scales_mask(DNNL_ARG_SRC, 0);
  }

  auto conv = convolution_forward(conv_pd);

  std::unordered_map<int, memory> conv_args;
  conv_args.insert({DNNL_ARG_SRC, src_mem});
  conv_args.insert({DNNL_ARG_WEIGHTS, weights_mem});
  conv_args.insert({DNNL_ARG_DST, dst_mem});
  if (alpha != 1.0f) {
    conv_args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, src_scale_mem});
  }

  conv.execute(stream, conv_args);
  if (conv_pd.dst_desc() != user_dst_mem.get_desc()) {
    reorder(dst_mem, user_dst_mem).execute(stream, dst_mem, user_dst_mem);
  }
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
                                          int output_w) {
  PADDLE_ENFORCE_GE(
      num_args,
      3,
      phi::errors::InvalidArgument(
          "Expected number of argruments >= 3, but recived %d.", num_args));
  dnnl::engine engine = OneDNNHandle::GetInstance().GetOneDNNEngine();
  dnnl::stream stream = OneDNNHandle::GetInstance().GetOneDNNStream();
  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);
  void *_w = args[0].operator cinn_buffer_t *()->memory;
  void *_dy = args[1].operator cinn_buffer_t *()->memory;
  void *_dx = args[2].operator cinn_buffer_t *()->memory;

  tag tensor_format = static_cast<tag>(format);
  dt data_type = convert_to_onednn_dtype(v_args, num_args);
  memory::dims input_dims = {input_n, input_c, input_h, input_w};
  memory::dims output_dims = {output_n, output_c, output_h, output_w};
  memory::dims weights_dims = groups > 1
      ? memory::dims{groups, filter_n / groups, filter_c, filter_h, filter_w}
      : memory::dims{filter_n, filter_c, filter_h, filter_w};
  tag weights_format = groups > 1
      ? tensor_format == tag::nchw ? tag::goihw : tag::gohwi
      : tensor_format == tag::nchw ? tag::oihw : tag::ohwi;

  std::string hash_key =
      "conv2d backward data, layout=" +
      debug_onednn_tensor_format(tensor_format) +
      ", dtype=" + debug_onednn_tensor_dtype(data_type) + ", input_nchw={" +
      std::to_string(input_n) + "," + std::to_string(input_c) + "," +
      std::to_string(input_h) + "," + std::to_string(input_w) +
      "}, filter_nchw={" + std::to_string(filter_n) + "," +
      std::to_string(filter_c) + "," + std::to_string(filter_h) + "," +
      std::to_string(filter_w) + "}, output_nchw={" + std::to_string(output_n) +
      "," + std::to_string(output_c) + "," + std::to_string(output_h) + "," +
      std::to_string(output_w) + "}";

  VLOG(4) << hash_key;

  auto user_diff_dst_mem = dnnl::memory({output_dims, data_type, tensor_format}, engine, _dy);
  auto user_weights_mem = dnnl::memory({weights_dims, data_type, weights_format}, engine, _w);
  auto user_diff_src_mem = dnnl::memory({input_dims, data_type, tensor_format}, engine, _dx);

  auto src_md = memory::desc(input_dims, data_type, tag::any);
  auto weights_md = memory::desc(weights_dims, data_type, tag::any);
  auto dst_md = memory::desc(output_dims, data_type, tag::any);

  memory::dims strides = {stride_h, stride_w};
  memory::dims dilation = {dilation_h - 1, dilation_w - 1};
  memory::dims padding_l = {pad_h, pad_w};
  memory::dims padding_r = {pad_h, pad_w};

  auto conv_fwd_pd = convolution_forward::primitive_desc(engine,
      prop_kind::forward_inference, algorithm::convolution_direct,
      src_md, weights_md, dst_md, strides, dilation,
      padding_l, padding_r);
  auto conv_bwd_data_pd = convolution_backward_data::primitive_desc(
      engine, algorithm::convolution_direct, src_md, weights_md,
      dst_md, strides, dilation, padding_l, padding_r, conv_fwd_pd);

  memory diff_dst_mem = user_diff_dst_mem, weights_mem = user_weights_mem, diff_src_mem = user_diff_src_mem;
  if (conv_bwd_data_pd.diff_dst_desc() != user_diff_dst_mem.get_desc()) {
    diff_dst_mem = dnnl::memory(conv_bwd_data_pd.diff_dst_desc(), engine);
    reorder(user_diff_dst_mem, diff_dst_mem).execute(stream, user_diff_dst_mem, diff_dst_mem);
  }
  if (conv_bwd_data_pd.weights_desc() != user_weights_mem.get_desc()) {
    weights_mem = dnnl::memory(conv_bwd_data_pd.weights_desc(), engine);
    reorder(user_weights_mem, weights_mem).execute(stream, user_weights_mem, weights_mem);
  }
  if (conv_bwd_data_pd.diff_src_desc() != user_diff_src_mem.get_desc()) {
    diff_src_mem = dnnl::memory(conv_bwd_data_pd.diff_src_desc(), engine);
  }

  auto conv_bwd_data = convolution_backward_data(conv_bwd_data_pd);

  std::unordered_map<int, memory> conv_bwd_data_args;
  conv_bwd_data_args.insert({DNNL_ARG_DIFF_DST, diff_dst_mem});
  conv_bwd_data_args.insert({DNNL_ARG_WEIGHTS, weights_mem});
  conv_bwd_data_args.insert({DNNL_ARG_DIFF_SRC, diff_src_mem});

  conv_bwd_data.execute(stream, conv_bwd_data_args);
  if (conv_bwd_data_pd.diff_src_desc() != user_diff_src_mem.get_desc()) {
    reorder(diff_src_mem, user_diff_src_mem).execute(stream, diff_src_mem, user_diff_src_mem);
  }
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
                                            int output_w) {
  PADDLE_ENFORCE_GE(
      num_args,
      3,
      phi::errors::InvalidArgument(
          "Expected number of argruments >= 3, but recived %d.", num_args));
  dnnl::engine engine = OneDNNHandle::GetInstance().GetOneDNNEngine();
  dnnl::stream stream = OneDNNHandle::GetInstance().GetOneDNNStream();
  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);
  void *_x = args[0].operator cinn_buffer_t *()->memory;
  void *_dy = args[1].operator cinn_buffer_t *()->memory;
  void *_dw = args[2].operator cinn_buffer_t *()->memory;

  tag tensor_format = static_cast<tag>(format);
  dt data_type = convert_to_onednn_dtype(v_args, num_args);
  memory::dims input_dims = {input_n, input_c, input_h, input_w};
  memory::dims output_dims = {output_n, output_c, output_h, output_w};
  memory::dims weights_dims = groups > 1
      ? memory::dims{groups, filter_n / groups, filter_c, filter_h, filter_w}
      : memory::dims{filter_n, filter_c, filter_h, filter_w};
  tag weights_format = groups > 1
      ? tensor_format == tag::nchw ? tag::goihw : tag::gohwi
      : tensor_format == tag::nchw ? tag::oihw : tag::ohwi;

  std::string hash_key =
      "conv2d backward filter, layout=" +
      debug_onednn_tensor_format(tensor_format) +
      ", dtype=" + debug_onednn_tensor_dtype(data_type) + ", input_nchw={" +
      std::to_string(input_n) + "," + std::to_string(input_c) + "," +
      std::to_string(input_h) + "," + std::to_string(input_w) +
      "}, filter_nchw={" + std::to_string(filter_n) + "," +
      std::to_string(filter_c) + "," + std::to_string(filter_h) + "," +
      std::to_string(filter_w) + "}, output_nchw={" + std::to_string(output_n) +
      "," + std::to_string(output_c) + "," + std::to_string(output_h) + "," +
      std::to_string(output_w) + "}";

  VLOG(4) << hash_key;

  auto user_diff_dst_mem = dnnl::memory({output_dims, data_type, tensor_format}, engine, _dy);
  auto user_src_mem = dnnl::memory({input_dims, data_type, tensor_format}, engine, _x);
  auto user_diff_weights_mem = dnnl::memory({weights_dims, data_type, weights_format}, engine, _dw);

  auto src_md = memory::desc(input_dims, data_type, tag::any);
  auto weights_md = memory::desc(weights_dims, data_type, tag::any);
  auto dst_md = memory::desc(output_dims, data_type, tag::any);

  memory::dims strides = {stride_h, stride_w};
  memory::dims dilation = {dilation_h - 1, dilation_w - 1};
  memory::dims padding_l = {pad_h, pad_w};
  memory::dims padding_r = {pad_h, pad_w};

  auto conv_fwd_pd = convolution_forward::primitive_desc(engine,
      prop_kind::forward_inference, algorithm::convolution_direct,
      src_md, weights_md, dst_md, strides, dilation,
      padding_l, padding_r);
  auto conv_bwd_weights_pd = convolution_backward_weights::primitive_desc(
      engine, algorithm::convolution_direct, src_md, weights_md,
      dst_md, strides, dilation, padding_l, padding_r, conv_fwd_pd);

  auto diff_dst_mem = user_diff_dst_mem, src_mem = user_src_mem, diff_weights_mem = user_diff_weights_mem;
  if (conv_bwd_weights_pd.diff_dst_desc() != user_diff_dst_mem.get_desc()) {
    diff_dst_mem = dnnl::memory(conv_bwd_weights_pd.diff_dst_desc(), engine);
    reorder(user_diff_dst_mem, diff_dst_mem).execute(stream, user_diff_dst_mem, diff_dst_mem);
  }
  if (conv_bwd_weights_pd.src_desc() != user_src_mem.get_desc()) {
    src_mem = dnnl::memory(conv_bwd_weights_pd.src_desc(), engine);
    reorder(user_src_mem, src_mem).execute(stream, user_src_mem, src_mem);
  }
  if (conv_bwd_weights_pd.diff_weights_desc() != user_diff_weights_mem.get_desc()) {
    diff_weights_mem = dnnl::memory(conv_bwd_weights_pd.diff_weights_desc(), engine);
  }

  auto conv_bwd_weights = convolution_backward_weights(conv_bwd_weights_pd);

  std::unordered_map<int, memory> conv_bwd_weights_args;
  conv_bwd_weights_args.insert({DNNL_ARG_DIFF_DST, diff_dst_mem});
  conv_bwd_weights_args.insert({DNNL_ARG_SRC, src_mem});
  conv_bwd_weights_args.insert({DNNL_ARG_DIFF_WEIGHTS, diff_weights_mem});

  conv_bwd_weights.execute(stream, conv_bwd_weights_args);
  if (conv_bwd_weights_pd.diff_weights_desc() != user_diff_weights_mem.get_desc()) {
    reorder(diff_weights_mem, user_diff_weights_mem).execute(stream, diff_weights_mem, user_diff_weights_mem);
  }
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
                                    int output_w) {
  PADDLE_ENFORCE_EQ(
      num_args,
      2,
      phi::errors::InvalidArgument(
          "Expected number of argruments is 2, but recived %d.", num_args));
  dnnl::engine engine = OneDNNHandle::GetInstance().GetOneDNNEngine();
  dnnl::stream stream = OneDNNHandle::GetInstance().GetOneDNNStream();
  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);

  void *_x = args[0].operator cinn_buffer_t *()->memory;
  void *_y = args[1].operator cinn_buffer_t *()->memory;

  dnnl::algorithm pool_mode = static_cast<dnnl::algorithm>(mode);
  tag tensor_format = static_cast<tag>(format);
  dt data_type = convert_to_onednn_dtype(v_args, num_args);

  std::string hash_key =
      "pool2d forward, layout=" + debug_onednn_tensor_format(tensor_format) +
      ", pool_type=" + debug_onednn_pool_mode(pool_mode) +
      ", dtype=" + debug_onednn_tensor_dtype(data_type) + ", input_nchw={" +
      std::to_string(input_n) + "," + std::to_string(input_c) + "," +
      std::to_string(input_h) + "," + std::to_string(input_w) +
      "}, kernel_hw={" + std::to_string(kernel_h) + "," +
      std::to_string(kernel_w) + "}, pad_hw={" + std::to_string(pad_h) + "," +
      std::to_string(pad_w) + "}, stride_hw={" + std::to_string(stride_h) +
      "," + std::to_string(stride_w) + "}, output_nchw={" +
      std::to_string(output_n) + "," + std::to_string(output_c) + "," +
      std::to_string(output_h) + "," + std::to_string(output_w) + "}";

  VLOG(4) << hash_key;

  memory::dims src_dims = {input_n, input_c, input_h, input_w};
  memory::dims dst_dims = {output_n, output_c, output_h, output_w};

  memory::dims kernel_dims = {kernel_h, kernel_w};
  memory::dims strides = {stride_h, stride_w};
  memory::dims padding_l = {pad_h, pad_w};
  memory::dims padding_r = {pad_h, pad_w};
  memory::dims dilation = {0, 0};

  auto src_md = dnnl::memory::desc(src_dims, data_type, tensor_format);
  auto dst_md = dnnl::memory::desc(dst_dims, data_type, tensor_format);
  auto src_mem = dnnl::memory(src_md, engine, _x);
  auto dst_mem = dnnl::memory(dst_md, engine, _y);

  auto pooling_pd = pooling_forward::primitive_desc(engine,
      prop_kind::forward_inference, pool_mode, src_md, dst_md,
      strides, kernel_dims, dilation, padding_l, padding_r);
  auto pooling = pooling_forward(pooling_pd);

  std::unordered_map<int, memory> pooling_args;
  pooling_args.insert({DNNL_ARG_SRC, src_mem});
  pooling_args.insert({DNNL_ARG_DST, dst_mem});

  pooling.execute(stream, pooling_args);
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
                                     int output_w) {
  PADDLE_ENFORCE_EQ(
      num_args,
      4,
      phi::errors::InvalidArgument(
          "Expected number of argruments is 4, but recived %d.", num_args));
  dnnl::engine engine = OneDNNHandle::GetInstance().GetOneDNNEngine();
  dnnl::stream stream = OneDNNHandle::GetInstance().GetOneDNNStream();
  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);

  void *_x = args[0].operator cinn_buffer_t *()->memory;
  void *_y = args[1].operator cinn_buffer_t *()->memory;
  void *_dy = args[2].operator cinn_buffer_t *()->memory;
  void *_dx = args[3].operator cinn_buffer_t *()->memory;

  dnnl::algorithm pool_mode = static_cast<dnnl::algorithm>(mode);
  tag tensor_format = static_cast<tag>(format);
  dt data_type = convert_to_onednn_dtype(v_args, num_args);

  std::string hash_key =
      "pool2d backward, layout=" + debug_onednn_tensor_format(tensor_format) +
      ", pool_type=" + debug_onednn_pool_mode(pool_mode) +
      ", dtype=" + debug_onednn_tensor_dtype(data_type) + ", input_nchw={" +
      std::to_string(input_n) + "," + std::to_string(input_c) + "," +
      std::to_string(input_h) + "," + std::to_string(input_w) +
      "}, kernel_hw={" + std::to_string(kernel_h) + "," +
      std::to_string(kernel_w) + "}, pad_hw={" + std::to_string(pad_h) + "," +
      std::to_string(pad_w) + "}, stride_hw={" + std::to_string(stride_h) +
      "," + std::to_string(stride_w) + "}, output_nchw={" +
      std::to_string(output_n) + "," + std::to_string(output_c) + "," +
      std::to_string(output_h) + "," + std::to_string(output_w) + "}";

  VLOG(4) << hash_key;

  memory::dims diff_src_dims = {input_n, input_c, input_h, input_w};
  memory::dims diff_dst_dims = {output_n, output_c, output_h, output_w};

  memory::dims kernel_dims = {kernel_h, kernel_w};
  memory::dims strides = {stride_h, stride_w};
  memory::dims padding_l = {pad_h, pad_w};
  memory::dims padding_r = {pad_h, pad_w};
  memory::dims dilation = {0, 0};

  auto diff_src_md = dnnl::memory::desc(diff_src_dims, data_type, tensor_format);
  auto diff_dst_md = dnnl::memory::desc(diff_dst_dims, data_type, tensor_format);
  auto diff_src_mem = dnnl::memory(diff_src_md, engine, _dx);
  auto diff_dst_mem = dnnl::memory(diff_dst_md, engine, _dy);

  auto pooling_fwd_pd = pooling_forward::primitive_desc(engine,
      prop_kind::forward, pool_mode, diff_src_md, diff_dst_md, 
      strides, kernel_dims, dilation, padding_l, padding_r);
  auto pooling_bwd_pd = pooling_backward::primitive_desc(engine,
      pool_mode, diff_src_md, diff_dst_md, strides, kernel_dims,
      dilation, padding_l, padding_r, pooling_fwd_pd);
  auto pooling_bwd = pooling_backward(pooling_bwd_pd);

  // workspace contains src and dst data
  auto workspace_md = pooling_bwd_pd.workspace_desc();
  auto workspace_mem = dnnl::memory(workspace_md, engine);
  auto workspace = (uint8_t *)workspace_mem.get_data_handle();
  ::sycl::queue q = dnnl::sycl_interop::get_queue(stream);
  q.memcpy(workspace, _x, diff_src_md.get_size());
  q.memcpy(workspace + diff_src_md.get_size(), _y, diff_dst_md.get_size());

  std::unordered_map<int, memory> pooling_bwd_args;
  pooling_bwd_args.insert({DNNL_ARG_DIFF_DST, diff_dst_mem});
  pooling_bwd_args.insert({DNNL_ARG_DIFF_SRC, diff_src_mem});
  pooling_bwd_args.insert({DNNL_ARG_WORKSPACE, workspace_mem});

  pooling_bwd.execute(stream, pooling_bwd_args);
}

#endif // CINN_WITH_DNNL

}  // namespace sycl
}  // namespace runtime
}  // namespace cinn
