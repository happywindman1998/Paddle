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
    sycl::queue *sycl_queue = SYCLBackendAPI::Global()->get_now_queue();
    
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

void cinn_gpu_onednn_matmul(const std::vector<int> &attrs,
                          cinn_buffer_t *lhs,
                          cinn_buffer_t *rhs,
                          cinn_buffer_t *bias,
                          cinn_buffer_t *output,
                          void* vqueue) {
  
  std::cout<<"============= call gpu onednn matmul ==============="<<std::endl;
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
  //onednn_stream.wait();
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
                      void *vqueue) {
  cinn::utils::RecordEvent record_run("cinn_call_onednn_matmul",
                                      cinn::utils::EventType::kInstruction);

  std::cout<<"============= call onednn matmul ==============="<<std::endl;
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
    
  //onednn_stream.wait();
}

#define GetAttrValue(attr_map, key_name, default_value)      \
  int key_name = 0;                                          \
  if (attr_map.count(#key_name) != 0) {                      \
    key_name = attr_map.find(#key_name)->second;             \
  } else if (default_value >= 0) {                           \
    key_name = default_value;                                \
  } else {                                                   \
    LOG(FATAL) << #key_name << " is not exist in attr_map!"; \
  }

void cinn_gpu_onednn_conv2d(const absl::flat_hash_map<std::string, int> &attr,
                           cinn_buffer_t *x,
                           cinn_buffer_t *w,
                           cinn_buffer_t *y,
                           void* stream,
                           common::Layout target) {
  
  cinn::utils::RecordEvent record_run("cinn_gpu_onednn_conv2d",
                                      cinn::utils::EventType::kInstruction);
  
  std::cout<<"============= call gpu onednn conv2d ==============="<<std::endl;

  dnnl::engine onednn_engine = OneDNNHandle::GetInstance().GetOneDNNEngine();
  dnnl::stream onednn_stream = OneDNNHandle::GetInstance().GetOneDNNStream();

  // Get tensor data layout
  memory::format_tag tensor_format;
  if (target == common::Layout::kNCHW) {
    tensor_format = memory::format_tag::nchw;
  } else if (target == common::Layout::kNHWC) {
    tensor_format = memory::format_tag::nhwc;
  } else {
    CINN_NOT_IMPLEMENTED
  }

  GetAttrValue(attr, input_n, -1);
  GetAttrValue(attr, input_c, -1);
  GetAttrValue(attr, input_h, -1);
  GetAttrValue(attr, input_w, -1);
  GetAttrValue(attr, weights_n, -1);
  GetAttrValue(attr, weights_c, -1);
  GetAttrValue(attr, weights_h, -1);
  GetAttrValue(attr, weights_w, -1);
  GetAttrValue(attr, pad_h, 0);
  GetAttrValue(attr, pad_w, 0);
  GetAttrValue(attr, stride_h, 1);
  GetAttrValue(attr, stride_w, 1);
  GetAttrValue(attr, dilation_h, 1);
  GetAttrValue(attr, dilation_w, 1);
  GetAttrValue(attr, groups, 1);
  GetAttrValue(attr, output_n, -1);
  GetAttrValue(attr, output_c, -1);
  GetAttrValue(attr, output_h, -1);
  GetAttrValue(attr, output_w, -1);

  // Tensor dimensions.
  const memory::dim N = input_n, // batch size
          IC = input_c, // input channels
          IH = input_h, // input height
          IW = input_w, // input width
          OC = output_c, // output channels
          KH = weights_h, // weights height
          KW = weights_w, // weights width
          PH_L = pad_h, // height padding: left
          PH_R = pad_h, // height padding: right
          PW_L = pad_w, // width padding: left
          PW_R = pad_w, // width padding: right
          SH = stride_h, // height-wise stride
          SW = stride_w, // width-wise stride
          OH = (IH - KH + PH_L + PH_R) / SH + 1, // output height
          OW = (IW - KW + PW_L + PW_R) / SW + 1; // output width

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

  // TODO: Get data type
  auto data_type = convert_to_onednn_dtype(x);

  // Get input and output memory handle
  void *_x = x->memory;
  void *_w = w->memory;
  void *_y = y->memory;

  // Create memory objects for tensor data (src, weights, dst). In this
  // example, NCHW layout is assumed for src and dst, and OIHW for weights.
  auto conv_src_mem = dnnl::memory({src_dims, data_type, tag::nchw}, onednn_engine, _x);
  auto conv_weights_mem = dnnl::memory({weights_dims, data_type, tag::oihw}, onednn_engine, _w);
  auto conv_dst_mem = dnnl::memory({dst_dims, data_type, tag::nchw}, onednn_engine, _y);

  // Create memory descriptors with format_tag::any for the primitive. This
  // enables the convolution primitive to choose memory layouts for an
  // optimized primitive implementation, and these layouts may differ from the
  // ones provided by the user.
  auto conv_src_md = dnnl::memory::desc(src_dims, data_type, tag::any);
  auto conv_weights_md = dnnl::memory::desc(weights_dims, data_type, tag::any);
  auto conv_dst_md = dnnl::memory::desc(dst_dims, data_type, tag::any);


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
          strides_dims, padding_dims_l, padding_dims_r);
  
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

void cinn_gpu_onednn_conv2d_backward_data(
    const absl::flat_hash_map<std::string, int>& attr,
    cinn_buffer_t* w,
    cinn_buffer_t* dy,
    cinn_buffer_t* dx,
    void* stream) {}

void cinn_gpu_onednn_conv2d_backward_filter(
    const absl::flat_hash_map<std::string, int>& attr,
    cinn_buffer_t* x,
    cinn_buffer_t* dy,
    cinn_buffer_t* dw,
    void* stream) {}

void cinn_gpu_onednn_pool2d(const std::vector<int>& attrs,
                           const std::vector<std::string>& str_attrs,
                           cinn_buffer_t* input,
                           cinn_buffer_t* output,
                           void* stream ) {}

void cinn_gpu_onednn_softmax(const std::vector<int>& attrs,
                            cinn_buffer_t* input,
                            cinn_buffer_t* output,
                            void* stream ) {}




void cinn_call_onednn_conv2d_common(void* v_args,
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

  cinn::utils::RecordEvent record_run("cinn_call_onednn_conv2d_common",
                                      cinn::utils::EventType::kInstruction);
  
  std::cout<<"============= cinn call onednn conv2d common ==============="<<std::endl;

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
    CINN_NOT_IMPLEMENTED
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

void cinn_call_onednn_conv2d_forward(void* v_args,
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
  
  cinn_call_onednn_conv2d_common(v_args,
                                num_args,
                                format,
                                alpha,
                                beta,
                                input_n,
                                input_c,
                                input_h,
                                input_w,
                                filter_n,
                                filter_c,
                                filter_h,
                                filter_w,
                                pad_h,
                                pad_w,
                                stride_h,
                                stride_w,
                                dilation_h,
                                dilation_w,
                                groups,
                                output_n,
                                output_c,
                                output_h,
                                output_w,
                                stream);
}

void cinn_call_onednn_conv2d_backward_data(void* v_args,
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
  // TODO reuse onednn_common(...)
}

void cinn_call_onednn_conv2d_backward_filter(void* v_args,
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

}

void cinn_call_onednn_pool2d_common(void* v_args,
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
  
  
  cinn::utils::RecordEvent record_run("cinn_call_onednn_pool2d_common",
                                      cinn::utils::EventType::kInstruction);
  
  std::cout<<"============= cinn call onednn pool2d common ==============="<<std::endl;

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
    //CINN_NOT_IMPLEMENTED
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
    //CINN_NOT_IMPLEMENTED
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

void cinn_call_onednn_pool2d_forward(void* v_args,
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
  cinn_call_onednn_pool2d_common(v_args,
                                num_args,
                                mode,
                                format,
                                alpha,
                                beta,
                                input_n,
                                input_c,
                                input_h,
                                input_w,
                                kernel_h,
                                kernel_w,
                                pad_h,
                                pad_w,
                                stride_h,
                                stride_w,
                                output_n,
                                output_c,
                                output_h,
                                output_w,
                                stream);
}

void cinn_call_onednn_pool2d_backward(void* v_args,
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
                                     void* stream) {}

void cinn_call_onednn_softmax_common(void* v_args,
                                     int num_args,
                                     int mode,
                                     int format,
                                     float alpha,
                                     float beta,
                                     int input_n,
                                     int input_c,
                                     int input_h,
                                     int input_w,
                                     int output_n,
                                     int output_c,
                                     int output_h,
                                     int output_w,
                                     void* stream) {

  cinn::utils::RecordEvent record_run("cinn_call_onednn_softmax_common",
                                      cinn::utils::EventType::kInstruction);
  
  std::cout<<"============= cinn call onednn softmax common ==============="<<std::endl;

  dnnl::engine onednn_engine = OneDNNHandle::GetInstance().GetOneDNNEngine();
  dnnl::stream onednn_stream = OneDNNHandle::GetInstance().GetOneDNNStream();

  // Get softmax mode
  dnnl::algorithm softmax_mode;
  static_cast<dnnl::algorithm>(mode);
  if (mode == static_cast<int>(dnnl::algorithm::softmax_accurate)) {
    softmax_mode = dnnl::algorithm::softmax_accurate;
  } else if (mode == static_cast<int>(dnnl::algorithm::softmax_log)) {
    softmax_mode = dnnl::algorithm::softmax_log;
  }  else {
    CINN_NOT_IMPLEMENTED
  }

  // Get tensor data layout
  memory::format_tag tensor_format;
  if (format == static_cast<int>(memory::format_tag::nchw)) {
    tensor_format = memory::format_tag::nc;
  } else if (format == static_cast<int>(memory::format_tag::nhwc)) {
    tensor_format = memory::format_tag::nhwc;
  } else {
    tensor_format = memory::format_tag::nc;
    //std::cout<<"common::layout is: "<<format<<std::endl;
    CINN_NOT_IMPLEMENTED
  }

  // Get input and output memory handle
  cinn_pod_value_t *args = static_cast<cinn_pod_value_t *>(v_args);
  void *_x = args[0].operator cinn_buffer_t *()->memory;
  void *_y = args[1].operator cinn_buffer_t *()->memory;

  // TODO: Get data type
  auto data_type = convert_to_onednn_dtype(v_args, num_args);

  // Tensor dimensions.
  const memory::dim N = input_n, // batch size
            IC = input_c; // channels

  // Source (src) and destination (dst) tensors dimensions.
  memory::dims src_dims = {N, IC};

  // Create src memory descriptor and memory object.
  auto src_md = memory::desc(src_dims, data_type, tensor_format);
  auto dst_md = memory::desc(src_dims, data_type, tensor_format);
  auto src_mem = memory(src_md, onednn_engine, _x);
  auto dst_mem = memory(dst_md, onednn_engine, _y);

  // Softmax axis.
  const int axis = 1;

  // Create primitive descriptor.
  auto softmax_pd = softmax_forward::primitive_desc(onednn_engine,
          prop_kind::forward_inference, softmax_mode, src_md,
          dst_md, axis);

  // Create the primitive.
  auto softmax_prim = softmax_forward(softmax_pd);

  // Primitive arguments. Set up in-place execution by assigning src as DST.
  std::unordered_map<int, memory> softmax_args;
  softmax_args.insert({DNNL_ARG_SRC, src_mem});
  softmax_args.insert({DNNL_ARG_DST, src_mem});

  // Primitive execution.
  softmax_prim.execute(onednn_stream, softmax_args);


}

void cinn_call_onednn_softmax_forward(void* v_args,
                                     int num_args,
                                     int mode,
                                     int format,
                                     float alpha,
                                     float beta,
                                     int input_n,
                                     int input_c,
                                     int input_h,
                                     int input_w,
                                     int output_n,
                                     int output_c,
                                     int output_h,
                                     int output_w,
                                     void* stream) {

  cinn_call_onednn_softmax_common(v_args,
                                  num_args,
                                  mode,
                                  format,
                                  alpha,
                                  beta,
                                  input_n,
                                  input_c,
                                  input_h,
                                  input_w,
                                  output_n,
                                  output_c,
                                  output_h,
                                  output_w,
                                  stream);

}

void cinn_call_onednn_softmax_backward(void* v_args,
                                      int num_args,
                                      int mode,
                                      int format,
                                      float alpha,
                                      float beta,
                                      int input_n,
                                      int input_c,
                                      int input_h,
                                      int input_w,
                                      int output_n,
                                      int output_c,
                                      int output_h,
                                      int output_w,
                                      void* stream) {}


} // namespace Sycl
} // namespace runtime
} // namespace cinn