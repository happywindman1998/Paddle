#include <glog/logging.h>
#include <vector>
#include <unordered_map>

#include <algorithm>
#include <string>

#include "paddle/cinn/backends/extern_func_jit_register.h"
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/runtime/sycl/onednn_util.h"
#include "paddle/cinn/runtime/custom_function.h"
#include "paddle/cinn/runtime/flags.h"
#include "paddle/cinn/utils/profiler.h"
#include "paddle/cinn/utils/timer.h"

#include "dnnl.hpp"
#include "dnnl_sycl.hpp"

using namespace dnnl;

using tag = memory::format_tag;
using dt = memory::data_type;

// Read from memory, write to handle
inline void read_from_dnnl_memory(void *handle, dnnl::memory &mem) {
  dnnl::engine eng = mem.get_engine();
  size_t size = mem.get_desc().get_size();

  if (!handle) throw std::runtime_error("handle is nullptr.");

  auto mkind = dnnl::sycl_interop::get_memory_kind(mem);
  if (mkind == dnnl::sycl_interop::memory_kind::buffer) {
    auto buffer = dnnl::sycl_interop::get_buffer<uint8_t>(mem);
    auto src = buffer.get_host_access();
    uint8_t *src_ptr = src.get_pointer();
    if (!src_ptr)
        throw std::runtime_error("get_pointer returned nullptr.");
    for (size_t i = 0; i < size; ++i)
        ((uint8_t *)handle)[i] = src_ptr[i];
  } else {
    assert(mkind == dnnl::sycl_interop::memory_kind::usm);
    uint8_t *src_ptr = (uint8_t *)mem.get_data_handle();
    if (!src_ptr)
        throw std::runtime_error("get_data_handle returned nullptr."); 
    auto sycl_queue
            = dnnl::sycl_interop::get_queue(dnnl::stream(eng));
    sycl_queue.memcpy(handle, src_ptr, size).wait();
  }
  return;
}

// Read from handle, write to memory
inline void write_to_dnnl_memory(void *handle, dnnl::memory &mem) {
  dnnl::engine eng = mem.get_engine();
  size_t size = mem.get_desc().get_size();

  if (!handle) throw std::runtime_error("handle is nullptr.");
  auto mkind = dnnl::sycl_interop::get_memory_kind(mem);
  if (mkind == dnnl::sycl_interop::memory_kind::buffer) {
    auto buffer = dnnl::sycl_interop::get_buffer<uint8_t>(mem);
    auto dst = buffer.get_host_access();
    uint8_t *dst_ptr = dst.get_pointer();
    if (!dst_ptr)
        throw std::runtime_error("get_pointer returned nullptr.");
    for (size_t i = 0; i < size; ++i)
        dst_ptr[i] = ((uint8_t *)handle)[i];
  } else {
    assert(mkind == dnnl::sycl_interop::memory_kind::usm);
    uint8_t *dst_ptr = (uint8_t *)mem.get_data_handle();
    if (!dst_ptr)
        throw std::runtime_error("get_data_handle returned nullptr.");
    auto sycl_queue
            = dnnl::sycl_interop::get_queue(dnnl::stream(eng));
    sycl_queue.memcpy(dst_ptr, handle, size).wait();
  }
  return;
}

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
    dnnl::engine tmp_engine(dnnl::engine::kind::gpu, 0);
    onednn_engine = tmp_engine;

    // Create dnnl::stream.
    dnnl::stream tmp_stream(tmp_engine);
    onednn_stream = tmp_stream;
  }

  dnnl::engine onednn_engine;
  dnnl::stream onednn_stream;
};

void cinn_gpu_onednn_mul(const std::vector<int> &attrs,
                         cinn_buffer_t *input1,
                         cinn_buffer_t *input2,
                         cinn_buffer_t *output) {
  
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

  // Allocate buffers
  std::vector<float> a_data(M*K, 0.5);
  std::vector<float> b_data(K*N, 0.5);
  std::vector<float> c_data(M*N, 0);
  
  // Source (A), weights (B), and destination (C) matrix dimensions.
  memory::dims a_dims = {M, K};
  memory::dims b_dims = {K, N};
  memory::dims c_dims = {M, N};

  auto a_md = memory::desc(a_dims, dt::f32, tag::ab);
  auto b_md = memory::desc(b_dims, dt::f32, tag::ab);
  auto c_md = memory::desc(c_dims, dt::f32, tag::ab);
  
  auto a_mem = memory(a_md, onednn_engine);
  auto b_mem = memory(b_md, onednn_engine);
  
  // Write data to memory object's handles.
  write_to_dnnl_memory(a_data.data(), a_mem);
  write_to_dnnl_memory(b_data.data(), b_mem);

  // Create primitive descriptor.
  auto matmul_pd = matmul::primitive_desc(onednn_engine, a_md, b_md, c_md);
  auto c_mem = memory(matmul_pd.dst_desc(), onednn_engine);

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
                      void *stream) {
  cinn::utils::RecordEvent record_run("cinn_call_onednn",
                                      cinn::utils::EventType::kInstruction);
  dnnl::engine onednn_engine = OneDNNHandle::GetInstance().GetOneDNNEngine();
  dnnl::stream onednn_stream = OneDNNHandle::GetInstance().GetOneDNNStream();
  
  //float *x_data = reinterpret_cast<float *>(input1->memory);
  //float *y_data = reinterpret_cast<float *>(input2->memory);
  //float *out_data = reinterpret_cast<float *>(output->memory);
  int M = a3;
  int N = a4;
  int K= b4;

  // Allocate buffers
  std::vector<float> a_data(M*K, 0.5);
  std::vector<float> b_data(K*N, 0.5);
  std::vector<float> c_data(M*N, 0);
  
  // Source (A), weights (B), and destination (C) matrix dimensions.
  memory::dims a_dims = {M, K};
  memory::dims b_dims = {K, N};
  memory::dims c_dims = {M, N};

  auto a_md = memory::desc(a_dims, dt::f32, tag::ab);
  auto b_md = memory::desc(b_dims, dt::f32, tag::ab);
  auto c_md = memory::desc(c_dims, dt::f32, tag::ab);
  
  auto a_mem = memory(a_md, onednn_engine);
  auto b_mem = memory(b_md, onednn_engine);
  
  // Write data to memory object's handles.
  write_to_dnnl_memory(a_data.data(), a_mem);
  write_to_dnnl_memory(b_data.data(), b_mem);

  // Create primitive descriptor.
  auto matmul_pd = matmul::primitive_desc(onednn_engine, a_md, b_md, c_md);
  auto c_mem = memory(matmul_pd.dst_desc(), onednn_engine);

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