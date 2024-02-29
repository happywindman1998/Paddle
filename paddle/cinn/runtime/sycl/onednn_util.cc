#include <glog/logging.h>
#include <vector>
#include <unordered_map>

#include <algorithm>
#include <string>

#include "paddle/cinn/backends/extern_func_jit_register.h"
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/runtime/onednn/onednn_util.h"
#include "paddle/cinn/runtime/custom_function.h"
#include "paddle/cinn/runtime/flags.h"
#include "paddle/cinn/utils/profiler.h"
#include "paddle/cinn/utils/timer.h"

using namespace dnnl;

using tag = memory::format_tag;
using dt = memory::data_type;

// Read from handle, write to memory
static inline void write_to_dnnl_memory(void *handle, dnnl_memory_t mem) {
    dnnl_engine_t eng;
    dnnl_engine_kind_t eng_kind;
    const_dnnl_memory_desc_t md;

    CHECK(dnnl_memory_get_engine(mem, &eng));
    CHECK(dnnl_engine_get_kind(eng, &eng_kind));
    CHECK(dnnl_memory_get_memory_desc(mem, &md));
    size_t bytes = dnnl_memory_desc_get_size(md);
    
    void *mapped_ptr = NULL;
    CHECK(dnnl_memory_map_data(mem, &mapped_ptr));
    if (mapped_ptr) {
        for (size_t i = 0; i < bytes; ++i) {
            ((char *)mapped_ptr)[i] = ((char *)handle)[i];
        }
    }
    CHECK(dnnl_memory_unmap_data(mem, mapped_ptr));
    return;
}

// Read from memory, write to handle
static inline void read_from_dnnl_memory(void *handle, dnnl_memory_t mem) {
    dnnl_engine_t eng;
    dnnl_engine_kind_t eng_kind;
    const_dnnl_memory_desc_t md;

    CHECK(dnnl_memory_get_engine(mem, &eng));
    CHECK(dnnl_engine_get_kind(eng, &eng_kind));
    CHECK(dnnl_memory_get_memory_desc(mem, &md));
    size_t bytes = dnnl_memory_desc_get_size(md);

    void *mapped_ptr = NULL;
    CHECK(dnnl_memory_map_data(mem, &mapped_ptr));
    if (mapped_ptr) memcpy(handle, mapped_ptr, bytes);
    CHECK(dnnl_memory_unmap_data(mem, mapped_ptr));
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
    dnnl::engine engine(dnnl_gpu, 0);
    onednn_engine = engine;

    // Create dnnl::stream.
    dnnl::stream engine_stream(engine);
    onednn_stream = engine_stream;
  }

  dnnl::engine onednn_engine;
  dnnl::stream onednn_stream;
};

void cinn_gpu_onednn_mul(const std::vector<int> &attrs,
                         cinn_buffer_t *input1,
                         cinn_buffer_t *input2,
                         cinn_buffer_t *output,
                         cudaStream_t stream) {
  
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
  std::vector<float> a_data(M*k, 0.5);
  std::vector<float> b_data(K*N, 0.5);
  std::vector<float> c_data(M*N, 0);
  
  // Source (A), weights (B), and destination (C) matrix dimensions.
  memory::dims a_dims = {M, K};
  memory::dims b_dims = {K, N};
  memory::dims c_dims = {M, N};

  auto a_md = memory::desc(a_dims, dt::f32, tag::ab);
  auto b_md = memory::desc(b_dims, dt::f32, tag::ab);
  
  auto a_mem = memory(a_md, engine);
  auto b_mem = memory(b_md, engine);
  
  // Write data to memory object's handles.
  write_to_dnnl_memory(a_data.data(), a_mem);
  write_to_dnnl_memory(b_data.data(), b_mem);

  // Create primitive descriptor.
  auto matmul_pd = matmul::primitive_desc(engine, a_md, b_md, c_md);
  auto c_mem = memory(matmul_pd.dst_desc(), engine)

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