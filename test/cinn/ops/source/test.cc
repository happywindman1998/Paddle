#include <stdint.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#define CINN_CUDA_FP16
#define CINN_WITH_ROCM
//#define __HIPCC__

struct  float16 {
  uint16_t x;
    float16() = default;
  float16(const float16& o) = default;
  float16& operator=(const float16& o) = default;
  float16(float16&& o) = default;
  float16& operator=(float16&& o) = default;
  ~float16() = default;

#ifdef CINN_CUDA_FP16
  __host__ __device__ inline explicit float16(const half& h) {
#if defined(CINN_WITH_CUDA) || defined(CINN_WITH_ROCM)
#if defined(CINN_WITH_ROCM) || CUDA_VERSION >= 9000
    x = reinterpret_cast<__half_raw*>(const_cast<half*>(&h))->x;
#else
    x = h.x;
#endif  // CUDA_VERSION >= 9000
#endif
  }
#endif  // CINN_CUDA_FP16
};

__host__ __device__ inline bool operator>(const float16& a, const float16& b) {
    return true;
}

// #if defined(__HIPCC__)
// __device__ inline bool operator>(const float16& a, const float16& b) {
//   return __hgt(a.to_half(), b.to_half());
// }
// __host__ inline bool operator>(const float16& a, const float16& b) {
//   return static_cast<float>(a) > static_cast<float>(b);
// }
// #else  // __HIPCC__
// __host__ __device__ inline bool operator>(const float16& a, const float16& b) {
// #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
//   return __hgt(a.to_half(), b.to_half());
// #else
//   return static_cast<float>(a) > static_cast<float>(b);
// #endif
// }
// #endif  // __HIPCC__


__host__ __device__ inline float16 max(const float16& a, const float16& b) {
  return a > b ? a : b;
}

__device__ inline float16 fp16_max(float16 a, float16 b) { return max(a, b); }

int main(){

    return 0;
}
