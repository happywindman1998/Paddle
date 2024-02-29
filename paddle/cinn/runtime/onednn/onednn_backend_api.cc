#include "paddle/cinn/runtime/onednn/onednn_backend_api.h"
#include <glog/logging.h>
namespace cinn {
namespace runtime {
namespace onednn {
OneDNNBackendAPI* OneDNNBackendAPI::Global() {
  static auto* inst = new OneDNNBackendAPI();
  return inst;
}

void OneDNNBackendAPI::set_device(int device_id) {}

int OneDNNBackendAPI::get_device_property(DeviceProperty device_property,
                            std::optional<int> device_id) {}

void* OneDNNBackendAPI::malloc(size_t numBytes){
  return nullptr;
}

void OneDNNBackendAPI::free(void* data) {}

void OneDNNBackendAPI::memset(void* data, int value, size_t numBytes){}

void OneDNNBackendAPI::memcpy(void* dest, const void* src, size_t numBytes, MemcpyType type){}

void OneDNNBackendAPI::device_sync(){}



}  // namespace onednn
}  // namespace runtime
}  // namespace cinn