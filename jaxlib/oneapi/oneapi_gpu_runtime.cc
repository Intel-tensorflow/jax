/* Copyright 2026 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// OneAPI/SYCL GPU runtime wrappers for JAX.
// Reference: xla/stream_executor/sycl/sycl_gpu_runtime.cc

#include "jaxlib/oneapi/oneapi_gpu_runtime.h"

namespace jax {
namespace oneapi {

std::error_code SyclMemcpyAsync(void *dst, const void *src, size_t byte_count,
                                SyclMemcpyKind /*kind*/,
                                ::sycl::queue *stream) {
  if (byte_count == 0) {
    return SyclSuccess();
  }
  if (dst == nullptr || src == nullptr || stream == nullptr) {
    LOG(ERROR) << "SyclMemcpyAsync: null pointer (dst=" << dst
               << ", src=" << src << ", stream=" << stream << ")";
    return std::make_error_code(std::errc::invalid_argument);
  }
  JAX_ONEAPI_RETURN_IF_ERROR(stream->memcpy(dst, src, byte_count),
                             "SyclMemcpyAsync");
  return SyclSuccess();
}

std::error_code SyclGetLastError() { return SyclSuccess(); }

std::error_code SyclStreamSynchronize(::sycl::queue *stream) {
  if (stream == nullptr) {
    LOG(ERROR) << "SyclStreamSynchronize: null stream (sycl::queue*)";
    return std::make_error_code(std::errc::invalid_argument);
  }
  JAX_ONEAPI_RETURN_IF_ERROR(stream->wait(), "SyclStreamSynchronize");
  return SyclSuccess();
}

}  // namespace oneapi
}  // namespace jax
