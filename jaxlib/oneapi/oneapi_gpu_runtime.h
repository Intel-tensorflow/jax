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
//
// SYCL wrappers (SyclMemcpyAsync, SyclGetLastError,
// SyclStreamSynchronize) that use SYCL queue APIs and exception handling
// to return std::error_code. Detailed exception messages are preserved
// via LOG(ERROR) before returning the error code.
// vendor.h aliases these to the gpuXxx (e.g. gpuMemcpyAsync) abstractions.
//
// Reference: xla/stream_executor/sycl/sycl_gpu_runtime.cc

#ifndef JAXLIB_ONEAPI_ONEAPI_GPU_RUNTIME_H_
#define JAXLIB_ONEAPI_ONEAPI_GPU_RUNTIME_H_

#include <cstdint>
#include <exception>
#include <string>
#include <system_error>

#include <sycl/sycl.hpp>

#include "absl/base/optimization.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

// Wraps a SYCL expression in try-catch. On exception, logs the detailed
// exception message and returns the error code (std::error_code).
// Use in functions returning std::error_code.
// Usage: JAX_ONEAPI_RETURN_IF_ERROR(q->memcpy(dst, src, n), "SyclMemcpyAsync");
#define JAX_ONEAPI_RETURN_IF_ERROR(expr, name)                                 \
  do {                                                                         \
    try {                                                                      \
      (expr);                                                                  \
    } catch (const ::sycl::exception &e) {                                     \
      LOG(ERROR) << name << " failed: " << e.what() << ", file=" << __FILE__   \
                 << ", line=" << __LINE__;                                     \
      return e.code();                                                         \
    } catch (const std::exception &e) {                                        \
      LOG(ERROR) << name << " failed: " << e.what() << ", file=" << __FILE__   \
                 << ", line=" << __LINE__;                                     \
      return std::make_error_code(std::errc::io_error);                        \
    } catch (...) {                                                            \
      LOG(ERROR) << name << " failed: unknown exception"                       \
                 << ", file=" << __FILE__ << ", line=" << __LINE__;            \
      return std::make_error_code(std::errc::io_error);                        \
    }                                                                          \
  } while (0)

// Wraps a SYCL expression in try-catch. On exception, logs the error.
// Use in void functions (e.g., kernel launch wrappers).
// Usage: JAX_ONEAPI_LOG_IF_ERROR(q->submit([&](handler& h) {...}),
// "LaunchKernel");
#define JAX_ONEAPI_LOG_IF_ERROR(expr, name)                                    \
  do {                                                                         \
    try {                                                                      \
      (expr);                                                                  \
    } catch (const ::sycl::exception &e) {                                     \
      LOG(ERROR) << name << " failed: " << e.what() << ", file=" << __FILE__   \
                 << ", line=" << __LINE__;                                     \
    } catch (const std::exception &e) {                                        \
      LOG(ERROR) << name << " failed: " << e.what() << ", file=" << __FILE__   \
                 << ", line=" << __LINE__;                                     \
    } catch (...) {                                                            \
      LOG(ERROR) << name << " failed: unknown exception"                       \
                 << ", file=" << __FILE__ << ", line=" << __LINE__;            \
    }                                                                          \
  } while (0)

// Wraps a SYCL expression, assigns result to lhs. On exception, logs the
// detailed message and returns the error code (std::error_code).
// Use when capturing sycl::event from q->submit().
// Usage:
// sycl::event ev;
// JAX_ONEAPI_ASSIGN_WITH_NAME_OR_RETURN(ev, q->submit([&](handler& h) {
//     h.parallel_for(...);
// }), "LaunchKernel");
#define JAX_ONEAPI_ASSIGN_WITH_NAME_OR_RETURN(lhs, expr, name)                 \
  do {                                                                         \
    try {                                                                      \
      (lhs) = (expr);                                                          \
    } catch (const ::sycl::exception &e) {                                     \
      LOG(ERROR) << name << " failed: " << e.what() << ", file=" << __FILE__   \
                 << ", line=" << __LINE__;                                     \
      return e.code();                                                         \
    } catch (const std::exception &e) {                                        \
      LOG(ERROR) << name << " failed: " << e.what() << ", file=" << __FILE__   \
                 << ", line=" << __LINE__;                                     \
      return std::make_error_code(std::errc::io_error);                        \
    } catch (...) {                                                            \
      LOG(ERROR) << name << " failed: unknown exception"                       \
                 << ", file=" << __FILE__ << ", line=" << __LINE__;            \
      return std::make_error_code(std::errc::io_error);                        \
    }                                                                          \
  } while (0)

namespace jax {
namespace oneapi {

// q->memcpy We do not need to specify in which direction the copy is meant to
// happen but we keep the enum for API compatibility with call sites that pass
// gpuMemcpyDeviceToDevice etc. at call sites.
enum SyclMemcpyKind {
  SyclMemcpyHostToHost = 0,
  SyclMemcpyHostToDevice = 1,
  SyclMemcpyDeviceToHost = 2,
  SyclMemcpyDeviceToDevice = 3,
};

// Wraps sycl::queue::memcpy in a try-catch and returns std::error_code.
// The `kind` parameter is accepted for API compatibility but is not used
// by SYCL (memcpy direction is implicit in the source and destination
// pointers).
std::error_code SyclMemcpyAsync(void *dst, const void *src, size_t byte_count,
                                SyclMemcpyKind kind, ::sycl::queue *stream);

// Placeholder for gpuGetLastError. SYCL has no equivalent, returns success.
std::error_code SyclGetLastError();

// Stream synchronization wrapper.
// Wraps sycl::queue::wait() in a try-catch and returns std::error_code.
std::error_code SyclStreamSynchronize(::sycl::queue *stream);

// Maps std::error_code to a descriptive string.
inline std::string SyclGetErrorString(std::error_code error) {
  return absl::StrCat(error.message(), " [code=", error.value(),
                      ", category=", error.category().name(), "]");
}

inline std::error_code SyclSuccess() {
  return sycl::make_error_code(sycl::errc::success);
}

}  // namespace oneapi
}  // namespace jax

#endif  // JAXLIB_ONEAPI_ONEAPI_GPU_RUNTIME_H_
