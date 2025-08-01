/* Copyright 2025 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "jaxlib/mosaic/dialect/tpu/tpu_dialect.h"
#include "xla/mlir/utils/error_util.h"

namespace mlir::tpu {
namespace {

using ::testing::_;
using ::testing::HasSubstr;
using ::testing::status::StatusIs;

class TpuOpsVerificationTest : public ::testing::Test {
 protected:
  TpuOpsVerificationTest()
      : context_([]() {
          DialectRegistry registry;
          registry.insert<arith::ArithDialect, func::FuncDialect,
                          memref::MemRefDialect, TPUDialect>();
          return registry;
        }()),
        builder_(UnknownLoc::get(&context_), &context_) {
    context_.loadAllAvailableDialects();
    context_.printOpOnDiagnostic(true);
  }
  ~TpuOpsVerificationTest() {
    for (int i = ops_.size() - 1; i >= 0; --i) {
      ops_[i]->erase();
    }
  }

  template <typename OpTy, typename... Args>
  OpTy Create(Args&&... args) {
    OpTy op = builder_.create<OpTy>(std::forward<Args>(args)...);
    ops_.push_back(op.getOperation());
    return op;
  }

  template <typename OpTy>
  absl::Status VerifyOp(OpTy op) {
    BaseScopedDiagnosticHandler diag(&context_);
    if (op.verify().succeeded()) {
      return absl::OkStatus();
    }
    return diag.ConsumeStatus();
  }

  Type i32() { return builder_.getI32Type(); }

  MemRefType GetMemRefType(
      ArrayRef<int64_t> shape, Type element_type,
      std::optional<MemorySpace> memory_space = std::nullopt) {
    return MemRefType::get(
        shape, element_type, nullptr,
        memory_space.has_value()
            ? MemorySpaceAttr::get(builder_.getContext(), *memory_space)
            : Attribute());
  }

  Value AllocaI32(ArrayRef<int64_t> shape,
                  std::optional<MemorySpace> memory_space = std::nullopt) {
    return Create<memref::AllocaOp>(GetMemRefType(shape, i32(), memory_space))
        .getMemref();
  }

  Value AllocaSemaphore() {
    return Create<tpu::AllocaSemaphoreOp>(
               GetMemRefType({}, SemaphoreType::get(builder_.getContext()),
                             MemorySpace::kSemaphoreMem))
        .getResult();
  }

  Value ConstantI8Vector(ArrayRef<int64_t> shape, ArrayRef<int8_t> values) {
    return Create<arith::ConstantOp>(
               /*result=*/VectorType::get(shape, builder().getI8Type()),
               /*value=*/dyn_cast<TypedAttr>(
                   builder().getDenseI8ArrayAttr(values)))
        .getResult();
  }

  Value ConstantI32Vector(ArrayRef<int64_t> shape, ArrayRef<int32_t> values) {
    return Create<arith::ConstantOp>(
               /*result=*/VectorType::get(shape, i32()),
               /*value=*/dyn_cast<TypedAttr>(
                   builder().getDenseI32ArrayAttr(values)))
        .getResult();
  }

  ImplicitLocOpBuilder& builder() { return builder_; }

 private:
  MLIRContext context_;
  ImplicitLocOpBuilder builder_;
  std::vector<Operation*> ops_;
};

class TpuOpsVectorSubcoreVerificationTest : public TpuOpsVerificationTest {
 protected:
  TpuOpsVectorSubcoreVerificationTest() {
    auto func_op = Create<func::FuncOp>("vector_kernel",
                                        builder().getFunctionType({}, {}));
    func_op->setAttr(
        TPUDialect::GetCoreTypeKey(),
        CoreTypeAttr::get(builder().getContext(), CoreType::kScVectorSubcore));
    builder().setInsertionPointToStart(func_op.addEntryBlock());
  }
};

TEST_F(TpuOpsVerificationTest, VectorLoadVerificationWorks) {
  auto c0 = Create<arith::ConstantIndexOp>(0);
  Value memref = AllocaI32({8});
  auto vl = Create<VectorLoadOp>(
      /*result=*/VectorType::get({8}, i32()),
      /*base=*/memref,
      /*indices=*/ValueRange{c0},
      /*strides=*/builder().getDenseI32ArrayAttr({}),
      /*mask=*/nullptr);

  ASSERT_OK(VerifyOp(vl));
}

TEST_F(TpuOpsVerificationTest,
       VectorLoadRankOfStridesDoesNotMatchBaseMemrefRank) {
  auto c0 = Create<arith::ConstantIndexOp>(0);
  Value memref = AllocaI32({8});
  auto vl = Create<VectorLoadOp>(
      /*result=*/VectorType::get({8}, i32()),
      /*base=*/memref,
      /*indices=*/ValueRange{c0},
      /*strides=*/builder().getDenseI32ArrayAttr({1, 1, 1, 1}),
      /*mask=*/nullptr);
  ASSERT_THAT(VerifyOp(vl), StatusIs(_, HasSubstr("Expected 1 strides.")));
}

TEST_F(TpuOpsVerificationTest, VectorLoadStridesFeatureNotImplemented) {
  auto c0 = Create<arith::ConstantIndexOp>(0);
  Value memref = AllocaI32({8});
  auto vl = Create<VectorLoadOp>(
      /*result=*/VectorType::get({8}, i32()),
      /*base=*/memref,
      /*indices=*/ValueRange{c0},
      /*strides=*/builder().getDenseI32ArrayAttr({1}),
      /*mask=*/nullptr);
  ASSERT_THAT(
      VerifyOp(vl),
      StatusIs(
          _, HasSubstr("Not implemented: general vector load with strides.")));
}

TEST_F(TpuOpsVerificationTest, VectorLoadBaseAndResultTypesDoNotMatch) {
  auto c0 = Create<arith::ConstantIndexOp>(0);
  Value memref = AllocaI32({8});
  auto vl = Create<VectorLoadOp>(
      /*result=*/VectorType::get({8}, builder().getF32Type()),
      /*base=*/memref,
      /*indices=*/ValueRange{c0},
      /*strides=*/builder().getDenseI32ArrayAttr({}),
      /*mask=*/nullptr);

  ASSERT_THAT(
      VerifyOp(vl),
      StatusIs(_,
               HasSubstr("Expected base and result element type to match.")));
}

TEST_F(TpuOpsVerificationTest,
       VectorLoadRankOfIndicesDoesNotMatchBaseMemrefRank) {
  auto c0 = Create<arith::ConstantIndexOp>(0);
  Value memref = AllocaI32({8});
  auto vl = Create<VectorLoadOp>(
      /*result=*/VectorType::get({8}, i32()),
      /*base=*/memref,
      /*indices=*/ValueRange{c0, c0, c0},
      /*strides=*/builder().getDenseI32ArrayAttr({}),
      /*mask=*/nullptr);

  ASSERT_THAT(VerifyOp(vl), StatusIs(_, HasSubstr("Expected 1 indices.")));
}

TEST_F(TpuOpsVerificationTest, VectorLoadValidMaskSucceeds) {
  auto c0 = Create<arith::ConstantIndexOp>(0);
  Value memref = AllocaI32({8, 128});
  Value mask = ConstantI32Vector(/*shape=*/{8, 1},
                                 /*values=*/{1, 1, 1, 1, 1, 1, 1, 1});
  auto vl = Create<VectorLoadOp>(
      /*result=*/VectorType::get({8, 128}, i32()),
      /*base=*/memref,
      /*indices=*/ValueRange{c0, c0},
      /*strides=*/builder().getDenseI32ArrayAttr({}),
      /*mask=*/mask);

  ASSERT_OK(VerifyOp(vl));
}

TEST_F(TpuOpsVerificationTest, VectorLoadMaskInvalidResultBitWidth) {
  auto c0 = Create<arith::ConstantIndexOp>(0);
  auto memref = Create<memref::AllocaOp>(
      MemRefType::get({8, 128}, builder().getI64Type()));
  Value mask = ConstantI32Vector(/*shape=*/{8, 1},
                                 /*values=*/{1, 1, 1, 1, 1, 1, 1, 1});
  auto vl = Create<VectorLoadOp>(
      /*result=*/VectorType::get({8, 128}, builder().getI64Type()),
      /*base=*/memref.getMemref(),
      /*indices=*/ValueRange{c0, c0},
      /*strides=*/builder().getDenseI32ArrayAttr({}),
      /*mask=*/mask);

  ASSERT_THAT(
      VerifyOp(vl),
      StatusIs(
          _, HasSubstr(
                 "Not implemented: masked load with non-32-bit element type")));
}

TEST_F(TpuOpsVerificationTest,
       VectorLoadMaskNotBroadcastableToResultShapeInvalidMinor) {
  auto c0 = Create<arith::ConstantIndexOp>(0);
  Value memref = AllocaI32({8, 128});
  Value mask = ConstantI32Vector(/*shape=*/{8, 2},
                                 /*values=*/{1});
  auto vl = Create<VectorLoadOp>(
      /*result=*/VectorType::get({8, 128}, i32()),
      /*base=*/memref,
      /*indices=*/ValueRange{c0, c0},
      /*strides=*/builder().getDenseI32ArrayAttr({}),
      /*mask=*/mask);

  ASSERT_THAT(
      VerifyOp(vl),
      StatusIs(
          _, HasSubstr(
                 "Expected mask shape to be broadcastable to result shape.")));
}

TEST_F(TpuOpsVerificationTest,
       VectorLoadMaskNotBroadcastableToResultShapeInvalidMajor) {
  auto c0 = Create<arith::ConstantIndexOp>(0);
  Value memref = AllocaI32({8, 128});
  Value mask = ConstantI32Vector(/*shape=*/{5, 1},
                                 /*values=*/{1});
  auto vl = Create<VectorLoadOp>(
      /*result=*/VectorType::get({8, 128}, i32()),
      /*base=*/memref,
      /*indices=*/ValueRange{c0, c0},
      /*strides=*/builder().getDenseI32ArrayAttr({}),
      /*mask=*/mask);

  ASSERT_THAT(
      VerifyOp(vl),
      StatusIs(
          _, HasSubstr(
                 "Expected mask shape to be broadcastable to result shape.")));
}

TEST_F(TpuOpsVerificationTest, UnpackSubelementsValidIndex) {
  Value source = ConstantI8Vector(/*shape=*/{4, 8}, /*values=*/{1});
  auto unpack = Create<UnpackSubelementsOp>(
      /*output=*/VectorType::get({16}, builder().getI16Type()), source,
      /*index=*/builder().getI32IntegerAttr(1),
      /*pack_format=*/
      PackFormatAttr::get(builder().getContext(), PackFormat::kInterleaved));
  ASSERT_OK(VerifyOp(unpack));
}

TEST_F(TpuOpsVerificationTest, UnpackSubelementsInvalidIndex) {
  Value source = ConstantI8Vector(/*shape=*/{4, 8}, /*values=*/{1});
  auto unpack = Create<UnpackSubelementsOp>(
      /*output=*/VectorType::get({16}, builder().getI16Type()), source,
      /*index=*/builder().getI32IntegerAttr(4),
      /*pack_format=*/
      PackFormatAttr::get(builder().getContext(), PackFormat::kInterleaved));
  ASSERT_THAT(
      VerifyOp(unpack),
      StatusIs(
          _, HasSubstr("Index must be between 0 and the packing factor (2)")));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest, DmaElementTypeMismatch) {
  auto dma = Create<EnqueueDMAOp>(
      /*source=*/AllocaI32({1024, 256, 128}, MemorySpace::kHbm),
      /*source_semaphore=*/AllocaSemaphore(),
      /*target=*/
      Create<memref::AllocaOp>(GetMemRefType({1024, 256, 128},
                                             builder().getI64Type(),
                                             MemorySpace::kHbm))
          .getMemref(),
      /*target_semaphore=*/AllocaSemaphore(),
      /*device_id=*/nullptr,
      /*core_id=*/nullptr);

  ASSERT_THAT(
      VerifyOp(dma),
      StatusIs(_, HasSubstr("DMA source and target element type mismatch")));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest, DmaDynamicRankMismatch) {
  auto dma = Create<EnqueueDMAOp>(
      /*source=*/AllocaI32({ShapedType::kDynamic, 256, 128}, MemorySpace::kHbm),
      /*source_semaphore=*/AllocaSemaphore(),
      /*target=*/
      AllocaI32({ShapedType::kDynamic, ShapedType::kDynamic, 128},
                MemorySpace::kHbm),
      /*target_semaphore=*/AllocaSemaphore(),
      /*device_id=*/nullptr,
      /*core_id=*/nullptr);

  ASSERT_THAT(VerifyOp(dma),
              StatusIs(_, HasSubstr("DMA source and target shape mismatch.")));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest,
       IndirectDmaHbmChunkGatherVerificationWorks) {
  auto dma = Create<EnqueueIndirectDMAOp>(
      /*source=*/AllocaI32({1024, 256, 128}, MemorySpace::kHbm),
      /*target=*/AllocaI32({64, 256, 128}, MemorySpace::kVmem),
      /*offsets=*/AllocaI32({64}, MemorySpace::kVmem),
      /*semaphore=*/AllocaSemaphore(),
      /*offset_filter=*/nullptr,
      /*add=*/false);

  ASSERT_OK(VerifyOp(dma));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest,
       IndirectDmaVmemSharedChunkGatherVerificationWorks) {
  auto dma = Create<EnqueueIndirectDMAOp>(
      /*source=*/AllocaI32({1024, 256, 128}, MemorySpace::kVmemShared),
      /*target=*/AllocaI32({64, 256, 128}, MemorySpace::kVmem),
      /*offsets=*/AllocaI32({64}, MemorySpace::kVmem),
      /*semaphore=*/AllocaSemaphore(),
      /*offset_filter=*/nullptr,
      /*add=*/false);

  ASSERT_OK(VerifyOp(dma));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest,
       IndirectDmaSublaneGatherVerificationWorks) {
  auto dma = Create<EnqueueIndirectDMAOp>(
      /*source=*/AllocaI32({1024, 256, 128}, MemorySpace::kHbm),
      /*target=*/AllocaI32({64, 32, 128}, MemorySpace::kVmem),
      /*offsets=*/AllocaI32({64, 32}, MemorySpace::kVmem),
      /*semaphore=*/AllocaSemaphore(),
      /*offset_filter=*/nullptr,
      /*add=*/false);

  ASSERT_OK(VerifyOp(dma));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest,
       IndirectDmaElementGatherVerificationWorks) {
  auto dma = Create<EnqueueIndirectDMAOp>(
      /*source=*/AllocaI32({1024, 256, 128}, MemorySpace::kHbm),
      /*target=*/AllocaI32({64, 32, 128}, MemorySpace::kVmem),
      /*offsets=*/AllocaI32({64, 32, 128}, MemorySpace::kVmem),
      /*semaphore=*/AllocaSemaphore(),
      /*offset_filter=*/nullptr,
      /*add=*/false);

  ASSERT_OK(VerifyOp(dma));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest,
       IndirectDmaFilteredGatherVerificationWorks) {
  Value offset_filter =
      Create<arith::ConstantOp>(builder().getIntegerAttr(i32(), -1))
          .getResult();
  auto dma = Create<EnqueueIndirectDMAOp>(
      /*source=*/AllocaI32({1024, 256, 128}, MemorySpace::kHbm),
      /*target=*/AllocaI32({64, 32, 128}, MemorySpace::kVmem),
      /*offsets=*/AllocaI32({64, 32}, MemorySpace::kVmem),
      /*semaphore=*/AllocaSemaphore(),
      /*offset_filter=*/offset_filter,
      /*add=*/false);

  ASSERT_OK(VerifyOp(dma));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest,
       IndirectDmaVectorGatherVerificationWorks) {
  Value vector_of_offsets =
      ConstantI32Vector(/*shape=*/{8},
                        /*values=*/{0, 1, 2, 3, 4, 5, 6, 7});
  auto dma = Create<EnqueueIndirectDMAOp>(
      /*source=*/AllocaI32({1024, 32, 128}, MemorySpace::kHbm),
      /*target=*/AllocaI32({8, 32, 128}, MemorySpace::kVmem),
      /*offsets=*/vector_of_offsets,
      /*semaphore=*/AllocaSemaphore(),
      /*offset_filter=*/nullptr,
      /*add=*/false);

  ASSERT_OK(VerifyOp(dma));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest,
       IndirectDmaHbmScatterVerificationWorks) {
  auto dma = Create<EnqueueIndirectDMAOp>(
      /*source=*/AllocaI32({64, 32, 128}, MemorySpace::kVmem),
      /*target=*/AllocaI32({1024, 256, 128}, MemorySpace::kHbm),
      /*offsets=*/AllocaI32({64, 32}, MemorySpace::kVmem),
      /*semaphore=*/AllocaSemaphore(),
      /*offset_filter=*/nullptr,
      /*add=*/false);

  ASSERT_OK(VerifyOp(dma));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest,
       IndirectDmaVmemSharedScatterVerificationWorks) {
  auto dma = Create<EnqueueIndirectDMAOp>(
      /*source=*/AllocaI32({64, 32, 128}, MemorySpace::kVmem),
      /*target=*/AllocaI32({1024, 256, 128}, MemorySpace::kVmemShared),
      /*offsets=*/AllocaI32({64, 32}, MemorySpace::kVmem),
      /*semaphore=*/AllocaSemaphore(),
      /*offset_filter=*/nullptr,
      /*add=*/false);

  ASSERT_OK(VerifyOp(dma));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest,
       IndirectDmaVectorScatterVerificationWorks) {
  Value vector_of_offsets = ConstantI32Vector(
      /*shape=*/{16},
      /*values=*/{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
  auto dma = Create<EnqueueIndirectDMAOp>(
      /*source=*/AllocaI32({16, 32, 128}, MemorySpace::kVmem),
      /*target=*/AllocaI32({1024, 32, 128}, MemorySpace::kHbm),
      /*offsets=*/vector_of_offsets,
      /*semaphore=*/AllocaSemaphore(),
      /*offset_filter=*/nullptr,
      /*add=*/false);

  ASSERT_OK(VerifyOp(dma));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest,
       IndirectDmaScatterAddVerificationWorks) {
  auto dma = Create<EnqueueIndirectDMAOp>(
      /*source=*/AllocaI32({64, 32, 128}, MemorySpace::kVmem),
      /*target=*/AllocaI32({1024, 256, 128}, MemorySpace::kHbm),
      /*offsets=*/AllocaI32({64, 32}, MemorySpace::kVmem),
      /*semaphore=*/AllocaSemaphore(),
      /*offset_filter=*/nullptr,
      /*add=*/true);

  ASSERT_OK(VerifyOp(dma));
}

TEST_F(TpuOpsVerificationTest, IndirectDmaOnUnsupportedCore) {
  std::vector<CoreType> unsupported_cores = {CoreType::kScScalarSubcore,
                                             CoreType::kTc};
  for (CoreType unsupported_core : unsupported_cores) {
    auto func_op = Create<func::FuncOp>("scalar_kernel",
                                        builder().getFunctionType({}, {}));
    func_op->setAttr(
        TPUDialect::GetCoreTypeKey(),
        CoreTypeAttr::get(builder().getContext(), unsupported_core));
    builder().setInsertionPointToStart(func_op.addEntryBlock());
    auto dma = Create<EnqueueIndirectDMAOp>(
        /*source=*/AllocaI32({1024, 256, 128}, MemorySpace::kHbm),
        /*target=*/AllocaI32({64, 32, 128}, MemorySpace::kVmem),
        /*offsets=*/AllocaI32({64, 32}, MemorySpace::kVmem),
        /*semaphore=*/AllocaSemaphore(),
        /*offset_filter=*/nullptr,
        /*add=*/false);

    ASSERT_THAT(
        VerifyOp(dma),
        StatusIs(_, HasSubstr("Enqueue indirect DMA is supported only on "
                              "the SC vector subcore")));
  }
}

TEST_F(TpuOpsVerificationTest, IndirectDmaOnUnsupportedTc) {
  auto func_op =
      Create<func::FuncOp>("tc_kernel", builder().getFunctionType({}, {}));
  func_op->setAttr(TPUDialect::GetCoreTypeKey(),
                   CoreTypeAttr::get(builder().getContext(), CoreType::kTc));
  builder().setInsertionPointToStart(func_op.addEntryBlock());
  auto dma = Create<EnqueueIndirectDMAOp>(
      /*source=*/AllocaI32({1024, 256, 128}, MemorySpace::kHbm),
      /*target=*/AllocaI32({64, 32, 128}, MemorySpace::kVmem),
      /*offsets=*/AllocaI32({64, 32}, MemorySpace::kVmem),
      /*semaphore=*/AllocaSemaphore(),
      /*offset_filter=*/nullptr,
      /*add=*/false);

  ASSERT_THAT(VerifyOp(dma),
              StatusIs(_, HasSubstr("Enqueue indirect DMA is supported only on "
                                    "the SC vector subcore")));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest,
       IndirectDmaGatherSourceAndTargetTypeMismatch) {
  auto dma = Create<EnqueueIndirectDMAOp>(
      /*source=*/AllocaI32({1024, 256, 128}, MemorySpace::kHbm),
      /*target=*/
      Create<memref::AllocaOp>(GetMemRefType({64, 32, 128},
                                             builder().getI64Type(),
                                             MemorySpace::kVmem))
          .getMemref(),
      /*offsets=*/AllocaI32({64, 32}, MemorySpace::kVmem),
      /*semaphore=*/AllocaSemaphore(),
      /*offset_filter=*/nullptr,
      /*add=*/false);

  ASSERT_THAT(
      VerifyOp(dma),
      StatusIs(_, HasSubstr("Source and target element type mismatch")));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest, IndirectDmaWithoutLocalMem) {
  auto dma = Create<EnqueueIndirectDMAOp>(
      /*source=*/AllocaI32({1024, 256, 128}, MemorySpace::kHbm),
      /*target=*/AllocaI32({64, 32, 128}, MemorySpace::kHbm),
      /*offsets=*/AllocaI32({64, 32}, MemorySpace::kVmem),
      /*semaphore=*/AllocaSemaphore(),
      /*offset_filter=*/nullptr,
      /*add=*/false);

  ASSERT_THAT(VerifyOp(dma),
              StatusIs(_, HasSubstr("The transfer must be between HBM and "
                                    "VMEM, or between VMEM_SHARED and VMEM")));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest, IndirectDmaOffsetsNotInVmem) {
  auto dma = Create<EnqueueIndirectDMAOp>(
      /*source=*/AllocaI32({1024, 256, 128}, MemorySpace::kHbm),
      /*target=*/AllocaI32({64, 32, 128}, MemorySpace::kVmem),
      /*offsets=*/AllocaI32({64, 32}, MemorySpace::kHbm),
      /*semaphore=*/AllocaSemaphore(),
      /*offset_filter=*/nullptr,
      /*add=*/false);

  ASSERT_THAT(VerifyOp(dma),
              StatusIs(_, HasSubstr("Offsets memref must be in VMEM")));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest, IndirectDma1DSemaphore) {
  auto dma = Create<EnqueueIndirectDMAOp>(
      /*source=*/AllocaI32({1024, 256, 128}, MemorySpace::kHbm),
      /*target=*/AllocaI32({64, 32, 128}, MemorySpace::kVmem),
      /*offsets=*/AllocaI32({64, 32}, MemorySpace::kVmem),
      /*semaphore=*/
      Create<tpu::AllocaSemaphoreOp>(
          GetMemRefType({1}, SemaphoreType::get(builder().getContext()),
                        MemorySpace::kSemaphoreMem))
          .getResult(),
      /*offset_filter=*/nullptr,
      /*add=*/false);

  ASSERT_THAT(VerifyOp(dma),
              StatusIs(_, HasSubstr("Semaphore must be rank 0")));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest,
       IndirectDmaGatherTargetShapeInvalid) {
  auto dma = Create<EnqueueIndirectDMAOp>(
      /*source=*/AllocaI32({1024, 256, 128}, MemorySpace::kHbm),
      /*target=*/AllocaI32({512, 32, 128}, MemorySpace::kVmem),
      /*offsets=*/AllocaI32({64, 32}, MemorySpace::kVmem),
      /*semaphore=*/AllocaSemaphore(),
      /*offset_filter=*/nullptr,
      /*add=*/false);

  ASSERT_THAT(
      VerifyOp(dma),
      StatusIs(_,
               HasSubstr(
                   "Offsets shape (64, 32) must match the majormost dimensions "
                   "of the target (gather result) shape (512, 32, 128)")));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest,
       IndirectDmaVectorGatherTargetShapeInvalid) {
  Value vector_of_offsets =
      ConstantI32Vector(/*shape=*/{8},
                        /*values=*/{0, 1, 2, 3, 4, 5, 6, 7});
  auto dma = Create<EnqueueIndirectDMAOp>(
      /*source=*/AllocaI32({1024, 32, 128}, MemorySpace::kHbm),
      /*target=*/AllocaI32({512, 32, 128}, MemorySpace::kVmem),
      /*offsets=*/vector_of_offsets,
      /*semaphore=*/AllocaSemaphore(),
      /*offset_filter=*/nullptr,
      /*add=*/false);

  ASSERT_THAT(
      VerifyOp(dma),
      StatusIs(
          _, HasSubstr("Offsets shape (8) must match the majormost dimensions "
                       "of the target (gather result) shape (512, 32, 128)")));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest,
       IndirectDmaGatherOperandShapeInvalid) {
  auto dma = Create<EnqueueIndirectDMAOp>(
      /*source=*/AllocaI32({1024, 256, 512}, MemorySpace::kHbm),
      /*target=*/AllocaI32({64, 32, 128}, MemorySpace::kVmem),
      /*offsets=*/AllocaI32({64, 32}, MemorySpace::kVmem),
      /*semaphore=*/AllocaSemaphore(),
      /*offset_filter=*/nullptr,
      /*add=*/false);

  ASSERT_THAT(
      VerifyOp(dma),
      StatusIs(_,
               HasSubstr(
                   "1 minormost dimension of the source (gather operand) shape "
                   "(1024, 256, 512) must match the minormost dimension of "
                   "the target (gather result) shape (64, 32, 128)")));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest,
       IndirectDmaScatterUpdatesShapeInvalid) {
  auto dma = Create<EnqueueIndirectDMAOp>(
      /*source=*/AllocaI32({512, 32, 128}, MemorySpace::kVmem),
      /*target=*/AllocaI32({1024, 256, 128}, MemorySpace::kHbm),
      /*offsets=*/AllocaI32({64, 32}, MemorySpace::kVmem),
      /*semaphore=*/AllocaSemaphore(),
      /*offset_filter=*/nullptr,
      /*add=*/false);

  ASSERT_THAT(
      VerifyOp(dma),
      StatusIs(_,
               HasSubstr(
                   "Offsets shape (64, 32) must match the majormost dimensions "
                   "of the source (scatter updates) shape (512, 32, 128)")));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest,
       IndirectDmaScatterOperandShapeInvalid) {
  auto dma = Create<EnqueueIndirectDMAOp>(
      /*source=*/AllocaI32({64, 32, 128}, MemorySpace::kVmem),
      /*target=*/AllocaI32({1024, 256, 512}, MemorySpace::kHbm),
      /*offsets=*/AllocaI32({64, 32}, MemorySpace::kVmem),
      /*semaphore=*/AllocaSemaphore(),
      /*offset_filter=*/nullptr,
      /*add=*/false);

  ASSERT_THAT(
      VerifyOp(dma),
      StatusIs(
          _, HasSubstr(
                 "1 minormost dimension of the source (scatter updates) shape "
                 "(64, 32, 128) must match the minormost dimension of the "
                 "target (scatter operand) shape (1024, 256, 512)")));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest,
       IndirectDmaVectorScatterOperandShapeInvalid) {
  Value vector_of_offsets =
      ConstantI32Vector(/*shape=*/{8},
                        /*values=*/{0, 1, 2, 3, 4, 5, 6, 7});
  auto dma = Create<EnqueueIndirectDMAOp>(
      /*source=*/AllocaI32({8, 32, 128}, MemorySpace::kVmem),
      /*target=*/AllocaI32({1024, 96, 512}, MemorySpace::kHbm),
      /*offsets=*/vector_of_offsets,
      /*semaphore=*/AllocaSemaphore(),
      /*offset_filter=*/nullptr,
      /*add=*/false);

  ASSERT_THAT(
      VerifyOp(dma),
      StatusIs(
          _, HasSubstr(
                 "2 minormost dimensions of the source (scatter updates) shape "
                 "(8, 32, 128) must match the minormost dimensions of the "
                 "target (scatter operand) shape (1024, 96, 512)")));
}

TEST_F(TpuOpsVerificationTest, IndirectDmaWaitOnUnsupportedCoreInvalid) {
  static constexpr std::array<CoreType, 2> unsupported_cores = {
      CoreType::kScScalarSubcore, CoreType::kTc};
  for (CoreType unsupported_core : unsupported_cores) {
    SCOPED_TRACE(testing::Message()
                 << "Testing unsupported core type: "
                 << stringifyCoreType(unsupported_core).str());
    auto func_op = Create<func::FuncOp>("scalar_kernel",
                                        builder().getFunctionType({}, {}));
    func_op->setAttr(
        TPUDialect::GetCoreTypeKey(),
        CoreTypeAttr::get(builder().getContext(), unsupported_core));
    builder().setInsertionPointToStart(func_op.addEntryBlock());
    auto wait = Create<WaitIndirectDMAOp>(
        /*semaphore=*/AllocaSemaphore(),
        /*src=*/AllocaI32({1024, 256, 128}, MemorySpace::kHbm),
        /*dst=*/AllocaI32({64, 256, 128}, MemorySpace::kVmem));

    ASSERT_THAT(VerifyOp(wait),
                StatusIs(_, HasSubstr("Wait indirect DMA is supported only on "
                                      "the SC vector subcore")));
  }
}

TEST_F(TpuOpsVectorSubcoreVerificationTest,
       IndirectDmaWaitGatherVerificationWorks) {
  auto wait = Create<WaitIndirectDMAOp>(
      /*semaphore=*/AllocaSemaphore(),
      /*src=*/AllocaI32({1024, 256, 128}, MemorySpace::kHbm),
      /*dst=*/AllocaI32({64, 256, 128}, MemorySpace::kVmem));

  ASSERT_OK(VerifyOp(wait));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest,
       IndirectDmaWaitScatterVerificationWorks) {
  auto wait = Create<WaitIndirectDMAOp>(
      /*semaphore=*/AllocaSemaphore(),
      /*source=*/AllocaI32({64, 32, 128}, MemorySpace::kVmem),
      /*target=*/AllocaI32({1024, 256, 128}, MemorySpace::kVmemShared));

  ASSERT_OK(VerifyOp(wait));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest,
       IndirectDmaWaitWithoutLocalMemInvalid) {
  auto wait = Create<WaitIndirectDMAOp>(
      /*semaphore=*/AllocaSemaphore(),
      /*src=*/AllocaI32({1024, 256, 128}, MemorySpace::kHbm),
      /*dst=*/AllocaI32({64, 256, 128}, MemorySpace::kHbm));

  ASSERT_THAT(VerifyOp(wait),
              StatusIs(_, HasSubstr("The transfer must be between HBM and "
                                    "VMEM, or between VMEM_SHARED and VMEM")));
}

TEST_F(TpuOpsVectorSubcoreVerificationTest,
       IndirectDmaWaitInvalidSemaphoreRank) {
  auto wait = Create<WaitIndirectDMAOp>(
      /*semaphore=*/Create<tpu::AllocaSemaphoreOp>(
          GetMemRefType({8}, SemaphoreType::get(builder().getContext()),
                        MemorySpace::kSemaphoreMem))
          .getResult(),
      /*source=*/AllocaI32({64, 32, 128}, MemorySpace::kVmem),
      /*target=*/AllocaI32({1024, 256, 128}, MemorySpace::kVmemShared));

  ASSERT_THAT(
      VerifyOp(wait),
      StatusIs(_, HasSubstr("Indirect DMA wait semaphore must be rank 0")));
}
}  // namespace
}  // namespace mlir::tpu
