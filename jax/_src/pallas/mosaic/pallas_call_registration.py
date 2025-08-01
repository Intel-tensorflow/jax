# Copyright 2023 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains registrations for pallas_call on TPU."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import os
import tempfile
from typing import cast

import jax
from jax import dtypes
from jax._src import config
from jax._src import core as jax_core
from jax._src import frozen_dict
from jax._src import sharding_impls
from jax._src import tpu_custom_call
from jax._src.interpreters import mlir
from jax._src.lib.mlir import ir
from jax._src.pallas import core as pallas_core
from jax._src.pallas.mosaic import core as tpu_core
from jax._src.pallas.mosaic import lowering
from jax._src.pallas.mosaic import verification
from jax.experimental import mosaic
from jax.experimental.mosaic.dialects import tpu


def _maybe_cast_to_int(x: jax.Array | jax_core.AbstractValue):
  """Casts boolean values to integers.

  We perform this cast because Mosaic does not directly support bool values
  for Memrefs. Instead, we load bools as integers and cast them to bools
  after loading from a memref inside of the kernel.
  """
  assert isinstance(
      x, (jax.Array, jax_core.ShapedArray, jax_core.DShapedArray)
  ), type(x)
  if isinstance(x, jax.Array):
    if dtypes.issubdtype(x.dtype, jax.numpy.bool_):
      return x.astype(lowering.BOOL_MEMREF_TYPE)
    return x
  else:
    if dtypes.issubdtype(x.dtype, jax.numpy.bool_):
      return jax_core.ShapedArray(x.shape, lowering.BOOL_MEMREF_TYPE)
    return x

_DUMP_PROMELA_TO = config.string_flag(
    "jax_pallas_dump_promela_to",
    default=os.getenv("JAX_PALLAS_DUMP_PROMELA_TO", ""),
    help=(
        "If set, dumps a Promela model of the kernel to the specified"
        " directory. The model can verify that the kernel is free of data"
        " races, deadlocks, etc."
    ),
)


def _get_memory_space_from_aval(
    out_aval: jax_core.AbstractValue,
) -> tpu_custom_call.MemorySpace | None:
  if not isinstance(out_aval, jax_core.ShapedArray):
    raise ValueError('Memory spaces not defined for non-ShapedArrays')
  if not isinstance(out_aval, pallas_core.ShapedArrayWithMemorySpace):
    # If we are passed a regular old ShapedArray, we don't constrain the
    # memory space
    return None
  # If we are passed an aval with an explicit memory space tag, we use it
  # to constrain the memory space.
  match out_aval.memory_space:
    case None:
      return None
    case tpu_core.MemorySpace.ANY:
      return None
    case tpu_core.MemorySpace.HBM:
      return tpu_custom_call.MemorySpace.HBM
    case tpu_core.MemorySpace.VMEM:
      return tpu_custom_call.MemorySpace.VMEM
    case tpu_core.MemorySpace.SMEM:
      return tpu_custom_call.MemorySpace.SMEM
    case tpu_core.MemorySpace.SEMAPHORE:
      return tpu_custom_call.MemorySpace.SEMAPHORE_MEM
  return None


def _get_memory_spaces_from_avals(
    avals: Sequence[jax_core.AbstractValue],
) -> tuple[tpu_custom_call.MemorySpace | None, ...] | None:
  memory_spaces = None
  if any(
      isinstance(aval, pallas_core.ShapedArrayWithMemorySpace) for aval in avals
  ):
    memory_spaces = tuple(map(_get_memory_space_from_aval, avals))
  return memory_spaces


def pallas_call_tpu_lowering_rule(
    ctx: mlir.LoweringRuleContext,
    *in_nodes,
    jaxpr: jax_core.Jaxpr,
    grid_mapping: pallas_core.GridMapping,
    mesh: pallas_core.Mesh | None,
    input_output_aliases: tuple[tuple[int, int], ...],
    debug: bool,
    interpret: bool,
    compiler_params: dict[str, pallas_core.CompilerParams],
    cost_estimate: pallas_core.CostEstimate | None,
    out_avals: tuple[jax_core.AbstractValue, ...],
    metadata: frozen_dict.FrozenDict[str, str] | None,
):
  """Lowers a pallas_call to a Mosaic TPU custom call."""
  del mesh, interpret  # Unused.

  debug_info = jaxpr._debug_info
  if debug:
    print(f"\nThe kernel jaxpr for pallas_call {debug_info.func_src_info}:")
    print(jaxpr)

  if "mosaic_tpu" in compiler_params:
    mosaic_params = cast(
        tpu_core.CompilerParams, compiler_params["mosaic_tpu"]
    )
  else:
    mosaic_params = tpu_core.CompilerParams()

  jax_mesh = None
  axis_context = ctx.module_context.axis_context
  if axis_context is not None:
    if isinstance(axis_context, sharding_impls.SPMDAxisContext):
      jax_mesh = axis_context.mesh
  mlir_ctx = mlir.JaxIrContext()
  mlir_ctx.append_dialect_registry(mlir.upstream_dialects)
  mlir_ctx.load_all_available_dialects()
  tpu.register_dialect(mlir_ctx)

  def lower_module(for_verification: bool):
    if for_verification:
      mlir_ctx.allow_unregistered_dialects = True
    with mlir_ctx, ir.Location.unknown(mlir_ctx):
      return lowering.lower_jaxpr_to_module(
          ctx,
          grid_mapping,
          jaxpr,
          dimension_semantics=mosaic_params.dimension_semantics,
          kernel_type=mosaic_params.kernel_type,
          mesh=jax_mesh,
          for_verification=for_verification,
          dynamic_shape_replacement_enabled=pallas_core.dynamic_shapes_export_enabled(),
      )

  mosaic_module, extra_args = lower_module(for_verification=False)
  if debug:
    print(f"\nThe Mosaic module for pallas_call {debug_info.func_src_info}:")
    print(mosaic_module)
  num_extra_args = len(extra_args)
  num_dyn_bounds = grid_mapping.num_dynamic_grid_bounds
  input_output_aliases = tuple(
      (a[0] + num_dyn_bounds + num_extra_args, a[1])
      for a in input_output_aliases
  )

  if promela_dump_path := _DUMP_PROMELA_TO.value:
    num_devices = 1 if jax_mesh is None else jax_mesh.devices.size
    num_cores = (
        jax.devices()[0].num_cores
        if jax_mesh is None
        else jax_mesh.devices[0].num_cores
    )
    verification_module, _ = lower_module(for_verification=True)
    model = verification.export_promela_model(
        verification_module, num_devices, num_cores
    )
    if promela_dump_path == "stdout":
      print(f"The Promela model for pallas_call {debug_info.func_src_info}:")
      print(model)
    else:
      if promela_dump_path == "sponge":
        promela_dump_path = os.getenv("TEST_UNDECLARED_OUTPUTS_DIR", "")
        if not promela_dump_path:
          raise ValueError(
              "TEST_UNDECLARED_OUTPUTS_DIR must be set when"
              " --jax_pallas_dump_promela_to=sponge"
          )
      dump_ctx = tempfile.NamedTemporaryFile(
          mode="w",
          prefix=mlir.sanitize_name(debug_info.func_name) + "-",
          suffix=".pml",
          dir=promela_dump_path, delete=False,
      )
      with dump_ctx as f:
        f.write(model)

  # Replace in_avals to physical avals.
  # This step is required for mapping logical types to physical types.
  # (e.g. PRNG key -> uint32[2])
  physical_avals = [jax_core.physical_aval(aval) for aval in ctx.avals_in]
  ctx = ctx.replace(avals_in=physical_avals)

  # Booleans are loaded into the kernel as integers.
  def _maybe_cast_inputs(*args):
    args = [_maybe_cast_to_int(x) for x in args]
    return args
  kernel_in_avals = [_maybe_cast_to_int(x) for x in ctx.avals_in]
  kernel_out_avals = [_maybe_cast_to_int(x) for x in out_avals]
  cast_ctx = ctx.replace(avals_out=kernel_in_avals)
  in_nodes = mlir.lower_fun(_maybe_cast_inputs)(cast_ctx, *in_nodes)

  # Dynamic grid bounds have to go at the front.
  dynamic_grid_args, args = in_nodes[:num_dyn_bounds], in_nodes[num_dyn_bounds:]
  kernel_ctx = ctx.replace(avals_in=kernel_in_avals, avals_out=kernel_out_avals)
  output_memory_spaces = _get_memory_spaces_from_avals(out_avals)
  input_memory_spaces = None
  if any(
      isinstance(aval, pallas_core.ShapedArrayWithMemorySpace)
      for aval in ctx.avals_in
  ):
    # TODO(sharadmv): Support dynamic grid bounds and extra args.
    if num_dyn_bounds != 0 or len(extra_args) > 0:
      raise NotImplementedError(
          "Dynamic grid bounds and extra args are not supported when"
          " specifying memory spaces for inputs."
      )
    input_memory_spaces = _get_memory_spaces_from_avals(ctx.avals_in)
  if cost_estimate is not None:
    mosaic_cost_estimate = cast(
        tpu_custom_call.CostEstimate, dataclasses.asdict(cost_estimate)
    )
  else:
    mosaic_cost_estimate = None
  if input_memory_spaces is None and output_memory_spaces is not None:
    input_memory_spaces_list: list[tpu_custom_call.MemorySpace | None] = [
        None,
    ] * len(ctx.avals_in)
    for input_output_alias in input_output_aliases:
      input_memory_spaces_list[input_output_alias[0]] = output_memory_spaces[
          input_output_alias[1]
      ]
    input_memory_spaces = tuple(input_memory_spaces_list)
  out_nodes = mosaic.lower_module_to_custom_call(
      kernel_ctx,
      *dynamic_grid_args,
      *extra_args,
      *args,
      module=mosaic_module,
      out_type=kernel_out_avals,
      kernel_name=mlir.sanitize_name(debug_info.func_name),
      cost_estimate=mosaic_cost_estimate,
      vmem_limit_bytes=mosaic_params.vmem_limit_bytes,
      flags=mosaic_params.flags,
      allow_input_fusion=mosaic_params.allow_input_fusion,
      input_output_aliases=input_output_aliases,
      serialization_format=mosaic_params.serialization_format,
      internal_scratch_in_bytes=mosaic_params.internal_scratch_in_bytes,
      collective_id=mosaic_params.collective_id,
      has_side_effects=mosaic_params.has_side_effects,
      output_memory_spaces=output_memory_spaces,
      disable_bounds_checks=mosaic_params.disable_bounds_checks,
      input_memory_spaces=input_memory_spaces,
      metadata=dict(metadata) if metadata is not None else None,
  )
  _maybe_cast_to_bool = lambda x, aval: x.astype(
      jax.numpy.bool_) if aval.dtype == jax.numpy.bool_ else x
  def _maybe_cast_outputs(*args):
    args = [_maybe_cast_to_bool(x, aval) for x, aval in zip(args, out_avals)]
    return args
  cast_ctx = ctx.replace(avals_in=kernel_out_avals)
  return mlir.lower_fun(_maybe_cast_outputs)(cast_ctx, *out_nodes)
