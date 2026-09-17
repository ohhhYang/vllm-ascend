# mypy: ignore-errors
import math

import torch
import vllm.model_executor.models.config
from vllm.logger import logger
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.config import MambaModelConfig
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE, get_dtype_size


def _using_kv_store(vllm_config) -> bool:
    """
    Check whether AscendStoreConnector is used.
    In the scenario where only PD separation is used, mamba_cache_mode is not automatically set to align.
    """
    if not vllm_config.kv_transfer_config:
        return False
    if vllm_config.kv_transfer_config.kv_connector == "AscendStoreConnector":
        return True
    if vllm_config.kv_transfer_config.kv_connector == "MultiConnector":
        kv_connector_extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
        if not kv_connector_extra_config:
            return False
        if connectors := kv_connector_extra_config.get("connectors"):
            return any(connector.get("kv_connector") == "AscendStoreConnector" for connector in connectors)
    return False


@classmethod
def verify_and_update_config(cls, vllm_config) -> None:
    """
    Ensure that page size of attention layers is greater than or
    equal to the mamba layers. If not, automatically set the attention
    block size to ensure that it is. If the attention page size is
    strictly greater than the mamba page size, we pad the mamba page size
    to make them equal.

    Args:
        vllm_config: vLLM Config
    """
    using_kv_store_with_hybrid = not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager and _using_kv_store(
        vllm_config
    )
    logger.debug("Using kv store: %s", using_kv_store_with_hybrid)
    # Enable FULL_AND_PIECEWISE by default
    MambaModelConfig.verify_and_update_config(vllm_config)

    cache_config = vllm_config.cache_config
    model_config = vllm_config.model_config
    parallel_config = vllm_config.parallel_config

    if cache_config.cache_dtype == "auto":
        kv_cache_dtype = model_config.dtype
    else:
        kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

    kernel_block_size = 128
    model_cls, _ = ModelRegistry.resolve_model_cls(
        model_config.architecture,
        model_config=model_config,
    )

    # get mamba block size
    mamba_shapes = model_cls.get_mamba_state_shape_from_config(vllm_config)
    mamba_dtypes = model_cls.get_mamba_state_dtype_from_config(vllm_config)
    # xlite graph mode manages GDN states itself (slot-major bf16 buffers in
    # the xlite adapter, see xlite.py _adapt_hybrid_kv_caches) and never reads
    # the vllm-side mamba pages. The model card's mamba_ssm_dtype=float32 then
    # only inflates the mamba page (3MiB -> block_size 1536, ~20% KV
    # fragmentation tax) for zero benefit. Keep the accounting consistent by
    # declaring the ssm state bf16 while xlite owns the states.
    if cache_config.mamba_ssm_cache_dtype != "bfloat16":
        import os
        if os.environ.get("XLITE_MAMBA_SSM_BF16", "1") == "0":
            _xlite_on = False
        else:
            try:
                from vllm_ascend.ascend_config import get_ascend_config
                _xlite_graph = getattr(
                    get_ascend_config(), "xlite_graph_config", None)
                _xlite_on = bool(_xlite_graph and getattr(_xlite_graph, "enabled", False))
            except Exception:
                _xlite_on = False
            if not _xlite_on:
                _add = getattr(vllm_config, "additional_config", None) or {}
                _xc = _add.get("xlite_graph_config") if isinstance(_add, dict) else None
                _xlite_on = bool(_xc and _xc.get("enabled", False))
        if _xlite_on:
            logger.info(
                "xlite graph mode owns GDN states in bf16; overriding "
                "mamba_ssm_cache_dtype %s -> bfloat16 to halve the mamba "
                "page and relax the block_size inflation.",
                cache_config.mamba_ssm_cache_dtype)
            cache_config.mamba_ssm_cache_dtype = "bfloat16"
            mamba_dtypes = tuple(
                torch.bfloat16 if d == torch.float32 else d for d in mamba_dtypes)
    mamba_sizes = []
    for shape, dtype in zip(mamba_shapes, mamba_dtypes):
        mamba_sizes.append(math.prod(shape) * get_dtype_size(dtype))
    ssm_block_page_size, conv_block_page_size = max(mamba_sizes), min(mamba_sizes)

    # Pure linear attention models (e.g. bailing 2.5) have only SSM state,
    # no conv block. Detected by a single 3-D mamba shape (ssm only, no conv).
    # Example shape: MambaSpec(shapes=((8, 128, 128),), mamba_type='linear_attention')
    if len(mamba_shapes) == 1 and len(mamba_shapes[0]) == 3:
        conv_block_page_size = 0

    # NOTE(zxr): because of the limit of Ascend Hardware, we need to keep
    # all cache tensors contiguous, so we align the page size of ssm_block
    # and single attn_block
    if model_config.use_mla:
        attn_num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        kv_lora_rank = model_config.hf_text_config.kv_lora_rank
        qk_rope_head_dim = model_config.hf_text_config.qk_rope_head_dim
        attn_single_token_k_page_size = kv_lora_rank * attn_num_kv_heads * get_dtype_size(kv_cache_dtype)
        attn_rope_token_page_size = qk_rope_head_dim * attn_num_kv_heads * get_dtype_size(kv_cache_dtype)
        attn_token_page_size = attn_single_token_k_page_size + attn_rope_token_page_size
    else:
        attn_num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        attn_head_size = model_config.get_head_size()
        attn_single_token_k_page_size = attn_head_size * attn_num_kv_heads * get_dtype_size(kv_cache_dtype)
        attn_token_page_size = 2 * attn_head_size * attn_num_kv_heads * get_dtype_size(kv_cache_dtype)

    attn_block_size = kernel_block_size * cdiv(ssm_block_page_size, kernel_block_size * attn_single_token_k_page_size)
    assert attn_single_token_k_page_size * attn_block_size == ssm_block_page_size, (
        "Cannot align ssm_page_size and attn_page_size."
    )

    # override attention block size if either (a) the
    # user has not set it or (b) the user has set it
    # too small.
    if cache_config.block_size is None or cache_config.block_size < attn_block_size:
        cache_config.block_size = attn_block_size
        logger.info(
            "Setting attention block size to %d tokens to ensure that attention page size is >= mamba page size.",
            attn_block_size,
        )

    # compute new attention page size
    attn_page_size = cache_config.block_size * attn_token_page_size

    # pad mamba page size for conv_blocks
    if (
        cache_config.mamba_page_size_padded is None
        or cache_config.mamba_page_size_padded != attn_page_size + conv_block_page_size
    ):
        cache_config.mamba_page_size_padded = attn_page_size + conv_block_page_size
        mamba_padding_pct = 100 * conv_block_page_size / cache_config.mamba_page_size_padded
        logger.info(
            "Padding mamba page size by %.2f%% to ensure "
            "that mamba page size and attention page size are "
            "exactly equal.",
            mamba_padding_pct,
        )
    # The extract_hidden_states connector (ExampleHiddenStatesConnector) only
    # manages the dedicated hidden-state cache-only layer; it does not migrate
    # mamba KV blocks across instances, so it does not require the block-aligned
    # mamba cache mode. Forcing "align" for it would route hybrid models onto
    # vLLM's fused GPU postprocess Triton kernel (introduced in vLLM #40172),
    # which the Ascend Triton backend cannot compile. Leave the mode as vLLM
    # derived it (e.g. "none" when prefix caching is off) for this case.
    spec_config = vllm_config.speculative_config
    is_extract_hidden_states = (
        spec_config is not None and getattr(spec_config, "method", None) == "extract_hidden_states"
    )
    if using_kv_store_with_hybrid and not is_extract_hidden_states:
        if cache_config.mamba_cache_mode == "none":
            cache_config.mamba_cache_mode = "align"
        else:
            assert cache_config.mamba_cache_mode == "align", (
                "mamba_cache_mode only support 'align' when kv_transfer enabled now!"
            )
    if cache_config.enable_prefix_caching and cache_config.mamba_cache_mode == "align":
        cache_config.mamba_block_size = cache_config.block_size
    else:
        cache_config.mamba_block_size = model_config.max_model_len


vllm.model_executor.models.config.HybridAttentionMambaModelConfig.verify_and_update_config = verify_and_update_config
