from abc import ABC, abstractmethod
from turtle import forward
from typing import Any, Dict, List, Optional, Tuple, Type
from dataclasses import dataclass

import torch

from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
)
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
    get_num_prefill_decode_query_kv_tokens,
)
from vllm.vllm_flash_attn import (
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
    sparse_attn_func,
)
from vllm.attention.backends.utils import get_seq_len_block_table_args
from vllm.vllm_flash_attn.flash_attn_interface import (
    DEFAULT_FA_VERSION,
    maybe_contiguous,
)

# TODO: Metadata can hold V/S sizes?
# TODO: Must support pass-through to flash-attn backend when SWA used!


class MInferenceFlashAttentionInterface(ABC):
    """Base class for attention implementations (both prefilling and KV compression)"""

    def __init__(self):
        self.sparsity_statistics = []
        self.layer_sparsity_statistics = []

    def reset_sparsity_statistics(self):
        """Reset the accumulated sparsity statistics."""
        self.sparsity_statistics = []
        self.layer_sparsity_statistics = []

    def sync_and_calc_layer_stats(self):
        layer_sparsity = torch.stack(self.layer_sparsity_statistics).mean(
            dim=0, keepdim=True
        )

        if get_tensor_model_parallel_world_size() > 1:
            layer_sparsity = tensor_model_parallel_all_gather(layer_sparsity)

        self.sparsity_statistics.append(layer_sparsity.mean().item())
        self.layer_sparsity_statistics = []

    def calculate_sparsity(self) -> float:
        return sum(self.sparsity_statistics) / len(self.sparsity_statistics)

    @staticmethod
    def sum_over_diagonals(matrix: torch.Tensor) -> torch.Tensor:
        """Efficiently sum values along diagonals of the attention matrix.

        This function computes the sum of values along each diagonal of a 4D attention matrix.
        It uses an efficient strided implementation to avoid explicit diagonal extraction.

        Args:
            matrix: Input attention matrix of shape (batch_size, num_heads, queries, keys)
                   where queries and keys are sequence lengths

        Returns:
            Tensor of shape (batch_size, num_heads, queries + keys - 1) containing the
            summed values for each diagonal. The diagonals are ordered from top-right
            to bottom-left, with the main diagonal at index queries-1.
        """
        batch_size, num_heads, queries, keys = matrix.shape
        zero_matrix = torch.zeros(
            (batch_size, num_heads, queries, queries), device=matrix.device
        )
        matrix_padded = torch.cat((zero_matrix, matrix, zero_matrix), -1)

        matrix_strided = matrix_padded.as_strided(
            (batch_size, num_heads, queries, queries + keys),
            (
                num_heads * queries * (2 * queries + keys),
                queries * (2 * queries + keys),
                2 * queries + keys + 1,
                1,
            ),
        )
        return torch.sum(matrix_strided, 2)[:, :, 1:]

    # def kv_compress(
    #     self,
    #     queries: torch.Tensor,  # [num_tokens, num_heads, head_size]
    #     keys: torch.Tensor,  # [num_tokens, num_kv_heads, head_size]
    #     values: torch.Tensor,  # [num_tokens, num_kv_heads, head_size]
    # ) -> tuple[torch.Tensor, torch.Tensor]:
    #     """Compress KV cache after prefilling (default: no compression)"""
    #     return keys, values


# MInferenceAttention


class MInferenceAttentionnBackend(FlashAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "MInferenceAttention"

    @staticmethod
    def get_impl_cls() -> Type["MInferenceFlashAttentionImpl"]:
        return MInferenceFlashAttentionImpl

    # @staticmethod
    # def get_metadata_cls() -> Type["AttentionMetadata"]:
    #     return MInferenceAttentionMetadata

    # @staticmethod
    # def get_builder_cls() -> Type["MInferenceAttentionMetadataBuilder"]:
    #     return MInferenceAttentionMetadataBuilder


# @dataclass
# class MInferenceAttentionMetadata(FlashAttentionMetadata):
#     pass


class MInferenceFlashAttentionImpl(FlashAttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        attn_type: str = AttentionType.DECODER,
        **extra_impl_args,
    ) -> None:
        self.super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            blocksparse_params,
            logits_soft_cap,
            attn_type,
        )
        self.sparse_attention_config = extra_impl_args.get(
            "sparse_attention_config", None
        )
        self.sparse_attention_threshold = extra_impl_args.get(
            "sparse_attention_threshold", 0
        )
        self.last_q_size = extra_impl_args.get("last_q_size", 64)
        if self.sparse_attention_config is None:
            raise AttributeError("No sparse attention config passed to MInferenceImpl!")

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        fp8_out_scale: Optional[torch.Tensor] = None,
        output: Optional[torch.Tensor] = None,
    ):
        ## START COPIED FROM FlashAttentionImpl.forward ##
        # NOTE(woosuk): FlashAttention does not support FP8 KV cache.
        assert (
            layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0
        ), "key/v_scale is not supported in FlashAttention."

        assert output is not None, "Output tensor must be provided."

        attn_type = self.attn_type
        if attn_type == AttentionType.ENCODER and (
            not attn_metadata.is_all_encoder_attn_metadata_set
        ):
            raise AttributeError(
                "Encoder attention requires setting encoder metadata attributes."
            )
        elif attn_type == AttentionType.ENCODER_DECODER and (
            not attn_metadata.is_all_cross_attn_metadata_set
        ):
            raise AttributeError(
                "Encoder/decoder cross-attention "
                "requires setting cross-attention "
                "metadata attributes."
            )

        kv_cache_dtype: str = self.kv_cache_dtype
        softmax_scale: float = self.scale
        window_size = self.sliding_window
        alibi_slopes: Optional[torch.Tensor] = self.alibi_slopes
        logits_soft_cap: Optional[float] = self.logits_soft_cap

        if kv_cache.numel() > 0:
            key_cache = kv_cache[0]
            value_cache = kv_cache[1]
            # We skip updating the KV cache under two conditions:
            #  a. When the Attention Type is ENCODER. In this phase, we compute
            #     only the encoder attention without updating the cache.
            #  b. When both Key and Value are None. This occurs during
            #     cross-attention computation in the decoding phase, where the
            #     KV cache is already populated with the cross-attention
            #     tensor. Thus, we skip cache updates during this time.
            if (
                (attn_type != AttentionType.ENCODER)
                and (key is not None)
                and (value is not None)
            ):
                if attn_type == AttentionType.ENCODER_DECODER:
                    # Update cross-attention KV cache (prefill-only)
                    updated_slot_mapping = attn_metadata.cross_slot_mapping
                else:
                    # Update self-attention KV cache (prefill/decode)
                    updated_slot_mapping = attn_metadata.slot_mapping

                # Reshape the input keys and values and store them in the cache.
                # If kv_cache is not provided, the new key and value tensors are
                # not cached. This happens during the initial memory
                # profiling run.
                torch.ops._C_cache_ops.reshape_and_cache_flash(
                    key,
                    value,
                    kv_cache[0],
                    kv_cache[1],
                    updated_slot_mapping.flatten(),  # type: ignore[union-attr]
                    kv_cache_dtype,
                    layer._k_scale,
                    layer._v_scale,
                )

        attn_type = self.attn_type
        (num_prefill_query_tokens, num_prefill_kv_tokens, num_decode_query_tokens) = (
            get_num_prefill_decode_query_kv_tokens(attn_metadata, attn_type)
        )
        decode_query = query[num_prefill_query_tokens:]
        decode_output = output[num_prefill_query_tokens:]
        # QKV for prefill.
        query = query[:num_prefill_query_tokens]
        prefill_output = output[:num_prefill_query_tokens]
        assert query.shape[0] == num_prefill_query_tokens
        assert decode_query.shape[0] == num_decode_query_tokens
        ## END COPIED FROM FlashAttentionImpl.forward ##

        if prefill_meta := attn_metadata.prefill_metadata:
            # Profiling run, use normal attn.
            if (
                kv_cache is None
                or prefill_meta.block_tables is None
                or prefill_meta.block_tables.numel() == 0
            ):
                # normal attention, called during the profiling run.
                out = flash_attn_varlen_func(
                    q=query,
                    k=key,
                    v=value,
                    cu_seqlens_q=prefill_meta.seq_start_loc,
                    cu_seqlens_k=prefill_meta.seq_start_loc,
                    max_seqlen_q=prefill_meta.max_prefill_seq_len,
                    max_seqlen_k=prefill_meta.max_prefill_seq_len,
                    softmax_scale=self.scale,
                    causal=True,
                    window_size=self.sliding_window,
                    alibi_slopes=self.alibi_slopes,
                )
                assert output[:num_prefill_query_tokens].shape == out.shape
                output[:num_prefill_query_tokens] = out
            else:
                # attn w/ kv cache: k/v cache should have 0 numel here.
                assert (
                    key_cache.numel() == 0 and value_cache.numel() == 0
                ), "k/v caches not empty before sparse prefill!"  # TODO: Won't these have values during subsequent chunk prefill?
                assert (
                    attn_type == AttentionType.DECODER
                ), "Only decoder-only models support prefix caching"
                assert prefill_meta.seq_lens is not None
                max_seq_len = max(prefill_meta.seq_lens)
                self._sparse_flash_attn_prefill(
                    q=query,
                    k=key_cache,
                    v=value_cache,
                    cu_seqlens_q=prefill_meta.query_start_loc,
                    max_seqlen_q=prefill_meta.max_query_len,
                    seqused_k=prefill_meta.seq_lens_tensor,
                    max_seqlen_k=max_seq_len,
                    softmax_scale=softmax_scale,
                    causal=True,
                    window_size=window_size,
                    alibi_slopes=alibi_slopes,
                    block_table=prefill_meta.block_tables,
                    softcap=logits_soft_cap,
                    out=prefill_output,
                    fa_version=self.vllm_flash_attn_version,
                )

                # flash_attn_varlen_func(  # TODO: need a sparse prefill func
                #     q=query,
                #     k=key_cache,
                #     v=value_cache,
                #     cu_seqlens_q=prefill_meta.query_start_loc,
                #     max_seqlen_q=prefill_meta.max_query_len,
                #     seqused_k=prefill_meta.seq_lens_tensor,
                #     max_seqlen_k=max_seq_len,
                #     softmax_scale=softmax_scale,
                #     causal=True,
                #     window_size=window_size,
                #     alibi_slopes=alibi_slopes,
                #     block_table=prefill_meta.block_tables,
                #     softcap=logits_soft_cap,
                #     out=prefill_output,
                #     fa_version=self.vllm_flash_attn_version,
                # )

        if decode_meta := attn_metadata.decode_metadata:
            # Decoding run. No sparsity during decoding.
            # Use flash_attn_varlen_func kernel for speculative decoding
            # because different queries might have different lengths.
            assert decode_meta.max_decode_query_len is not None
            # use only for actual varlen decoding
            if decode_meta.max_decode_query_len > 1:
                assert (
                    attn_type == AttentionType.DECODER
                ), "Only decoder-only models support max_decode_query_len > 1"
                flash_attn_varlen_func(
                    q=decode_query,
                    k=key_cache,
                    v=value_cache,
                    cu_seqlens_q=decode_meta.query_start_loc,
                    max_seqlen_q=decode_meta.max_decode_query_len,
                    seqused_k=decode_meta.seq_lens_tensor,
                    max_seqlen_k=decode_meta.max_decode_seq_len,
                    softmax_scale=softmax_scale,
                    causal=True,
                    window_size=window_size,
                    alibi_slopes=alibi_slopes,
                    softcap=logits_soft_cap,
                    block_table=decode_meta.block_tables,
                    out=decode_output,
                    fa_version=self.vllm_flash_attn_version,
                )
            else:
                # Use flash_attn_with_kvcache for normal decoding.
                (
                    seq_lens_arg,
                    _,
                    block_tables_arg,
                ) = get_seq_len_block_table_args(decode_meta, False, attn_type)
                flash_attn_with_kvcache(
                    q=decode_query.unsqueeze(1),
                    k_cache=key_cache,
                    v_cache=value_cache,
                    block_table=block_tables_arg,
                    cache_seqlens=seq_lens_arg,
                    softmax_scale=softmax_scale,
                    causal=True,
                    window_size=window_size,
                    alibi_slopes=alibi_slopes,
                    softcap=logits_soft_cap,
                    out=decode_output.unsqueeze(1),
                    fa_version=self.vllm_flash_attn_version,
                )

    def _sparse_flash_attn_prefill(
        self,
        q,
        k,
        v,
        max_seqlen_q,
        cu_seqlens_q,
        max_seqlen_k,
        cu_seqlens_k=None,  # only used for non-paged prefill
        seqused_k=None,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size: Optional[List[int]] = None,
        softcap=0.0,  # 0.0 means deactivated
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        block_table=None,
        chunk_size: int = 8192,
        local_size: int = 4096,  # match SWA window
        *,
        return_softmax_lse=False,
        out=None,
        fa_version: int = DEFAULT_FA_VERSION,
    ) -> torch.Tensor:
        """dropout_p should be set to 0.0 during evaluation
        Supports multi-query and grouped-query attention (MQA/GQA) by passing in K, V with fewer heads
        than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
        For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
        0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

        If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
        For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
            1 1 1 1 0
            1 1 1 1 1
        If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
            0 0
            0 0
            0 0
            1 0
            1 1
        If the row of the mask is all zero, the output will be zero.

        If window_size != (-1, -1), implements sliding window local attention. Query at position i
        will only attend to keys between
        [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

        Arguments:
            q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
            k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
            v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
            cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
            of the sequences in the batch, used to index into q.
            cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
            of the sequences in the batch, used to index into kv.
            max_seqlen_q: int. Maximum query sequence length in the batch.
            max_seqlen_k: int. Maximum key sequence length in the batch.
            dropout_p: float. Dropout probability.
            softmax_scale: float. The scaling of QK^T before applying softmax.
                Default to 1 / sqrt(headdim).
            causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
            window_size: (left, right). If not (-1, -1), implements sliding window local attention.
            softcap: float. Anything > 0 activates softcapping attention.
            alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
                (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
                is added to the attention score of query i and key j.
            deterministic: bool. Whether to use the deterministic implementation of the backward pass,
                which is slightly slower and uses more memory. The forward pass is always deterministic.
            return_attn_probs: bool. Whether to return the attention probabilities. This option is for
            testing only. The returned probabilities are not guaranteed to be correct
            (they might not have the right scaling).
        Return:
            out: (total, nheads, headdim).
            softmax_lse [optional, if return_softmax_lse=True]: (nheads, total_q_seqlen). The
                logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
                normalization factor).
        """
        assert (
            cu_seqlens_k is not None or seqused_k is not None
        ), "cu_seqlens_k or seqused_k must be provided"
        assert (
            cu_seqlens_k is None or seqused_k is None
        ), "cu_seqlens_k and seqused_k cannot be provided at the same time"
        assert (
            block_table is None or seqused_k is not None
        ), "seqused_k must be provided if block_table is provided"
        if alibi_slopes is not None:
            raise ValueError("MInference Attention does not support alibi_slopes")
        if not causal:
            raise ValueError("MInference Attention does not support causal=False")

        # custom op does not support non-tuple input
        real_window_size: Tuple[int, int]
        if (
            window_size != (-1, -1) and window_size is not None
        ):  # TODO: We want to support swa!
            raise NotImplementedError("Sparse SWA not implemented!")
            assert len(window_size) == 2
            real_window_size = (window_size[0], window_size[1])
        else:
            real_window_size = (-1, -1)

        all_outputs = []
        # loop through each sequence in the batch to determine critical tokens
        # TODO: extract method
        for i in range(0, len(cu_seqlens_q) - 1):
            qs = cu_seqlens_q[i]
            qe = cu_seqlens_q[i : i + 2][-1]
            ks = cu_seqlens_k[i]
            ke = cu_seqlens_k[i : i + 2][-1]

            current_q = q[qs:qe]
            seq_len = len(current_q)
            if seq_len < self.sparse_threshold:
                # TODO: Route this seq to dense attn.
                sparsity_enabled = True
                continue
            if (
                block_table is None
            ):  # NOTE: For chunked prefill, may have a block_table already populated
                current_k = k[ks:ke]
                current_v = v[ks:ke]
                current_block_table = None
            else:
                current_block_table = block_table[i]
                current_k = k
                current_v = v

            if current_q.shape[0] == 0:
                raise RuntimeError("How did this happen :)")  # TODO
                # continue

            if current_k.shape[0] == 0:
                all_outputs.append(
                    torch.zeros(
                        (current_q.shape[0], current_q.shape[1], v.shape[2]),
                        device=q.device,
                        dtype=q.dtype,
                    )
                )
                continue

            current_output = torch.empty_like(current_q)
            group_size = int(
                current_q.size(-2) / current_k.size(-2)
            )  # num. q heads per k/v head
            num_device_q_heads = current_q.size(
                -2
            )  # NOTE: For TP we may not have all heads on this device
            heads_vertical_size = torch.empty(
                size=(num_device_q_heads,), dtype=torch.int32
            )
            heads_slash_size = torch.empty(
                size=(num_device_q_heads,), dtype=torch.int32
            )
            for head_id in range(current_q.size(-2)):
                (
                    ty,  # TODO: Don't think we need type, remove.
                    vertical_size,
                    slash_size,
                    _,
                ) = self.sparse_attention_config[head_id]
                assert ty == "vertical_and_slash", "only support slash mode"

                if vertical_size == 30:
                    vertical_size += 100  # TODO: Bit hacky, should be removed?
                heads_vertical_size[head_id] = vertical_size
                heads_slash_size[head_id] = slash_size

            ### BEGIN _dual_chunk_flash_attn_prefill_func logic ###
            flash_results = []
            chunk_len = (
                chunk_size - local_size
            )  # TODO: Not sure we need this, I think chunks will simply be processed with cache in subsquent calls to backend
            if block_table is not None:
                block_size = v.shape[1]
                if chunk_len % block_size != 0:
                    raise ValueError("chunk_len must be divisible by block_size.")
            else:
                block_size = 1

            # TODO: yarn scaling
            # if self.original_max_position_embeddings > 0:
            #     softmax_scale = softmax_scale * scaling_factor

        k_length = ke - ks
        begin = k_length - q.shape[0]
        while begin < k_length:
            flash_per_chunk = []
            last_q_size = min(qe - qs, self.last_q_size)


# BEGIN PIOTR WORK
class VerticalAndSlashAttentionMInference(FlashAttentionImpl):
    """Combines vertical and diagonal patterns for efficient sparse attention.

    Implements the Vertical and Slash attention mechanism that selects important tokens based on:
    1) Vertical patterns - Top-k tokens that receive high attention across all queries
    2) Diagonal patterns - Diagonal stripes that capture local dependencies
    """

    def __init__(
        self,
        vertical_size: int = 64,
        slash_size: int = 128,
        approximation_size: int = 64,
    ):
        """Initialize Vertical and Slash attention.

        Args:
            vertical_size (int): Number of vertical tokens to select
            slash_size (int): Number of diagonal stripes to select
        """
        super().__init__()
        self.vertical_size = vertical_size
        self.slash_size = slash_size
        self.approximation_size = approximation_size

    def __call__(
        self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        attn_output, sparsity = vertical_and_slash_kernel(
            queries,
            keys,
            values,
            vertical_size=min(self.vertical_size, queries.shape[2]),
            slash_size=min(self.slash_size, queries.shape[2]),
            last_q=min(self.approximation_size, queries.shape[2]),
        )
        self.layer_sparsity_statistics.append(sparsity)
        return attn_output
