from abc import ABC
from typing import Any, Dict, List, Optional, Tuple, Type
import math
import triton
import triton.language as tl

import torch
import torch.nn.functional as F

from vllm._custom_ops import convert_vertical_slash_indexes_mergehead, convert_vertical_slash_indexes
from vllm.attention.backends.abstract import (
    AttentionLayer,
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
    get_num_prefill_decode_query_kv_tokens,
    _get_causal_option,
    _get_query_key_seq_metadata,
)
from vllm.vllm_flash_attn import (
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
    sparse_attn_func,
)
from vllm.attention.backends.utils import get_seq_len_block_table_args
from vllm.vllm_flash_attn.flash_attn_interface import (
    DEFAULT_FA_VERSION,
)


class MInferenceFlashAttentionBackend(FlashAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "MInferenceFlashAttention"

    @staticmethod
    def get_impl_cls() -> Type["MInferenceFlashAttentionImpl"]:
        return MInferenceFlashAttentionImpl


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
        super().__init__(
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
        if not hasattr(self, "vllm_flash_attn_version"):
            self.vllm_flash_attn_version = 2
        self.sparse_attention_config = extra_impl_args.get("sparse_attention_config")
        self.sparse_attention_threshold = extra_impl_args.get("sparse_attention_threshold", 0)
        self.layer_idx = extra_impl_args["layer_idx"]
        self.last_q_size = extra_impl_args.get("last_q_size", 64)
        arange = torch.arange(self.last_q_size, device="cuda")
        self.last_q_mask = arange[None, None, :, None] >= arange[None, None, None, :]
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
        # TODO: Small fraction (0.002%) of output values are unchanged through the attn op below. Consider overwritting with zeros.
        # NOTE: Zero init completely removes the unchanged elements. It seems like a coicidence that several elements are re-written with the same value.
        # output[:] = torch.zeros_like(query)

        ## NOTE: START COPIED FROM FlashAttentionImpl.forward ##
        # NOTE(woosuk): FlashAttention does not support FP8 KV cache.

        assert layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0, "key/v_scale is not supported in FlashAttention."

        assert output is not None, "Output tensor must be provided."

        attn_type = self.attn_type
        if attn_type == AttentionType.ENCODER and (not attn_metadata.is_all_encoder_attn_metadata_set):
            raise AttributeError("Encoder attention requires setting encoder metadata attributes.")
        elif attn_type == AttentionType.ENCODER_DECODER and (not attn_metadata.is_all_cross_attn_metadata_set):
            raise AttributeError("Encoder/decoder cross-attention requires setting cross-attention metadata attributes.")

        kv_cache_dtype: str = self.kv_cache_dtype
        softmax_scale: float = self.scale
        window_size = self.sliding_window
        is_swa = window_size is not None and window_size != (-1, -1)
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
            if (attn_type != AttentionType.ENCODER) and (key is not None) and (value is not None):
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
        (num_prefill_query_tokens, num_prefill_kv_tokens, num_decode_query_tokens) = get_num_prefill_decode_query_kv_tokens(attn_metadata, attn_type)
        decode_query = query[num_prefill_query_tokens:]
        decode_output = output[num_prefill_query_tokens:]
        # QKV for prefill.
        query = query[:num_prefill_query_tokens]
        prefill_output = output[:num_prefill_query_tokens]
        assert query.shape[0] == num_prefill_query_tokens
        assert decode_query.shape[0] == num_decode_query_tokens
        ## NOTE: END COPIED FROM FlashAttentionImpl.forward ##

        if prefill_meta := attn_metadata.prefill_metadata:
            # We deal with prefill sequences in this batch first.
            if kv_cache.numel() == 0 or prefill_meta.block_tables is None or prefill_meta.block_tables.numel() == 0:
                # Standard attn. no kv cache
                # When block_tables are not filled, it means q and k are the
                # prompt, and they have the same length.
                q_seq_start_loc, q_seq_len, k_seq_start_loc, k_seq_len = _get_query_key_seq_metadata(prefill_meta, True, attn_type)
                key = key[:num_prefill_kv_tokens]  # remove any decode context keys if present in batch
                value = value[:num_prefill_kv_tokens]

                if is_swa:
                    # We do not support SWA for sparse attn.
                    flash_attn_varlen_func(
                        q=query,
                        k=key,
                        v=value,
                        cu_seqlens_q=q_seq_start_loc,
                        cu_seqlens_k=k_seq_start_loc,
                        max_seqlen_q=q_seq_len,
                        max_seqlen_k=k_seq_len,
                        softmax_scale=softmax_scale,
                        causal=_get_causal_option(attn_type),
                        window_size=window_size,
                        alibi_slopes=alibi_slopes,
                        softcap=logits_soft_cap,
                        out=prefill_output,
                        fa_version=self.vllm_flash_attn_version,
                    )
                else:
                    # sparse attn no prefix
                    orig_seq_len = None
                    if hasattr(prefill_meta, "orig_seq_len"):
                        orig_seq_len = prefill_meta.orig_seq_len
                    self._sparse_flash_attn_prefill(
                        q=query,
                        k=key,
                        v=value,
                        cu_seqlens_q=prefill_meta.query_start_loc,
                        cu_seqlens_k=prefill_meta.seq_start_loc,
                        max_seqlen_q=prefill_meta.max_query_len,
                        orig_seq_lens=orig_seq_len,
                        softmax_scale=softmax_scale,
                        causal=True,
                        window_size=window_size,
                        alibi_slopes=alibi_slopes,
                        block_table=prefill_meta.block_tables,
                        softcap=logits_soft_cap,
                        out=prefill_output,
                        fa_version=self.vllm_flash_attn_version,
                    )
                    # TODO: Remove pitor code added here after debuggin completed. 
                    # NOTE: Current implementation matches exactly (i.e., assertion will pass at the end)
                    # pitor_q = query.unsqueeze(0).reshape(prefill_meta.num_prefills, -1, query.shape[1], query.shape[2]).permute(0, 2, 1, 3)
                    # pitor_k = key.unsqueeze(0).reshape(prefill_meta.num_prefills, -1, key.shape[1], key.shape[2]).permute(0, 2, 1, 3)
                    # pitor_v = value.unsqueeze(0).reshape(prefill_meta.num_prefills, -1, value.shape[1], value.shape[2]).permute(0, 2, 1, 3)
                    # pitor_out, _ = self._pitor_attn(pitor_q, pitor_k, pitor_v, block_table=prefill_meta.block_tables)
                    # pitor_out = pitor_out.view(query.shape[1]*prefill_meta.num_prefills, query.shape[0], query.shape[2]).transpose(0, 1)
                    # prefill_output[prefill_meta.query_start_loc[0]:prefill_meta.query_start_loc[-1]] = pitor_out
                    # assert torch.allclose(prefill_output, pitor_out)
            else:
                # prefix-enabled attn, kv_cache populated. 
                assert attn_type == AttentionType.DECODER, "Only decoder-only models support prefix caching"
                assert prefill_meta.seq_lens is not None
                max_seq_len = max(prefill_meta.seq_lens)
                if is_swa:
                    # prefix enabled flash attn. for SWA. Note args differ from above.
                    flash_attn_varlen_func(  # noqa
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
                else:
                    # prefix enabled sparse attn
                    orig_seq_len = None
                    if hasattr(prefill_meta, "orig_seq_len"):
                        # TODO: Currently unused but can be used to toggle sparse attn based on min. seq len
                        orig_seq_len = prefill_meta.orig_seq_len
                    self._sparse_flash_attn_prefill(
                        q=query,
                        k=key_cache,
                        v=value_cache,
                        cu_seqlens_q=prefill_meta.query_start_loc,
                        cu_seqlens_k=prefill_meta.seq_start_loc,
                        max_seqlen_q=prefill_meta.max_query_len,
                        orig_seq_lens=orig_seq_len,
                        softmax_scale=softmax_scale,
                        causal=True,
                        window_size=window_size,
                        alibi_slopes=alibi_slopes,
                        block_table=prefill_meta.block_tables,
                        softcap=logits_soft_cap,
                        out=prefill_output,
                        fa_version=self.vllm_flash_attn_version,
                    )

        # From here on we simply use dense attn. for decode. Same as flash attn.
        if decode_meta := attn_metadata.decode_metadata:
            # Decoding run. No sparsity during decoding.
            # Use flash_attn_varlen_func kernel for speculative decoding
            # because different queries might have different lengths.
            assert decode_meta.max_decode_query_len is not None
            # use only for actual varlen decoding
            if decode_meta.max_decode_query_len > 1:
                assert attn_type == AttentionType.DECODER, "Only decoder-only models support max_decode_query_len > 1"
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
        orig_seq_lens: List[int],  # required to determine if we need sparse attn.
        cu_seqlens_k=None,  # only used for non-paged prefill
        seqused_k=None,  # TODO: Should be able to remove this?
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size: Optional[List[int]] = None,
        softcap=0.0,  # 0.0 means deactivated
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        block_table=None,
        *,
        return_softmax_lse=False,
        out=None,
        fa_version: int = DEFAULT_FA_VERSION,
    ) -> torch.Tensor:
        """
        NOTE: Unused args remain for future bypass into Flash Attn. for orig_seq_len < sparsity threshold.

        dropout_p should be set to 0.0 during evaluation
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
            k: (num_blocks, page_block_size, nheads_k, headdim) if block_table is not None
            v: (num_blocks, page_block_size, nheads_v, headdim) if block_table is not None
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
        assert cu_seqlens_k is not None or seqused_k is not None, "cu_seqlens_k or seqused_k must be provided"
        assert cu_seqlens_k is None or seqused_k is None, "cu_seqlens_k and seqused_k cannot be provided at the same time"
        if alibi_slopes is not None:
            raise ValueError("MInference Attention does not support alibi_slopes")
        if not causal:
            raise ValueError("MInference Attention does not support causal=False")

        if window_size != (-1, -1) and window_size is not None:
            raise NotImplementedError("Sparse SWA not implemented!")

        all_outputs = []
        # loop through each sequence in the batch to determine critical tokens
        for i in range(0, len(cu_seqlens_q) - 1):
            qs = cu_seqlens_q[i]
            qe = cu_seqlens_q[i : i + 2][-1]
            ks = cu_seqlens_k[i]
            ke = cu_seqlens_k[i : i + 2][-1]
            k_length = ke - ks
            current_orig_seq_len = None
            if orig_seq_lens is not None:
                # TODO: need to propograte from metadata, always None for now.
                current_orig_seq_len = orig_seq_lens[i]
            current_q = q[qs:qe]
            if block_table.numel() == 0:
                current_k = k[ks:ke]
                current_v = v[ks:ke]
                current_block_table = None
            else:
                current_block_table = block_table[i]
                current_k = k
                current_v = v

            if current_orig_seq_len is not None and current_orig_seq_len < self.sparse_attention_threshold:
                # TODO: Route this seq to dense attn. or use conditional branches as per DCA
                raise NotImplementedError("Need to implement a dense attn bypass")

            if current_q.shape[0] == 0:
                continue

            if current_k.shape[0] == 0:
                all_outputs.append(
                    torch.zeros(
                        (current_q.shape[0], current_q.shape[1], v.shape[2]),
                        device=q.device,
                        dtype=q.dtype,
                    )
                )
                continue

            group_size = int(current_q.size(-2) / current_k.size(-2))  # num. q heads per k/v head
            num_device_q_heads = current_q.size(-2)  # NOTE: For TP we may not have all heads on this device
            heads_vertical_size = torch.empty(size=(num_device_q_heads,), dtype=torch.int32)
            heads_slash_size = torch.empty(size=(num_device_q_heads,), dtype=torch.int32)
            for head_id in range(current_q.size(-2)):
                (
                    sparsity_type,
                    vertical_size,
                    slash_size,
                    _,
                ) = self.sparse_attention_config[self.layer_idx][head_id]
                assert sparsity_type == "vertical_and_slash", "We only support Vertical and Slash sparsity."

                if vertical_size == 30:
                    # TODO: Parameterize this min. num vert (30->130).
                    vertical_size += 100
                heads_vertical_size[head_id] = vertical_size
                heads_slash_size[head_id] = slash_size

            ### NOTE: BEGIN _dual_chunk_flash_attn_prefill_func logic ###
            # # TODO: yarn scaling -> Need to know original_max_position_embeddings
            # if self.original_max_position_embeddings > 0:
            #     softmax_scale = softmax_scale * scaling_factor

            # If prefix-caching is enabled, retrieve all key/value chunks from cache
            if current_block_table is not None:
                # reshape to (seq_len, num_k_head, headdim) and extract only the keys currently populated.
                # The last block table page may not be completely filled. 
                current_k = k[current_block_table].view(-1, *k.shape[-2:])[:k_length]
                current_v = v[current_block_table].view(-1, *v.shape[-2:])[:k_length]

            # reshape for GQA to (seq_len, num_q_head, headdim)
            num_device_k_heads, head_dim = current_k.shape[-2:]
            current_k = current_k.unsqueeze(2).repeat(1, 1, group_size, 1).reshape(-1, num_device_k_heads * group_size, head_dim)
            current_v = current_v.unsqueeze(2).repeat(1, 1, group_size, 1).reshape(-1, num_device_k_heads * group_size, head_dim)

            # calc approx. atten to determine vertical/slash indicies
            last_q_size = min(qe - qs, self.last_q_size)
            # qk will have shape (query_heads, last_q_size, k_length)
            qk = (current_q.transpose(0, 1)[:, -last_q_size:] * softmax_scale) @ current_k.permute(1, 2, 0)
            # apply attn. scores to -inf for causally masked elements
            qk[:, :, -last_q_size:] = torch.where(
                self.last_q_mask[..., -last_q_size:, -last_q_size:].to(qk.device),
                qk[:, :, -last_q_size:],
                -torch.inf,
            )
            qk = F.softmax(qk, dim=-1, dtype=torch.float32)

            # get per head attn. score sums across each key index
            vertical_attn_score_sums = qk.sum(-2, keepdim=True)
            # heuristic from Qwen1M -> Always keep first 30 keys for vertical (prefix)
            # TODO: Parameterize width of prefix to always keep
            vertical_attn_score_sums[..., :30] = torch.inf
            vertical_attn_score_sums = vertical_attn_score_sums.squeeze(dim=1)

            # vertical indices
            num_query_heads = qk.shape[0]
            max_slash_topk = torch.max(heads_slash_size).item()
            max_vertical_topk = torch.max(heads_vertical_size).item()
            # Handle case where num_keys < max_vertical_topk
            max_vertical_topk = min(vertical_attn_score_sums.shape[-1], max_vertical_topk)
            max_slash_topk = min(vertical_attn_score_sums.shape[-1], max_slash_topk)

            vertical_topk_buffer = torch.topk(vertical_attn_score_sums, max_vertical_topk, -1).indices
            slash_topk_buffer = torch.empty(size=(num_query_heads, max_slash_topk), dtype=torch.int64, device=qk.device)

            # Get per head slash scores and max num. slashes based on prompt size
            for head_i in range(num_query_heads):
                #  (nqheads=1, lastq, k_len)
                head_score = qk[head_i : head_i + 1]
                slash_scores = _sum_all_diagonal_matrix(head_score)
                if head_score.size(1) != 1:
                    # drop right up corner -> (1, k_length)
                    slash_scores = slash_scores[..., : -last_q_size + 1]
                # heuristic from Qwen1M -> always keep last 100 slash indicies
                # TODO: Parameterize last slash scores to always keep.
                slash_scores[..., -100:] = torch.inf

                head_slash_size = heads_slash_size[head_i]
                head_slash_size = min(head_slash_size, vertical_attn_score_sums.size(-1))
                slash_topk = torch.topk(slash_scores, head_slash_size, -1).indices
                # （nheads, max_topk）
                slash_topk_buffer[head_i, :head_slash_size] = slash_topk

                # reset heads topk
                heads_slash_size[head_i] = head_slash_size
                heads_vertical_size[head_i] = min(heads_vertical_size[head_i], max_vertical_topk)

            int32_max = torch.iinfo(torch.int32).max
            int32_min = torch.iinfo(torch.int32).min
            vertical_indicies = torch.full(
                (num_query_heads, max_vertical_topk),
                int32_max,
                dtype=torch.int64,
                device=q.device,
            )
            slash_indicies = torch.full(
                (num_query_heads, max_slash_topk),
                int32_min,
                dtype=torch.int64,
                device=q.device,
            )
            vertical_indices_count = torch.empty(size=(num_query_heads,), dtype=torch.int32, device=q.device)
            slash_indices_count = torch.empty(size=(num_query_heads,), dtype=torch.int32, device=q.device)

            # Select vertical/slash indicies per head:
            for head_i in range(num_query_heads):
                # Get topk indices for each head and assign to buffers
                vertical_topk = vertical_topk_buffer[head_i, : heads_vertical_size[head_i]]
                slash_topk = slash_topk_buffer[head_i, : heads_slash_size[head_i]]

                vertical_indices_count[head_i] = vertical_topk.shape[0]
                slash_indices_count[head_i] = slash_topk.shape[0]
                vertical_indicies[head_i] = vertical_topk
                slash_indicies[head_i] = slash_topk

            ### NOTE: END _dual_chunk_flash_attn_prefill_func logic ###

            # Reshape for flash attn
            q_len, q_heads, h_dim = current_q.shape
            current_q = current_q.unsqueeze(0).transpose(1, 2)
            current_k = current_k.unsqueeze(0).transpose(1, 2)
            current_v = current_v.unsqueeze(0).transpose(1, 2)

            seq_output, _ = _vertical_slash_sparse_attention(
                current_q,
                current_k,
                current_v,
                vertical_indicies,
                slash_indicies,
                softmax_scale,
                causal,
                vertical_indices_count=vertical_indices_count,
                slash_indices_count=slash_indices_count,
            )
            seq_output = seq_output.view(q_heads, q_len, h_dim).transpose(0, 1)  # (qlen,nhead,h_dim)
            out[qs:qe] = seq_output
        return out

    # TODO: Remove pitor code here once satisfied that above implementation matches
    # NOTE: _pitor_attn as implmeneted here does not support chunked prefill, prefix-enabled cache or batched attn without padding.
    def _pitor_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_table: Optional[torch.Tensor] = None,
    ):
        def _triton_mixed_sparse_attention(
            q: torch.Tensor,  # [BATCH, N_HEADS, N_CTX, D_HEAD]
            k: torch.Tensor,  # [BATCH, N_HEADS, N_CTX, D_HEAD]
            v: torch.Tensor,  # [BATCH, N_HEADS, N_CTX, D_HEAD]
            seqlens: torch.Tensor,  # [BATCH, ]
            block_count: torch.Tensor,  # [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M)]
            block_offset: torch.Tensor,  # [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M), NNZ_S]
            column_count: torch.Tensor,  # [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M)]
            column_index: torch.Tensor,  # [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M), NNZ_V]
            sm_scale: float,
            block_size_M: int = 64,
            block_size_N: int = 64,
        ) -> torch.Tensor:
            # shape constraints
            Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
            assert Lq == Lk and Lk == Lv
            assert Lk in {16, 32, 64, 128}
            o = torch.zeros_like(q)
            grid = (triton.cdiv(q.shape[2], block_size_M), q.shape[0] * q.shape[1], 1)
            dtype = tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16
            _triton_mixed_sparse_attn_fwd_kernel[grid](
                q,
                k,
                v,
                seqlens,
                sm_scale,
                block_count,
                block_offset,
                column_count,
                column_index,
                o,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                k.stride(3),
                v.stride(0),
                v.stride(1),
                v.stride(2),
                v.stride(3),
                o.stride(0),
                o.stride(1),
                o.stride(2),
                o.stride(3),
                q.shape[0],
                q.shape[1],
                q.shape[2],
                block_count.shape[-1],
                block_offset.shape[-1],
                column_index.shape[-1],
                BLOCK_M=block_size_M,
                BLOCK_N=block_size_N,
                BLOCK_DMODEL=Lk,
                dtype=dtype,
                num_warps=4,
                num_stages=1,
            )

            return o

        def vertical_slash_sparse_attention(
            query: torch.Tensor,  # [BATCH, N_HEADS, N_CTX, D_HEAD]
            key: torch.Tensor,  # [BATCH, N_HEADS, N_CTX, D_HEAD]
            value: torch.Tensor,  # [BATCH, N_HEADS, N_CTX, D_HEAD]
            v_idx: torch.Tensor,  # [BATCH, N_HEADS, NNZ_V]
            s_idx: torch.Tensor,  # [BATCH, N_HEADS, NNZ_S]
            block_size_M: int = 64,
            block_size_N: int = 64,
        ):
            batch_size, num_heads, context_size, head_dim = query.shape
            pad = block_size_M - (context_size & (block_size_M - 1))
            query = torch.nn.functional.pad(query, [0, 0, 0, pad, 0, 0, 0, 0])
            key = torch.nn.functional.pad(key, [0, 0, 0, pad, 0, 0, 0, 0])
            value = torch.nn.functional.pad(value, [0, 0, 0, pad, 0, 0, 0, 0])

            if head_dim not in [16, 32, 64, 128, 256, 512]:
                target_dim = 2 ** math.ceil(math.log2(head_dim)) - head_dim
                query = torch.nn.functional.pad(query, [0, target_dim, 0, 0, 0, 0, 0, 0])
                key = torch.nn.functional.pad(key, [0, target_dim, 0, 0, 0, 0, 0, 0])
                value = torch.nn.functional.pad(value, [0, target_dim, 0, 0, 0, 0, 0, 0])

            v_idx = v_idx.to(torch.int32).reshape((batch_size, num_heads, -1)).sort(dim=-1, descending=False)[0]
            s_idx = s_idx.to(torch.int32).reshape((batch_size, num_heads, -1)).sort(dim=-1, descending=True)[0]
            seqlens = torch.tensor([context_size], dtype=torch.int32, device=query.device)
            sm_scale = head_dim**-0.5
            block_count, block_offset, column_count, column_index = torch.ops._C.minference_convert_vertical_slash_indexes(
                seqlens,
                v_idx,
                s_idx,
                context_size,
                block_size_M,
                block_size_N,
            )
            # batch_size, num_heads, context_size, head_dim = query.shape
            # sparsity = calc_sparsity(block_count, column_count, context_size)
            out = _triton_mixed_sparse_attention(
                query,
                key,
                value,
                seqlens,
                block_count,
                block_offset,
                column_count,
                column_index,
                sm_scale,
                block_size_M,
                block_size_N,
            )
            return out[..., :context_size, :head_dim], None

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
            zero_matrix = torch.zeros((batch_size, num_heads, queries, queries), device=matrix.device)
            matrix_padded = torch.cat((zero_matrix, matrix, zero_matrix), -1)

            matrix_strided = matrix_padded.as_strided(
                (batch_size, num_heads, queries, queries + keys),
                (num_heads * queries * (2 * queries + keys), queries * (2 * queries + keys), 2 * queries + keys + 1, 1),
            )
            return torch.sum(matrix_strided, 2)[:, :, 1:]

        if block_table.numel() != 0:
            raise NotImplementedError("Doesn't support prefix cache")
        last_q = self.last_q_size
        inf_value: float = float("inf")
        topk_vertical_inf: int = 30
        topk_slash_inf: int = 100
        LAST_Q_MASK = self.last_q_mask
        vertical_size = min(2048, q.shape[2])
        slash_size = min(2048, q.shape[2])

        _, _, q_len, d = q.shape

        # GQA fix
        group_size = q.shape[1] // k.shape[1]
        bs, num_device_k_heads, seq_len, head_dim = k.shape
        if self.layer_idx == 3:
            print("pitor breakpoint")
        k = k.unsqueeze(2).repeat(1, 1, group_size, 1, 1).reshape(bs, num_device_k_heads * group_size, seq_len, head_dim)
        v = v.unsqueeze(2).repeat(1, 1, group_size, 1, 1).reshape(bs, num_device_k_heads * group_size, seq_len, head_dim)
        # END GQA fix

        # Compute scaled dot-product attention
        qk = torch.einsum("bhmk, bhnk -> bhmn", q[:, :, -last_q:, :], k) / math.sqrt(d)
        qk[:, :, :, -last_q:] = torch.where(LAST_Q_MASK[..., -last_q:, -last_q:], qk[:, :, :, -last_q:], -inf_value)
        qk = torch.nn.functional.softmax(qk, dim=-1, dtype=torch.float32).to(q.dtype)

        assert topk_vertical_inf <= vertical_size
        assert topk_slash_inf <= slash_size

        # Compute top verticals
        vertical = qk.sum(-2, keepdim=True)
        vertical[..., :topk_vertical_inf] = inf_value
        vertical_topk = torch.topk(vertical, vertical_size, -1).indices

        # # Compute top slashes
        slash = sum_over_diagonals(qk)[..., : -last_q + 1]
        slash[..., -topk_slash_inf:] = inf_value

        # Qwen-1m backend
        # NOTE: Slash index calculated differently compared to Pitor's codebase
        # slash = torch.topk(slash, slash_size, -1).indices
        # sm_scale = head_dim**-0.5
        # vertical_indices_count = torch.tensor([vertical_size] * q.shape[1], dtype=torch.int32, device=q.device)
        # slash_indices_count = torch.tensor([slash_size] * q.shape[1], dtype=torch.int32, device=q.device)
        # global vs_args
        # return _vertical_slash_sparse_attention(
        #     q, k, v, vertical_topk, slash, sm_scale, causal=True, vertical_indices_count=vertical_indices_count, slash_indices_count=slash_indices_count
        # )

        # pitor backend -> same result as above
        slash = (q_len - 1) - torch.topk(slash, slash_size, -1).indices
        return vertical_slash_sparse_attention(q, k, v, vertical_topk, slash)


def _vertical_slash_sparse_attention(
    query: torch.Tensor,  # [BATCH, N_HEADS, N_CTX, D_HEAD]
    key: torch.Tensor,  # [BATCH, N_HEADS, N_KV_CTX, D_HEAD]
    value: torch.Tensor,  # [BATCH, N_HEADS, N_KV_CTX, D_HEAD]
    v_idx: torch.Tensor,  # [BATCH, N_HEADS, NNZ_V]
    s_idx: torch.Tensor,  # [BATCH, N_HEADS, NNZ_S]
    softmax_scale: float,
    causal: bool = True,
    block_size_M: int = 64,
    block_size_N: int = 64,
    vertical_indices_count: torch.Tensor = None,  # [N_HEADS,]
    slash_indices_count: torch.Tensor = None,
):
    batch_size, num_heads, context_size, head_dim = query.shape
    _, _, kv_seq_len, _ = key.shape

    if head_dim not in [16, 32, 64, 128, 256, 512]:
        target_dim = 2 ** math.ceil(math.log2(head_dim)) - head_dim
        query = F.pad(query, [0, target_dim, 0, 0, 0, 0, 0, 0])
        key = F.pad(key, [0, target_dim, 0, 0, 0, 0, 0, 0])
        value = F.pad(value, [0, target_dim, 0, 0, 0, 0, 0, 0])

    v_idx = v_idx.to(torch.int32).reshape((batch_size, num_heads, -1)).sort(dim=-1, descending=False)[0]
    s_idx = s_idx.to(torch.int32).reshape((batch_size, num_heads, -1)).sort(dim=-1, descending=True)[0]
    q_seqlens = torch.tensor([context_size], dtype=torch.int32, device=query.device)
    kv_seqlens = torch.tensor([kv_seq_len], dtype=torch.int32, device=query.device)

    if vertical_indices_count is not None and slash_indices_count is not None:
        (
            block_count,
            block_offset,
            column_count,
            column_index,
        ) = convert_vertical_slash_indexes_mergehead(
            q_seqlens,
            kv_seqlens,
            v_idx,
            s_idx,
            vertical_indices_count,
            slash_indices_count,
            context_size,
            block_size_M,
            block_size_N,
            causal,
        )
    else:
        (
            block_count,
            block_offset,
            column_count,
            column_index,
        ) = convert_vertical_slash_indexes(
            q_seqlens,
            kv_seqlens,
            v_idx,
            s_idx,
            context_size,
            block_size_M,
            block_size_N,
            causal,
        )

    q = query.transpose(1, 2).contiguous()
    k = key.transpose(1, 2).contiguous()
    v = value.transpose(1, 2).contiguous()
    out, lse = sparse_attn_func(
        q,
        k,
        v,
        block_count,
        block_offset,
        column_count,
        column_index,
        causal=causal,
        softmax_scale=softmax_scale,
        return_softmax_lse=True,
    )
    out = out.transpose(1, 2).contiguous()
    softmax_lse = lse.reshape(*lse.shape, 1)
    return (out[..., :context_size, :head_dim], softmax_lse[..., :context_size, :])


def _sum_all_diagonal_matrix(mat: torch.tensor):
    h, n, m = mat.shape
    # Zero matrix used for padding
    zero_mat = torch.zeros((h, n, n), device=mat.device)
    # pads the matrix on left and right
    mat_padded = torch.cat((zero_mat, mat, zero_mat), -1)
    # Change the strides
    mat_strided = mat_padded.as_strided((1, n, n + m), (n * (2 * n + m), 2 * n + m + 1, 1))
    # Sums the resulting matrix's columns
    sum_diags = torch.sum(mat_strided, 1)
    return sum_diags[:, 1:]  # drop left bottom corner


# TODO: pitor triton -> To be removed
@triton.jit
def _triton_mixed_sparse_attn_fwd_kernel(
    Q,
    K,
    V,
    seqlens,
    sm_scale,
    block_count,
    block_offset,
    column_count,
    column_index,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    Z,
    H,
    N_CTX,
    NUM_ROWS,
    NNZ_S,
    NNZ_V,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    dtype: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    seqlen = tl.load(seqlens + off_hz // H)
    if start_m * BLOCK_M >= seqlen:
        return

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    qo_offset = (off_hz // H) * stride_qz + (off_hz % H) * stride_qh
    kv_offset = (off_hz // H) * stride_kz + (off_hz % H) * stride_kh

    q_ptrs = Q + qo_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + kv_offset + offs_d[:, None] * stride_kk
    v_ptrs = V + kv_offset + offs_d[None, :] * stride_vk
    o_ptrs = Out + qo_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok

    num_blks = tl.load(block_count + off_hz * NUM_ROWS + start_m)
    blks_ptr = block_offset + (off_hz * NUM_ROWS + start_m) * NNZ_S
    num_cols = tl.load(column_count + off_hz * NUM_ROWS + start_m)
    cols_ptr = column_index + (off_hz * NUM_ROWS + start_m) * NNZ_V

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    q = tl.load(q_ptrs)
    q = (q * qk_scale).to(dtype)

    # loop over k, v and update accumulator
    m_mask = offs_m[:, None] < seqlen

    for block_index in range(num_blks):
        start_n = tl.load(blks_ptr + block_index)
        cols = start_n + offs_n
        n_mask = cols < seqlen
        # -- load k, v --
        k = tl.load(k_ptrs + cols[None, :] * stride_kn, mask=n_mask[None, :], other=0.0)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn, mask=n_mask[:, None], other=0.0)
        # -- compute qk --
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        causal_mask = cols[None, :] <= offs_m[:, None]
        qk = tl.where(m_mask & causal_mask, qk, float("-inf"))
        qk += tl.dot(q, k)
        # -- compute scaling constant --
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        # -- scale and update acc --
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(dtype), v)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    for start_n in range(0, num_cols, BLOCK_N):
        n_mask = start_n + offs_n < num_cols
        cols = tl.load(cols_ptr + start_n + offs_n, mask=n_mask, other=0)
        # -- load k, v --
        k = tl.load(k_ptrs + cols[None, :] * stride_kn, mask=n_mask[None, :], other=0.0)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn, mask=n_mask[:, None], other=0.0)
        # -- compute qk --
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.where(m_mask & n_mask, qk, float("-inf"))
        qk += tl.dot(q, k)
        # -- compute scaling constant --
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        # -- scale and update acc --
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(dtype), v)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    # write back O
    acc /= l_i[:, None]
    # acc = tl.where(m_mask, acc / l_i[:, None], 0.0)
    tl.store(o_ptrs, acc.to(dtype), mask=m_mask)
