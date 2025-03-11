# ruff: noqa: E501, F841
"""PyTorch Cohere Command 3 model."""
from ast import Attribute
import copy
from typing import Iterable, List, Optional, Protocol, Set, Tuple, Union

import torch
from torch import nn
from transformers import CohereConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
    ReplicatedLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.pooler import Pooler, PoolingType
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
    row_parallel_weight_loader,
)
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.utils import set_weight_attrs
from vllm.sequence import IntermediateTensors, PoolerOutput

from .interfaces import SupportsLoRA, SupportsPP
from .utils import is_pp_missing_parameter, maybe_offload_to_cpu, maybe_prefix, extract_layer_index, make_layers, make_empty_intermediate_tensors_factory
from .commandr import (
    LayerNorm,
    CohereMLP,
    CohereDecoderLayer,
    CohereModel,
    CohereAttention,
    CohereForCausalLM,
    get_sampler,
    make_layers,
    get_rope
)


class SparseCohereAttention(CohereAttention):
    """GQA w/ MInference, SWA, and dense variants definable by configs"""

    def __init__(
        self,
        config: CohereConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        nn.Module.__init__(self)
        assert hasattr(config, "sparse_attention_config"), \
            "Must provide a sparse_attention_config.json file in model directory!"
        self.sparse_attention_config = config.sparse_attention_config
        tp_size = get_tensor_model_parallel_world_size()
        self.config = config
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.num_heads = self.total_num_heads // tp_size
        self.head_dim = self.hidden_size // self.total_num_heads
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = getattr(
            config, "model_max_length", None) or getattr(
                config, "max_position_embeddings", 8192)
        self.rope_theta = config.rope_theta
        self.rope_scaling = getattr(config, "rope_scaling", None)
        self.use_qk_norm = getattr(config, "use_qk_norm", False)
        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            base=self.rope_theta,
            rope_scaling=self.rope_scaling,
            is_neox_style=False,
        )

        # Model v2 has interleaved sliding windows, v1 does not
        interleaved_sliding_window = getattr(config,
                                             "interleaved_sliding_window",
                                             None)
        self.v1 = interleaved_sliding_window is None

        layer_idx = extract_layer_index(prefix)
        layer_has_sliding_window = (
            getattr(config, "sliding_window_pattern", False)
            and (layer_idx + 1) % self.config.sliding_window_pattern != 0)

        self.sliding_window = (interleaved_sliding_window
                               if layer_has_sliding_window else None)
        extra_impl_args = {}
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              per_layer_sliding_window=self.sliding_window,
                              prefix=f"{prefix}.attn",
                              ### BEGIN extra_impl_args ###
                              sparse_attention_config = config.sparse_attention_config,
                              # TODO: What about sparse_threshold? Add to hf config
                              ### END extra_impl_args ###
                              )
        if self.use_qk_norm:
            self.q_norm = LayerNorm(param_shape=(self.num_heads,
                                                 self.head_dim),
                                    eps=config.layer_norm_eps)
            self.k_norm = LayerNorm(param_shape=(self.num_kv_heads,
                                                 self.head_dim),
                                    eps=config.layer_norm_eps)

    def _apply_qk_norm(self, q, k):
        q = q.view(*q.shape[:-1], -1, self.head_dim)
        k = k.view(*k.shape[:-1], -1, self.head_dim)
        q, _ = self.q_norm(q)
        k, _ = self.k_norm(k)
        q = q.view(*q.shape[:-2], -1)
        k = k.view(*k.shape[:-2], -1)
        return q, k

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.use_qk_norm:
            q, k = self._apply_qk_norm(q, k)
        if self.v1 or self.sliding_window:
            q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class SparseCohereDecoderLayer(CohereDecoderLayer):

    def __init__(self,
                 config: CohereConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.self_attn = SparseCohereAttention(config,
                                         cache_config,
                                         quant_config=quant_config,
                                         prefix=f"{prefix}.self_attn")

        self.mlp = CohereMLP(config, quant_config=quant_config)
        self.input_layernorm = LayerNorm(param_shape=(config.hidden_size),
                                         eps=config.layer_norm_eps)

@support_torch_compile
class SparseCohereModel(nn.Module):
    # TODO: Inherit from CohereModule?

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        self.config = config
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size,
                                                   config.hidden_size)
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: SparseCohereDecoderLayer(
                config, cache_config, quant_config, prefix=prefix),
            prefix=f"{prefix}.layers")
        self.norm = LayerNorm(param_shape=(config.hidden_size),
                              eps=config.layer_norm_eps)
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i - self.start_layer],
                attn_metadata,
                residual,
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class SparseCohereForCausalLM(CohereForCausalLM):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # Init nn.Module grandparent, the CohereForCausalLM init logic otherwise
        # is the same below, with the exception of using SparseCohereModel
        nn.Module.__init__(self)  # we do not want to invoke immediate parent init
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        self.config = config
        # currently all existing command R models have `tie_word_embeddings`
        # enabled
        assert config.tie_word_embeddings
        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
        self.quant_config = quant_config
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size,
                                                scale=config.logit_scale)
        self.model = SparseCohereModel(vllm_config=vllm_config,
                                 prefix=maybe_prefix(prefix, "model"))
        self.sampler = get_sampler()
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)