from pathlib import Path
from typing import Optional, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    TextIteratorStreamer,
    Wav2Vec2FeatureExtractor,
)
from transformers.generation.utils import (
    GenerateBeamDecoderOnlyOutput,
    GenerateBeamEncoderDecoderOutput,
    GenerateDecoderOnlyOutput,
    GenerateEncoderDecoderOutput,
)

try:
    from .asr_config import ASRConfig
except ImportError:
    from asr_config import ASRConfig  # type: ignore[no-redef]

class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, bias=False, dropout_rate=0.05):
        super().__init__()
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        w12_out = self.w12(x)
        x_gate, x_val = w12_out.chunk(2, dim=-1)
        x = F.silu(x_gate) * x_val
        x = self.w3(x)
        x = self.dropout(x)
        return x

class MoEAudioProjector(nn.Module):
    """
    Stable Single-Stage MoE Projector (Sparse Upcycling).
    
    Strategy:
    - Starts as a 'Dense' model (all experts identical) to prevent router collapse.
    - Uses Noise Injection + Z-Loss to gradually diverge experts during training.
    - No complex curriculum required; trainable end-to-end immediately.
    """
    def __init__(self, config):
        super().__init__()
        self.k = getattr(config, "projector_pool_stride", 2)
        self.num_experts = getattr(config, "num_experts", 8)
        self.top_k = getattr(config, "moe_top_k", 2)
        
        # Stability Hyperparameters (Critical for Single-Stage)
        self.router_z_loss_coef = 1e-3 
        self.router_aux_loss_coef = 1e-2
        self.noise_std = 0.01  # Jitter for router exploration

        in_dim = config.encoder_dim * self.k
        out_dim = config.llm_dim
        expert_hidden_dim = getattr(config, "projector_hidden_dim", 512)

        # Normalization
        self.ln_pre = nn.LayerNorm(in_dim, eps=1e-6)
        self.ln_post = nn.LayerNorm(out_dim, eps=1e-6)

        # Router (Bias=True helps with initial load balancing)
        self.router = nn.Linear(in_dim, self.num_experts, bias=True)

        # Experts
        # Shared Expert (Always active, captures common features)
        self.shared_expert = SwiGLU(in_dim, expert_hidden_dim, out_dim)
        
        # Routed Experts (Specialists)
        self.experts = nn.ModuleList()

        self.last_aux_loss = 0.0

        # --- CRITICAL: SPARSE UPCYCLING INITIALIZATION ---
        # This is the "magic" that enables single-stage training.
        # Instead of random experts (which confuses the router), we start
        # with valid weights and slightly perturb them.
        with torch.no_grad():
            # 1. Initialize Shared Expert well (Xavier/Kaiming)
            nn.init.xavier_normal_(self.shared_expert.w12.weight)
            nn.init.xavier_normal_(self.shared_expert.w3.weight)

            # 2. CLONE Shared Weights to All Experts
            # At Step 0, the model acts like a dense Ensemble.
            # The router can pick *any* expert and get a valid result.
            for expert in self.experts:
                expert.w12.weight.copy_(self.shared_expert.w12.weight)
                expert.w3.weight.copy_(self.shared_expert.w3.weight)
                
                # 3. Symmetry Breaking: Add noise to force divergence
                # This pushes experts slightly apart so the router has a reason to choose.
                expert.w12.weight.add_(torch.randn_like(expert.w12.weight) * 0.02)
                expert.w3.weight.add_(torch.randn_like(expert.w3.weight) * 0.02)

            # 4. Zero-Init Router: Start with maximum entropy (equal probability)
            self.router.weight.zero_()
            self.router.bias.zero_()

    def forward(self, x):
        # x shape:
        batch_size, seq_len, dim = x.size()
        
        # 1. Pooling/Stride Logic
        remainder = seq_len % self.k
        if remainder:
            pad_len = self.k - remainder
            x = F.pad(x, (0, 0, 0, pad_len), value=0.0)
        
        # Reshape for pooling:
        x = x.view(batch_size, -1, dim * self.k)
        new_seq_len = x.shape[1]
        
        # Flatten for Expert Routing:
        x_flat = x.view(-1, dim * self.k)
        norm_x = self.ln_pre(x_flat)

        # --- SHARED EXPERT (Always Active) ---
        shared_out = self.shared_expert(norm_x)

        # --- ROUTING ---
        router_logits = self.router(norm_x)

        # Noise Injection (Training Only)
        # Adds stochasticity to prevent the router from getting stuck early on
        if self.training:
            router_logits = router_logits + torch.randn_like(router_logits) * self.noise_std

        routing_probs = F.softmax(router_logits, dim=-1)
        
        # Top-K Selection
        top_k_weights, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1)
        
        # Re-normalize weights to sum to 1
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-20)

        # --- ROUTED EXPERTS DISPATCH ---
        # Create output tensor
        routed_out = torch.zeros_like(shared_out)

        # Naive Loop Dispatch (Compatible with all PyTorch versions)
        # For K=2, this loops twice. 
        for k in range(self.top_k):
            # Get indices and weights for the k-th choice
            indices_k = top_k_indices[:, k] 
            weights_k = top_k_weights[:, k].unsqueeze(-1) 

            # Iterate over experts (could be optimized with scatter/gather kernels)
            for expert_idx, expert in enumerate(self.experts):
                # Find tokens that chose this expert for their k-th slot
                mask = (indices_k == expert_idx)
                if mask.any():
                    expert_input = norm_x[mask]
                    expert_output = expert(expert_input)
                    
                    # Add weighted output to the buffer
                    # We use index_add_ or masked assignment
                    routed_out[mask] += expert_output * weights_k[mask]

        # --- LOSS CALCULATION ---
        self.last_aux_loss = 0.0
        if self.training:
            # 1. Load Balancing Loss (Prevent collapse to 1 expert)
            # P(x) * f(x)
            tokens_per_expert = torch.histc(
                top_k_indices.float(), bins=self.num_experts, min=0, max=self.num_experts-1
            )
            fraction_routed = tokens_per_expert / top_k_indices.numel()
            prob_sum = routing_probs.mean(dim=0)
            
            balance_loss = (fraction_routed * prob_sum).sum() * self.num_experts
            
            # 2. Router Z-Loss (Prevent huge logits -> instability)
            # log(sum(exp(x))^2)
            z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()

            self.last_aux_loss = (
                self.router_aux_loss_coef * balance_loss +
                self.router_z_loss_coef * z_loss
            )

        # --- COMBINE & RESHAPE ---
        final_out = self.ln_post(shared_out + routed_out)
        final_out = final_out.view(batch_size, new_seq_len, -1)

        return final_out, self.last_aux_loss
    
class ASRModel(PreTrainedModel):
    config_class = ASRConfig
    base_model_prefix = "model"
    main_input_name = "input_values"
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True
    _is_loading_from_pretrained: bool = False
    _pretrained_model_path: Optional[str] = None

    TASK_PROMPTS = {
        "transcribe": "Transcribe: <audio>",
        "continue": "Continue: <audio>",
        "describe": "Describe: <audio>",
        "emotion": "Emotion: <audio>",
    }

    @staticmethod
    def _create_feature_extractor(audio_model_id: str, training_mode: bool = False):
        is_whisper = "whisper" in audio_model_id.lower()
        if is_whisper:
            from transformers import WhisperConfig, WhisperFeatureExtractor

            encoder_config = WhisperConfig.from_pretrained(audio_model_id)
            num_mel_bins = encoder_config.num_mel_bins

            return WhisperFeatureExtractor.from_pretrained(
                audio_model_id,
                feature_size=num_mel_bins,
                apply_spec_augment=training_mode,
                # Optimized SpecAugment for frozen-backbone training (keep minimal)
                mask_feature_prob=0.05,       
                mask_feature_length=15,       
                mask_feature_min_masks=0,     
                mask_time_prob=0.05,          
                mask_time_length=20,          
                mask_time_min_masks=1,        
            )
        return Wav2Vec2FeatureExtractor.from_pretrained(audio_model_id)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        from transformers import AutoFeatureExtractor

        config = kwargs.pop("config", None)
        if config is None:
            config = ASRConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        kwargs["feature_extractor"] = AutoFeatureExtractor.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )

        cls._is_loading_from_pretrained = True
        cls._pretrained_model_path = pretrained_model_name_or_path

        try:
            from safetensors.torch import load_file
            from transformers.utils.hub import cached_file

            model = cls(config, **kwargs)

            subfolder = kwargs.get("subfolder")
            revision = kwargs.get("revision")
            cache_kwargs = {}
            if subfolder:
                cache_kwargs["subfolder"] = subfolder
            if revision:
                cache_kwargs["revision"] = revision

            model_file = cached_file(
                pretrained_model_name_or_path,
                "model.safetensors",
                _raise_exceptions_for_missing_entries=False,
                **cache_kwargs,
            )

            if not model_file:
                raise FileNotFoundError(
                    f"model.safetensors not found in {pretrained_model_name_or_path}. "
                )

            state_dict = load_file(model_file)
            model.load_state_dict(state_dict, strict=False, assign=True)

            # Let Accelerate handle dtype and device management
            device = kwargs.get("device")
            if device is not None:
                model = model.to(device)

            return model
        finally:
            cls._is_loading_from_pretrained = False
            del cls._pretrained_model_path

    def __init__(self, config: ASRConfig, **kwargs):
        super().__init__(config)

        feature_extractor = kwargs.pop("feature_extractor", None)
        self.system_prompt = config.system_prompt

        self.encoder = self._create_encoder(config)

        is_whisper = "whisper" in config.audio_model_id.lower() or (
            hasattr(self.encoder.config, "model_type")
            and "whisper" in self.encoder.config.model_type.lower()
        )

        if is_whisper:
            self.main_input_name = "input_features"
        else:
            self.main_input_name = "input_values"

        if feature_extractor is not None:
            self.feature_extractor = feature_extractor
        else:
            training_mode = getattr(config, "use_specaugment", False)
            self.feature_extractor = self._create_feature_extractor(
                config.audio_model_id, training_mode=training_mode
            )

        self.decoder = self._create_decoder(config)
        self.generation_config = self.decoder.generation_config

        config_params = [
            "max_new_tokens", "min_new_tokens", "num_beams", "do_sample",
            "temperature", "top_k", "top_p", "repetition_penalty",
            "length_penalty", "no_repeat_ngram_size", "early_stopping", "use_cache"
        ]
        for param in config_params:
            if hasattr(config, param) and getattr(config, param) is not None:
                setattr(self.generation_config, param, getattr(config, param))

        self._init_tokenizer()

        # Store encoder type for forward pass
        self.is_whisper_encoder = "whisper" in config.audio_model_id.lower() or (
            hasattr(self.encoder.config, "model_type") and "whisper" in self.encoder.config.model_type.lower()
        )

        from types import SimpleNamespace

        encoder_dim = config.encoder_dim
        if encoder_dim is None:
            if hasattr(self.encoder.config, "hidden_size"):
                encoder_dim = self.encoder.config.hidden_size
            elif hasattr(self.encoder.config, "d_model"):
                encoder_dim = self.encoder.config.d_model
            else:
                raise ValueError("Could not auto-detect encoder_dim. Please specify in config.")

        llm_dim = config.llm_dim
        if llm_dim is None:
            if hasattr(self.decoder.config, "hidden_size"):
                llm_dim = self.decoder.config.hidden_size
            elif hasattr(self.decoder.config, "d_model"):
                llm_dim = self.decoder.config.d_model
            else:
                raise ValueError("Could not auto-detect llm_dim. Please specify in config.")

        proj_kwargs = {k: v for k, v in vars(config).items() if k.startswith('projector_')}

        proj_kwargs.update({
            "encoder_dim": encoder_dim,
            "llm_dim": llm_dim,
            "num_experts": 8,
            "moe_top_k": 2,
            "projector_hidden_dim": llm_dim // 4
        })

        projector_config = SimpleNamespace(**proj_kwargs)

        self.projector = MoEAudioProjector(projector_config)

        target_dtype = getattr(torch, config.model_dtype)
        self.projector = self.projector.to(dtype=target_dtype)

        self.label_smoothing = getattr(config, "label_smoothing", 0.1)
        self.loss_fct = nn.CrossEntropyLoss(
            ignore_index=-100,
            label_smoothing=self.label_smoothing
        )

        self._no_split_modules = self.decoder._no_split_modules

    @classmethod
    def _create_encoder(cls, config: ASRConfig):
        target_dtype = getattr(torch, config.model_dtype)

        encoder_kwargs = {
            "attn_implementation": config.attn_implementation,
            "dtype": target_dtype,
            "low_cpu_mem_usage": True,
        }
        if not cls._is_loading_from_pretrained:
            encoder_kwargs["device_map"] = "auto"

        if "whisper" in config.audio_model_id.lower():
            from transformers import WhisperModel
            full_model = WhisperModel.from_pretrained(config.audio_model_id, **encoder_kwargs)
            encoder = full_model.encoder
            del full_model
        else:
            encoder = AutoModel.from_pretrained(config.audio_model_id, **encoder_kwargs)

        encoder = encoder.to(dtype=target_dtype)
        encoder.requires_grad_(False)
        encoder.eval()

        return encoder

    @classmethod
    def _create_decoder(cls, config: ASRConfig):
        target_dtype = getattr(torch, config.model_dtype)

        decoder_kwargs = {
            "attn_implementation": config.attn_implementation,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "dtype": target_dtype,
        }

        decoder = AutoModelForCausalLM.from_pretrained(config.text_model_id, **decoder_kwargs)

        decoder = decoder.to(dtype=target_dtype)
        decoder.config.use_cache = config.use_cache
        decoder.requires_grad_(False)
        decoder.eval()

        return decoder

    def _init_weights(self, module):
        pass

    def can_generate(self) -> bool:
        return True

    @property
    def _tied_weights_keys(self):
        if hasattr(self.decoder, "_tied_weights_keys"):
            return [f"decoder.{k}" for k in self.decoder._tied_weights_keys]
        return []

    def _init_tokenizer(self):
        model_path = (
            self.__class__._pretrained_model_path
            if self._is_loading_from_pretrained
            else self.config.text_model_id
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        if (
            self.tokenizer.pad_token is None
            or self.tokenizer.pad_token_id == self.tokenizer.eos_token_id
        ) and "<|finetune_right_pad_id|>" in self.tokenizer.get_vocab():
            self.tokenizer.pad_token = "<|finetune_right_pad_id|>"

        existing_special = self.tokenizer.additional_special_tokens or []

        if "<audio>" not in existing_special:
            special_tokens = {"additional_special_tokens": existing_special + ["<audio>"]}
            num_added_tokens = self.tokenizer.add_special_tokens(special_tokens)
            if num_added_tokens > 0:
                self.decoder.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)

        current_embed_size = self.decoder.get_input_embeddings().weight.shape[0]
        expected_size = len(self.tokenizer)
        if current_embed_size != expected_size:
            self.decoder.resize_token_embeddings(expected_size, mean_resizing=False)

        self.audio_token_id = self.tokenizer.convert_tokens_to_ids("<audio>")
        self.tokenizer.padding_side = "right"

        for cfg in [self.config.text_config, self.decoder.config, self.generation_config]:
            if isinstance(cfg, dict):
                cfg["pad_token_id"] = self.tokenizer.pad_token_id
                cfg["eos_token_id"] = self.tokenizer.eos_token_id
                cfg["bos_token_id"] = self.tokenizer.bos_token_id
            else:
                cfg.pad_token_id = self.tokenizer.pad_token_id
                cfg.eos_token_id = self.tokenizer.eos_token_id
                cfg.bos_token_id = self.tokenizer.bos_token_id

    def get_processor(self):
        try:
            from .asr_processing import ASRProcessor
        except ImportError:
            from asr_processing import ASRProcessor 

        return ASRProcessor(feature_extractor=self.feature_extractor, tokenizer=self.tokenizer)

    def state_dict(self, *args, **kwargs):
        return self._get_trainable_state_dict()

    def _get_trainable_state_dict(self):
        state = {}
        projector_state = self.projector.state_dict()
        for name, tensor in projector_state.items():
            state[f"projector.{name}"] = tensor
        return state

    def get_input_embeddings(self):
        return self.decoder.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.decoder.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.decoder.get_output_embeddings()

    def set_output_embeddings(self, value):
        self.decoder.set_output_embeddings(value)

    def _pool_attention_mask(self, mask: torch.Tensor, k: int) -> torch.Tensor:
        """Pool attention mask to match projector stride.

        If any frame in the stride window is 1 (valid), the pooled frame is 1.
        Uses max pooling: 1=valid, 0=pad.
        """
        batch_size, seq_len = mask.shape

        # Ensure mask is long dtype from the start
        mask = mask.long()

        # Pad to match projector logic if needed
        remainder = seq_len % k
        if remainder > 0:
            pad_len = k - remainder
            mask = torch.nn.functional.pad(mask, (0, pad_len), value=0)

        # Reshape and max over the stride dimension
        # [Batch, Seq_Pooled, K] -> max(dim=-1)
        # Using .max() on long tensor preserves dtype
        pooled = mask.view(batch_size, -1, k).max(dim=-1).values

        return pooled

    def _encode_audio(
        self,
        input_values: torch.Tensor,
        audio_attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            if self.is_whisper_encoder:
                audio_features = self.encoder(
                    input_features=input_values,
                    attention_mask=audio_attention_mask,
                ).last_hidden_state
            else:
                audio_features = self.encoder(
                    input_values=input_values,
                    attention_mask=audio_attention_mask,
                ).last_hidden_state

        audio_embeds, aux_loss = self.projector(audio_features)

        return audio_embeds, aux_loss

    def _get_audio_expansion_details(self, input_ids: torch.Tensor, num_audio_tokens: int) -> dict:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        audio_mask = input_ids == self.audio_token_id

        token_counts = torch.where(audio_mask, num_audio_tokens, 1)
        cumsum_counts = torch.cumsum(token_counts, dim=1)
        new_start_positions = torch.cat(
            [
                torch.zeros(batch_size, 1, dtype=torch.long, device=device),
                cumsum_counts[:, :-1],
            ],
            dim=1,
        )
        new_seq_len = seq_len - 1 + num_audio_tokens

        return {
            "new_seq_len": new_seq_len,
            "new_start_positions": new_start_positions,
            "audio_mask": audio_mask,
        }

    def _expand_tensor_for_audio(
        self,
        input_ids: torch.Tensor,
        tensor_to_expand: Optional[torch.Tensor],
        num_audio_tokens: int,
        fill_value: Union[int, float] = -100,
        audio_fill_value: Union[int, float, torch.Tensor] = -100,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        details = self._get_audio_expansion_details(input_ids, num_audio_tokens)
        new_seq_len = details["new_seq_len"]
        new_start_positions = details["new_start_positions"]
        audio_mask = details["audio_mask"]

        is_expanding_input_ids = tensor_to_expand is None
        if is_expanding_input_ids:
            tensor_to_expand = input_ids
            fill_value = self.tokenizer.pad_token_id
            audio_fill_value = self.audio_token_id

        expanded = torch.full(
            (batch_size, new_seq_len),
            fill_value,
            dtype=tensor_to_expand.dtype,
            device=device,
        )

        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, seq_len)
        non_audio_mask = ~audio_mask
        expanded[batch_indices[non_audio_mask], new_start_positions[non_audio_mask]] = (
            tensor_to_expand[non_audio_mask]
        )

        audio_positions = audio_mask.int().argmax(dim=1)
        audio_new_start = new_start_positions[
            torch.arange(batch_size, device=device), audio_positions
        ]
        audio_token_indices = torch.arange(num_audio_tokens, device=device).unsqueeze(0)
        audio_positions_expanded = audio_new_start.unsqueeze(1) + audio_token_indices
        batch_idx_expanded = (
            torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, num_audio_tokens)
        )

        # Handle both scalar and tensor audio_fill_value
        if isinstance(audio_fill_value, torch.Tensor):
            # audio_fill_value is a tensor with shape [batch_size, num_audio_tokens]
            # Broadcast it to the audio positions
            expanded[batch_idx_expanded, audio_positions_expanded] = audio_fill_value
        else:
            needs_audio_fill = audio_fill_value != fill_value
            if needs_audio_fill:
                expanded[batch_idx_expanded, audio_positions_expanded] = audio_fill_value

        return expanded

    def _expand_audio_tokens(self, input_ids: torch.Tensor, num_audio_tokens: int) -> torch.Tensor:
        return self._expand_tensor_for_audio(input_ids, None, num_audio_tokens)

    def _expand_for_audio_tokens(
        self,
        input_ids: torch.Tensor,
        tensor_to_expand: torch.Tensor,
        num_audio_tokens: int,
        fill_value: Union[int, float],
        audio_fill_value: Union[int, float, torch.Tensor] = None,
    ) -> torch.Tensor:
        if audio_fill_value is None:
            audio_fill_value = fill_value
        return self._expand_tensor_for_audio(
            input_ids, tensor_to_expand, num_audio_tokens, fill_value, audio_fill_value
        )

    def _prepare_audio_inputs_embeds(
        self, expanded_input_ids: torch.Tensor, audio_embeds: torch.Tensor
    ) -> torch.Tensor:
        inputs_embeds = self.decoder.get_input_embeddings()(expanded_input_ids)
        special_audio_mask = (expanded_input_ids == self.audio_token_id).unsqueeze(-1)
        special_audio_mask = special_audio_mask.expand_as(inputs_embeds)
        # Ensure audio_embeds matches the dtype of inputs_embeds
        audio_embeds = audio_embeds.to(inputs_embeds.dtype)
        audio_embeds_flat = audio_embeds.reshape(-1, audio_embeds.shape[-1])
        return inputs_embeds.masked_scatter(special_audio_mask, audio_embeds_flat)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_values: Optional[torch.Tensor] = None,
        input_features: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        num_items_in_batch: Optional[int] = None,
        **kwargs,
    ):
        audio_inputs = input_values if input_values is not None else input_features
        if audio_inputs is not None:
            if input_ids is None:
                raise ValueError("forward() requires both audio inputs and input_ids.")

            audio_attention_mask = kwargs.pop("audio_attention_mask", None)
            kwargs.pop("past_key_values", None)
            use_cache = kwargs.pop("use_cache", None)

            audio_embeds, aux_loss = self._encode_audio(
                input_values=audio_inputs,
                audio_attention_mask=audio_attention_mask,
            )

            num_audio_tokens = audio_embeds.shape[1]

            # Pool the audio attention mask to match final audio_embeds shape
            # Calculate total stride (encoder + projector downsampling)
            if audio_attention_mask is not None:
                total_stride = audio_attention_mask.shape[1] // audio_embeds.shape[1]
                pooled_audio_mask = self._pool_attention_mask(
                    audio_attention_mask,
                    total_stride
                )
            else:
                pooled_audio_mask = torch.ones(
                    audio_embeds.shape[:2],
                    device=audio_embeds.device,
                    dtype=torch.long
                )

            expanded_input_ids = self._expand_audio_tokens(input_ids, num_audio_tokens)
            inputs_embeds = self._prepare_audio_inputs_embeds(expanded_input_ids, audio_embeds)

            if attention_mask is not None:
                # Use pooled audio mask instead of hardcoded 1
                full_attention_mask = self._expand_for_audio_tokens(
                    input_ids, attention_mask, num_audio_tokens, fill_value=1, audio_fill_value=pooled_audio_mask
                )
            else:
                full_attention_mask = None

            if labels is not None:
                labels = self._expand_for_audio_tokens(
                    input_ids, labels, num_audio_tokens, fill_value=-100
                )
        else:
            inputs_embeds = self.decoder.get_input_embeddings()(input_ids)
            full_attention_mask = attention_mask
            use_cache = kwargs.pop("use_cache", None)

        # 2. Run Decoder (LLM)
        outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=None, 
            use_cache=use_cache if use_cache is not None else False,
            **kwargs,
        )

        # 3. Calculate Losses
        if labels is not None:
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_labels = shift_labels.view(-1)

            loss = self.loss_fct(flat_logits, flat_labels)

            if audio_inputs is not None:
                loss = loss + 0.01 * aux_loss

            from transformers.modeling_outputs import CausalLMOutputWithPast
            outputs = CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        return outputs

    @torch.no_grad()
    def generate(
        self,
        input_values: Optional[torch.Tensor] = None,
        input_features: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        task: Optional[str] = None,
        **generate_kwargs,
    ) -> Union[
        torch.Tensor,
        GenerateDecoderOnlyOutput,
        GenerateEncoderDecoderOutput,
        GenerateBeamDecoderOnlyOutput,
        GenerateBeamEncoderDecoderOutput,
    ]:
        audio_inputs = input_values if input_values is not None else input_features
        if audio_inputs is None:
            raise ValueError("input_values or input_features must be provided for generation")

        audio_embeds, _ = self._encode_audio(
            audio_inputs,
            audio_attention_mask=audio_attention_mask
        )
        batch_size = audio_embeds.shape[0]
        device = audio_embeds.device

        if system_prompt is None:
            system_prompt = self.system_prompt

        if user_prompt is None:
            user_prompt = (
                self.TASK_PROMPTS.get(task, self.config.user_prompt or "Transcribe: <audio>")
                or "Transcribe: <audio>"
            )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": user_prompt,
            }
        )

        prompt_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(device)

        if len(prompt_ids.shape) == 1:
            prompt_ids = prompt_ids.unsqueeze(0)

        if prompt_ids.shape[0] == 1 and batch_size > 1:
            prompt_ids = prompt_ids.expand(batch_size, -1)

        if not (prompt_ids == self.audio_token_id).any():
            raise ValueError("Audio token <audio> not found in prompt")

        num_audio_tokens = audio_embeds.shape[1]
        expanded_prompt_ids = self._expand_audio_tokens(prompt_ids, num_audio_tokens)
        inputs_embeds = self._prepare_audio_inputs_embeds(expanded_prompt_ids, audio_embeds)

        # Pool the audio attention mask to match final audio_embeds shape
        # Calculate total stride (encoder + projector downsampling)
        if audio_attention_mask is not None:
            total_stride = audio_attention_mask.shape[1] // audio_embeds.shape[1]
            pooled_audio_mask = self._pool_attention_mask(
                audio_attention_mask,
                total_stride
            )
        else:
            pooled_audio_mask = torch.ones(
                audio_embeds.shape[:2],
                device=audio_embeds.device,
                dtype=torch.long
            )

        # Create attention mask for text tokens (all valid)
        text_attention_mask = torch.ones_like(prompt_ids)

        # Expand to include audio tokens with pooled audio mask
        attention_mask = self._expand_for_audio_tokens(
            prompt_ids,
            text_attention_mask,
            num_audio_tokens,
            fill_value=1,
            audio_fill_value=pooled_audio_mask
        )
        
        config_params = [
            "max_new_tokens", "min_new_tokens", "num_beams", "do_sample",
            "temperature", "top_k", "top_p", "repetition_penalty",
            "length_penalty", "no_repeat_ngram_size", "early_stopping",
        ]
        for param in config_params:
            if hasattr(self.config, param) and getattr(self.config, param) is not None:
                generate_kwargs.setdefault(param, getattr(self.config, param))

        generate_kwargs.setdefault("use_cache", True)
        generate_kwargs.setdefault(
            "eos_token_id", self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        )
        generate_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        prompt_length = expanded_prompt_ids.shape[1]

        generated_ids = self.decoder.generate(
            input_ids=expanded_prompt_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        return generated_ids[:, prompt_length:]

    def save_pretrained(self, save_directory: Union[str, Path], **kwargs):
        import shutil
        from pathlib import Path as PathlibPath

        save_dir = PathlibPath(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        actual_vocab_size = self.decoder.config.vocab_size
        self.config.vocab_size = actual_vocab_size
        self.config.text_config.vocab_size = actual_vocab_size

        if hasattr(self.encoder.config, "num_mel_bins"):
            self.config.audio_config.num_mel_bins = self.encoder.config.num_mel_bins

        feature_extractor = self.feature_extractor
        tokenizer = self.tokenizer
        del self.feature_extractor
        del self.tokenizer

        try:
            super().save_pretrained(save_dir, **kwargs)
        finally:
            self.feature_extractor = feature_extractor
            self.tokenizer = tokenizer

        self.tokenizer.save_pretrained(save_dir)

        if hasattr(self.encoder.config, "num_mel_bins"):
            num_mel_bins = self.encoder.config.num_mel_bins
            self.feature_extractor.feature_size = num_mel_bins
            self.feature_extractor.num_mel_bins = num_mel_bins
            if hasattr(self.feature_extractor, "n_mels"):
                self.feature_extractor.n_mels = num_mel_bins
            self.feature_extractor.nb_max_frames = 3000

        self.get_processor().save_pretrained(save_dir)

        src_dir = PathlibPath(__file__).parent
        for asr_file in src_dir.glob("asr_*.py"):
            shutil.copy(asr_file, save_dir / asr_file.name)

AutoConfig.register("asr_model", ASRConfig)
AutoModel.register(ASRConfig, ASRModel)