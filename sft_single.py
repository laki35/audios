from __future__ import annotations
import argparse
import json
import math
import shutil
import time
import glob
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.utils.dataclasses import DistributedDataParallelKwargs
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from transformers.utils import cached_file

DEFAULT_MODEL_PATH = Path("models/MOSS-TTS-Nano")
DEFAULT_CODEC_PATH = Path("models/MOSS-Audio-Tokenizer-Nano")

SCHEDULER_CHOICES = (
    "linear",
    "cosine",
    "cosine_with_restarts",
    "polynomial",
    "constant",
    "constant_with_warmup",
    "inverse_sqrt",
)

MODEL_SUPPORT_FILES = (
    "__init__.py",
    "configuration_moss_tts_nano.py",
    "gpt2_decoder.py",
    "modeling_moss_tts_nano.py",
    "prompting.py",
    "tokenization_moss_tts_nano.py",
)

USER_ROLE_PREFIX = "user\n"
USER_TEMPLATE_REFERENCE_PREFIX = "<user_inst>\n- Reference(s):\n"
USER_TEMPLATE_SUFFIX = "\n</user_inst>"
ASSISTANT_TURN_PREFIX = "\n"
ASSISTANT_ROLE_PREFIX = "assistant\n"

OPTIONAL_MESSAGE_FIELDS = (
    ("instruction", "Instruction"),
    ("tokens", "Tokens"),
    ("quality", "Quality"),
    ("sound_event", "Sound Event"),
    ("ambient_sound", "Ambient Sound"),
    ("language", "Language"),
)

def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

def resolve_jsonl_paths(spec: str | List[str]) -> List[Path]:
    if isinstance(spec, (list, tuple)):
        raw_tokens = [str(item).strip() for item in spec if str(item).strip()]
    else:
        raw_tokens = [item.strip() for item in str(spec).split(",") if item.strip()]
    paths: List[Path] = []
    for token in raw_tokens:
        if any(ch in token for ch in "*?[]"):
            matches = [Path(match) for match in sorted(glob.glob(token))]
            paths.extend(match for match in matches if match.suffix == ".jsonl")
            continue
        path = Path(token)
        if path.is_dir():
            paths.extend(sorted(child for child in path.iterdir() if child.suffix == ".jsonl"))
            continue
        paths.append(path)
    deduped: List[Path] = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(path)
    if not deduped:
        raise ValueError(f"No JSONL files found for input spec: {spec}")
    return deduped

def load_jsonl_spec(spec: str | List[str]) -> tuple[List[Path], List[Dict[str, Any]]]:
    paths = resolve_jsonl_paths(spec)
    records: List[Dict[str, Any]] = []
    for path in paths:
        records.extend(load_jsonl(path))
    return paths, records

def format_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def format_duration(seconds: float) -> str:
    return str(timedelta(seconds=max(0, int(seconds))))

def encode_text(tokenizer, text: str) -> List[int]:
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return list(tokenizer.encode(text))

def normalize_audio_codes(value: Any, field_name: str) -> torch.LongTensor:
    tensor = torch.as_tensor(value, dtype=torch.long)
    if tensor.ndim != 2:
        raise ValueError("audio_codes must have shape (T, n_vq)")
    return tensor.cpu().contiguous()

def normalize_audio_code_list(
    value: Any,
    field_name: str,
    *,
    allow_none: bool = False,
) -> Optional[List[Optional[torch.LongTensor]]]:
    if value in (None, "", []):
        return None
    if torch.is_tensor(value):
        return [normalize_audio_codes(value, field_name)]
    if isinstance(value, list):
        if not value:
            return None
        if allow_none and any(item is None for item in value):
            return [
                None if item is None else normalize_audio_codes(item, "field_name")
                for index, item in enumerate(value)
            ]
        first_item = value[0]
        if torch.is_tensor(first_item):
            return [normalize_audio_codes(item, "field_name") for index, item in enumerate(value)]
        if isinstance(first_item, list):
            if first_item and isinstance(first_item[0], list):
                return [normalize_audio_codes(item, "field_name") for index, item in enumerate(value)]
            return [normalize_audio_codes(value, field_name)]
    raise TypeError("Unsupported field_name type")

class MossTTSNanoSFTDataset(Dataset):
    def __init__(
        self,
        records: List[Dict[str, Any]],
        *,
        tokenizer,
        model_config,
        max_length: int,
    ) -> None:
        self.records = list(records)
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.max_length = int(max_length)
        if self.max_length < 8:
            raise ValueError("max_length must be >= 8")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self._build_example(self.records[index], index=index)

    def _build_example(self, record: Dict[str, Any], *, index: int) -> Dict[str, torch.Tensor]:
        if "text" not in record or not str(record["text"]).strip():
            raise ValueError(f"Record {index} is missing text field")
        if "audio_codes" not in record:
            raise ValueError(f"Record {index} is missing audio_codes")
        target_codes = self._normalize_codes_to_model_width(
            normalize_audio_codes(record["audio_codes"], "audio_codes"),
            field_name="audio_codes",
            index=index,
        )
        reference_codes = self._resolve_reference_codes(record, index=index)
        prompt_rows = self._build_prompt_rows(record=record, reference_codes=reference_codes)
        target_rows = self._build_audio_rows(
            target_codes,
            slot_token_id=self.model_config.audio_assistant_slot_token_id,
        )
        end_rows = self._build_text_rows([self.model_config.audio_end_token_id])
        full_sequence = torch.cat([prompt_rows, target_rows, end_rows], dim=0)
        prompt_length = int(prompt_rows.shape[0])
        if prompt_length >= self.max_length:
            raise ValueError(f"Record {index} prompt length >= max_length")
        if full_sequence.shape[0] > self.max_length:
            full_sequence = full_sequence[: self.max_length]
        seq_len = int(full_sequence.shape[0])
        if seq_len < 2:
            raise ValueError(f"Record {index} sequence is too short")
        return {
            "full_input_ids": full_sequence,
            "seq_len": torch.tensor(seq_len, dtype=torch.long),
            "prompt_length": torch.tensor(prompt_length, dtype=torch.long),
        }

    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch_size = len(batch)
        row_width = self.model_config.n_vq + 1
        full_input_ids = torch.full(
            (batch_size, self.max_length, row_width),
            int(self.model_config.audio_pad_token_id),
            dtype=torch.long,
        )
        full_input_ids[:, :, 0] = int(self.model_config.pad_token_id)
        full_attention_mask = torch.zeros((batch_size, self.max_length), dtype=torch.bool)
        loss_mask = torch.zeros((batch_size, self.max_length - 1), dtype=torch.bool)
        for batch_index, item in enumerate(batch):
            sequence = item["full_input_ids"]
            seq_len = int(item["seq_len"].item())
            prompt_length = int(item["prompt_length"].item())
            full_input_ids[batch_index, :seq_len, :] = sequence
            full_attention_mask[batch_index, :seq_len] = True
            loss_mask[batch_index, prompt_length - 1 : seq_len - 1] = True
        labels = full_input_ids[:, 1:, :].clone()
        labels = labels.masked_fill(~loss_mask.unsqueeze(-1), -100)
        labels = labels.masked_fill(~full_attention_mask[:, 1:].unsqueeze(-1), -100)
        labels[:, :, 1:] = labels[:, :, 1:].masked_fill(
            labels[:, :, 1:] == int(self.model_config.audio_pad_token_id),
            -100,
        )
        return {
            "input_ids": full_input_ids[:, :-1, :].contiguous(),
            "attention_mask": full_attention_mask[:, :-1].contiguous(),
            "labels": labels.contiguous(),
        }

    def _resolve_reference_codes(
        self,
        record: Dict[str, Any],
        *,
        index: int,
    ) -> Optional[List[Optional[torch.LongTensor]]]:
        if record.get("ref_audio_codes") is not None:
            codes_list = normalize_audio_code_list(record["ref_audio_codes"], "ref_audio_codes", allow_none=False)
            if codes_list is None:
                return None
            if len(codes_list) != 1:
                raise ValueError("Only supports a single ref_audio_codes")
            return [
                self._normalize_codes_to_model_width(codes_list[0], field_name="ref_audio_codes", index=index)
            ]
        if record.get("ref_audio") is not None:
            raise ValueError(f"Record {index} contains ref_audio but no ref_audio_codes")
        return None

    def _build_prompt_rows(
        self,
        *,
        record: Dict[str, Any],
        reference_codes: Optional[List[Optional[torch.LongTensor]]],
    ) -> torch.LongTensor:
        prefix_ids = [self.model_config.im_start_token_id] + encode_text(
            self.tokenizer,
            USER_ROLE_PREFIX + USER_TEMPLATE_REFERENCE_PREFIX,
        )
        suffix_text = self._build_suffix_text(record)
        assistant_prefix_ids = encode_text(self.tokenizer, USER_TEMPLATE_SUFFIX + ASSISTANT_TURN_PREFIX) + [
            self.model_config.im_start_token_id
        ] + encode_text(self.tokenizer, ASSISTANT_ROLE_PREFIX)
        sections = [self._build_text_rows(prefix_ids)]
        if reference_codes is None:
            sections.append(self._build_text_rows(encode_text(self.tokenizer, "None" + suffix_text)))
        else:
            sections.append(self._build_text_rows([self.model_config.audio_start_token_id]))
            for reference in reference_codes:
                if reference is None:
                    sections.append(self._build_text_rows(encode_text(self.tokenizer, "None")))
                    continue
                sections.append(
                    self._build_audio_rows(reference, slot_token_id=self.model_config.audio_user_slot_token_id)
                )
            sections.append(
                self._build_text_rows([self.model_config.audio_end_token_id] + encode_text(self.tokenizer, suffix_text))
            )
        sections.append(self._build_text_rows(assistant_prefix_ids + [self.model_config.audio_start_token_id]))
        return torch.cat(sections, dim=0)

    def _build_suffix_text(self, record: Dict[str, Any]) -> str:
        lines = [""]
        for field_name, display_name in OPTIONAL_MESSAGE_FIELDS:
            value = record.get(field_name)
            lines.append(f"- {display_name}:")
            lines.append("None" if value in (None, "") else str(value))
        lines.append("- Text:")
        lines.append(str(record["text"]))
        return "\n".join(lines)

    def _build_text_rows(self, token_ids: List[int]) -> torch.LongTensor:
        rows = torch.full(
            (len(token_ids), self.model_config.n_vq + 1),
            int(self.model_config.audio_pad_token_id),
            dtype=torch.long,
        )
        if token_ids:
            rows[:, 0] = torch.tensor(token_ids, dtype=torch.long)
        return rows

    def _build_audio_rows(self, audio_codes: torch.LongTensor, *, slot_token_id: int) -> torch.LongTensor:
        rows = torch.full(
            (int(audio_codes.shape[0]), self.model_config.n_vq + 1),
            int(self.model_config.audio_pad_token_id),
            dtype=torch.long,
        )
        if rows.shape[0] > 0:
            rows[:, 0] = int(slot_token_id)
            rows[:, 1:] = audio_codes
        return rows

    def _normalize_codes_to_model_width(
        self,
        codes: torch.LongTensor,
        *,
        field_name: str,
        index: int,
    ) -> torch.LongTensor:
        target_width = int(self.model_config.n_vq)
        source_width = int(codes.shape[1])
        if source_width > target_width:
            raise ValueError(f"Record {index} field n_vq exceeds target")
        if source_width == target_width:
            return codes
        padded = torch.full(
            (int(codes.shape[0]), target_width),
            int(self.model_config.audio_pad_token_id),
            dtype=torch.long,
        )
        if source_width > 0:
            padded[:, :source_width] = codes
        return padded

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple supervised finetuning for MOSS-TTS-Nano.")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--codec-path", type=str, default=str(DEFAULT_CODEC_PATH))
    parser.add_argument("--train-jsonl", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="output/moss_tts_nano_sft")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lr-scheduler-type", type=str, default="linear", choices=SCHEDULER_CHOICES)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-every-epochs", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--mixed-precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--attn-implementation", type=str, default="auto")
    parser.add_argument("--channelwise-loss-weight", type=str, default="1,32")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def validate_args(args: argparse.Namespace) -> None:
    if args.max_length <= 8:
        raise ValueError("max_length must be > 8")
    if args.per_device_batch_size <= 0:
        raise ValueError("per_device_batch_size must be > 0")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be > 0")
    if args.learning_rate <= 0:
        raise ValueError("learning_rate must be > 0")
    if args.weight_decay < 0:
        raise ValueError("weight_decay must be >= 0")
    if args.warmup_steps < 0:
        raise ValueError("warmup_steps must be >= 0")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be in [0, 1)")
    if args.num_epochs <= 0:
        raise ValueError("num_epochs must be > 0")
    if args.max_train_steps is not None and args.max_train_steps <= 0:
        raise ValueError("max_train_steps must be > 0")
    if args.max_grad_norm < 0:
        raise ValueError("max_grad_norm must be >= 0")
    if args.logging_steps <= 0:
        raise ValueError("logging_steps must be > 0")
    if args.save_every_epochs <= 0:
        raise ValueError("save_every_epochs must be > 0")
    if args.num_workers < 0:
        raise ValueError("num_workers must be >= 0")

def configure_torch_backends() -> None:
    if not torch.cuda.is_available():
        return
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    if hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(True)

def resolve_torch_dtype(mixed_precision: str) -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32

def resolve_accelerate_mixed_precision(mixed_precision: str) -> str:
    if not torch.cuda.is_available():
        return "no"
    return mixed_precision

def resolve_attn_implementation(requested: str, dtype: torch.dtype) -> str:
    if requested != "auto":
        return requested
    if not torch.cuda.is_available():
        return "eager"
    if dtype in {torch.float16, torch.bfloat16}:
        try:
            import flash_attn
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                return "flash_attention_2"
        except Exception:
            pass
    return "sdpa"

def resolve_warmup_steps(args: argparse.Namespace, num_training_steps: int) -> int:
    if args.warmup_steps > 0:
        return args.warmup_steps
    if args.warmup_ratio > 0:
        return math.ceil(num_training_steps * args.warmup_ratio)
    return 0

def parse_channelwise_loss_weight(spec: str, n_heads: int) -> List[float]:
    values = [float(item.strip()) for item in str(spec).split(",") if item.strip()]
    if len(values) == n_heads:
        resolved = values
    elif len(values) == 2 and n_heads > 1:
        text_weight, total_audio_weight = values
        per_audio_weight = total_audio_weight / float(n_heads - 1)
        resolved = [text_weight] + [per_audio_weight] * (n_heads - 1)
    else:
        raise ValueError(f"channelwise_loss_weight accepts {n_heads} or 2 values")
    if sum(resolved) <= 0:
        raise ValueError("channelwise_loss_weight must sum to positive")
    return resolved

def build_optimizer(model, args: argparse.Namespace) -> AdamW:
    return AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
    )

def unwrap_training_model(model):
    unwrapped = model
    while hasattr(unwrapped, "module"):
        unwrapped = unwrapped.module
    return unwrapped

def compute_supervised_loss(
    model,
    *,
    input_ids: torch.LongTensor,
    attention_mask: torch.BoolTensor,
    labels: torch.LongTensor,
    channelwise_loss_weight: List[float],
) -> torch.Tensor:
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    global_hidden_states = outputs.global_hidden_states
    if global_hidden_states is None:
        raise RuntimeError("Model forward did not return global_hidden_states")
    base_model = unwrap_training_model(model)
    batch_size, seq_len, hidden_size = global_hidden_states.shape
    n_vq = int(base_model.config.n_vq)
    flat_hidden = global_hidden_states.reshape(batch_size * seq_len, hidden_size)
    local_dtype = base_model.local_transformer.ln_f.weight.dtype
    flat_hidden = flat_hidden.to(dtype=local_dtype)
    flat_labels = labels.reshape(batch_size * seq_len, n_vq + 1)
    local_inputs = torch.zeros(
        (batch_size * seq_len, n_vq + 1, hidden_size),
        dtype=local_dtype,
        device=flat_hidden.device,
    )
    local_inputs[:, 0, :] = flat_hidden
    text_targets = flat_labels[:, 0]
    safe_text_targets = text_targets.masked_fill(text_targets.lt(0), int(base_model.config.pad_token_id))
    local_inputs[:, 1, :] = base_model.transformer.wte(safe_text_targets)
    audio_targets = flat_labels[:, 1:]
    for channel_index in range(n_vq - 1):
        teacher_ids = audio_targets[:, channel_index]
        valid_mask = (teacher_ids >= 0) & (teacher_ids < base_model.audio_embeddings[channel_index].num_embeddings)
        safe_ids = teacher_ids.masked_fill(~valid_mask, 0)
        channel_embeds = base_model.audio_embeddings[channel_index](safe_ids)
        channel_embeds = channel_embeds * valid_mask.unsqueeze(-1)
        local_inputs[:, channel_index + 2, :] = channel_embeds.to(dtype=local_dtype)
    local_attention_mask = torch.ones(
        (batch_size * seq_len, n_vq + 1),
        dtype=torch.bool,
        device=flat_hidden.device,
    )
    local_outputs = base_model.local_transformer(
        input_ids=None,
        attention_mask=local_attention_mask,
        position_ids=None,
        inputs_embeds=local_inputs,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        cu_seqlens=None,
        num_sequences=None,
    )
    local_hidden_states = local_outputs.last_hidden_state
    total_loss = torch.zeros((), device=flat_hidden.device, dtype=torch.float32)
    total_weight = 0.0
    text_logits = base_model.text_lm_head(local_hidden_states[:, 0, :])
    if (text_targets != -100).any():
        text_loss = F.cross_entropy(text_logits.float(), text_targets, ignore_index=-100)
        total_loss = total_loss + float(channelwise_loss_weight[0]) * text_loss.float()
        total_weight += float(channelwise_loss_weight[0])
    for channel_index in range(n_vq):
        channel_targets = audio_targets[:, channel_index]
        if not (channel_targets != -100).any():
            continue
        channel_logits = base_model.audio_lm_heads[channel_index](local_hidden_states[:, channel_index + 1, :])
        channel_loss = F.cross_entropy(channel_logits.float(), channel_targets, ignore_index=-100)
        total_loss = total_loss + float(channelwise_loss_weight[channel_index + 1]) * channel_loss.float()
        total_weight += float(channelwise_loss_weight[channel_index + 1])
    if total_weight <= 0:
        raise RuntimeError("All labels are ignored")
    return total_loss / total_weight

def resolve_asset(model_path: str, filename: str) -> Optional[Path]:
    model_path_obj = Path(model_path)
    if model_path_obj.is_dir():
        candidate = model_path_obj / filename
        return candidate if candidate.exists() else None
    try:
        resolved = cached_file(
            model_path,
            filename,
            _raise_exceptions_for_missing_entries=False,
        )
    except OSError:
        return None
    if resolved is None:
        return None
    return Path(resolved)

def save_checkpoint(
    *,
    accelerator: Accelerator,
    model,
    tokenizer,
    model_path: str,
    codec_path: str,
    output_dir: Path,
    train_args: Dict[str, Any],
    global_step: int,
    epoch: int,
) -> None:
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    unwrapped_model = unwrap_training_model(model)
    unwrapped_model.config.audio_tokenizer_pretrained_name_or_path = str(Path(codec_path).expanduser().resolve())
    unwrapped_model.config.save_pretrained(output_dir)
    state_dict = {
        key: value.detach().cpu()
        for key, value in unwrapped_model.state_dict().items()
    }
    torch.save(state_dict, output_dir / "pytorch_model.bin")
    tokenizer.save_pretrained(output_dir)
    for filename in MODEL_SUPPORT_FILES:
        src = resolve_asset(model_path, filename)
        if src is not None and src.exists():
            shutil.copy2(src, output_dir / filename)
    metadata = dict(train_args)
    metadata["saved_global_step"] = int(global_step)
    metadata["saved_epoch"] = int(epoch)
    metadata["saved_at"] = format_timestamp()
    metadata["checkpoint_dir"] = str(output_dir)
    with open(output_dir / "finetune_config.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

def main() -> None:
    args = parse_args()
    validate_args(args)
    configure_torch_backends()
    set_seed(args.seed)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=resolve_accelerate_mixed_precision(args.mixed_precision),
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[ddp_kwargs],
    )
    if accelerator.device.type != "cuda":
        raise EnvironmentError("MOSS-TTS-Nano finetuning requires CUDA")
    model_dtype = resolve_torch_dtype(args.mixed_precision)
    attn_implementation = resolve_attn_implementation(args.attn_implementation, model_dtype)
    records_paths, records = load_jsonl_spec(args.train_jsonl)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=model_dtype,
    )
    if hasattr(model, "_set_attention_implementation"):
        model._set_attention_implementation(attn_implementation)
    dataset = MossTTSNanoSFTDataset(
        records,
        tokenizer=tokenizer,
        model_config=model.config,
        max_length=args.max_length,
    )
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=dataset.collate_fn,
    )
    optimizer = build_optimizer(model, args)
    global_batch_size = (
        args.per_device_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    )
    micro_batches_per_epoch = math.ceil(len(dataset) / (args.per_device_batch_size * accelerator.num_processes))
    optimizer_steps_per_epoch = math.ceil(micro_batches_per_epoch / args.gradient_accumulation_steps)
    max_train_steps = args.max_train_steps or (args.num_epochs * optimizer_steps_per_epoch)
    warmup_steps = resolve_warmup_steps(args, max_train_steps)
    channelwise_loss_weight = parse_channelwise_loss_weight(
        args.channelwise_loss_weight,
        int(model.config.n_vq) + 1,
    )
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps,
    )
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )
    output_root = Path(args.output_dir)
    if accelerator.is_main_process:
        output_root.mkdir(parents=True, exist_ok=True)
    train_args_to_save = vars(args).copy()
    train_args_to_save["resolved_warmup_steps"] = warmup_steps
    train_args_to_save["resolved_channelwise_loss_weight"] = channelwise_loss_weight
    train_args_to_save["global_batch_size"] = global_batch_size
    train_args_to_save["records_paths"] = [str(path.resolve()) for path in records_paths]
    train_args_to_save["attn_implementation"] = attn_implementation
    accelerator.print(
        f"[{format_timestamp()}] [sft] loaded_records={len(dataset)} "
        f"device={accelerator.device} "
        f"num_processes={accelerator.num_processes} "
        f"global_batch_size={global_batch_size} "
        f"micro_batches_per_epoch={micro_batches_per_epoch} "
        f"optimizer_steps_per_epoch={optimizer_steps_per_epoch} "
        f"max_train_steps={max_train_steps} "
        f"warmup_steps={warmup_steps} "
        f"attn={attn_implementation} "
        f"model_dtype={model_dtype}"
    )
    global_step = 0
    completed_epochs = 0
    last_log_time = time.perf_counter()
    last_logged_step = 0
    for epoch in range(args.num_epochs):
        model.train()
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                loss = compute_supervised_loss(
                    model,
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    channelwise_loss_weight=channelwise_loss_weight,
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients and args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                if accelerator.sync_gradients:
                    optimizer.step()
                    if not getattr(optimizer, "step_was_skipped", False):
                        lr_scheduler.step()
                    optimizer.zero_grad()
            if accelerator.sync_gradients:
                global_step += 1
                if global_step % args.logging_steps == 0:
                    now = time.perf_counter()
                    steps_since_last_log = max(global_step - last_logged_step, 1)
                    elapsed = max(now - last_log_time, 1e-12)
                    last_log_time = now
                    last_logged_step = global_step
                    step_time = elapsed / steps_since_last_log
                    steps_per_sec = steps_since_last_log / elapsed
                    samples_per_sec = (global_batch_size * steps_since_last_log) / elapsed
                    eta_seconds = max(max_train_steps - global_step, 0) / steps_per_sec
                    logged_loss = accelerator.gather(loss.detach().float().reshape(1)).mean().item()
                    lr_val = lr_scheduler.get_last_lr()[0]
                    accelerator.print(
                        f"[{format_timestamp()}] "
                        f"epoch={epoch} step={global_step}/{max_train_steps} "
                        f"loss={logged_loss:.4f} "
                        f"lr={lr_val:.2e} "
                        f"step_time={step_time:.2f}s "
                        f"steps_per_sec={steps_per_sec:.3f} "
                        f"samples_per_sec={samples_per_sec:.2f} "
                        f"eta={format_duration(eta_seconds)}"
                    )
                if global_step >= max_train_steps:
                    break
        if (epoch + 1) % args.save_every_epochs == 0 or global_step >= max_train_steps:
            save_checkpoint(
                accelerator=accelerator,
                model=model,
                tokenizer=tokenizer,
                model_path=args.model_path,
                codec_path=args.codec_path,
                output_dir=output_root / f"checkpoint-epoch-{epoch + 1}",
                train_args=train_args_to_save,
                global_step=global_step,
                epoch=epoch + 1,
            )
        completed_epochs = epoch + 1
        if global_step >= max_train_steps:
            break
    save_checkpoint(
        accelerator=accelerator,
        model=model,
        tokenizer=tokenizer,
        model_path=args.model_path,
        codec_path=args.codec_path,
        output_dir=output_root / "checkpoint-last",
        train_args=train_args_to_save,
        global_step=global_step,
        epoch=completed_epochs,
    )
    accelerator.print(
        f"[{format_timestamp()}] [sft] finished "
        f"global_step={global_step} saved_epochs={completed_epochs} output_dir={output_root}"
    )

if __name__ == "__main__":
    main()
