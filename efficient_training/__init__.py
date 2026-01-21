"""
Efficient Training Module for TRM

This module provides memory-efficient training techniques:
- PEFT adapters (LoRA, DoRA, LoRA+, PiSSA)
- Quantization (QLoRA, NF4, LoftQ)
- Memory optimization (GaLore)
"""

from .adapters import (
    LoRALinear,
    DoRALinear,
    inject_lora_adapters,
    merge_lora_weights,
    pissa_init,
    apply_pissa_init,
    get_lora_params,
    get_lora_param_groups,
    count_lora_parameters,
)

from .quantization import (
    Linear4bit,
    QLoRALinear,
    quantize_model_nf4,
    prepare_for_qlora,
    loftq_init,
    estimate_memory_savings,
)

from .galore import (
    GaLoreProjector,
    GaLoreAdamW,
    estimate_galore_memory_savings,
)

from .memory_utils import (
    get_gpu_memory_info,
    clear_gpu_memory,
    memory_tracker,
    profile_model_memory,
    estimate_training_memory,
    print_memory_summary,
)

__all__ = [
    # Adapters
    "LoRALinear",
    "DoRALinear",
    "inject_lora_adapters",
    "merge_lora_weights",
    "pissa_init",
    "apply_pissa_init",
    "get_lora_params",
    "get_lora_param_groups",
    "count_lora_parameters",
    # Quantization
    "Linear4bit",
    "QLoRALinear",
    "quantize_model_nf4",
    "prepare_for_qlora",
    "loftq_init",
    "estimate_memory_savings",
    # GaLore
    "GaLoreProjector",
    "GaLoreAdamW",
    "estimate_galore_memory_savings",
    # Memory utils
    "get_gpu_memory_info",
    "clear_gpu_memory",
    "memory_tracker",
    "profile_model_memory",
    "estimate_training_memory",
    "print_memory_summary",
]

