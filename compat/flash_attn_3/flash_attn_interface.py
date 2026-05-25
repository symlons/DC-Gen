"""Fail-fast flash-attn-3 shim for code paths this project does not use.

Transformer Engine imports these symbols at module import time when the
flash-attn-3 package metadata is installed. The flow DiT FP8 experiment uses
TE Linear modules only, not TE DotProductAttention. If an FA3 attention path is
ever reached, raise a clear dependency error instead of silently running a stub.
"""


def _missing_flash_attn_3(*args, **kwargs):
    raise ModuleNotFoundError(
        "flash_attn_3.flash_attn_interface is required for Transformer Engine "
        "FlashAttention-3 attention kernels, but only package metadata is "
        "visible in this environment. Install a complete flash-attn-3 build "
        "or avoid TE attention modules."
    )


flash_attn_func = _missing_flash_attn_3
flash_attn_varlen_func = _missing_flash_attn_3
flash_attn_with_kvcache = _missing_flash_attn_3
_flash_attn_forward = _missing_flash_attn_3
_flash_attn_backward = _missing_flash_attn_3
