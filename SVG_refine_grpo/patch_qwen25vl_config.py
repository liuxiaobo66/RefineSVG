"""
Monkey-patches for Qwen VL configs in transformers 4.57.0.

Fixes:
1. Qwen2_5_VLConfig.__getattribute__ delegates to text_config, shadowing top-level
   attributes (architectures, model_type, etc.) with None values from text_config.
2. rope_scaling is None in 4.57.0 (moved to rope_parameters), but SGLang/vLLM and
   the modeling code expect rope_scaling to contain mrope_section.  Affects BOTH
   Qwen2.5-VL and Qwen3-VL.

Usage: import this module before any model loading code.
"""


def _patch_rope_scaling(config_cls):
    """Patch a VL config class so rope_parameters is copied to rope_scaling."""
    _original_init = config_cls.__init__

    def patched_init(self, *args, **kwargs):
        _original_init(self, *args, **kwargs)
        text_cfg = getattr(self, "text_config", None)
        if text_cfg is not None:
            rope_params = getattr(text_cfg, "rope_parameters", None) or text_cfg.__dict__.get("rope_parameters")
            rope_scaling = getattr(text_cfg, "rope_scaling", None) or text_cfg.__dict__.get("rope_scaling")
            if rope_scaling is None and rope_params is not None and "mrope_section" in rope_params:
                text_cfg.rope_scaling = dict(rope_params)

    config_cls.__init__ = patched_init


def apply_patch():
    # --- Qwen2.5-VL ---
    try:
        from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig

        # Fix 1: __getattribute__ delegation bug (Qwen2.5-VL only)
        if "__getattribute__" in Qwen2_5_VLConfig.__dict__:
            _EXCLUDED_KEYS = {
                "dtype",
                "_attn_implementation_internal",
                "architectures",
                "model_type",
                "tie_word_embeddings",
                "pad_token_id",
                "transformers_version",
                "_name_or_path",
            }

            def patched_getattribute(self, key):
                if "text_config" in super(Qwen2_5_VLConfig, self).__getattribute__("__dict__") and key not in _EXCLUDED_KEYS:
                    text_config = super(Qwen2_5_VLConfig, self).__getattribute__("text_config")
                    if key in text_config.__dict__:
                        return getattr(text_config, key)
                return super(Qwen2_5_VLConfig, self).__getattribute__(key)

            Qwen2_5_VLConfig.__getattribute__ = patched_getattribute

        # Fix 2: rope_scaling compat
        _patch_rope_scaling(Qwen2_5_VLConfig)
    except ImportError:
        pass

    # --- Qwen3-VL ---
    try:
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

        _patch_rope_scaling(Qwen3VLConfig)
    except ImportError:
        pass


apply_patch()
