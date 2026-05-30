import logging
import os
import sys

import torch

logger = logging.getLogger("sd-forge-int8-compat")


_SCRIPT_DIR = os.path.dirname(__file__)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)


_FLOAT_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}


def _is_floating_tensor(value) -> bool:
    return isinstance(value, torch.Tensor) and value.is_floating_point()


def _parameter_dtype(module) -> torch.dtype | None:
    weight = getattr(module, "weight", None)
    if _is_floating_tensor(weight) and weight.dtype in _FLOAT_DTYPES:
        return weight.dtype

    try:
        from backend import operations

        dtype = getattr(operations, "current_dtype", None)
    except Exception:
        dtype = None

    if dtype in _FLOAT_DTYPES:
        return dtype

    dtype = getattr(module, "compute_dtype", None)
    if dtype in _FLOAT_DTYPES:
        return dtype

    return None


def _cast_parameter_dtype(module, name: str, dtype: torch.dtype | None) -> None:
    if dtype is None:
        return

    param = getattr(module, name, None)
    if not _is_floating_tensor(param) or param.dtype == dtype:
        return

    converted = param.to(dtype=dtype)
    setattr(module, name, torch.nn.Parameter(converted, requires_grad=getattr(param, "requires_grad", False)))


def _cast_for_linear(value, x: torch.Tensor):
    if value is None:
        return None

    if _is_floating_tensor(value) and (value.dtype != x.dtype or value.device != x.device):
        return value.to(device=x.device, dtype=x.dtype, non_blocking=True)

    if isinstance(value, torch.Tensor) and value.device != x.device:
        return value.to(device=x.device, non_blocking=True)

    return value


def _install_linear_patch() -> None:
    from backend import operations

    linear_cls = operations.ForgeOperationsInt8.Linear
    if getattr(linear_cls, "_sd_forge_int8_compat_installed", False):
        return

    original_load = linear_cls._load_from_state_dict
    original_forward = linear_cls.forward
    get_weight_and_bias = operations.get_weight_and_bias
    linear = torch.nn.functional.linear
    tensor_type = torch.Tensor

    def patched_load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        target_dtype = _parameter_dtype(self)
        original_load(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

        if getattr(self, "_is_quantized", False):
            return

        _cast_parameter_dtype(self, "weight", target_dtype)
        _cast_parameter_dtype(self, "bias", target_dtype)

    def patched_forward(self, x: torch.Tensor):
        if getattr(self, "_is_quantized", False):
            return original_forward(self, x)

        need_cast = (
            getattr(self, "parameters_manual_cast", False)
            or len(getattr(self, "weight_function", [])) > 0
            or len(getattr(self, "bias_function", [])) > 0
        )
        if need_cast:
            return original_forward(self, x)

        weight, bias = get_weight_and_bias(self)

        if isinstance(weight, tensor_type):
            if weight.is_floating_point() and (weight.dtype != x.dtype or weight.device != x.device):
                weight = weight.to(device=x.device, dtype=x.dtype, non_blocking=True)
            elif weight.device != x.device:
                weight = weight.to(device=x.device, non_blocking=True)

        if isinstance(bias, tensor_type):
            if bias.is_floating_point() and (bias.dtype != x.dtype or bias.device != x.device):
                bias = bias.to(device=x.device, dtype=x.dtype, non_blocking=True)
            elif bias.device != x.device:
                bias = bias.to(device=x.device, non_blocking=True)

        return linear(x, weight, bias)

    linear_cls._load_from_state_dict = patched_load_from_state_dict
    linear_cls.forward = patched_forward
    linear_cls._sd_forge_int8_compat_original_load = original_load
    linear_cls._sd_forge_int8_compat_original_forward = original_forward
    linear_cls._sd_forge_int8_compat_installed = True


def _install_online_lora_patch() -> None:
    from backend import utils
    from backend.operations_int8 import INT8ModelPatcher

    if getattr(INT8ModelPatcher, "_sd_forge_int8_compat_installed", False):
        return

    def patched_process_online_loras(self):
        if not hasattr(self.model, "online_lora_layers"):
            utils.set_attr_raw(self.model, "online_lora_layers", set())

        for layer in self.model.online_lora_layers:
            if hasattr(layer, "forge_online_loras"):
                del layer.forge_online_loras
            if hasattr(layer, "lora_patches"):
                layer.lora_patches = []
        self.model.online_lora_layers.clear()

        for _name, module in self.model.named_modules():
            if hasattr(module, "lora_patches"):
                module.lora_patches = []

        for key, current_patches in self.online_patches.items():
            module_path = key.rsplit(".", 1)[0]

            try:
                module = utils.get_attr(self.model, module_path)
            except Exception:
                module = None

            if getattr(module, "_is_quantized", False):
                self.patch_weight_to_device(key, online=True)
                self.model.online_lora_layers.add(module)
                continue

            try:
                parent_layer, child_key, weight = utils.get_attr_with_parent(self.model, key)
                assert isinstance(weight, torch.nn.Parameter)
            except Exception:
                logger.error("Invalid LoRA Key for INT8 online patch: %s", key)
                continue

            if not hasattr(parent_layer, "forge_online_loras"):
                parent_layer.forge_online_loras = {}

            parent_layer.forge_online_loras.setdefault(child_key, []).extend(current_patches)
            self.model.online_lora_layers.add(parent_layer)

    INT8ModelPatcher._process_online_loras = patched_process_online_loras
    INT8ModelPatcher._sd_forge_int8_compat_installed = True


def install_patch() -> None:
    try:
        _install_linear_patch()
        _install_online_lora_patch()
        logger.info("Installed Forge INT8 dtype/LoRA compatibility patch")
    except Exception:
        logger.exception("Failed to install Forge INT8 compatibility patch")


install_patch()
