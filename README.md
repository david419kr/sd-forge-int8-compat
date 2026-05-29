# sd-forge-int8-compat

[Korean documentation](README.kr.md)

Compatibility patch extension for Forge Neo INT8 inference. It mitigates dtype mismatch failures when using Anima models with `Diffusion in Low Bits` set to `int8` or `int8 (fp16 LoRA)`.

## Problem

During Anima INT8 loading, some Linear layers can be excluded from INT8 quantization. Those fallback layers may keep the checkpoint's original `fp16` weights while the model input arrives as `bf16`, which can fail with:

```text
RuntimeError: expected mat1 and mat2 to have the same dtype, but got: struct c10::BFloat16 != struct c10::Half
```

## What It Does

- Casts non-quantized fallback Linear `weight` / `bias` tensors to the active load dtype.
- Adds a forward-time fallback for remaining Linear dtype mismatches.
- Adjusts online LoRA handling so LoRAs can apply to both INT8-quantized layers and INT8-excluded fallback layers.

## Usage

1. Place this folder at `extensions/sd-forge-int8-compat`.
2. Restart Forge Neo.
3. Select `int8` or `int8 (fp16 LoRA)` from `Diffusion in Low Bits`.
4. Test generation with an Anima checkpoint and your existing Anima LoRAs.

For LoRA workflows, start with `int8 (fp16 LoRA)`.

## Scope

This extension does not modify Forge Neo core files. It applies runtime monkey patches to Forge Neo's INT8 Linear and INT8 LoRA patcher paths.

If Forge Neo changes the internal `ForgeOperationsInt8` or `INT8ModelPatcher` structure, this extension should be reviewed again.

This does not add direct support for loading ComfyUI `INT8-Fast` pre-quantized `int8_rowwise` safetensors in Forge Neo.
