# sd-forge-int8-compat

Forge Neo의 `Diffusion in Low Bits`에서 `int8` / `int8 (fp16 LoRA)` 사용 시 Anima 모델과 LoRA가 dtype mismatch로 중단되는 문제를 완화하는 호환성 패치 extension입니다.

## 대상 문제

Anima INT8 로딩 중 일부 Linear 레이어가 INT8 양자화 제외 목록에 걸리면 checkpoint 원래 dtype인 `fp16` weight가 남을 수 있습니다. 이 상태에서 입력이 `bf16`으로 들어오면 다음과 같은 오류가 발생합니다.

```text
RuntimeError: expected mat1 and mat2 to have the same dtype, but got: struct c10::BFloat16 != struct c10::Half
```

## 동작

- INT8에서 양자화되지 않은 fallback Linear의 `weight` / `bias` dtype을 로딩 dtype에 맞춥니다.
- forward 시 남아 있는 dtype mismatch를 한 번 더 보정합니다.
- `int8 (fp16 LoRA)` 모드에서 INT8 양자화 레이어와 INT8 제외 레이어 모두 LoRA가 적용되도록 online LoRA 처리를 보정합니다.

## 사용법

1. 이 폴더를 `extensions/sd-forge-int8-compat`에 둡니다.
2. Forge Neo를 재시작합니다.
3. 상단 `Diffusion in Low Bits`에서 `int8` 또는 `int8 (fp16 LoRA)`를 선택합니다.
4. Anima checkpoint와 기존 Anima LoRA로 생성 테스트를 진행합니다.

LoRA를 사용할 때는 먼저 `int8 (fp16 LoRA)`를 권장합니다.

## 범위와 한계

이 extension은 Forge Neo 본체 파일을 수정하지 않고 런타임 monkey patch로 동작합니다. Forge Neo의 내부 `ForgeOperationsInt8` / `INT8ModelPatcher` 구조가 바뀌면 다시 확인이 필요합니다.

ComfyUI `INT8-Fast`용으로 사전 양자화된 `int8_rowwise` safetensors를 Forge Neo에서 직접 로드하도록 만드는 패치는 아닙니다.
