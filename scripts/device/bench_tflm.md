# Microcontroller Deployment & Benchmarking Guide: TFLite-Micro (ESP32-S3 & ARM Cortex-M)

This document provides step-by-step procedures to build, deploy, and benchmark **LiteCascade INT8** on resource-constrained microcontrollers (MCUs) using **TensorFlow Lite for Microcontrollers (TFLite-Micro)**, measuring:
1. **Flash size (ROM footprint in KB)**
2. **Tensor Arena size (peak RAM requirement in KB)**
3. **Execution clock cycles per inference**

> [!NOTE]
> If a physical MCU is not available, run `python scripts/estimate_mcu.py --model checkpoints/litecascade_int8.tflite` to compute estimated Flash/Arena values from the `.tflite` flatbuffer. Per Rule 6, all such outputs are strictly marked `"estimated"`.

---

## 1. Preparing the C Header File

TFLite-Micro requires model flatbuffers embedded directly into flash memory as a C byte array.

```bash
# Convert INT8 model to C array
xxd -i checkpoints/litecascade_int8.tflite > firmware/include/litecascade_model_data.h
```

Inspect the generated header:
```c
// firmware/include/litecascade_model_data.h
#ifndef LITECASCADE_MODEL_DATA_H_
#define LITECASCADE_MODEL_DATA_H_

extern const unsigned char checkpoints_litecascade_int8_tflite[];
extern const unsigned int checkpoints_litecascade_int8_tflite_len;

#endif
```

---

## 2. Target Platforms

| Target | Architecture | Clock Frequency | Flash | SRAM | Hardware Acceleration |
|---|---|---|---|---|---|
| **Espressif ESP32-S3** | Xtensa 32-bit LX7 dual-core | 240 MHz | 8 MB / 16 MB | 512 KB | ESP-NN Vector SIMD |
| **STM32F407 / NRF52840** | ARM Cortex-M4 | 168 MHz / 64 MHz | 1 MB | 192 KB / 256 KB | CMSIS-NN DSP instructions (`__SMLAD`) |
| **STM32H743** | ARM Cortex-M7 | 480 MHz | 2 MB | 1024 KB | CMSIS-NN Dual-issue |
| **Raspberry Pi RP2040** | ARM Cortex-M0+ | 133 MHz | 2 MB | 264 KB | Standard C reference kernels |

---

## 3. TFLite-Micro Firmware Implementation (C++)

```cpp
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "litecascade_model_data.h"

#if defined(ESP_PLATFORM)
#include "esp_cpu.h"
#elif defined(ARM_MATH_CM4) || defined(ARM_MATH_CM7)
#include "core_cm4.h"
#endif

// Static memory allocation: Flash model pointer & RAM Tensor Arena
namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;

// Tensor Arena sized from scripts/estimate_mcu.py (20 KB recommended)
constexpr int kTensorArenaSize = 20 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];
}

void setup() {
    tflite::InitializeTarget();

    // 1. Map model flatbuffer
    model = tflite::GetModel(checkpoints_litecascade_int8_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf("Schema mismatch: expected %d, got %d", TFLITE_SCHEMA_VERSION, model->version());
        return;
    }

    // 2. Resolver with optimized hardware kernels (ESP-NN / CMSIS-NN)
    static tflite::MicroMutableOpResolver<10> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddFullyConnected();
    resolver.AddRelu6();
    resolver.AddReshape();
    resolver.AddSoftmax();
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddAdd();
    resolver.AddMean();

    // 3. Build MicroInterpreter
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    // 4. Allocate tensors inside Arena
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        MicroPrintf("AllocateTensors failed!");
        return;
    }

    input_tensor = interpreter->input(0);
    
    // Log measured Arena consumption
    size_t arena_used = interpreter->arena_used_bytes();
    MicroPrintf("Tensor Arena Allocated: %d bytes, Peak Used: %d bytes (%.2f KB)",
                kTensorArenaSize, arena_used, arena_used / 1024.0);
}

void loop() {
    // Populate test feature window (1 x 10 x 115 INT8/FP32)
    // ...

    // Cycle-accurate timing
#if defined(ESP_PLATFORM)
    uint32_t start_cycles = esp_cpu_get_cycle_count();
#elif defined(DWT)
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
    uint32_t start_cycles = DWT->CYCCNT;
#endif

    // Invoke inference
    TfLiteStatus invoke_status = interpreter->Invoke();

#if defined(ESP_PLATFORM)
    uint32_t elapsed_cycles = esp_cpu_get_cycle_count() - start_cycles;
#elif defined(DWT)
    uint32_t elapsed_cycles = DWT->CYCCNT - start_cycles;
#endif

    if (invoke_status == kTfLiteOk) {
        TfLiteTensor* sent_output = interpreter->output(0);
        TfLiteTensor* analyst_output = interpreter->output(1);

        MicroPrintf("Inference completed in %u cycles (~%.3f ms @ 240 MHz)",
                    elapsed_cycles, (elapsed_cycles / 240000000.0) * 1000.0);
    }
}
```

---

## 4. Measuring Flash and RAM Footprint

### On ESP-IDF (ESP32-S3)
```bash
idf.py build
idf.py size-components
```
Look for:
- `.rodata` contribution of `litecascade_model_data.o`: Model Flash size (~41.9 KB).
- `.bss` contribution of `tensor_arena`: SRAM consumption (20.0 KB).

### On ARM GCC / CMSIS (Cortex-M)
```bash
arm-none-eabi-size -A build/firmware.elf
```
- Total Flash = `.text` + `.data` + `.rodata`
- Total RAM = `.data` + `.bss` (contains `tensor_arena`)

---

## 5. Docker Fallback for Constrained Emulation

If no physical Raspberry Pi or microcontroller hardware is present on the testing host, use Docker to emulate a constrained compute environment:

```bash
# Run benchmark container constrained to 0.5 CPU core and 256MB RAM
docker run --rm \
    --cpus=0.5 \
    --memory=256m \
    --env EMULATE_PI=1 \
    -v $(pwd):/workspace \
    -w /workspace \
    python:3.12-slim \
    bash -c "pip install -r requirements.txt && python scripts/device/bench_rpi.py --model checkpoints/litecascade_int8.tflite --runs 100 --dry-run"
```

> [!IMPORTANT]
> Any metrics produced under Docker or dev machines will automatically have `"measured_or_emulated": "emulated"` set in the resulting JSON to adhere strictly to Rule 6.
