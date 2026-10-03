"""Check and measure the memory footprint of the Custom UNet lane segmentation model.

สคริปต์แยกสำหรับตรวจสอบและวัดขนาด Memory Footprint ของโมเดล UNet:
- ขนาดพารามิเตอร์โมเดล (Model Parameters)
- ขนาดบัฟเฟอร์ (BatchNorm Buffers)
- ขนาดไฟล์ Checkpoint บน Disk (.pt)
- ขนาดเทนเซอร์ภาพนำเข้า (Input Tensor)
- ขนาดหน่วยความจำ Activation รายสเตจระหว่าง Forward Pass
- ขนาดหน่วยความจำจริงที่จองบน GPU (CUDA Allocated / Peak / Reserved) หรือ CPU RAM (Tracemalloc)

Usage:
    python check_memory.py
    python check_memory.py --checkpoint checkpoints/unet_lane/best.pt
    python check_memory.py --height 48 --width 64 --batch-size 1
"""
import argparse
import os
from pathlib import Path
import tracemalloc

import torch
from model import UNet


def format_bytes(size_bytes: int) -> str:
    """แปลงจำนวนไบต์เป็นหน่วยที่อ่านง่าย (Bytes, KB, MB)"""
    if size_bytes < 1024:
        return f"{size_bytes} Bytes"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes / (1024 ** 2):.2f} MB ({size_bytes / 1e6:.2f} MB Dec)"


def measure_inference_memory(checkpoint_path=None, height=48, width=64, batch_size=1, device_name=None):
    """วัดผลหน่วยความจำโมเดลและ Forward Pass ทั้งทางทฤษฎีและการรันจริง"""
    if device_name is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    print("=" * 70)
    print("    CUSTOM UNET LANE SEGMENTATION - MEMORY FOOTPRINT PROFILER    ")
    print("=" * 70)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Input Shape: (Batch={batch_size}, Channels=3, Height={height}, Width={width})")
    print("-" * 70)

    # 1. สร้างโมเดล
    model = UNet(in_channels=3, num_classes=1)

    # 2. ตรวจสอบไฟล์ Checkpoint บน Disk (ถ้ามี)
    disk_file_size = None
    if checkpoint_path and Path(checkpoint_path).exists():
        disk_file_size = os.path.getsize(checkpoint_path)
        chk = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = chk["model_state_dict"] if "model_state_dict" in chk else chk
        model.load_state_dict(state_dict)
        epoch_info = f" (Epoch: {chk.get('epoch', '-')})" if isinstance(chk, dict) else ""
        print(f"[1] Checkpoint File on Disk: {checkpoint_path}{epoch_info}")
        print(f"    -> Size on Disk: {format_bytes(disk_file_size)}")
    else:
        print(f"[1] Checkpoint File on Disk: {'Not specified or not found' if not checkpoint_path else checkpoint_path}")
        print(f"    -> Size on Disk: -")

    model.to(device)
    model.eval()

    # 3. คำนวณพารามิเตอร์โมเดล (Model Parameters)
    total_params = sum(p.numel() for p in model.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())

    print(f"\n[2] Model Static Weights & Buffers:")
    print(f"    -> Total Trainable Parameters : {total_params:,}")
    print(f"    -> Weights Memory (Float32)   : {format_bytes(param_bytes)}")
    print(f"    -> BatchNorm Buffers Memory   : {format_bytes(buffer_bytes)}")
    print(f"    -> Static Model In-Memory     : {format_bytes(param_bytes + buffer_bytes)}")

    # 4. คำนวณ Input Tensor
    input_elements = batch_size * 3 * height * width
    input_bytes = input_elements * 4  # float32 = 4 bytes
    print(f"\n[3] Input Tensor:")
    print(f"    -> Shape                      : ({batch_size}, 3, {height}, {width})")
    print(f"    -> Elements                   : {input_elements:,}")
    print(f"    -> Memory Footprint           : {format_bytes(input_bytes)}")

    # 5. คำนวณ Activation Memory รายสเตจระหว่าง Forward Pass (ทางทฤษฎี)
    dummy_input = torch.randn(batch_size, 3, height, width, device=device)
    activations = []
    with torch.no_grad():
        x1 = model.in_conv(dummy_input)
        activations.append(("in_conv (Enc 1)", tuple(x1.shape), x1.numel() * 4))
        x2 = model.down1(x1)
        activations.append(("down1   (Enc 2)", tuple(x2.shape), x2.numel() * 4))
        x3 = model.down2(x2)
        activations.append(("down2   (Enc 3)", tuple(x3.shape), x3.numel() * 4))
        x4 = model.down3(x3)
        activations.append(("down3   (Enc 4)", tuple(x4.shape), x4.numel() * 4))
        x5 = model.down4(x4)
        activations.append(("down4 (Bottleneck)", tuple(x5.shape), x5.numel() * 4))
        u1 = model.up1(x5, x4)
        activations.append(("up1     (Dec 1)", tuple(u1.shape), u1.numel() * 4))
        u2 = model.up2(u1, x3)
        activations.append(("up2     (Dec 2)", tuple(u2.shape), u2.numel() * 4))
        u3 = model.up3(u2, x2)
        activations.append(("up3     (Dec 3)", tuple(u3.shape), u3.numel() * 4))
        u4 = model.up4(u3, x1)
        activations.append(("up4     (Dec 4)", tuple(u4.shape), u4.numel() * 4))
        out = model.out_conv(u4)
        activations.append(("out_conv (Output)", tuple(out.shape), out.numel() * 4))

    total_activation_bytes = sum(item[2] for item in activations)

    print(f"\n[4] Layer-by-Layer Activation Memory (Forward Pass):")
    print(f"    {'Layer / Stage':<22} | {'Output Shape':<18} | {'Memory':<12}")
    print("    " + "-" * 58)
    for name, shape, b_size in activations:
        print(f"    {name:<22} | {str(shape):<18} | {format_bytes(b_size):<12}")
    print("    " + "-" * 58)
    print(f"    Total Intermediate Activations: {format_bytes(total_activation_bytes)}")

    # 6. ผลรวมทางทฤษฎี (Theoretical Total Footprint)
    theoretical_total = param_bytes + buffer_bytes + input_bytes + total_activation_bytes
    print(f"\n[5] Theoretical Minimum In-Memory Footprint:")
    print(f"    -> Weights + Buffers + Input + Activations = {format_bytes(theoretical_total)}")

    # 7. การวัดผลจริงที่ระดับ Runtime (Hardware Allocation)
    print(f"\n[6] Runtime Measured Memory Allocation:")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # ทำการรัน Forward ซ้ำเพื่อจับ Peak VRAM
        with torch.no_grad():
            _ = model(dummy_input)
            torch.cuda.synchronize()

        alloc = torch.cuda.memory_allocated()
        peak = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.memory_reserved()

        print(f"    -> PyTorch CUDA Memory Allocated : {format_bytes(alloc)}")
        print(f"    -> PyTorch CUDA Peak Allocated   : {format_bytes(peak)}")
        print(f"    -> PyTorch CUDA Memory Reserved  : {format_bytes(reserved)}")
    else:
        # บน CPU ใช้ tracemalloc ตรวจจับการจัดสรร RAM
        tracemalloc.start()
        with torch.no_grad():
            _ = model(dummy_input)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"    -> Traced Current CPU RAM        : {format_bytes(current)}")
        print(f"    -> Traced Peak CPU RAM           : {format_bytes(peak)}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Measure memory footprint of Custom UNet during inference.")
    parser.add_argument("--checkpoint", default="checkpoints/unet_lane/best.pt", help="Path to checkpoint .pt file")
    parser.add_argument("--height", type=int, default=48, help="Input height (default: 48)")
    parser.add_argument("--width", type=int, default=64, help="Input width (default: 64)")
    parser.add_argument("--batch-size", type=int, default=1, help="Inference batch size (default: 1)")
    parser.add_argument("--device", default=None, help="Device to run on (cuda or cpu)")
    args = parser.parse_args()

    measure_inference_memory(
        checkpoint_path=args.checkpoint,
        height=args.height,
        width=args.width,
        batch_size=args.batch_size,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
