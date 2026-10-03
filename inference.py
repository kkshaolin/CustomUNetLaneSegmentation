"""Run lane segmentation inference and save binary masks to disk.

Example:
    python inference.py --images-dir ./data/images/val \
        --checkpoint checkpoints/unet_lane/best.pt --run-name val_exp

สคริปต์สำหรับการนำโมเดลไปใช้งานจริง (Inference / Prediction)
- โหลดโมเดล UNet และค่าน้ำหนักที่ฝึกสอนแล้วจากไฟล์ Checkpoint (.pt)
- ประมวลผลภาพอินพุตและทำนายผลการแบ่งส่วนเส้นเลนถนน
- บันทึกผลลัพธ์ออกมาเป็นไฟล์ภาพ Binary Mask ขาว-ดำ (.png) ขนาดเท่ากับภาพต้นฉบับ
"""
# ==============================================================================
# ไลบรารีมาตรฐานและไลบรารีภายนอกที่จำเป็นสำหรับการทำนายผล
# ==============================================================================
import argparse  # สำหรับจัดการรับ Argument จาก Command-Line
from pathlib import Path  # สำหรับจัดการ File Path อย่างมีประสิทธิภาพ

import cv2  # OpenCV สำหรับการอ่าน บันทึกภาพ และปรับขนาดภาพ
import numpy as np  # NumPy สำหรับการจัดการอาร์เรย์ตัวเลข
import torch  # PyTorch สำหรับการรันโมเดล Deep Learning

# นำเข้าฟังก์ชันค้นหารูปภาพและสถาปัตยกรรมโมเดล UNet
from dataset import list_images
from model import UNet


def load_model(checkpoint_path, device):
    """โหลดสถาปัตยกรรม UNet และค่าน้ำหนักที่ผ่านการฝึกสอนแล้วจาก Checkpoint

    Args:
        checkpoint_path (str หรือ Path): path ของไฟล์ checkpoint (.pt)
        device (torch.device): อุปกรณ์ประมวลผล (CUDA หรือ CPU)

    Returns:
        torch.nn.Module: โมเดล UNet ที่โหลดน้ำหนักพร้อมใช้งานในโหมด Evaluation
    """
    # สร้างโครงสร้างโมเดลเปล่าขึ้นมาใหม่ (3 Input Channels, 1 Output Class)
    model = UNet(in_channels=3, num_classes=1)
    # โหลดไฟล์ Checkpoint จาก Disk พร้อมระบุ map_location และเปิด weights_only=True เพื่อความปลอดภัย
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    # ตรวจสอบโครงสร้าง Checkpoint (หากเก็บเป็น Dictionary ให้ดึงคีย์ "model_state_dict")
    state_dict = state["model_state_dict"] if "model_state_dict" in state else state
    # นำค่าน้ำหนักที่บันทึกไว้บรรจุเข้าสู่โมเดล
    model.load_state_dict(state_dict)
    # ส่งโมเดลไปยังอุปกรณ์ที่ประมวลผล (GPU/CPU)
    model.to(device)
    # ปรับสถานะโมเดลเป็นโหมดประเมินผล (Evaluation Mode) เพื่อปิดการปรับค่าใน BatchNorm
    model.eval()
    return model


def preprocess(image_rgb, img_size):
    """เตรียมข้อมูลภาพก่อนป้อนเข้าสู่โมเดล (Preprocessing Pipeline)

    ขั้นตอน:
    1. ปรับขนาดภาพให้เป็นขนาดที่โมเดลรองรับ (img_size, img_size) ด้วย INTER_AREA
    2. ทำ Normalization ปรับช่วงค่าสีพิกเซลจาก [0, 255] เป็น [0.0, 1.0]
    3. สลับแกนมิติภาพจาก (H, W, C) ให้เป็น (C, H, W)
    4. เพิ่มมิติ Batch Dimension เข้าไปด้านหน้า -> (1, C, H, W)

    Args:
        image_rgb (np.ndarray): ภาพอินพุตระบบสี RGB ขนาด (H, W, 3)
        img_size (int): ขนาดภาพเป้าหมาย (เช่น 48)

    Returns:
        torch.Tensor: Tensor ภาพขนาด (1, 3, img_size, img_size)
    """
    # ปรับขนาดภาพให้เป็น (img_size, img_size) โดยใช้ cv2.INTER_AREA เพื่อคุณภาพที่ดีที่สุดเมื่อย่อขนาด
    resized = cv2.resize(image_rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
    # แปลงเป็น float32, หาร 255.0, สลับแกนด้วย .permute, และเพิ่มมิติ Batch ด้วย .unsqueeze(0)
    tensor = torch.from_numpy(resized.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return tensor


@torch.no_grad()
def predict_mask(model, image_path, img_size, device, threshold=0.5):
    """ทำนายภาพ Mask เส้นเลนสำหรับภาพเดี่ยว 1 ภาพ

    ขั้นตอน:
    1. โหลดภาพต้นฉบับและแปลงสีเป็น RGB
    2. นำภาพเข้าฟังก์ชัน preprocess แล้วส่งเข้าโมเดลเพื่อหาค่า Logits
    3. แปลงค่า Logits เป็นความน่าจะเป็นด้วย Sigmoid
    4. ตัดเกณฑ์ Threshold เพื่อแยกพิกเซลเส้นเลน (255 สีขาว) และพื้นหลัง (0 สีดำ)
    5. ปรับขนาด Mask กลับไปเท่ากับขนาดเดิมของภาพต้นฉบับ

    Args:
        model (nn.Module): โมเดล UNet ที่โหลดค่าน้ำหนักแล้ว
        image_path (str หรือ Path): path ของไฟล์ภาพที่ต้องการทำนาย
        img_size (int): ขนาดภาพที่โมเดลรองรับ (เช่น 48)
        device (torch.device): อุปกรณ์ประมวลผล (CUDA หรือ CPU)
        threshold (float): เกณฑ์ความน่าจะเป็นในการตัดสินว่าเป็นเส้นเลน (ค่าเริ่มต้น 0.5)

    Returns:
        np.ndarray: หน้ากาก Binary Mask ขนาดเท่าภาพต้นฉบับ ชนิด uint8 (ค่า 0 หรือ 255)
    """
    # อ่านไฟล์ภาพต้นฉบับด้วย OpenCV (ค่าเริ่มต้นเป็น BGR)
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    # แปลงระบบสีจาก BGR เป็น RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]

    # เตรียมข้อมูลภาพและส่งไปยังอุปกรณ์ประมวลผล
    tensor = preprocess(image_rgb, img_size).to(device)
    # Forward Pass: ส่งเข้าโมเดลเพื่อคำนวณ Logits
    logits = model(tensor)
    # นำ Logits ผ่าน Sigmoid เพื่อแปลงเป็นความน่าจะเป็น แล้วดึงออกมาเป็น NumPy array บน CPU
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

    # ตัดเกณฑ์ความน่าจะเป็น: ถ้า prob > threshold ให้มีค่าเป็น 1 แล้วคูณ 255 (สีขาว), นอกนั้นเป็น 0 (สีดำ)
    mask_small = (prob > threshold).astype(np.uint8) * 255
    # ขยายหน้ากากขนาดเล็ก (48x48) กลับไปเป็นขนาดภาพดั้งเดิม (w, h) โดยใช้ INTER_NEAREST เพื่อคงขอบคมชัด
    mask_full = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask_full


def main():
    """ฟังก์ชันหลักสำหรับการรัน Inference ทั้งโฟลเดอร์ภาพและบันทึกผลลง Disk"""
    # กำหนด Arguments ที่รับจาก Command-Line
    parser = argparse.ArgumentParser(description="Lane segmentation inference -> binary masks.")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--img-size", type=int, default=48)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--run-name", default="exp")
    parser.add_argument("--output-root", default="run")
    args = parser.parse_args()

    # เลือกอุปกรณ์ประมวลผล (CUDA GPU หากมี หรือ CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # โหลดโมเดลจากไฟล์ Checkpoint
    model = load_model(args.checkpoint, device)

    # สร้างโฟลเดอร์ปลายทางสำหรับบันทึกภาพผลลัพธ์ (เช่น run/exp/masks)
    out_dir = Path(args.output_root) / args.run_name / "masks"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ค้นหารายการไฟล์รูปภาพทั้งหมดในโฟลเดอร์ที่ระบุ
    images = list_images(args.images_dir)
    if not images:
        print(f"No images found under {args.images_dir}")
        return

    # วนลูปประมวลผลรูปภาพทีละภาพ
    for image_path in images:
        # ทำนายหน้ากากเส้นเลน
        mask = predict_mask(model, image_path, args.img_size, device, args.threshold)
        # กำหนดชื่อไฟล์ผลลัพธ์เป็นชื่อเดียวกับภาพต้นฉบับแต่เปลี่ยนนามสกุลเป็น .png
        out_path = out_dir / f"{image_path.stem}.png"
        # บันทึกไฟล์ภาพ Mask ลง Disk
        cv2.imwrite(str(out_path), mask)

    # แสดงผลสรุปจำนวนภาพที่บันทึกสำเร็จ
    print(f"Saved {len(images)} masks to {out_dir}")


if __name__ == "__main__":
    main()
