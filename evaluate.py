"""Evaluate predicted lane masks against YOLO-seg ground truth via pixel-wise IoU.

Metrics reported:
  - Detection rate: fraction of images classified "detected" (IoU > threshold, default 0.6)
  - Average IoU of detected results (average IoU over only the "detected" images)

Example:
    python evaluate.py --images-dir ./data/images/val --labels-dir ./data/labels/val \
        --pred-masks-dir run/val_exp/masks

สคริปต์สำหรับประเมินประสิทธิภาพการแบ่งส่วนเส้นเลนถนน (Lane Segmentation Evaluation)
- เปรียบเทียบภาพผลการทำนาย (Predicted Mask) กับความจริง (Ground Truth จาก YOLO-seg)
- คำนวณค่า Intersection over Union (IoU) ในระดับพิกเซล (Pixel-wise IoU)
- รายงานอัตราการตรวจจับสำเร็จ (Detection Rate) และค่า IoU เฉลี่ย
"""
# ==============================================================================
# ไลบรารีมาตรฐานและไลบรารีภายนอกที่จำเป็นสำหรับการประเมินผล
# ==============================================================================
import argparse  # สำหรับรับคำสั่งและพารามิเตอร์จาก Command-Line
import json  # สำหรับการบันทึกรายงานผลการประเมินลงไฟล์ JSON
from pathlib import Path  # สำหรับจัดการ Path ไฟล์และไดเรกทอรี

import cv2  # สำหรับการอ่าน จัดการ และปรับขนาดภาพ Mask
import numpy as np  # สำหรับการคำนวณทางคณิตศาสตร์และตรรกศาสตร์ของเมทริกซ์ (Boolean Operations)

# นำเข้าฟังก์ชันช่วยเหลือเกี่ยวกับชุดข้อมูลจาก dataset.py
from dataset import list_images, load_yolo_seg_polygons, polygons_to_mask, label_path_for_image


def pixel_iou(pred_mask, gt_mask, eps=1e-6):
    """คำนวณค่า Pixel-wise IoU ระหว่าง Predicted Mask และ Ground Truth Mask

    Args:
        pred_mask (np.ndarray): ภาพหน้ากากที่โมเดลทำนายได้ (ค่า > 0 คือเส้นเลน)
        gt_mask (np.ndarray): ภาพหน้ากากความจริง (ค่า > 0 คือเส้นเลน)
        eps (float): ค่าคงที่ขนาดเล็กเพื่อป้องกันข้อผิดพลาดกรณีหารด้วยศูนย์

    Returns:
        float: ค่าความถูกต้อง IoU ระหว่าง 0.0 ถึง 1.0
    """
    # แปลงภาพเป็นอาร์เรย์ค่าความจริง (Boolean Array: True เมื่อพิกเซล > 0)
    pred = pred_mask > 0
    gt = gt_mask > 0

    # คำนวณส่วนทับซ้อน (Intersection): ตำแหน่งที่ทั้งคู่เป็นเส้นเลนพร้อมกัน (Logical AND)
    intersection = np.logical_and(pred, gt).sum()
    # คำนวณส่วนรวม (Union): ตำแหน่งที่มีเส้นเลนในภาพใดภาพหนึ่งหรือทั้งคู่ (Logical OR)
    union = np.logical_or(pred, gt).sum()

    # จัดการกรณีพิเศษ (Edge Case): ภาพไม่มีเส้นเลนทั้งในความเป็นจริงและที่ทำนายได้
    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    # คืนค่าสัดส่วน IoU = ส่วนทับซ้อน / ส่วนรวม
    return float(intersection) / float(union + eps)


def main():
    """ฟังก์ชันหลักสำหรับดำเนินการประเมินผลการตรวจจับเส้นเลน"""
    # กำหนดอาร์กิวเมนต์ที่รับผ่าน Command-Line
    parser = argparse.ArgumentParser(description="Evaluate lane masks via pixel-wise IoU.")
    parser.add_argument("--images-dir", required=True, help="Original images (for size + filenames).")
    parser.add_argument("--labels-dir", required=True, help="YOLO-seg polygon label directory.")
    parser.add_argument("--pred-masks-dir", required=True, help="Predicted mask PNGs from inference.py.")
    parser.add_argument("--iou-threshold", type=float, default=0.6)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    # โหลดรายการ path ของภาพต้นฉบับทั้งหมด
    images = list_images(args.images_dir)
    if not images:
        print(f"No images found under {args.images_dir}")
        return

    results = []
    # วนลูปประเมินผลรูปภาพทีละภาพ
    for image_path in images:
        # ค้นหาไฟล์ Prediction Mask ที่ตรงกับชื่อภาพ (.png)
        pred_path = Path(args.pred_masks_dir) / f"{image_path.stem}.png"
        if not pred_path.exists():
            print(f"Warning: missing prediction for {image_path.name}, skipping.")
            continue

        # อ่านภาพต้นฉบับเพื่อดึงขนาดความละเอียดจริง (ความสูง h, ความกว้าง w)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        h, w = image.shape[:2]

        # โหลด Ground Truth จากไฟล์ Label แบบ YOLO-seg แล้วแปลงเป็น Binary Mask ขนาดเท่าภาพจริง
        label_path = label_path_for_image(image_path, args.labels_dir)
        polygons = load_yolo_seg_polygons(label_path, w, h)
        gt_mask = polygons_to_mask(polygons, w, h)

        # อ่านภาพหน้ากากที่โมเดลทำนายได้ในรูปแบบภาพระดับเทา (Grayscale)
        pred_mask = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
        if pred_mask is None:
            print(f"Warning: could not read prediction {pred_path}, skipping.")
            continue
        # หากขนาดของ Prediction Mask ไม่ตรงกับภาพจริง ให้ปรับขนาดด้วย INTER_NEAREST เพื่อคงค่าพิกเซล
        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # คำนวณค่า IoU ระหว่างภาพทำนายและภาพความจริง
        iou = pixel_iou(pred_mask, gt_mask)
        # ตรวจสอบว่าภาพนี้ถือว่า "ตรวจพบสำเร็จ (Detected)" หรือไม่ (IoU สูงกว่าเกณฑ์ที่กำหนด เช่น 0.6)
        detected = iou > args.iou_threshold
        # บันทึกผลลัพธ์รายภาพ
        results.append({"image": image_path.name, "iou": iou, "detected": detected})

    # คำนวณสถิติและตัวชี้วัดภาพรวม
    n = len(results)
    detected_results = [r for r in results if r["detected"]]
    # คำนวณอัตราการตรวจจับสำเร็จ (Detection Rate)
    detection_rate = len(detected_results) / n if n else 0.0
    # คำนวณค่าเฉลี่ย IoU เฉพาะภาพที่ตรวจจับสำเร็จ
    avg_iou_detected = (sum(r["iou"] for r in detected_results) / len(detected_results)
                         if detected_results else 0.0)
    # คำนวณค่าเฉลี่ย IoU ของรูปภาพทั้งหมด
    avg_iou_all = sum(r["iou"] for r in results) / n if n else 0.0

    # จัดกลุ่มข้อมูลสรุปเป็น Dictionary
    summary = {
        "num_images": n,
        "num_detected": len(detected_results),
        "detection_rate": detection_rate,
        "avg_iou_detected": avg_iou_detected,
        "avg_iou_all": avg_iou_all,
        "iou_threshold": args.iou_threshold,
    }

    # แสดงผลสรุปผ่านหน้าจอคอนโซล
    print("=== Lane Detection Evaluation ===")
    print(f"Images evaluated         : {n}")
    print(f"Detected (IoU > {args.iou_threshold})   : {len(detected_results)} ({detection_rate * 100:.1f}%)")
    print(f"Average IoU (detected)   : {avg_iou_detected:.4f}")
    print(f"Average IoU (all images) : {avg_iou_all:.4f}")

    # หากระบุ path ไฟล์ JSON ให้บันทึกผลการประเมินลงไฟล์
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({"summary": summary, "per_image": results}, f, indent=2)
        print(f"Saved detailed results to {args.output_json}")


if __name__ == "__main__":
    main()
