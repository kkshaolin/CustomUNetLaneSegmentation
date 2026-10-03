"""Script for re-splitting an existing dataset into train, val, and test sets.

Splits images and YOLO-seg labels into standard directory structure:
    output_dir/images/train, val, test
    output_dir/labels/train, val, test
"""
from pathlib import Path
import random
import shutil

# 1. ตั้งค่าโฟลเดอร์และพารามิเตอร์ที่นี่
# ชี้ไปที่โฟลเดอร์หลัก เช่น โฟลเดอร์ที่เพิ่งรัน split ไปรอบแรก (ระบบจะดึงไฟล์จาก train, val, test มารวมกันให้อัตโนมัติ)
DATASET_ROOT = r"dataset"

# โฟลเดอร์ปลายทาง (สามารถตั้งเป็นชื่อใหม่ หรือตั้งเป็นโฟลเดอร์เดิมเพื่อเขียนทับได้)
OUTPUT_DIR   = r"dataset"

# สัดส่วนข้อมูลใหม่ (Train + Val ต้องน้อยกว่า 1.0 เพื่อเหลือพื้นที่ให้ Test)
TRAIN_RATIO  = 0.66  # ปรับสัดส่วน Train ตามต้องการ 
VAL_RATIO    = 0.17  # ปรับสัดส่วน Val ตามต้องการ 
                     # ส่วนที่เหลือ Test จะได้โดยอัตโนมัติ
SEED         = 42    # เลขสุ่มคงที่ (เปลี่ยนเลขได้หากต้องการให้สุ่มแบบใหม่)
# นามสกุลไฟล์รูปภาพที่ระบบรองรับ
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def collect_dataset(root_dir: Path):
    """รวบรวมไฟล์รูปภาพและ labels จากทุกโฟลเดอร์ย่อย (train, val, test) แบบ recursive"""
    valid_data_dict = {}

    # ค้นหารูปภาพทั้งหมดในโฟลเดอร์และโฟลเดอร์ย่อยทั้งหมด
    for img_path in root_dir.rglob("*"):
        if img_path.is_file() and img_path.suffix.lower() in IMG_EXTS:
            stem_name = img_path.stem

            # หากพบไฟล์ชื่อซ้ำจากการค้นหา จะยึดไฟล์แรกที่เจอ
            if stem_name in valid_data_dict:
                continue

            # ค้นหาไฟล์ label (.txt) ที่ตรงกัน
            # 1. ลองหาในตำแหน่งคู่ขนาน labels/ (ตามโครงสร้าง YOLO)
            possible_label_path = None
            if "images" in img_path.parts:
                parts = list(img_path.parts)
                idx = parts.index("images")
                parts[idx] = "labels"
                candidate = Path(*parts).with_suffix(".txt")
                if candidate.exists():
                    possible_label_path = candidate

            # 2. ถ้าไม่พบ ให้ลองหาไฟล์ .txt ในไดเรกทอรีเดียวกันกับรูป
            if possible_label_path is None:
                candidate = img_path.with_suffix(".txt")
                if candidate.exists():
                    possible_label_path = candidate

            # 3. ถ้ายังไม่พบ ให้ค้นหาไฟล์ชื่อเดียวกัน .txt จากทุกที่ใต้ root_dir
            if possible_label_path is None:
                found_labels = list(root_dir.rglob(f"{stem_name}.txt"))
                if found_labels:
                    possible_label_path = found_labels[0]

            if possible_label_path and possible_label_path.exists():
                valid_data_dict[stem_name] = (img_path, possible_label_path)
            else:
                print(f"Warning: ไม่พบไฟล์ label สำหรับรูปภาพ '{img_path.name}' (ข้ามรูปภาพนี้)")

    return list(valid_data_dict.values())


def main():
    # ตรวจสอบสัดส่วน
    total_ratio = TRAIN_RATIO + VAL_RATIO
    if total_ratio >= 1.0:
        raise ValueError("ผลรวมของ TRAIN_RATIO และ VAL_RATIO ต้องน้อยกว่า 1.0 เพื่อเหลือพื้นที่ให้ Test set")

    random.seed(SEED)

    root_path = Path(DATASET_ROOT)
    if not root_path.exists():
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์ต้นทาง: {root_path}")

    print(f"กำลังกวาดค้นหาไฟล์รูปภาพและ Label ทั้งหมดใน: {root_path.resolve()} ...")
    valid_data = collect_dataset(root_path)

    total_files = len(valid_data)
    if total_files == 0:
        raise ValueError("ไม่พบข้อมูลคู่ รูปภาพ+Label ในโฟลเดอร์ที่ระบุ")

    print(f"ตรวจพบรูปภาพที่มี Label สมบูรณ์ทั้งหมด: {total_files} คู่")

    # สลับลำดับข้อมูล
    valid_data.sort(key=lambda x: x[0].stem)
    random.shuffle(valid_data)

    # คำนวณจำนวนไฟล์ตามสัดส่วนใหม่
    train_count = int(total_files * TRAIN_RATIO)
    val_count = int(total_files * VAL_RATIO)
    test_count = total_files - train_count - val_count

    print(f"\nคำนวณการแบ่งสัดส่วนใหม่:")
    print(f"  - Train ({(TRAIN_RATIO * 100):.1f}%) : {train_count} ไฟล์")
    print(f"  - Val   ({(VAL_RATIO * 100):.1f}%) : {val_count} ไฟล์")
    print(f"  - Test  ({(test_count / total_files * 100):.1f}%) : {test_count} ไฟล์")

    splits = {
        "train": valid_data[:train_count],
        "val": valid_data[train_count : train_count + val_count],
        "test": valid_data[train_count + val_count :]
    }

    output_path = Path(OUTPUT_DIR)

    # หากโฟลเดอร์ปลายทางคือโฟลเดอร์เดิม ให้สร้างลงโฟลเดอร์ชั่วคราวก่อนเพื่อไม่ให้ไฟล์เขียนทับกันระหว่างย้าย
    is_same_dir = (output_path.resolve() == root_path.resolve())
    target_base = output_path if not is_same_dir else output_path.parent / f"{output_path.name}_temp"

    # สร้างโฟลเดอร์ปลายทาง
    for split_name in ["train", "val", "test"]:
        (target_base / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (target_base / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    # คัดลอกไฟล์
    for split_name, data_list in splits.items():
        print(f"กำลังจัดเตรียมข้อมูล {len(data_list)} ชุด สำหรับ '{split_name}'...")
        for img_path, label_path in data_list:
            dest_img = target_base / "images" / split_name / img_path.name
            dest_label = target_base / "labels" / split_name / label_path.name

            shutil.copy2(img_path, dest_img)
            shutil.copy2(label_path, dest_label)

    # หากใช้โฟลเดอร์ชั่วคราว ให้ลบโฟลเดอร์เก่าแล้วแทนที่ด้วยอันใหม่
    if is_same_dir:
        print("\nกำลังอัปเดตโฟลเดอร์ปลายทาง...")
        shutil.rmtree(output_path)
        target_base.rename(output_path)

    print(f"\nเสร็จสิ้น! จัดแบ่งชุดข้อมูลรอบใหม่เรียบร้อยที่:")
    print(f"  -> {output_path.resolve()}")


if __name__ == "__main__":
    main()