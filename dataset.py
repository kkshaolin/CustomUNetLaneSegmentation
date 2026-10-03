"""Dataset utilities for lane segmentation from Ultralytics YOLO-seg polygon labels.

Expected on-disk layout (Ultralytics convention):

    <data_root>/images/train/*.jpg
    <data_root>/images/val/*.jpg
    <data_root>/labels/train/*.txt
    <data_root>/labels/val/*.txt

Each label file has one line per polygon instance:

    class_id x1 y1 x2 y2 x3 y3 ... xn yn

with all coordinates normalized to [0, 1] relative to the *original* image
size. Since source images vary in resolution, masks are rasterized at the
original resolution first, then image and mask are resized together to the
network's fixed 48x48 input/output size.
"""
# ==============================================================================
# ไลบรารีมาตรฐานและไลบรารีภายนอกที่จำเป็นสำหรับการประมวลผลข้อมูล
# ==============================================================================
import random  # สำหรับการสุ่มค่าในการทำ Data Augmentation และสลับข้อมูล (Shuffle)
from pathlib import Path  # จัดการ Path ของระบบไฟล์อย่างเป็นระบบและข้ามแพลตฟอร์ม

import cv2  # OpenCV สำหรับการอ่าน เขียน ปรับขนาดภาพ และวาดรูปหลายเหลี่ยม (Polygon)
import numpy as np  # NumPy สำหรับการคำนวณเมทริกซ์และจัดการอาร์เรย์ตัวเลข
import torch  # PyTorch สำหรับการแปลงข้อมูลเป็น Tensor
from torch.utils.data import Dataset  # คลาสแม่มาตรฐานสำหรับสร้าง Custom Dataset ใน PyTorch

# กำหนดนามสกุลไฟล์รูปภาพที่รองรับในระบบ
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(images_dir):
    """ค้นหาและรวบรวม path ของไฟล์รูปภาพทั้งหมดในโฟลเดอร์ที่กำหนด

    Args:
        images_dir (str หรือ Path): ไดเรกทอรีที่เก็บไฟล์รูปภาพ

    Returns:
        list[Path]: รายชื่อไฟล์รูปภาพที่เรียงลำดับตามตัวอักษรแล้ว
    """
    images_dir = Path(images_dir)
    # ตรวจสอบว่าไดเรกทอรีที่ระบุมีอยู่จริงหรือไม่ หากไม่มีให้คืนลิสต์ว่าง
    if not images_dir.exists():
        return []
    # วนลูปอ่านไฟล์ทั้งหมดในโฟลเดอร์ คัดกรองเฉพาะไฟล์ที่มีนามสกุลตรงกับ IMG_EXTS
    # และใช้ sorted() เพื่อให้ลำดับของรูปภาพคงที่แน่นอนเสมอ (Deterministic)
    return sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS)


def label_path_for_image(image_path, labels_dir):
    """จับคู่ path ของไฟล์ภาพไปยัง path ของไฟล์ label (.txt) ที่ตรงกัน

    Args:
        image_path (str หรือ Path): path ของไฟล์ภาพต้นฉบับ
        labels_dir (str หรือ Path): ไดเรกทอรีที่เก็บไฟล์ label

    Returns:
        Path: path ของไฟล์ label ที่มีชื่อไฟล์เดียวกับรูปภาพ แต่เปลี่ยนนามสกุลเป็น .txt
    """
    return Path(labels_dir) / f"{Path(image_path).stem}.txt"


def load_yolo_seg_polygons(label_path, img_w, img_h, target_class=0):
    """Parse a YOLO-seg label file into a list of int32 pixel-coordinate polygons.

    อ่านไฟล์ label รูปแบบ YOLO Segmentation (.txt) และแปลงพิกัดจุด Polygon
    จากค่า Normalized [0.0, 1.0] ให้เป็นพิกัดพิกเซลจริงตามขนาดความกว้างและความสูงของภาพ
    พร้อมกรองเฉพาะ class_id ที่ต้องการ (ค่าเริ่มต้น 0 = lane)

    Args:
        label_path (str หรือ Path): path ของไฟล์ label (.txt)
        img_w (int): ความกว้างจริงของภาพ (พิกเซล)
        img_h (int): ความสูงจริงของภาพ (พิกเซล)
        target_class (int, optional): ID ของคลาสที่ต้องการโหลด (ค่าเริ่มต้น 0, ระบุ None หากต้องการทุกคลาส)

    Returns:
        list[np.ndarray]: รายการของ Polygon แต่ละอันเป็น numpy array ชนิด int32 ขนาด (N, 2)
    """
    label_path = Path(label_path)
    polygons = []
    # ถ้าไม่มีไฟล์ label (เช่น ภาพที่ไม่มีเส้นเลนหรือไม่มี annotation) ให้คืน list ว่าง
    if not label_path.exists():
        return polygons
    # เปิดอ่านไฟล์ label ทีละบรรทัด
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            # ตรวจสอบว่าบรรทัดมีข้อมูลครบถ้วนหรือไม่
            # ต้องมีอย่างน้อย 7 ค่า: class_id (1 ค่า) + พิกัด (x, y) อย่างน้อย 3 จุด (6 ค่า) เพื่อสร้างรูปหลายเหลี่ยม
            if len(parts) < 7:  # class_id + at least 3 (x, y) points
                continue
            # กรองเฉพาะคลาสที่ต้องการ (ค่าเริ่มต้น class 0: lane)
            if target_class is not None and int(parts[0]) != target_class:
                continue
            # แยกเฉพาะส่วนพิกัด (ตั้งแต่ index ที่ 1 เป็นต้นไป ข้าม class_id) แล้ว reshape เป็น (N, 2)
            coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
            # แปลงพิกัดแกน X จากค่าสัมพัทธ์ (0.0 - 1.0) เป็นพิกเซลจริงโดยคูณด้วยความกว้างของภาพ
            coords[:, 0] *= img_w
            # แปลงพิกัดแกน Y จากค่าสัมพัทธ์ (0.0 - 1.0) เป็นพิกเซลจริงโดยคูณด้วยความสูงของภาพ
            coords[:, 1] *= img_h
            # แปลงค่าเป็นจำนวนเต็ม int32 สำหรับใช้งานกับฟังก์ชันของ OpenCV
            polygons.append(coords.astype(np.int32))
    return polygons


def polygons_to_mask(polygons, img_w, img_h):
    """แปลงรายการ Polygon ให้กลายเป็น Binary Mask รูปแบบ 2 มิติ (ภาพขาวดำ 0 และ 1)

    Args:
        polygons (list[np.ndarray]): รายการพิกัดรูปหลายเหลี่ยมที่ได้จาก load_yolo_seg_polygons
        img_w (int): ความกว้างของหน้ากาก (พิกเซล)
        img_h (int): ความสูงของหน้ากาก (พิกเซล)

    Returns:
        np.ndarray: หน้ากาก Binary Mask ขนาด (img_h, img_w) ชนิด uint8 โดยพิกเซลเลนมีค่า 1 และพื้นหลังมีค่า 0
    """
    # สร้างภาพพื้นหลังสีดำขนาดเท่าภาพจริง โดยเริ่มต้นให้ทุกพิกเซลมีค่าเป็น 0
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    # หากมี Polygon ให้ระบายสีทับลงบน mask ด้วย cv2.fillPoly โดยกำหนดให้พิกเซลภายในรูปปิดมีค่าเป็น 1
    if polygons:
        cv2.fillPoly(mask, polygons, color=1)
    return mask


def augment_image(image):
    """Slight photometric augmentation to mimic outdoor sunlight/sensor noise.

    Applies, each with independent probability: a small per-channel white
    balance gain shift, a small brightness shift, and a light Gaussian blur.

    ฟังก์ชันทำ Data Augmentation ด้านแสงและสีของภาพ เพื่อจำลองความหลากหลายของสภาพแสงกลางแจ้ง
    และสัญญาณรบกวนของเซ็นเซอร์กล้องติดหน้ารถ:
    1. ปรับ White Balance รายช่องสี (Gain shift) สุ่มอิสระ โอกาส 50%
    2. ปรับความสว่างรวมของภาพ (Brightness shift) โอกาส 50%
    3. ทำภาพเบลอแบบ Gaussian Blur เล็กน้อย โอกาส 30%

    Args:
        image (np.ndarray): ภาพต้นฉบับรูปแบบ RGB ขนาด (H, W, 3) ชนิด uint8

    Returns:
        np.ndarray: ภาพที่ผ่านการปรับแต่งค่าสีและแสงแล้ว ชนิด uint8
    """
    # แปลงชนิดข้อมูลเป็น float32 เพื่อป้องกันปัญหาค่าล้น (overflow/underflow) ขณะคำนวณทางคณิตศาสตร์
    img = image.astype(np.float32)

    # 1. การปรับ White Balance (สุ่มปรับ gain ของแต่ละช่องสี R, G, B อิสระในช่วง 0.92 - 1.08) โอกาส 50%
    if random.random() < 0.5:
        gains = np.random.uniform(0.92, 1.08, size=3).astype(np.float32)
        img = img * gains[np.newaxis, np.newaxis, :]

    # 2. การปรับความสว่าง (เพิ่มหรือลดค่าความสว่างของทุกพิกเซลในช่วง -25.0 ถึง +25.0) โอกาส 50%
    if random.random() < 0.5:
        delta = np.random.uniform(-25.0, 25.0)
        img = img + delta

    # จำกัดช่วงของค่าพิกเซลให้อยู่ระหว่าง [0, 255] เสมอ แล้วแปลงกลับเป็น uint8
    img = np.clip(img, 0, 255).astype(np.uint8)

    # 3. การใส่ Gaussian Blur (สุ่มเลือกระหว่าง kernel 3x3 หรือ 5x5) โอกาส 30% เพื่อจำลองการหลุดโฟกัสหรือเลนส์มัว
    if random.random() < 0.3:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    return img


class LaneSegDataset(Dataset):
    """Custom PyTorch Dataset สำหรับการโหลดภาพและสร้าง Mask เส้นเลน

    Attributes:
        images (list[Path]): รายการ path ของไฟล์ภาพทั้งหมด
        labels_dir (Path): ไดเรกทอรีที่เก็บไฟล์ label (.txt)
        img_size (int): ขนาดความกว้างและความสูงเป้าหมายที่โมเดลต้องการ (ค่าเริ่มต้น 48 พิกเซล)
        augment (bool): สถานะเปิด/ปิดการทำ Data Augmentation (มักเปิดตอน Train และปิดตอน Val)
    """
    def __init__(self, images_dir, labels_dir, img_size=48, augment=False, target_class=0):
        # ค้นหาและเก็บรายชื่อไฟล์ภาพทั้งหมด
        self.images = list_images(images_dir)
        self.labels_dir = Path(labels_dir)
        self.img_size = img_size
        self.augment = augment
        self.target_class = target_class

    def __len__(self):
        """คืนค่าจำนวนตัวอย่างทั้งหมดในชุดข้อมูล"""
        return len(self.images)

    def __getitem__(self, idx):
        """ดึงตัวอย่างข้อมูลคู่ (ภาพ, หน้ากาก) ตาม index ที่ระบุ

        ขั้นตอนการทำงาน:
        1. โหลดภาพจาก disk และแปลงจาก BGR เป็น RGB
        2. อ่านไฟล์ label YOLO-seg แล้วแปลง Polygon เป็น Mask ขนาดเท่าภาพจริง (Rasterization)
        3. ทำ Augmentation ภาพ (หากเปิดใช้งาน)
        4. Resize ทั้งภาพและ Mask ให้มีขนาดเท่ากัน (img_size, img_size) เช่น 48x48
        5. แปลงเป็น PyTorch Tensor พร้อมจัดเรียงมิติให้เป็น (C, H, W) และ Normalize ค่าสีเป็น [0.0, 1.0]

        Args:
            idx (int): ลำดับ index ของข้อมูลที่ต้องการ

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - image_t: Tensor ภาพ ขนาด (3, img_size, img_size) ค่าอยู่ในช่วง [0.0, 1.0]
                - mask_t: Tensor หน้ากาก ขนาด (1, img_size, img_size) ค่าเป็น 0.0 หรือ 1.0
        """
        # อ่าน path ของรูปภาพตามลำดับ index
        image_path = self.images[idx]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        # OpenCV อ่านภาพเป็น BGR ตามค่าเริ่มต้น ต้องแปลงเป็น RGB เพื่อให้ตรงกับมาตรฐานโครงข่ายประสาท
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        # ค้นหา path ของไฟล์ label ที่จับคู่กับภาพนี้
        label_path = label_path_for_image(image_path, self.labels_dir)
        # โหลดพิกัด polygon ของเส้นเลนตามขนาดความละเอียดต้นฉบับ (กรองเฉพาะ target_class เช่น 0 = lane)
        polygons = load_yolo_seg_polygons(label_path, w, h, target_class=self.target_class)
        # สร้าง binary mask ที่ความละเอียดต้นฉบับก่อน เพื่อรักษาความคมชัดและรูปทรงของเส้นเลน
        mask = polygons_to_mask(polygons, w, h)

        # ทำ Data Augmentation กับรูปภาพ (เฉพาะเมื่อเปิด flag augment=True สำหรับชุด Train)
        if self.augment:
            image = augment_image(image)

        # ย่อ/ขยายรูปภาพและหน้ากากให้มีขนาดเท่ากับ input ของโมเดล (img_size x img_size)
        # ใช้ INTER_AREA สำหรับรูปภาพเพราะให้ผลลัพธ์ที่ดีและลด Moire effect เมื่อย่อขนาดภาพ
        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        # ใช้ INTER_NEAREST สำหรับ mask เพื่อรักษาระดับค่าพิกเซลให้เป็น 0 หรือ 1 อย่างแม่นยำ ไม่ให้เกิดค่าเฉลี่ยกึ่งกลาง
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        # แปลงรูปภาพเป็น PyTorch Tensor:
        # 1. แปลงเป็น float32 และหารด้วย 255.0 เพื่อ Normalize ค่าให้อยู่ในช่วง [0.0, 1.0]
        # 2. สลับมิติจาก (H, W, C) เป็น (C, H, W) ด้วย .permute(2, 0, 1) ตามรูปแบบมาตรฐานของ PyTorch Conv2d
        # 3. เรียก .contiguous() เพื่อให้ข้อมูลในหน่วยความจำเรียงต่อกันเป็นแถบเดียวอย่างมีประสิทธิภาพ
        image_t = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()

        # แปลงหน้ากากเป็น PyTorch Tensor:
        # 1. แปลงเป็น float32
        # 2. เพิ่มมิติ Channel ด้านหน้าด้วย .unsqueeze(0) จากขนาด (H, W) กลายเป็น (1, H, W)
        # 3. เรียก .contiguous() เพื่อจัดหน่วยความจำให้ต่อเนื่อง
        mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).contiguous()

        return image_t, mask_t
