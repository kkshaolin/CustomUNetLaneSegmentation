"""Train the custom UNet for single-class lane segmentation.

Dataset layout expected (Ultralytics YOLO-seg convention):

    <data-root>/images/train/*.jpg
    <data-root>/images/val/*.jpg      (optional; auto-split from train if absent)
    <data-root>/labels/train/*.txt
    <data-root>/labels/val/*.txt

Example:
    python train.py --data-root ./data --epochs 30 --batch-size 4

สคริปต์สำหรับการฝึกสอน (Train) โครงข่ายประสาทเทียม Custom UNet สำหรับงานแบ่งส่วนเส้นเลนถนน
- รองรับชุดข้อมูลโครงสร้างตามมาตรฐาน Ultralytics YOLO-seg
- มีระบบแบ่งชุดข้อมูลตรวจสอบ (Validation Split) อัตโนมัติหากไม่มีโฟลเดอร์ val แยกไว้
- ใช้ฟังก์ชัน Loss แบบผสมผสาน (BCEDiceLoss) เพื่อแก้ปัญหาความไม่สมดุลของข้อมูล (Class Imbalance)
- บันทึกประวัติการเทรนผ่าน TensorBoard และบันทึกโมเดลที่ดีที่สุด (best.pt)
"""
# ==============================================================================
# ไลบรารีมาตรฐานและไลบรารีภายนอกที่เกี่ยวข้องกับการฝึกสอนโครงข่ายประสาทเทียม
# ==============================================================================
import argparse  # สำหรับการรับและประมวลผลคำสั่งผ่าน Command-Line Arguments
import random  # สำหรับการสุ่มตัวเลขและการสลับลำดับดัชนีข้อมูล (Data Shuffling)
from pathlib import Path  # สำหรับจัดการ Path โฟลเดอร์และไฟล์อย่างเป็นระบบ

import torch  # เฟรมเวิร์ก PyTorch หลักสำหรับการคำนวณ Tensor และ Deep Learning
import torch.nn as nn  # โมดูล Neural Network ของ PyTorch สำหรับ Loss Functions และ Layers
from torch.utils.data import DataLoader, Subset  # คลาสจัดการการป้อนข้อมูลเป็น Batch และการแบ่ง Subset
from torch.utils.tensorboard import SummaryWriter  # สำหรับบันทึกกราฟและสถิติการเทรนไปยัง TensorBoard
from tqdm import tqdm  # สำหรับแสดงแถบความคืบหน้า (Progress Bar) ที่สวยงามในหน้าจอ Terminal

from dataset import LaneSegDataset  # คลาสโหลดชุดข้อมูลที่เราสร้างไว้ใน dataset.py
from model import UNet  # สถาปัตยกรรมโมเดล UNet ที่เราสร้างไว้ใน model.py


def dice_loss(logits, targets, eps=1e-6):
    """คำนวณ Dice Loss สำหรับการวัดความทับซ้อน (Overlap) ระหว่าง Prediction และ Ground Truth

    Dice Coefficient = (2 * |A ∩ B|) / (|A| + |B|)
    Dice Loss = 1 - Dice Coefficient

    Args:
        logits (torch.Tensor): ผลลัพธ์ดิบจากโมเดล (Raw Logits) รูปร่าง (Batch, 1, H, W)
        targets (torch.Tensor): หน้ากากความจริง (Ground Truth Mask) รูปร่าง (Batch, 1, H, W)
        eps (float): ค่าคงที่ขนาดเล็กเพื่อป้องกันการหารด้วยศูนย์ (Division by Zero)

    Returns:
        torch.Tensor: ค่าเฉลี่ย Dice Loss ของทั้ง Batch (สเกลาร์)
    """
    # แปลง Raw Logits ให้กลายเป็นค่าความน่าจะเป็น (Probability 0.0 - 1.0) ด้วย Sigmoid
    probs = torch.sigmoid(logits)
    # คลี่ (Flatten) มิติ spatial ให้เป็น 1 มิติต่อภาพ รูปร่างกลายเป็น (Batch, H*W)
    probs = probs.reshape(probs.size(0), -1)
    targets = targets.reshape(targets.size(0), -1)

    # คำนวณพื้นที่ทับซ้อน (Intersection) ระหว่างพิกเซลที่ทำนายและพิกเซลความจริง
    intersection = (probs * targets).sum(dim=1)
    # คำนวณผลรวมพื้นที่ทั้งหมด (Sum of Cardinalities)
    union = probs.sum(dim=1) + targets.sum(dim=1)

    # คำนวณค่าสัมประสิทธิ์ Dice สำหรับแต่ละตัวอย่างใน Batch
    dice = (2 * intersection + eps) / (union + eps)
    # คืนค่า Dice Loss = 1 - Dice โดยเฉลี่ยจากทุกตัวอย่างใน Batch
    return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    """BCE handles pixel-wise classification; Dice compensates for the heavy
    background/lane class imbalance typical of thin lane markings.

    ฟังก์ชันคำนวณ Loss แบบผสม (Hybrid Loss) รวมระหว่าง:
    1. Binary Cross-Entropy with Logits (BCE): ตรวจสอบความถูกต้องของการจำแนกประเภทรายพิกเซล
    2. Dice Loss: ควบคุมรูปทรงความต่อเนื่องและพื้นที่ทับซ้อน ป้องกันปัญหาเส้นเลนมีพิกเซลน้อยมากเมื่อเทียบกับพื้นหลัง
    """

    def __init__(self, bce_weight=0.5):
        """กำหนดค่าน้ำหนักระหว่าง BCE และ Dice Loss

        Args:
            bce_weight (float): สัดส่วนน้ำหนักของ BCE Loss (ค่าที่เหลือ 1 - bce_weight จะเป็นของ Dice Loss)
        """
        super().__init__()
        # BCEWithLogitsLoss รวมเลเยอร์ Sigmoid เข้ากับ BCE Loss ทำให้มีความเสถียรเชิงตัวเลขสูงกว่า
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight

    def forward(self, logits, targets):
        """คำนวณค่า Loss รวมตามน้ำหนักถ่วง

        Args:
            logits (torch.Tensor): ค่า Raw Logits จากโมเดล
            targets (torch.Tensor): ค่า Ground Truth Mask

        Returns:
            torch.Tensor: ผลรวมถ่วงน้ำหนักของ BCE และ Dice Loss
        """
        return (self.bce_weight * self.bce(logits, targets)
                + (1 - self.bce_weight) * dice_loss(logits, targets))


@torch.no_grad()
def compute_iou(logits, targets, threshold=0.5, eps=1e-6):
    """คำนวณค่า Intersection over Union (IoU หรือ Jaccard Index) เพื่อประเมินความแม่นยำ

    สูตร: IoU = |A ∩ B| / |A ∪ B| = Intersection / (Area_A + Area_B - Intersection)

    ประดับด้วย @torch.no_grad() เพื่อปิดการคำนวณ Gradient ประหยัดแรมและเร่งความเร็วในการประเมิน

    Args:
        logits (torch.Tensor): ผลลัพธ์ดิบจากโมเดล
        targets (torch.Tensor): หน้ากากความจริง (Ground Truth)
        threshold (float): เกณฑ์ความน่าจะเป็นในการตัดสินว่าเป็นเส้นเลน (ค่าเริ่มต้น 0.5)
        eps (float): ค่าคงที่ขนาดเล็กป้องกันการหารด้วยศูนย์

    Returns:
        float: ค่าเฉลี่ย IoU ของทั้ง Batch เป็นเลขทศนิยม Python
    """
    # แปลง Logits เป็น Probability แล้วตัดเกณฑ์ threshold เพื่อได้ Binary Mask (0 หรือ 1)
    preds = (torch.sigmoid(logits) > threshold).float()
    # คลี่ข้อมูลเป็น 1 มิติต่อรูปภาพ รูปร่าง (Batch, H*W)
    preds = preds.reshape(preds.size(0), -1)
    targets = targets.reshape(targets.size(0), -1)

    # คำนวณส่วนทับซ้อน (Intersection)
    intersection = (preds * targets).sum(dim=1)
    # คำนวณส่วนรวม (Union) = ผลรวมพิกเซลทั้งสอง - ส่วนที่ทับซ้อนกัน
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection

    # คำนวณค่า IoU ของแต่ละภาพ แล้วหาค่าเฉลี่ยทั้ง Batch
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()


def build_datasets(args):
    """เตรียมและจัดการสร้างชุดข้อมูลสำหรับ Training และ Validation

    ขั้นตอนการทำงาน:
    1. ตรวจสอบว่าผู้ใช้มีโฟลเดอร์ images/val แยกไว้เฉพาะหรือไม่
    2. หากมี: โหลด train dataset (เปิด Augmentation) และ val dataset (ปิด Augmentation) ตรงๆ
    3. หากไม่มี: ทำการแบ่ง (Auto-split) จากชุด train ตามสัดส่วน val_split ที่กำหนด (เช่น 10%)
       โดยใช้การสลับลำดับแบบกำหนด Seed คงที่ เพื่อให้การแบ่งข้อมูลเหมือนกันทุกครั้ง

    Args:
        args (argparse.Namespace): พารามิเตอร์ที่รับมาจากหน้าจอคำสั่ง

    Returns:
        tuple: (train_dataset, val_dataset)
    """
    # กำหนด Path มาตรฐานสำหรับรูปภาพและ labels
    train_images = Path(args.data_root) / "images" / "train"
    train_labels = Path(args.data_root) / "labels" / "train"
    val_images = Path(args.data_root) / "images" / "val"
    val_labels = Path(args.data_root) / "labels" / "val"

    # กรณีที่ 1: มีโฟลเดอร์ val และมีไฟล์ภาพอยู่ข้างใน
    if val_images.exists() and any(val_images.iterdir()):
        # ชุด Train เปิดการทำ Augmentation เพื่อเพิ่มความหลากหลาย
        train_ds = LaneSegDataset(train_images, train_labels, args.img_size, augment=True)
        # ชุด Val ปิดการทำ Augmentation เพื่อประเมินผลบนข้อมูลสภาพจริง
        val_ds = LaneSegDataset(val_images, val_labels, args.img_size, augment=False)
        return train_ds, val_ds

    # กรณีที่ 2: ไม่มีโฟลเดอร์ val แยกไว้ ต้องแบ่งอัตโนมัติจากชุด train
    # No explicit val split provided: carve one out of the train set.
    augmented = LaneSegDataset(train_images, train_labels, args.img_size, augment=True)
    plain = LaneSegDataset(train_images, train_labels, args.img_size, augment=False)
    n = len(augmented)
    if n == 0:
        raise RuntimeError(f"No training images found under {train_images}")

    # สร้างรายการ index ข้อมูลทั้งหมดแล้วสลับลำดับด้วย Random Seed
    indices = list(range(n))
    random.Random(args.seed).shuffle(indices)
    # คำนวณจำนวนข้อมูลสำหรับ Validation
    val_len = max(1, int(n * args.val_split))
    val_idx, train_idx = indices[:val_len], indices[val_len:]

    # สร้าง Subset โดยส่วน Train ใช้ dataset แบบ augmented และส่วน Val ใช้ dataset แบบ plain
    train_ds = Subset(augmented, train_idx)
    val_ds = Subset(plain, val_idx)
    return train_ds, val_ds


def run_epoch(model, loader, criterion, device, optimizer=None):
    """ประมวลผลการทำงาน 1 รอบ Epoch (ใช้ได้ทั้งการ Train และการ Validation)

    Args:
        model (nn.Module): โมเดล UNet
        loader (DataLoader): ตัวโหลดข้อมูล Batch
        criterion (nn.Module): ฟังก์ชัน Loss (BCEDiceLoss)
        device (torch.device): อุปกรณ์ประมวลผล (CUDA หรือ CPU)
        optimizer (torch.optim.Optimizer, optional): Optimizer (ถ้าไม่ระบุ = โหมด Validation)

    Returns:
        tuple[float, float]: (ค่า Loss เฉลี่ย, ค่า IoU เฉลี่ย)
    """
    # ตรวจสอบว่าอยู่ในโหมด Train หรือ Validation จากการส่ง optimizer เข้ามา
    is_train = optimizer is not None
    model.train(is_train)  # กำหนดโหมดการทำงานของโมเดล (Train หรือ Eval)

    total_loss, total_iou, n_batches = 0.0, 0.0, 0
    # ถ้าเป็น Train ให้เปิดการคำนวณ Gradient แต่ถ้าเป็น Validation ให้ปิด Gradient
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, masks in loader:
            # ย้ายข้อมูลรูปภาพและหน้ากากไปยัง Device ที่กำหนด (GPU/CPU)
            images, masks = images.to(device), masks.to(device)
            # Forward Pass: ส่งภาพเข้าโมเดลเพื่อคำนวณ Logits
            logits = model(images)
            # คำนวณค่า Loss
            loss = criterion(logits, masks)

            # หากอยู่ในโหมด Train ให้ทำกระบวนการ Backpropagation และอัปเดตค่าน้ำหนัก
            if is_train:
                optimizer.zero_grad()  # ล้างค่า Gradient เก่า
                loss.backward()  # คำนวณ Gradient ย้อนกลับ
                optimizer.step()  # ปรับปรุงค่าน้ำหนักโมเดลด้วย Optimizer

            # สะสมค่าสถิติ
            total_loss += loss.item()
            total_iou += compute_iou(logits, masks)
            n_batches += 1

    # ป้องกันการหารด้วยศูนย์ในกรณีที่ loader ว่างเปล่า
    n_batches = max(n_batches, 1)
    return total_loss / n_batches, total_iou / n_batches


def main():
    """ฟังก์ชันหลักสำหรับการเริ่มกระบวนการฝึกสอนโมเดล (Training Pipeline)"""
    # ตั้งค่าตัวรับอาร์กิวเมนต์จาก Command-Line
    parser = argparse.ArgumentParser(description="Train custom UNet for lane segmentation.")
    parser.add_argument("--data-root", required=True, help="Root dir containing images/ and labels/.")
    parser.add_argument("--img-size", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4, help="Recommended 2-4 for local machines.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.1, help="Used only if no images/val split exists.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--log-dir", default="runs")
    parser.add_argument("--run-name", default="unet_lane")
    args = parser.parse_args()

    # ตั้งค่า Seed เพื่อให้ผลลัพธ์สามารถจำลองซ้ำได้เหมือนเดิม (Reproducibility)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # ตรวจสอบและเลือกใช้อุปกรณ์ประมวลผล (ใช้ GPU หากมี หรือใช้ CPU หากไม่มี)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # สร้างและแบ่งชุดข้อมูล
    train_ds, val_ds = build_datasets(args)
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    # สร้าง DataLoader สำหรับป้อนข้อมูลเข้าสู่โมเดลทีละ Batch
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, drop_last=False)

    # สร้างโมเดล UNet และย้ายไปยังอุปกรณ์ประมวลผล
    model = UNet(in_channels=3, num_classes=1).to(device)
    # ฟังก์ชัน Loss รวม BCE + Dice
    criterion = BCEDiceLoss()
    # กำหนด Optimizer เป็น Adam พร้อมกำหนด Learning Rate
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # สร้างไดเรกทอรีสำหรับเก็บ Checkpoint ของโมเดล
    checkpoint_dir = Path(args.checkpoint_dir) / args.run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # สร้าง SummaryWriter สำหรับบันทึก Log ลง TensorBoard
    writer = SummaryWriter(log_dir=str(Path(args.log_dir) / args.run_name))

    best_val_loss = float("inf")
    # วนลูปการฝึกสอนตามจำนวน Epochs
    for epoch in tqdm(range(1, args.epochs + 1), desc="Epochs"):
        # รันรอบ Training
        train_loss, train_iou = run_epoch(model, train_loader, criterion, device, optimizer)
        # รันรอบ Validation
        val_loss, val_iou = run_epoch(model, val_loader, criterion, device, optimizer=None)

        # บันทึกค่า Loss และ IoU ไปยัง TensorBoard เพื่อดูผ่านเว็บบราวเซอร์
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("IoU/train", train_iou, epoch)
        writer.add_scalar("IoU/val", val_iou, epoch)

        # แสดงผลลัพธ์ของแต่ละ Epoch ออกทางหน้าจอ
        print(f"Epoch {epoch:03d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_iou={train_iou:.4f} | "
              f"val_loss={val_loss:.4f} val_iou={val_iou:.4f}")

        # บันทึกสถานะโมเดลล่าสุด (last.pt) ในทุก Epoch
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict()},
                   checkpoint_dir / "last.pt")
        # บันทึกสถานะโมเดลที่ดีที่สุด (best.pt) เมื่อได้ค่า val_loss ต่ำที่สุด
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict()},
                       checkpoint_dir / "best.pt")

    # ปิด Writer หลังเทรนเสร็จสิ้น
    writer.close()
    print(f"Training complete. Checkpoints saved under {checkpoint_dir}")


if __name__ == "__main__":
    main()
