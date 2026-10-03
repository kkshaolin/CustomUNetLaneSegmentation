"""Custom UNet built from scratch for single-class lane segmentation.

Input:  (2, 3, 48, 64) RGB images (Batch Size = 2, Channels = 3, Height = 48, Width = 64)
Output: (2, 1, 48, 64) binary lane mask logits

โครงสร้างโครงข่ายประสาทเทียมแบบ Custom UNet สำหรับงานแบ่งส่วนเส้นเลนถนน (Lane Segmentation)
- ออกแบบมาสำหรับอินพุตภาพขนาด 48x64 พิกเซล (Height = 48, Width = 64, Channels = 3 RGB)
- เอาต์พุตเป็น Binary Mask ขนาด 48x64 พิกเซล (1 Channel)
- ส่งออกค่าเป็น Raw Logits (ยังไม่ผ่าน Sigmoid เพื่อความเสถียรในการคำนวณ Loss)
"""

# นำเข้าโมดูลและไลบรารี PyTorch สำหรับสร้าง Deep Learning Model
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv3x3 -> BN -> ReLU) x2, preserves spatial size.

    บล็อกพื้นฐาน (Basic Building Block) ของ UNet:
    ทำการ Convolution 3x3 -> Batch Normalization -> ReLU ต่อกัน 2 รอบ
    โดยใช้ padding=1 เพื่อรักษามิติเชิงพื้นที่ (ความกว้างและความสูง H x W) ของ Feature Map ให้เท่าเดิมเสมอ
    """

    def __init__(self, in_ch, out_ch):
        """กำหนดชั้นเลเยอร์ในบล็อก DoubleConv

        Args:
            in_ch (int): จำนวน Channel ขาเข้า (Input Channels)
            out_ch (int): จำนวน Channel ขาออก (Output Channels)
        """
        super().__init__()
        # ประกอบเลเยอร์แบบเรียงตามลำดับ (Sequential)
        self.block = nn.Sequential(
            # รอบที่ 1: Conv2d kernel 3x3, stride=1, padding=1 เพื่อคงขนาดมิติ (H, W)
            # ตั้งค่า bias=False เนื่องจากมี BatchNorm2d ตามหลัง ซึ่งมีค่า bias/affine ในตัวอยู่แล้ว
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            # ปรับข้อมูลให้อยู่ในสเกลมาตรฐาน (Batch Normalization) ช่วยให้โมเดลเทรนได้เร็วและเสถียรขึ้น
            nn.BatchNorm2d(out_ch),
            # ฟังก์ชันกระตุ้นแบบไม่เชิงเส้น (Activation Function) โดยใช้ inplace=True เพื่อประหยัดหน่วยความจำ
            nn.ReLU(inplace=True),

            # รอบที่ 2: Conv2d 3x3 อีกครั้ง เพื่อสกัดฟีเจอร์ที่ซับซ้อนขึ้นในระดับความละเอียดเดิม
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """ประมวลผลข้อมูลส่งผ่านบล็อก DoubleConv

        Args:
            x (torch.Tensor): Tensor ขาเข้า รูปร่าง (Batch, in_ch, H, W)

        Returns:
            torch.Tensor: Tensor ขาออก รูปร่าง (Batch, out_ch, H, W)
        """
        return self.block(x)


class Down(nn.Module):
    """MaxPool2x2 followed by DoubleConv.

    บล็อกฝั่ง Encoder (Contracting Path):
    ทำหน้าที่ลดขนาดเชิงพื้นที่ (Downsampling) ลงครึ่งหนึ่งเพื่อบีบอัดข้อมูล
    พร้อมทั้งเพิ่มจำนวน Feature Channels เพื่อสกัดข้อมูลเชิงบริบท (Contextual Information)
    """

    def __init__(self, in_ch, out_ch):
        """กำหนดเลเยอร์สำหรับบล็อก Downsampling

        Args:
            in_ch (int): จำนวน Channel ขาเข้า
            out_ch (int): จำนวน Channel ขาออก
        """
        super().__init__()
        # MaxPool2d ขนาด 2x2 พร้อม stride=2 จะลดขนาดความกว้างและความสูงลงครึ่งหนึ่ง (H/2, W/2)
        self.pool = nn.MaxPool2d(2)
        # ตามด้วย DoubleConv เพื่อประมวลผลฟีเจอร์และปรับขนาด Channel ให้เป็น out_ch
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        """ประมวลผลการลดขนาดภาพและการสกัดฟีเจอร์

        Args:
            x (torch.Tensor): Tensor ขาเข้า รูปร่าง (Batch, in_ch, H, W)

        Returns:
            torch.Tensor: Tensor ขาออก รูปร่าง (Batch, out_ch, H/2, W/2)
        """
        # ลดขนาดภาพด้วย MaxPool2d ก่อน แล้วจึงส่งเข้า DoubleConv
        return self.conv(self.pool(x))


class Up(nn.Module):
    """ConvTranspose2x2 upsample, concat with encoder skip, then DoubleConv.

    บล็อกฝั่ง Decoder (Expansive Path):
    ทำหน้าที่ขยายขนาดเชิงพื้นที่ (Upsampling) กลับขึ้นมา 2 เท่า
    และนำฟีเจอร์ที่มีรายละเอียดสูงจากฝั่ง Encoder (Skip Connection) มาเชื่อมต่อ (Concatenate)
    เพื่อกู้คืนข้อมูลตำแหน่งและขอบเขตของเส้นเลนที่แม่นยำ
    """

    def __init__(self, in_ch, out_ch):
        """กำหนดเลเยอร์สำหรับบล็อก Upsampling

        Args:
            in_ch (int): จำนวน Channel ขาเข้าจากระดับก่อนหน้า
            out_ch (int): จำนวน Channel ขาออกหลังการประมวลผล
        """
        super().__init__()
        # Transposed Convolution (Deconvolution) ขยายขนาดภาพขึ้น 2 เท่า (H*2, W*2) และลด channel ลงครึ่งหนึ่ง
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        # เนื่องจากมีการ Concatenate ฟีเจอร์จาก skip (out_ch) และฟีเจอร์ที่ขยายแล้ว (out_ch)
        # รวมกันเป็น out_ch * 2 จึงต้องใช้ DoubleConv ที่รับ out_ch * 2 แล้วแปลงกลับเป็น out_ch
        self.conv = DoubleConv(out_ch * 2, out_ch)

    def forward(self, x, skip):
        """ประมวลผลการขยายภาพ ผสาน Skip Connection และลด Channel

        Args:
            x (torch.Tensor): Feature Map จากเลเยอร์ลึกกว่า รูปร่าง (Batch, in_ch, H, W)
            skip (torch.Tensor): Feature Map จาก Encoder ระดับเดียวกัน รูปร่าง (Batch, out_ch, H*2, W*2)

        Returns:
            torch.Tensor: Feature Map ขาออก รูปร่าง (Batch, out_ch, H*2, W*2)
        """
        # 1. ขยายขนาด spatial dimension ขึ้น 2 เท่า
        x = self.up(x)
        # 2. นำฟีเจอร์จาก skip connection มาต่อกับ x ในแนวแกน Channel (dim=1)
        x = torch.cat([skip, x], dim=1)
        # 3. ส่งผ่าน DoubleConv เพื่อผสานข้อมูลและลดจำนวน Channels
        return self.conv(x)


class UNet(nn.Module):
    """4-level UNet sized for 48x64 inputs (48x64 -> 24x32 -> 12x16 -> 6x8 -> 3x4 at the bottleneck).

    สถาปัตยกรรม UNet ขนาด 4 ระดับ ออกแบบให้เหมาะกับภาพอินพุต 48x64 พิกเซล (Height=48, Width=64)
    ลำดับการเปลี่ยนขนาดเชิงพื้นที่: 48x64 -> 24x32 -> 12x16 -> 6x8 -> 3x4 (ที่ Bottleneck)
    """

    def __init__(self, in_channels=3, num_classes=1, base_ch=32):
        """สร้างสถาปัตยกรรมโครงข่ายประสาท UNet

        Args:
            in_channels (int): จำนวนช่องสัญญาณของภาพนำเข้า (RGB = 3)
            num_classes (int): จำนวนคลาสที่ต้องการแบ่งส่วน (1 คลาสสำหรับเส้นเลน)
            base_ch (int): จำนวน Channel เริ่มต้นของเลเยอร์แรก (ค่าเริ่มต้น 32)
        """
        super().__init__()
        # ----------------------------------------------------------------------
        # ฝั่ง Encoder (Contracting Path): สกัดข้อมูลและลดขนาดภาพ
        # ----------------------------------------------------------------------
        self.in_conv = DoubleConv(in_channels, base_ch)        # ระดับที่ 0: ขนาดคงที่ 48x64, base_ch (32)
        self.down1 = Down(base_ch, base_ch * 2)                # ระดับที่ 1: ย่อลงเหลือ 24x32, base_ch*2 (64)
        self.down2 = Down(base_ch * 2, base_ch * 4)            # ระดับที่ 2: ย่อลงเหลือ 12x16, base_ch*4 (128)
        self.down3 = Down(base_ch * 4, base_ch * 8)            # ระดับที่ 3: ย่อลงเหลือ 6x8,   base_ch*8 (256)
        self.down4 = Down(base_ch * 8, base_ch * 16)           # ระดับที่ 4 (Bottleneck): ย่อลงเหลือ 3x4, base_ch*16 (512)

        # ----------------------------------------------------------------------
        # ฝั่ง Decoder (Expansive Path): ขยายขนาดภาพกลับและกู้คืนรายละเอียด
        # ----------------------------------------------------------------------
        self.up1 = Up(base_ch * 16, base_ch * 8)               # ขยายจาก 3x4   -> 6x8
        self.up2 = Up(base_ch * 8, base_ch * 4)                # ขยายจาก 6x8   -> 12x16
        self.up3 = Up(base_ch * 4, base_ch * 2)                # ขยายจาก 12x16 -> 24x32
        self.up4 = Up(base_ch * 2, base_ch)                    # ขยายจาก 24x32 -> 48x64

        # ----------------------------------------------------------------------
        # เลเยอร์ส่งออก (Output Classification Layer)
        # ----------------------------------------------------------------------
        # ใช้ Convolution 1x1 เพื่อฉาย (Project) จาก base_ch (32) ไปเป็นจำนวนคลาสเป้าหมาย (1 สำหรับ binary lane)
        self.out_conv = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, x):
        """ประมวลผลภาพอินพุตผ่านโครงข่ายประสาท UNet แบบครบวงจร

        Args:
            x (torch.Tensor): ภาพนำเข้า รูปร่าง (Batch, 3, 48, 64) เช่น (2, 3, 48, 64)

        Returns:
            torch.Tensor: ผลลัพธ์ Raw Logits รูปร่าง (Batch, 1, 48, 64) เช่น (2, 1, 48, 64)
        """
        # ขั้นตอนที่ 1: ส่งผ่าน Encoder พร้อมเก็บ Feature Maps (x1 ถึง x4) ไว้ใช้สำหรับ Skip Connections
        x1 = self.in_conv(x)   # Shape: (B, 32, 48, 64)
        x2 = self.down1(x1)    # Shape: (B, 64, 24, 32)
        x3 = self.down2(x2)    # Shape: (B, 128, 12, 16)
        x4 = self.down3(x3)    # Shape: (B, 256, 6, 8)
        x5 = self.down4(x4)    # Shape: (B, 512, 3, 4) - จุดคอดคอด (Bottleneck)

        # ขั้นตอนที่ 2: ส่งผ่าน Decoder พร้อมนำฟีเจอร์จาก Skip Connection มารวมในแต่ละระดับ
        x = self.up1(x5, x4)   # นำ x5 (Bottleneck) ขยายและผสานกับ x4 -> ได้ Shape: (B, 256, 6, 8)
        x = self.up2(x, x3)    # ขยายและผสานกับ x3 -> ได้ Shape: (B, 128, 12, 16)
        x = self.up3(x, x2)    # ขยายและผสานกับ x2 -> ได้ Shape: (B, 64, 24, 32)
        x = self.up4(x, x1)    # ขยายและผสานกับ x1 -> ได้ Shape: (B, 32, 48, 64)

        # ขั้นตอนที่ 3: แปลงเป็นผลลัพธ์สุดท้าย (Logits) ผ่าน Conv 1x1 -> ได้ Shape: (B, 1, 48, 64)
        return self.out_conv(x)


# ==============================================================================
# ส่วนทดสอบการทำงานของโมเดลเมื่อรันสคริปต์นี้โดยตรง (Smoke Test & Architecture Check)
# ==============================================================================
if __name__ == "__main__":
    # สร้างอินสแตนซ์ของโมเดล UNet
    model = UNet()
    # สร้าง Tensor จำลองข้อมูลภาพอินพุต (Batch Size = 2, Channels = 3, Height = 48, Width = 64)
    dummy = torch.randn(2, 3, 48, 64)
    # ส่งข้อมูลจำลองเข้าโมเดลเพื่อตรวจสอบ Forward Pass
    out = model(dummy)
    # แสดงขนาดมิติของ Input และ Output
    print(f"Input:  {tuple(dummy.shape)}")
    print(f"Output: {tuple(out.shape)}")
    # คำนวณและแสดงจำนวนพารามิเตอร์ที่สามารถเรียนรู้ได้ทั้งหมดในโมเดล
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,}")
