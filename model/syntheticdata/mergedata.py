import pandas as pd
import glob
import os
from sklearn.utils import shuffle

# ==================== 1. ตั้งค่า ====================
# รูปแบบชื่อไฟล์ที่ต้องการรวม (ปรับให้ตรงกับไฟล์ที่สร้างจาก newociloopseedv1.py)
# ถ้าใช้ไฟล์แบบ Simple (4 features) ใช้คำนี้:
INPUT_PATTERN = 'ac_training_seed*.csv' 
# ถ้าใช้ไฟล์แบบเต็ม (มี ocicalc) ใช้คำนี้แทน:
# INPUT_PATTERN = 'realistic_ac_training_data_seed*_with_ocicalc.csv'

OUTPUT_FILE = 'ac_training_final_clean.csv'

# คอลัมน์ที่ต้องการเก็บไว้ (ตามตกลง 4 Features + 1 Label)
FEATURES = ['OCI', 'delta_OCI', 'temp_indoor', 'humi_indoor']
TARGET = 'ac_mode_class'
ALL_COLUMNS = FEATURES + [TARGET]

# ==================== 2. รวมไฟล์ทั้งหมด ====================
print("🔍 กำลังค้นหาไฟล์ CSV...")
csv_files = glob.glob(INPUT_PATTERN)

if len(csv_files) == 0:
    print(f"❌ ไม่พบไฟล์ที่ตรงกับรูปแบบ '{INPUT_PATTERN}'")
    print("   โปรดตรวจสอบชื่อไฟล์หรือตำแหน่งที่รันสคริปต์")
    exit()

print(f"✅ พบไฟล์จำนวน {len(csv_files)} ไฟล์")

dfs = []
total_rows = 0

for file in csv_files:
    try:
        df = pd.read_csv(file)
        
        # ตรวจสอบว่ามีคอลัมน์ครบไหม
        missing_cols = [col for col in ALL_COLUMNS if col not in df.columns]
        if missing_cols:
            print(f"  - ข้าม {file} (ขาดคอลัมน์: {missing_cols})")
            continue
        
        # เลือกเฉพาะคอลัมน์ที่ต้องการ (ทิ้งคอลัมน์เกินเช่น person_count, N_norm ฯลฯ)
        df = df[ALL_COLUMNS]
        
        # กรองข้อมูลผิดปกติเบื้องต้น (Optional)
        df = df[(df['OCI'] >= 0) & (df['OCI'] <= 1)]
        df = df[(df['temp_indoor'] > 10) & (df['temp_indoor'] < 50)]
        
        dfs.append(df)
        total_rows += len(df)
        print(f"  - โหลด {file} ({len(df)} แถว)")
        
    except Exception as e:
        print(f"  - Error reading {file}: {e}")

if len(dfs) == 0:
    print("❌ ไม่มีข้อมูลที่ใช้ได้")
    exit()

# ==================== 3. รวมเป็นก้อนเดียว ====================
print(f"\n🔗 กำลังรวมข้อมูลทั้งหมด...")
combined_df = pd.concat(dfs, ignore_index=True)
print(f"   ข้อมูลรวมก่อน Shuffle: {len(combined_df)} แถว")

# ==================== 4. สลับแถว (Shuffle) ====================
print("🔀 กำลังสลับลำดับข้อมูล (Shuffle)...")
# สำคัญมาก: ต้อง Shuffle เพื่อให้แต่ละ Class กระจายตัวเท่ากันเมื่อแบ่ง Train/Test
combined_df = shuffle(combined_df, random_state=42).reset_index(drop=True)

# ==================== 5. ตรวจสอบ Class Balance ====================
print("\n📈 ตรวจสอบความสมดุลของ Class:")
class_dist = combined_df[TARGET].value_counts().sort_index()
total = len(combined_df)

print(f"   ข้อมูลรวมทั้งหมด: {total} แถว")
for cls, count in class_dist.items():
    pct = (count / total) * 100
    bar = "█" * int(pct / 2)  # แสดงกราฟแท่งง่ายๆ
    print(f"   Class {cls}: {count} แถว ({pct:.2f}%) {bar}")

# เตือนถ้าไม่สมดุล
min_pct = (class_dist.min() / total) * 100
if min_pct < 10:
    print("\n⚠️  คำเตือน: มี Class น้อยกว่า 10% อาจต้องใช้ class_weight='balanced' ตอนเทรน")
else:
    print("\n✅ Class Balance ดี พร้อมสำหรับเทรนโมเดล!")

# ==================== 6. บันทึกไฟล์ ====================
combined_df.to_csv(OUTPUT_FILE, index=False)
print(f"\n💾 บันทึกข้อมูลรวมเป็น '{OUTPUT_FILE}' เรียบร้อย")
print(f"   ขนาดไฟล์: {os.path.getsize(OUTPUT_FILE) / 1024:.2f} KB")

print("\n✅ เสร็จสิ้นขั้นตอนการเตรียมข้อมูล!")
print(f"   ขั้นตอนถัดไป: ใช้ไฟล์ '{OUTPUT_FILE}' สำหรับเทรนโมเดล")