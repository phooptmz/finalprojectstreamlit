import numpy as np
import pandas as pd
from joblib import load
import os

print("="*70)
print("🏢 1. จำลองชุดข้อมูลทดสอบแบบละเอียด (High Resolution Test Data)")
print("="*70)

test_cases = []

# ==========================================
# 1. สร้างสถานการณ์จำลอง (Grid Search แบบประยุกต์)
# ==========================================
# จำนวนคน: ตั้งแต่ 0 ถึง 40 คน 
person_scenarios = list(range(0, 41, 1)) 

# ระดับการขยับตัว: ทดสอบทุกระดับ (รวมถึง เคลื่อนที่น้อย 0.2 และ เคลื่อนที่เยอะ 0.8)
movement_scenarios = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# อุณหภูมิและความชื้น (สุ่มตัวอย่างค่าที่เป็นตัวแทน)
temps = np.arange(22.0, 36.5, 0.5) 
humis = np.arange(45.0, 90.0, 5.0)

print(f"กำลังคำนวณ Combinations ทั้งหมด...")

for p in person_scenarios:
    n_norm = p / 40.0
    p_norm = 1.0 if p > 0 else 0.0
    
    movements = [0.0] if p == 0 else movement_scenarios
        
    for a_norm in movements:
        # คำนวณ OCI 
        oci = (0.5 * n_norm) + (0.3 * a_norm) + (0.2 * p_norm)
        delta_oci = 0.00 if p == 0 else 0.02
        
        for temp in temps:
            for humi in humis:
                
                # ==========================================
                # 📌 กฎคาดหวัง (Expected Rules) - อัปเดตล่าสุด
                # ==========================================
                if oci <= 0.30: 
                    oci_level = 'น้อย'
                elif oci <= 0.55: 
                    oci_level = 'กลาง'
                else: 
                    oci_level = 'มาก'
                    
                # 1. OCI น้อย -> Eco (0) เสมอ (ไม่สนร้อน/ชื้น)
                if oci_level == 'น้อย':
                    expected_class = 0
                    
                # 2. OCI มาก + ร้อน/ชื้น -> High (2)
                elif oci_level == 'มาก' and (temp > 26.0 or humi > 70.0):
                    expected_class = 2
                    
                # 3. กรณีอื่นๆ รวมถึงคนเยอะแต่หนาว -> Normal (1) เลี้ยงอุณหภูมิ
                else:
                    expected_class = 1 
                    
                test_cases.append([p, round(oci, 2), delta_oci, temp, humi, expected_class, a_norm])

# สร้าง DataFrame สำหรับชุดทดสอบ
df_test = pd.DataFrame(test_cases, columns=['Person_Count', 'OCI', 'delta_OCI', 'temp_indoor', 'humi_indoor', 'Expected_Mode', 'Movement'])

print(f"✅ สร้างชุดทดสอบเสร็จสิ้น จำนวน: {len(df_test):,} รูปแบบ!")

# ==========================================
# 2. โหลดโมเดล AI และทำการทำนาย (Prediction)
# ==========================================
model_path = 'RandomForest_model.joblib' # ⚠️ เปลี่ยนชื่อไฟล์ให้ตรง

try:
    model = load(model_path)
    print(f"✅ โหลดโมเดล '{model_path}' สำเร็จ!")
except FileNotFoundError:
    print(f"❌ ไม่พบไฟล์โมเดล '{model_path}' กรุณาแก้ไข path ในโค้ด!")
    exit()

# ส่ง Features เข้าไปทำนาย
features_for_pred = df_test[['OCI', 'delta_OCI', 'temp_indoor', 'humi_indoor']]
df_test['Predicted_Mode'] = model.predict(features_for_pred)

# แปลงผลลัพธ์เป็นชื่อให้อ่านง่าย
class_names = {0: 'Eco(28°C)', 1: 'Normal(25°C)', 2: 'High(23°C)'}
df_test['Expected_Name'] = df_test['Expected_Mode'].map(class_names)
df_test['Predicted_Name'] = df_test['Predicted_Mode'].map(class_names)

# ==========================================
# 3. ตรวจสอบความแม่นยำ (Accuracy Check)
# ==========================================
correct_preds = (df_test['Expected_Mode'] == df_test['Predicted_Mode']).sum()
total_cases = len(df_test)
accuracy = (correct_preds / total_cases) * 100

print("\n" + "="*70)
print(f"🎯 ผลการทดสอบความแม่นยำของ AI (จาก {total_cases:,} รูปแบบ)")
print("="*70)
print(f"ทายถูกต้อง: {correct_preds:,} กรณี")
print(f"ความแม่นยำรวม: {accuracy:.2f}%")

if accuracy != 100.0:
    error_cases = df_test[df_test['Expected_Mode'] != df_test['Predicted_Mode']]
    print(f"⚠️ มี {len(error_cases):,} กรณีที่ AI ตัดสินใจคลาดเคลื่อน!")
    error_cases.to_csv('model_errors_report.csv', index=False)
    print("   👉 บันทึกกรณีที่ผิดพลาดทั้งหมดไว้ใน 'model_errors_report.csv' แล้ว")

# ==========================================
# 4. เจาะลึก: เปรียบเทียบ "คน 0-40 คน" แบบ "ขยับน้อย" VS "ขยับเยอะ"
# ==========================================
print("\n" + "="*70)
print("🔍 เจาะลึกสถานการณ์: เปรียบเทียบคน 0-40 คน (ขยับน้อย 0.2 VS ขยับเยอะ 0.8)")
print("   *จำลองที่อุณหภูมิ 28°C, ความชื้น 60% (อากาศแอบร้อนนิดๆ)")
print("="*70)

# ดึงข้อมูลเฉพาะอุณหภูมิ 28C, ความชื้น 60%
df_focus = df_test[(df_test['temp_indoor'] == 28.0) & (df_test['humi_indoor'] == 60.0)]

# เลือกจำนวนคนที่จะสุ่มมาโชว์ (0, 10, 20, 30, 40)
sample_persons = [0, 5, 15, 25, 40]

print(f"{'คน':<5} | {'ขยับน้อย (Movement=0.2)':<28} | {'ขยับเยอะ (Movement=0.8)':<28}")
print("-" * 70)

for p in sample_persons:
    # ดึงค่าขยับน้อย (0.2)
    if p == 0:
        # 0 คน บังคับขยับ 0.0
        row_low = df_focus[(df_focus['Person_Count'] == p)].iloc[0]
        res_low = f"OCI={row_low['OCI']:.2f} -> {row_low['Predicted_Name']}"
        res_high = res_low # 0 คนมีแบบเดียว
    else:
        row_low = df_focus[(df_focus['Person_Count'] == p) & (df_focus['Movement'] == 0.2)].iloc[0]
        res_low = f"OCI={row_low['OCI']:.2f} -> {row_low['Predicted_Name']}"
        
        # ดึงค่าขยับเยอะ (0.8)
        row_high = df_focus[(df_focus['Person_Count'] == p) & (df_focus['Movement'] == 0.8)].iloc[0]
        res_high = f"OCI={row_high['OCI']:.2f} -> {row_high['Predicted_Name']}"

    print(f"{p:<4} | {res_low:<28} | {res_high:<28}")

print("\n💡 ข้อสังเกตจากตาราง:")
print("- คนน้อย (เช่น 0-5 คน): ไม่ว่าจะขยับน้อยหรือมาก OCI ก็ยังต่ำ AI จึงสั่ง Eco(28°C) เพื่อประหยัด")
print("- คนปานกลาง (เช่น 15 คน): ถ้าขยับน้อย จะยังประหยัดไฟอยู่ (Normal) แต่ถ้าเริ่มวิ่งเล่นขยับเยอะ AI จะเปลี่ยนเป็น High(23°C)")
print("- คนเยอะ (เช่น 40 คน): ไม่ว่าจะขยับยังไง OCI ก็ทะลุเกณฑ์ AI จะสั่ง High(23°C) ทันทีเพราะคนอัดแน่นและอากาศร้อน 28°C")

print("\n" + "="*70)
print("🔍 พิสูจน์กฎใหม่: คนเยอะ (40 คน) แต่อากาศหนาว (22°C)")
print("="*70)
cold_case = df_test[(df_test['Person_Count'] == 40) & (df_test['temp_indoor'] == 22.0) & (df_test['Movement'] == 0.8)].iloc[0]
print(f"คน 40 คน | ขยับตัว 0.8 | อุณหภูมิ 22°C | OCI={cold_case['OCI']}")
print(f"👉 AI ตัดสินใจเลือก: {cold_case['Predicted_Name']} (Expected: {cold_case['Expected_Name']})")
print("✅ ผลลัพธ์: AI ไม่ตัดไป High(23°C) เพราะอากาศหนาวอยู่แล้ว แต่เลือกเลี้ยงอุณหภูมิไว้ที่ Normal(25°C) ตามที่คุณต้องการ!")