import numpy as np
import pandas as pd
from joblib import load
import os

# ==========================================
# 1. สร้างชุดข้อมูลทดสอบ (0-40 คน, เคลื่อนที่น้อย/มาก, อุณหภูมิ 22-35 องศา)
# ==========================================
print("⚙️ กำลังสร้างสถานการณ์ทดสอบตั้งแต่ 0 ถึง 40 คน และอุณหภูมิ 22-35 องศา...")
test_cases = []

# กรณีที่ 1: ห้องว่าง (0 คน) - ให้ไล่อุณหภูมิตั้งแต่ 22 ถึง 35 องศาด้วย
for temp in range(22, 36):
    test_cases.append({
        'Person_Count': 0,
        'Movement_Level': 'None',
        'OCI': 0.0,
        # เพิ่ม Noise เล็กน้อยจำลองความคลาดเคลื่อนเซ็นเซอร์ตอนห้องว่าง
        'delta_OCI': round(np.random.normal(0, 0.005), 4), 
        'temp_indoor': float(temp),
        'humi_indoor': 60.0
    })

# กรณีที่ 2: มีคน 1 ถึง 40 คน
for p in range(1, 41):
    # คำนวณ N(t) และ P(t) ตามสมการ OCI
    n_norm = p / 40.0
    p_norm = 1.0  # มีคนอยู่ = 1.0 เสมอ
    
    # กำหนดค่าการเคลื่อนไหว (Activity Norm)
    movements = [
        {'label': 'Low (Moving Little)', 'a_norm': 0.05},
        {'label': 'High (Moving a Lot)', 'a_norm': 0.7}
    ]
    
    for move in movements:
        # คำนวณ OCI (ค่า OCI จะคงที่สำหรับจำนวนคนและระดับการเคลื่อนไหวนั้นๆ)
        oci = (0.5 * n_norm) + (0.3 * move['a_norm']) + (0.2 * p_norm)
        
        # วนลูปอุณหภูมิตั้งแต่ 22 ถึง 35 องศา
        for temp in range(22, 36):
            test_cases.append({
                'Person_Count': p, 
                'Movement_Level': move['label'],
                'OCI': round(oci, 4), 
                # เพิ่ม Noise จำลองการขยับตัวที่ทำให้ OCI แกว่งขึ้นลงเล็กน้อย
                'delta_OCI': round(np.random.normal(0, 0.015), 4), 
                'temp_indoor': float(temp),  # เปลี่ยนอุณหภูมิตามลูป
                'humi_indoor': 60.0
            })

df_test = pd.DataFrame(test_cases)
print(f"📊 สร้างข้อมูลจำลองสำเร็จ! (ทั้งหมด {len(df_test)} สถานการณ์)\n")

# ==========================================
# 2. โหลดโมเดลและทำนายผล
# ==========================================
# ⚠️ เปลี่ยนชื่อโมเดลตรงนี้ให้ตรงกับโมเดลที่คุณเทรนไว้ (เช่น Random_Forest_model.joblib หรือ CatBoost_model.joblib)
MODEL_PATH = "RandomForest_model.joblib" 

if not os.path.exists(MODEL_PATH):
    print(f"❌ ไม่พบไฟล์โมเดล: {MODEL_PATH}")
    print("   โปรดตรวจสอบชื่อไฟล์โมเดลในโฟลเดอร์ให้ตรงกับตัวแปร MODEL_PATH")
else:
    print(f"⏳ กำลังโหลดโมเดล: {MODEL_PATH} ...")
    model = load(MODEL_PATH)
    print("🔍 กำลังจำแนกโหมดแอร์...\n ")
    
    # เลือกเฉพาะฟีเจอร์ที่ใช้เทรน
    features_to_predict = df_test[['OCI', 'delta_OCI', 'temp_indoor', 'humi_indoor']]
    predictions = model.predict(features_to_predict)

    # แปลงผลลัพธ์จากตัวเลขเป็นชื่อโหมด (ลบช่องว่างส่วนเกินออกเพื่อให้แสดงผลสวยงาม)
    class_names = {0: "Eco", 1: "Normal", 2: "High"}
    predicted_labels = [class_names.get(int(p), f"Class {p}") for p in predictions]

    df_test['Predicted_Mode'] = predicted_labels

    # ==========================================
    # 3. สรุปผลลัพธ์ข้อมูลจำลองพื้นฐาน
    # ==========================================
    print("✅ ผลการทดสอบ (ตัวอย่างบางส่วน): ")
    print("-" * 85)
    
    # แสดงผลเฉพาะกรณี 1 คน (ตามที่ต้องการตรวจสอบ)
    print("🌡️ ตัวอย่างผลลัพธ์: คนในห้อง 1 คน (เคลื่อนที่น้อย vs มาก) อุณหภูมิ 22-35 องศา")
    person_1_data = df_test[df_test['Person_Count'] == 1]
    # เลือกคอลัมน์สำคัญมาแสดง
    cols_to_show = ['Person_Count', 'Movement_Level', 'temp_indoor', 'OCI', 'delta_OCI', 'Predicted_Mode'] # เพิ่ม delta_OCI เข้ามาแสดงผล
    print(person_1_data[cols_to_show].to_string(index=False))
    
    print("\n... (ข้อมูลกลางและข้อมูลอื่นๆ ถูกบันทึกลงไฟล์ CSV) ...")
    
    # แสดงผลกรณี 40 คน (บางตัวอย่าง)
    print("\n🌡️ ตัวอย่างผลลัพธ์: คนในห้อง 40 คน (บางอุณหภูมิ)")
    person_40_data = df_test[df_test['Person_Count'] == 40]
    print(person_40_data[cols_to_show].head(10).to_string(index=False))
    
    print("-" * 85)

    # สรุปภาพรวมว่า แต่ละการเคลื่อนไหว สั่งเปิดโหมดอะไรไปกี่ครั้ง
    print("\n📊 สรุปจำนวนครั้งที่แอร์ถูกสั่งงาน (แยกตามระดับการเคลื่อนไหว): ")
    summary = df_test.groupby(['Movement_Level', 'Predicted_Mode']).size().reset_index(name='Count')
    print(summary.to_string(index=False))
    
    # สรุปแยกตามอุณหภูมิ (เพิ่มใหม่เพื่อดูแนวโน้มอุณหภูมิ)
    print("\n📊 สรุปจำนวนครั้งที่แอร์ถูกสั่งงาน (แยกตามอุณหภูมิ): ")
    temp_summary = df_test.groupby(['temp_indoor', 'Predicted_Mode']).size().reset_index(name='Count')
    print(temp_summary.to_string(index=False))

    # บันทึกเป็นไฟล์ CSV สำหรับเอาไปเปิดดูใน Excel
    output_file = "comprehensive_model_test_temp_22to35.csv"
    df_test.to_csv(output_file, index=False)
    print(f"\n💾 บันทึกผลลัพธ์แบบละเอียดทั้งหมดลงไฟล์: {output_file} ")

    # ==========================================
    # 4. ทดสอบสถานการณ์พิเศษ (Edge Cases & Stress Tests)
    # ==========================================
    print("\n" + "="*85)
    print("🚨 ทดสอบสถานการณ์พิเศษ (EDGE CASES) เพื่อค้นหาจุดอ่อนโมเดล")
    print("="*85)

    edge_cases = []

    # --- Scenario A: ผลกระทบของความชื้น (Humidity Test) ---
    # ลองที่คน 20 คน เคลื่อนที่ปานกลาง อุณหภูมิ 26 องศา (ก้ำกึ่งระหว่าง Normal กับ High)
    oci_mid = (0.5 * (20/40.0)) + (0.3 * 0.3) + (0.2 * 1.0)
    for humi in [40.0, 60.0, 85.0]:
        edge_cases.append({
            'Scenario': f'Humidity Impact ({humi}%)',
            'OCI': round(oci_mid, 4), 'delta_OCI': round(np.random.normal(0, 0.01), 4), 
            'temp_indoor': 26.0, 'humi_indoor': humi
        })

    # --- Scenario B: คนแห่เข้า/ออก กะทันหัน (Delta OCI Test) ---
    # สถานการณ์: อุณหภูมิ 25 องศา, ค่า OCI ปัจจุบันเท่ากันคือ 0.35
    # (ใน Edge case ตรงนี้เรา fix delta_OCI เป็นค่าที่เหวี่ยงแรงๆ เพื่อเทสเฉพาะจุด)
    edge_cases.append({
        'Scenario': 'Sudden Entry (คนแห่เข้ามา)',
        'OCI': 0.35, 'delta_OCI': 0.15,
        'temp_indoor': 25.0, 'humi_indoor': 60.0
    })
    edge_cases.append({
        'Scenario': 'Sudden Exit (คนเพิ่งออกไป)',
        'OCI': 0.35, 'delta_OCI': -0.15,
        'temp_indoor': 25.0, 'humi_indoor': 60.0
    })

    # --- Scenario C: ความขัดแย้งสุดขั้ว (Contradictions) ---
    edge_cases.append({
        'Scenario': 'Empty but HOT (ไม่มีคน แต่ห้องร้อนจัด 35°C)',
        'OCI': 0.0, 'delta_OCI': 0.0, # ไม่มีคน ก็ไม่ควรมี noise การขยับ
        'temp_indoor': 35.0, 'humi_indoor': 60.0
    })
    edge_cases.append({
        'Scenario': 'Full but FREEZING (คนแน่น ขยับเยอะ แต่ห้องหนาว 20°C)',
        'OCI': 0.95, 'delta_OCI': round(np.random.normal(0, 0.015), 4), 
        'temp_indoor': 20.0, 'humi_indoor': 60.0
    })

    # ทำนายผล Edge Cases
    df_edge = pd.DataFrame(edge_cases)
    edge_features = df_edge[['OCI', 'delta_OCI', 'temp_indoor', 'humi_indoor']]
    edge_preds = model.predict(edge_features)
    df_edge['Predicted_Mode'] = [class_names.get(int(p), f"Class {p}") for p in edge_preds]

    # พิมพ์ผลลัพธ์
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df_edge.to_string(index=False))
    print("\n" + "="*85)