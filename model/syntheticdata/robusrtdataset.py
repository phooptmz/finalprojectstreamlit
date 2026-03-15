import numpy as np
import pandas as pd
import os

def generate_robust_dataset(n_samples=100000):
    """
    สร้างชุดข้อมูลจำลองแบบสุ่มอิสระ (Independent Monte Carlo Simulation)
    กฎ: 
    1. OCI น้อย -> Eco (0) เสมอ
    2. OCI มาก + ร้อน/ชื้น -> High (2)
    3. กรณีอื่นๆ รวมถึงคนเยอะแต่หนาว -> Normal (1) เลี้ยงอุณหภูมิ
    """
    np.random.seed(42)
    print(f"⚙️ กำลังสร้างชุดข้อมูล Robust Dataset จำนวน {n_samples:,} แถว (สุ่มแบบปรับสมดุล)...")

    # 1. ปรับการสุ่มจำนวนคน ให้มีโอกาสเจอ "ห้องว่าง" หรือ "คนน้อย" เยอะขึ้น
    p_rand = np.random.rand(n_samples)
    person_counts = np.zeros(n_samples, dtype=int)
    
    mask_low = (p_rand > 0.35) & (p_rand <= 0.65)
    mask_high = p_rand > 0.65
    person_counts[mask_low] = np.random.randint(1, 11, size=mask_low.sum())
    person_counts[mask_high] = np.random.randint(11, 41, size=mask_high.sum())
    
    # 2. ปรับการสุ่มการเคลื่อนไหว (Activity)
    a_norms = np.random.uniform(0.0, 1.0, n_samples)
    a_norms = np.where(person_counts == 0, 0.0, a_norms)
    
    # 3. สุ่มอุณหภูมิและความชื้น 
    temp_indoor = np.random.uniform(20.0, 36.0, n_samples)
    humi_indoor = np.random.uniform(45.0, 80.0, n_samples)
    
    delta_oci = np.where(person_counts == 0, 
                         np.random.normal(0, 0.002, n_samples), 
                         np.random.normal(0, 0.02, n_samples))

    # 4. คำนวณค่า OCI
    n_norms = person_counts / 40.0
    p_norms = np.where(person_counts > 0, 1.0, 0.0)
    oci_values = (0.5 * n_norms) + (0.3 * a_norms) + (0.2 * p_norms)

    # 5. กำหนดเฉลย (Labels) แบบ 3 Classes แบบเด็ดขาด
    labels = []
    for i in range(n_samples):
        oci = oci_values[i]
        temp = temp_indoor[i]
        humi = humi_indoor[i]
        
        # กำหนดระดับ OCI
        if oci <= 0.30: 
            oci_level = 'น้อย'
        elif oci <= 0.55: 
            oci_level = 'กลาง'
        else: 
            oci_level = 'มาก'
            
        # --- กฎการจำแนก 3 ระดับ ---
        
        # 📌 กฎข้อ 1: ถ้า OCI น้อย (คน 0 หรือคนน้อยมาก) ให้เป็น Eco เสมอ (ไม่สนร้อน/ชื้น)
        if oci_level == 'น้อย':
            labels.append(0)
            
        # 📌 กฎข้อ 2: ถ้า OCI มาก (คนเยอะ) และ ห้องร้อน/ชื้นจัด ให้เป็น High
        elif oci_level == 'มาก' and (temp > 26.0 or humi > 70.0):
            labels.append(2)
            
        # 📌 กฎข้อ 3: กรณีที่เหลือให้เลี้ยงแอร์ไว้ที่ Normal 
        # (รวมถึงกรณี OCI มาก แต่อากาศหนาว <=24 ด้วย ก็จะตกที่ Class 1 นี้)
        else:
            labels.append(1)

    # 6. สร้าง DataFrame สำหรับ Training (ตั้งทศนิยม 2 ตำแหน่ง)
    df = pd.DataFrame({
        'Person_Count': person_counts, 
        'OCI': np.round(oci_values, 2),
        'delta_OCI': np.round(delta_oci, 2),
        'temp_indoor': np.round(temp_indoor, 2),
        'humi_indoor': np.round(humi_indoor, 2),
        'ac_mode_class': labels
    })

    return df

if __name__ == "__main__":
    df_train = generate_robust_dataset(n_samples=100000)
    
    print("\n📊 สัดส่วนโหมดแอร์ที่ AI จะได้เรียนรู้ (3 Classes - Balanced):")
    dist = df_train['ac_mode_class'].value_counts().sort_index()
    class_names = {
        0: 'Standby / Eco+ (28°C)',
        1: 'Normal (25°C)',
        2: 'High / Rapid Cool (23°C)'
    }
    for cls, count in dist.items():
        pct = (count / len(df_train)) * 100
        print(f"  Class {cls} ({class_names[cls]}): {count:,} แถว ({pct:.1f}%)")

    # ตัดคอลัมน์ Person_Count ออก
    df_for_training = df_train[['OCI', 'delta_OCI', 'temp_indoor', 'humi_indoor', 'ac_mode_class']]

    output_dir = "combined_data"
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"{output_dir}/ac_training_comfort_balanced.csv"
    df_for_training.to_csv(filename, index=False)
    
    print(f"\n✅ สร้างและบันทึกไฟล์ '{filename}' สำเร็จแล้ว!")