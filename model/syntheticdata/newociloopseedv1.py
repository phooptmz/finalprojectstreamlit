import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import time
import os

# ==================== OCICalculator Class ====================
class OCICalculator:
    def __init__(self):
        self.N_MAX_CAPACITY = 40.0
        self.A_MAX_MOVE = 50.0
        self.ALPHA_N = 0.2
        self.ALPHA_A = 0.1
        self.BETA_P = 0.5
        self.W_ALPHA = 0.5
        self.W_BETA = 0.3
        self.W_GAMMA = 0.2
        
        self.ema_n = 0.0
        self.ema_a = 0.0
        self.prev_p = 0.0
        self.prev_oci = 0.0
        self.prev_centroids = {}

    def update(self, person_count, current_centroids):
        # --- 1. คำนวณ A(t) ---
        sum_dist_pixels = 0.0
        for pid, (cx, cy) in current_centroids.items():
            if pid in self.prev_centroids:
                px, py = self.prev_centroids[pid]
                dist = np.sqrt((cx - px)**2 + (cy - py)**2)
                sum_dist_pixels += dist
        self.prev_centroids = current_centroids

        if person_count > 0:
            avg_dist = sum_dist_pixels / person_count
            a_raw = avg_dist / self.A_MAX_MOVE
        else:
            a_raw = 0.0
        a_raw = min(a_raw, 1.0)
        self.ema_a = (self.ALPHA_A * a_raw) + ((1 - self.ALPHA_A) * self.ema_a)
        a_norm = self.ema_a

        # --- 2. คำนวณ N(t) ---
        self.ema_n = (self.ALPHA_N * person_count) + ((1 - self.ALPHA_N) * self.ema_n)
        n_norm = min(self.ema_n / self.N_MAX_CAPACITY, 1.0)

        # --- 3. คำนวณ P(t) ---
        i_t = 1.0 if person_count > 0 else 0.0
        p_t = (self.BETA_P * self.prev_p) + ((1 - self.BETA_P) * i_t)
        self.prev_p = p_t
        p_norm = min(p_t, 1.0)

        # --- 4. คำนวณ OCI ---
        oci = (self.W_ALPHA * n_norm) + (self.W_BETA * a_norm) + (self.W_GAMMA * p_norm)
        delta_oci = oci - self.prev_oci
        self.prev_oci = oci

        return {
            "OCI": oci,
            "Delta_OCI": delta_oci,
            "Raw_Move": sum_dist_pixels,
            "Raw_People": person_count
        }


# ==================== ✅ Class Assignment ด้วย OCI และอุณหภูมิ ====================
def assign_class_with_hysteresis(oci, current_temp, prev_class, 
                                  min_time_in_mode=120,
                                  current_time=0,
                                  last_change_time=0):
    
    # --- 1. หาค่าโหมด "อุดมคติ" (Ideal Target) ที่ควรจะเป็น ณ วินาทีนี้ ---
    if current_temp <= 24.0:
        recommended_class = 0    # ❄️ ถ้าห้องเย็น 24 องศาหรือต่ำกว่า บังคับโหมด Eco เสมอ
    else:
        if oci < 0.28:
            recommended_class = 0    # Eco (0)
        elif oci < 0.45:
            recommended_class = 1    # Normal (1)
        else:
            recommended_class = 2    # High (2)
    
    # --- 2. Hysteresis Logic (หน่วงเวลาสำหรับการทำงานของคอมเพรสเซอร์แอร์) ---
    time_since_last_change = current_time - last_change_time
    
    if recommended_class != prev_class:
        if time_since_last_change >= min_time_in_mode:
            final_class = recommended_class
            new_change_time = current_time
        else:
            final_class = prev_class
            new_change_time = last_change_time
    else:
        final_class = recommended_class
        new_change_time = last_change_time
    
    # ส่งคืนทั้งโหมดอุดมคติ (ใช้สอน AI) และโหมดจริง (ใช้คุมอุณหภูมิ)
    return recommended_class, final_class, new_change_time


# ==================== Data Generation Function ====================
def generate_ac_dataset(seed, n_frames=5000, time_step=60):
    np.random.seed(seed)
    
    # --- 1. จำลองจำนวนคนและ Movement ---
    time_array = np.arange(n_frames)
    daily_pattern = 0.5 + 0.5 * np.sin(2 * np.pi * time_array / (n_frames / 3))
    noise = np.random.normal(0, 0.3, n_frames)
    
    person_count_raw = (daily_pattern + noise) * 35  
    person_count = np.clip(person_count_raw, 0, 40).astype(int)
    
    base_movement_per_person = np.random.uniform(5, 50, n_frames)
    activity_noise = np.random.normal(1, 0.3, n_frames)
    movement_per_person = np.clip(base_movement_per_person * activity_noise, 0, 100)
    
    # --- 2. คำนวณ OCI ---
    oci_calc = OCICalculator()
    oci_values = []
    delta_oci_values = []
    prev_centroids = {}
    
    for frame_idx in range(n_frames):
        current_person_count = person_count[frame_idx]
        current_centroids = {}
        
        for person_id in range(current_person_count):
            if person_id in prev_centroids:
                px, py = prev_centroids[person_id]
                move_dist = movement_per_person[frame_idx]
                angle = np.random.uniform(0, 2 * np.pi)
                new_x = px + move_dist * np.cos(angle)
                new_y = py + move_dist * np.sin(angle)
                new_x = np.clip(new_x, 0, 640)
                new_y = np.clip(new_y, 0, 320)
                current_centroids[person_id] = (new_x, new_y)
            else:
                current_centroids[person_id] = (
                    np.random.uniform(0, 640),
                    np.random.uniform(0, 320)
                )
        
        result = oci_calc.update(current_person_count, current_centroids)
        oci_values.append(result['OCI'])
        delta_oci_values.append(result['Delta_OCI'])
        prev_centroids = current_centroids
    
    oci_final = np.array(oci_values)
    delta_oci = np.array(delta_oci_values)
    
    # --- 3. การตั้งค่าสภาพแวดล้อมทางกายภาพ & จำลองการทำงานร่วมกัน ---
    room_area_sqm = 64.0    
    room_height_m = 2.8     
    room_volume = room_area_sqm * room_height_m 
    ac_btu = 48000.0        
    
    outdoor_temp = 35.0     
    outdoor_humi = 75.0     
    current_temp = 32.0     # อุณหภูมิเริ่มต้น
    current_humi = 65.0
    
    k_ac_cool_max = (ac_btu / 12000.0) * (84.0 / room_volume) * 1.5 
    k_ac_dry_max = 0.20 * (ac_btu / 12000.0) 
    k_insulation = 0.03 * (room_area_sqm / 30.0) 
    k_people_heat = 0.50 * (84.0 / room_volume) 
    k_people_humi = 0.15 * (84.0 / room_volume)
    k_humi_env = 0.02 
    
    target_temp_config = { 0: 27.0, 1: 24.0, 2: 23 }
    
    labels = []
    temp_data_final = []
    humi_data_final = []
    
    current_class = 1
    last_change_time = 0
    
    # วนลูปเฟรมเพื่อประเมินสถานการณ์แบบวินาทีต่อวินาที
    for i in range(n_frames):
        oci = oci_final[i]
        current_time = i * time_step
        
        # 3.1 ตัดสินใจเลือกโหมดแอร์ โดยดูอุณหภูมิปัจจุบันประกอบด้วย
        ideal_class, final_class, last_change_time = assign_class_with_hysteresis(
            oci=oci,
            current_temp=current_temp,
            prev_class=current_class,
            min_time_in_mode=120, 
            current_time=current_time,
            last_change_time=last_change_time
        )
        
        # ⚠️ ให้ AI เรียนรู้จากสิ่งที่ "ควรจะเป็น" (ideal_class) แบบไม่มีดีเลย์
        labels.append(ideal_class) 
        
        # แต่คอมเพรสเซอร์แอร์ในซิมูเลเตอร์ จะทำงานตาม final_class (ติดดีเลย์ Hysteresis)
        current_class = final_class  
        
        # 3.2 แอร์ทำงานตามโหมดที่ผ่านการหน่วงเวลาแล้ว (current_class)
        target_temp = target_temp_config[current_class]
        heat_gain = (k_insulation * (outdoor_temp - current_temp)) + (k_people_heat * oci)
        temp_diff = current_temp - target_temp
        
        if temp_diff > 0:
            cooling_power = min(temp_diff * 0.4, 1.0) 
        else:
            cooling_power = 0.1 
            
        cooling = k_ac_cool_max * cooling_power
        
        current_temp += (heat_gain - cooling)
        current_temp += np.random.normal(0, 0.03) 
        
        humi_gain = (k_humi_env * (outdoor_humi - current_humi)) + (k_people_humi * oci)
        drying = k_ac_dry_max * cooling_power
        current_humi += (humi_gain - drying)
        current_humi += np.random.normal(0, 0.1) 
        current_humi = np.clip(current_humi, 40, 80)
        
        temp_data_final.append(current_temp)
        humi_data_final.append(current_humi)

    # --- 4. สร้าง DataFrame ตาม Features ที่ต้องการ ---
    df_sim = pd.DataFrame({
        'OCI': np.round(oci_final, 4),
        'delta_OCI': np.round(delta_oci, 4),
        'temp_indoor': np.round(temp_data_final, 2),
        'humi_indoor': np.round(humi_data_final, 2),
        'ac_mode_class': labels
    })
    
    return df_sim


# ==================== Main Execution ====================
if __name__ == "__main__":
    seeds = range(58, 68)  # 10 seeds = ประมาณ 50,000 แถว
    print(f"กำลังรันการจำลองด้วย seed ตั้งแต่ {seeds.start} ถึง {seeds.stop-1}...")
    print("กราฟแต่ละ seed จะแสดง 2 วินาทีก่อนบันทึกไฟล์...\n")
    
    all_dfs = []
    
    # สร้างโฟลเดอร์สำหรับเก็บไฟล์ CSV ถ้ายากรวมให้เป็นระเบียบ
    output_dir = "combined_data"
    os.makedirs(output_dir, exist_ok=True)
    
    for seed in seeds:
        print(f"กำลังรัน seed = {seed}...", end=" ")
        
        df = generate_ac_dataset(seed, n_frames=5000, time_step=60)
        all_dfs.append(df)
        
        # Plot กราฟ
        plt.figure(figsize=(14, 10))
        
        plt.subplot(4, 1, 1)
        plt.plot(df['OCI'], label='OCI', color='red', linewidth=2)
        plt.title(f'Seed {seed}: OCI Value')
        plt.ylabel('OCI')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(4, 1, 2)
        plt.plot(df['temp_indoor'], label='Temp', color='blue', alpha=0.8)
        plt.title(f'Seed {seed}: Room Temperature Response')
        plt.ylabel('Temperature (°C)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(4, 1, 3)
        plt.plot(df['ac_mode_class'], label='Ideal AC Class (AI Label)', color='purple', drawstyle='steps-pre', linewidth=1.5)
        plt.title(f'Seed {seed}: AC Mode Classification (Based on Rules)')
        plt.ylabel('AC Mode')
        plt.yticks([0, 1, 2], ['Eco', 'Normal', 'High'])
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(4, 1, 4)
        class_counts = df['ac_mode_class'].value_counts().sort_index()
        counts_for_plot = [class_counts.get(0, 0), class_counts.get(1, 0), class_counts.get(2, 0)]
        plt.bar([0, 1, 2], counts_for_plot, color=['green', 'cyan', 'red'])
        plt.title(f'Seed {seed}: Class Distribution')
        plt.ylabel('Count')
        plt.xticks([0, 1, 2], ['Eco', 'Normal', 'High'])
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(2.0)
        plt.close()
        
        filename = f'{output_dir}/ac_training_seed{seed}_oci_only.csv'
        df.to_csv(filename, index=False)
        print(f"✓ บันทึก '{filename}' ({len(df)} แถว)")
        
        dist = df['ac_mode_class'].value_counts().sort_index()
        print(f"  Class Dist: Eco={dist.get(0,0)}, Norm={dist.get(1,0)}, High={dist.get(2,0)}")
    
    # รวมทุกไฟล์
    print("\n🔗 กำลังรวมข้อมูลทั้งหมด...")
    combined_df = pd.concat(all_dfs, ignore_index=True)

    print("🔀 กำลังสลับข้อมูล (Shuffle)...")
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    combined_filename = f'{output_dir}/ac_training_combined_oci_only.csv'
    combined_df.to_csv(combined_filename, index=False)
    print(f"✓ บันทึกไฟล์รวม '{combined_filename}' ({len(combined_df)} แถว)")
    
    # แสดงสรุป
    print("\n📊 Class Distribution (รวมทุก seed):")
    total_dist = combined_df['ac_mode_class'].value_counts().sort_index()
    total_pct = (total_dist / len(combined_df) * 100).round(2)
    for cls, cnt in total_dist.items():
        print(f"  Class {cls}: {cnt} แถว ({total_pct.get(cls, 0)}%)")
    
    print("\n📋 Features ใน Dataset:")
    print(f"  {list(combined_df.columns)}")
    print("\n✅ สร้างข้อมูลเรียบร้อย! สามารถนำไฟล์ไปเทรนโมเดลต่อได้เลยครับ")