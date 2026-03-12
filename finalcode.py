import streamlit as st
from streamlit import fragment
import cv2
import numpy as np
import time
import threading
import os
import base64
from ultralytics import YOLO
import gc
import broadlink
import socket
import pandas as pd
import joblib
import subprocess
import sys
import shutil
import json
import torch
import collections
from streamlit.runtime.scriptrunner import add_script_run_ctx

# ==================== 0. Configuration & Pi 5 Prep ====================
CONFIG = {
    "IS_RASPBERRY_PI": False,  # เปลี่ยนเป็น True เมื่อจะย้ายไปรันบน Pi
    "USE_ONNX": False,         # เปลี่ยนเป็น True เมื่อแปลงโมเดลเป็น ONNX แล้ว
    "TARGET_FPS": 2,          # ลด FPS ลงเล็กน้อยเพื่อประหยัดทรัพยากรบน Pi
    "TRACK_BUFFER_SEC": 2.0    # เวลาที่จำ ID ไว้แม้คนหายไปชั่วครู่
}

# ==================== 1. Core Logic Classes ====================
class OCICalculator:
    def __init__(self):
        self.N_MAX_CAPACITY = 40.0
        self.A_MAX_MOVE = 50.0
        self.ALPHA_N = 0.2
        self.ALPHA_A = 0.1
        self.BETA_P = 0.8
        self.W_ALPHA = 0.5
        self.W_BETA = 0.3
        self.W_GAMMA = 0.2

        self.ema_n = 0.0
        self.ema_a = 0.0
        self.prev_p = 0.0
        self.prev_oci = 0.0
        self.prev_centroids = {}

    def update(self, person_count, current_centroids):
        # --- 1. คำนวณ A(t) ตามสูตรใหม่ ---
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
            "N_norm": n_norm,
            "A_norm": a_norm,
            "P_norm": p_norm,
            "OCI": oci,
            "Delta_OCI": delta_oci,
            "Raw_Move": sum_dist_pixels,
            "Raw_People": person_count,
            "Smoothed_People": int(round(self.ema_n))
        }

class VideoCaptureThread:
    def __init__(self, src):
        self.src = src
        self.lock = threading.Lock()
        self.frame = None
        self.ret = False
        self.running = True
        self.reconnecting = False
        self.error_count = 0
        self.reconnect_delay = 1.0
        
        self.capture = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while self.running:
            if self.capture.isOpened():
                try:
                    grabbed = self.capture.grab()
                    if grabbed:
                        for _ in range(4): 
                            self.capture.grab()
                            
                        ret, frame = self.capture.retrieve()
                        if ret and frame is not None and frame.size > 0:
                            with self.lock:
                                self.ret = True
                                self.frame = frame
                            self.error_count = 0 
                        else:
                            self.error_count += 1
                    else:
                        self.error_count += 1
                    
                    if self.error_count > 50:
                        print("Too many bad frames, reconnecting...")
                        self._reconnect()
                        self.error_count = 0
                except Exception as e:
                    print(f"Error in thread: {e}")
                    self.error_count += 1
            else:
                self._reconnect()
                time.sleep(0.01)

    def _reconnect(self):
        if self.reconnecting:
            return
        self.reconnecting = True
        with self.lock:
            self.ret = False
            self.frame = None
        
        if self.capture:
            self.capture.release()
        
        print(f"Reconnecting to {self.src}...")
        time.sleep(self.reconnect_delay)
        self.reconnect_delay = min(self.reconnect_delay * 1.5, 10.0) 
        
        self.capture = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.reconnecting = False

    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.ret, self.frame.copy()
            return False, None

    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.capture.release()

class BackgroundSensorUpdater:
    def __init__(self):
        self.temperature_indoor = None
        self.humidity_indoor = None
        self.last_update = None
        self.running = False
        self.thread = None
        self.DEVTYPE = 0x520c
        self.DEVICE_MAC = None
        self.DEVICE_IP = None
        self.device = None

    def set_device_config(self, ip, mac, devtype):
        self.DEVICE_IP = ip
        self.DEVICE_MAC = mac
        self.DEVTYPE = devtype

    def start(self, interval=1):
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, args=(interval,), daemon=True)
        add_script_run_ctx(self.thread)
        self.thread.start()

    def stop(self):
        self.running = False

    def _update_loop(self, interval):
        while self.running:
            self.update_indoor()
            self.last_update = time.time()
            time.sleep(interval)

    def update_indoor(self):
        if not self.DEVICE_IP or not self.DEVICE_MAC:
            return
            
        try:
            with BROADLINK_LOCK:
                if self.device is None:
                    self.device = broadlink.rm4(host=(self.DEVICE_IP, 80), mac=self.DEVICE_MAC, devtype=self.DEVTYPE)
                    self.device.auth()

                sensors = self.device.check_sensors()
                if sensors:
                    self.temperature_indoor = sensors.get("temperature")
                    self.humidity_indoor = sensors.get("humidity")
        except Exception as e:
            self.device = None


# ==================== 2. Global Config & Constants ====================
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;udp|"
    "fflags;nobuffer+discardcorrupt|"
    "flags;low_delay|"
    "strict;experimental|"
    "reorder_queue_size;0|"
    "max_delay;100000|"
    "error_resilience;1"
)

MODEL_CONFIG = {
    "USE_YOLO_V11": True,
    "USE_ONNX": True,
}

if MODEL_CONFIG["USE_YOLO_V11"]:
    if MODEL_CONFIG["USE_ONNX"]:
        YOLO_MODEL_PATH = "yolo11n.onnx"
    else:
        YOLO_MODEL_PATH = "yolo11n.pt"
else:
    if MODEL_CONFIG["USE_ONNX"]:
        YOLO_MODEL_PATH = "yolov8n.onnx"
    else:
        YOLO_MODEL_PATH = "yolov8n.pt"

RF_MODEL_PATH = "LightGBM_model.joblib"
CONFIDENCE_THRESHOLD = 0.35 
TARGET_WIDTH = 640
TARGET_HEIGHT_TOTAL = 720
IR_CODE_DIR = r"/home/npu/Downloads/streamlit/ircode"
CAMERA_USER = "admin"
CAMERA_PASS = "2601zaza"
DEVTYPE = 0x520c
DEVICE_MAC = bytes.fromhex("E87072ABC3E9")
BROADLINK_LOCK = threading.Lock()

def format_uptime(start_time):
    if start_time is None:
        return "00:00:00"
    elapsed = int(time.time() - start_time)
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def get_scan_time_str():
    if not st.session_state.get('scan_completed', False):
        cam_ok = st.session_state.get('camera_connected', False)
        rm4_ok = st.session_state.get('rm4_connected', False)
        
        # คำนวณเวลาที่ใช้ไป
        st.session_state.scan_elapsed_time = time.time() - st.session_state.scan_start_time
        
        # ถ้าเชื่อมต่อครบทั้งคู่ ให้ล็อกสถานะเพื่อหยุดเวลา
        if cam_ok and rm4_ok:
            st.session_state.scan_completed = True
            
    # ส่งคืนค่าเป็น String ทศนิยม 1 ตำแหน่ง
    return f"{st.session_state.scan_elapsed_time:.1f}s"

# ==================== 3. Helper Functions ====================
def update_metrics_display(main_ph, sidebar_ph):
    with main_ph.container():
        col1, col2, col3, col4, col5 , col6 = st.columns(6)
        t_in = sensor.temperature_indoor
        h_in = sensor.humidity_indoor
        temp_disp = f"{t_in:.1f}°C" if t_in else "--"
        humi_disp = f"{h_in:.1f}%" if h_in else "--"
        
        with col1:
            clean_metric("People Detected", st.session_state.get('smoothed_person_count', 0), "Active Tracking", "")
        with col2:
            clean_metric("Room Temp", temp_disp, f"Humidity: {humi_disp}", "")
        with col3:
            status = "RUNNING" if st.session_state.get('run_detection', False) else "STANDBY"
            clean_metric("System Status", status, "AI Ready" if model else "Loading...", "")
        with col4:
            ai_decision = st.session_state.get('ai_decision', 'Waiting...')
            ai_class = st.session_state.get('ai_class', 'Auto Mode')
            clean_metric("AI Decision", ai_decision, ai_class, "")
        with col5:
            oci_val = st.session_state.get('oci_metrics', {}).get('OCI', 0.0)
            clean_metric("Activity (OCI)", f"{oci_val:.2f}", "Composite Index", "")
        
        with col6:
            uptime_str = format_uptime(st.session_state.get('system_start_time'))
            clean_metric("Uptime", uptime_str, "HH:MM:SS", "")
        
        with st.expander("View Technical Details"):
            metrics = st.session_state.get('oci_metrics', {})
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.write(f"N (People): {metrics.get('N_norm', 0.0):.2f}")
                st.write(f"A (Move): {metrics.get('A_norm', 0.0):.2f}")
            with m_col2:
                st.write(f"P (Presence): {metrics.get('P_norm', 0.0):.2f}")
                st.write(f"Delta OCI: {metrics.get('Delta_OCI', 0.0):.3f}")

    with sidebar_ph.container():
        col_s1, col_s2 = st.columns(2)

        scan_time = get_scan_time_str()

        with col_s1:
            cam_status = ":green[ON]" if st.session_state.get('camera_connected', False) else ":red[OFF]"
            st.markdown(f"**Cam:** {cam_status} , {scan_time}")
        with col_s2:
            rm4_status = ":green[ON]" if st.session_state.get('rm4_connected', False) else ":red[OFF]"
            st.markdown(f"**RM4:** {rm4_status} , {scan_time}")

@st.cache_resource
def load_yolo_model(model_path):
    try:
        if model_path.endswith('.onnx'):
            model = YOLO(model_path, task='detect')
        else:
            model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading YOLO: {e}")
        return None

@st.cache_resource
def load_rf_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        return None

@st.cache_resource
def load_ir_codes():
    codes = {}
    mapping = {
        20: "20c.bin", 21: "21c.bin", 22: "22.bin", 23: "23.bin",
        24: "24.bin", 25: "25.bin", 26: "26.bin", 27: "27.bin", 28: "28.bin",
        "Close": "close.bin"
    }
    for key, filename in mapping.items():
        path = os.path.join(IR_CODE_DIR, filename)
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    codes[key] = f.read()
            except:
                pass
    return codes

AC_IR_CODES = load_ir_codes()

def send_ac_command(ip, mac, command_key):
    if not ip or not mac:
        return False, "No Broadlink connection"
        
    if isinstance(command_key, float):
        command_key = int(round(command_key))
        
    ir_data = AC_IR_CODES.get(command_key)
    if ir_data is None:
        print(f"DEBUG: No IR code found for key {command_key}")
        return False, f"No IR code for: {command_key}"
        
    try: 
        with BROADLINK_LOCK:
            updater = st.session_state.get('sensor_updater')
            if updater and updater.device:
                # ดึง Device ที่ Auth ไว้แล้วมาสั่งงาน
                updater.device.send_data(ir_data)
                pass
            else:
                device = broadlink.rm4(host=(ip, 80), mac=mac, devtype=DEVTYPE)
                device.auth()
                device.send_data(ir_data)
                pass
            
                
            print(f"DEBUG: Successfully sent IR command for {command_key}")
            return True, "Success"
            
    except Exception as e:
        error_msg = f"Broadlink Error: {str(e)}"
        print(f"DEBUG: {error_msg}")
        return False, error_msg
    
def process_frame_with_custom_tracker(frame, model, prev_tracks=None):
    if 'id_map' not in st.session_state:
        st.session_state.id_map = {}
    if 'lost_ids' not in st.session_state:
        st.session_state.lost_ids = {}
    if 'last_seen' not in st.session_state:
        st.session_state.last_seen = {}

    results = model.track(
        frame,
        persist=True,
        tracker="my_tracker.yaml", 
        conf=0.35,
        iou=0.45,
        imgsz=640,
        verbose=False,
        classes=[0]
    )
    
    annotated_frame = frame.copy()
    person_count = 0
    current_centroids = {}
    current_time = time.time()
    active_raw_ids = set()

    if results and results[0].boxes and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()
        
        for box, raw_id in zip(boxes, track_ids):
            active_raw_ids.add(raw_id)
            x1, y1, x2, y2 = map(int, box[:4])
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            
            if raw_id not in st.session_state.id_map:
                matched_display_id = None
                if cx < 320 and 360 <= cy < 470:
                    for d_id, data in list(st.session_state.lost_ids.items()):
                        if data['zone'] == 'top_right' and (current_time - data['time']) < 3.0:
                            matched_display_id = d_id
                            break
                elif cx >= 320 and 250 < cy <= 360:
                    for d_id, data in list(st.session_state.lost_ids.items()):
                        if data['zone'] == 'bot_left' and (current_time - data['time']) < 3.0:
                            matched_display_id = d_id
                            break

                if matched_display_id is not None:
                    st.session_state.id_map[raw_id] = matched_display_id
                    del st.session_state.lost_ids[matched_display_id]
                else:
                    st.session_state.id_map[raw_id] = raw_id
            
            display_id = st.session_state.id_map[raw_id]
            current_centroids[display_id] = (cx, cy)
            st.session_state.last_seen[display_id] = {'cx': cx, 'cy': cy, 'time': current_time}
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"ID:{display_id}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            person_count += 1

    active_display_ids = {st.session_state.id_map[r_id] for r_id in active_raw_ids if r_id in st.session_state.id_map}
    
    for d_id, data in list(st.session_state.last_seen.items()):
        if d_id not in active_display_ids:
            cx, cy = data['cx'], data['cy']
            if cx >= 320 and 250 < cy <= 360:
                st.session_state.lost_ids[d_id] = {'zone': 'top_right', 'time': current_time}
            elif cx < 320 and 360 <= cy < 470:
                st.session_state.lost_ids[d_id] = {'zone': 'bot_left', 'time': current_time}
            del st.session_state.last_seen[d_id]

    for d_id in list(st.session_state.lost_ids.keys()):
        if current_time - st.session_state.lost_ids[d_id]['time'] > 3.0:
            del st.session_state.lost_ids[d_id]

    return annotated_frame, person_count, 0.0, current_centroids

def rtsp_works(url, timeout=0.2):
    cap = cv2.VideoCapture(url)
    t0 = time.time()
    ok = False
    while time.time() - t0 < timeout:
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                ok = True
                break
        time.sleep(0.05)
    cap.release()
    return ok

def find_broadlink_in_range(subnet_prefix, start_host, end_host, devtype, mac, timeout=0.1):
    original_timeout = socket.getdefaulttimeout()
    for i in range(int(start_host), int(end_host) + 1):
        ip = f"{subnet_prefix}{i}"
        try:
            socket.setdefaulttimeout(timeout)
            device = broadlink.rm4(host=(ip, 80), mac=mac, devtype=devtype)
            device.auth()
            socket.setdefaulttimeout(original_timeout)
            return ip, device.mac
        except:
            pass
    socket.setdefaulttimeout(original_timeout)
    return None

# ==================== 4. UI Layout & CSS ====================
st.set_page_config(
    page_title="Smart AC Control",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.block-container { padding-top: 1rem; padding-bottom: 2rem; }
header { visibility: hidden; }
.metric-card {
    background-color: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 15px;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    backdrop-filter: blur(5px);
    margin-bottom: 10px;
}
.metric-label { font-size: 0.9rem; color: #aaaaaa; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 1px;}
.metric-value { font-size: 2.0rem; font-weight: 700; color: #ffffff; }
.metric-sub { font-size: 0.8rem; color: #22c55e; margin-top: 4px;}
section[data-testid="stSidebar"] {
    background-color: rgba(17, 17, 17, 0.95);
    border-right: 1px solid rgba(255, 255, 255, 0.1);
}
.stButton > button {
    border-radius: 8px;
    height: 3em;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

def clean_metric(label, value, sub="", icon=""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{icon} {label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

# ==================== 5. State & Initialization ====================
if 'yolo_model' not in st.session_state or st.session_state.yolo_model is None:
    st.session_state.yolo_model = load_yolo_model(YOLO_MODEL_PATH)
if 'rf_model' not in st.session_state or st.session_state.rf_model is None:
    st.session_state.rf_model = load_rf_model(RF_MODEL_PATH)
if 'sensor_updater' not in st.session_state:
    st.session_state.sensor_updater = BackgroundSensorUpdater()
    st.session_state.sensor_updater.start(interval=1)
sensor = st.session_state.sensor_updater
if 'oci_calculator' not in st.session_state:
    st.session_state.oci_calculator = OCICalculator()


st.session_state.oci_metrics = {}
if 'stream_url' not in st.session_state:
    st.session_state.stream_url = None
if 'no_person_start_ts' not in st.session_state:
    st.session_state.no_person_start_ts = None
if 'no_person_required_secs' not in st.session_state:
    st.session_state.no_person_required_secs = 30
if "run_detection" not in st.session_state:
    st.session_state.run_detection = False
if "person_count" not in st.session_state:
    st.session_state.person_count = 0
if 'camera_connected' not in st.session_state:
    st.session_state.camera_connected = False
if 'rm4_connected' not in st.session_state:
    st.session_state.rm4_connected = False
if 'camera_ip' not in st.session_state:
    st.session_state.camera_ip = None
if 'rm4_ip' not in st.session_state:
    st.session_state.rm4_ip = None
if 'last_sent_cmd' not in st.session_state:
    st.session_state.last_sent_cmd = None
if st.session_state.get("_force_close_target_temp", False):
    st.session_state["target_temp_selector"] = "Close"
    st.session_state["_force_close_target_temp"] = False
if st.session_state.get("_force_auto_target_temp", False):
    st.session_state["target_temp_selector"] = "Auto"
    st.session_state["_force_auto_target_temp"] = False
if 'last_metrics_update' not in st.session_state:
    st.session_state.last_metrics_update = 0
if 'smoothed_person_count' not in st.session_state:
    st.session_state.smoothed_person_count = 0
if 'system_start_time' not in st.session_state:
    st.session_state.system_start_time = None
if 'scan_start_time' not in st.session_state:
    st.session_state.scan_start_time = time.time()
if 'scan_elapsed_time' not in st.session_state:
    st.session_state.scan_elapsed_time = 0.0
if 'scan_completed' not in st.session_state:
    st.session_state.scan_completed = False

model = st.session_state.yolo_model
rf_model = st.session_state.rf_model

def background_device_scanner():
    while True:
        found_cam_ip = None
        subnet = "192.168.1."
        for i in range(100, 115):
            ip = f"{subnet}{i}"
            try:
                s = socket.create_connection((ip, 554), timeout=0.2)
                s.close()
                if rtsp_works(f"rtsp://{CAMERA_USER}:{CAMERA_PASS}@{ip}:554/onvif2", timeout=0.2):
                    found_cam_ip = ip
                    break
            except:
                continue
        
        if found_cam_ip:
            st.session_state.camera_ip = found_cam_ip
            st.session_state.stream_url = f"rtsp://{CAMERA_USER}:{CAMERA_PASS}@{found_cam_ip}:554/onvif1"
            st.session_state.camera_connected = True
        else:
            st.session_state.camera_connected = False
        
        if not st.session_state.get('rm4_connected'):
            res = find_broadlink_in_range("192.168.1.", 100, 115, DEVTYPE, DEVICE_MAC, timeout=0.05)
            if res:
                found_ip, found_mac = res
                st.session_state.rm4_ip = found_ip
                sensor.set_device_config(found_ip, found_mac, DEVTYPE)
                st.session_state.rm4_connected = True
            else:
                st.session_state.rm4_connected = False
        
        time.sleep(1)

if "background_scanner_started" not in st.session_state:
    st.session_state.background_scanner_started = True
    scanner_thread = threading.Thread(target=background_device_scanner, daemon=True)
    add_script_run_ctx(scanner_thread)
    scanner_thread.start()


# ==================== 6. Sidebar (Control Center) ====================
with st.sidebar:
    st.markdown("## Control Center")
    sidebar_status_placeholder = st.empty()

    st.divider()

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        start_click = st.button("START", type="primary", width='stretch')
    with col_btn2:
        stop_click = st.button("STOP", width='stretch')

    st.markdown("### Temperature Control")
    temp_options = ["Auto"] + ["Close"] + [f"{i}°C" for i in range(20, 29)]
    target_temp = st.selectbox("Set Target:", options=temp_options, index=0, key="target_temp_selector", label_visibility="collapsed")

    if target_temp != "Auto" and target_temp != "Close":
        try:
            manual_temp_int = int(target_temp.replace("°C", ""))
            if st.session_state.get('rm4_connected') and st.session_state.get('last_sent_cmd') != manual_temp_int:
                ok, msg = send_ac_command(st.session_state.rm4_ip, DEVICE_MAC, manual_temp_int)
                if ok:
                    st.toast(f"Manual: Set {manual_temp_int}°C")
                    st.session_state.last_sent_cmd = manual_temp_int
        except:
            pass
    elif target_temp == "Close":
        if st.session_state.get('rm4_connected') and st.session_state.get('last_sent_cmd') != "Close":
            ok, msg = send_ac_command(st.session_state.rm4_ip, DEVICE_MAC, "Close")
            if ok:
                st.toast("Manual: Close AC")
                st.session_state.last_sent_cmd = "Close"

    with st.expander("Advanced Settings"):
        st.session_state.no_person_required_secs = st.number_input(
            "Auto Close Delay (Secs)", 
            min_value=10, 
            max_value=28800, 
            value=st.session_state.no_person_required_secs, 
            step=10
        )
        st.info("System is running in Direct RTSP Mode.")


# ==================== 7. Main Dashboard ====================
st.markdown("### AI Smart Air Condition System")
metrics_placeholder = st.empty()
prediction_placeholder = st.empty()
update_metrics_display(metrics_placeholder, sidebar_status_placeholder)

@st.fragment(run_every=1.0)
def idle_status_updater():
    with sidebar_status_placeholder.container():

        scan_time = get_scan_time_str()

        col_s1, col_s2 = st.columns(2)
        cam_status = ":green[ON]" if st.session_state.get('camera_connected', False) else ":red[OFF]"
        col_s1.markdown(f"Cam:  {cam_status} , {scan_time}")
        rm4_status = ":green[ON]" if st.session_state.get('rm4_connected', False) else ":red[OFF]"
        col_s2.markdown(f"RM4:  {rm4_status} , {scan_time}")

st.divider()
st.markdown("##### Live Monitor")
col_cam1, col_cam2 = st.columns(2)
with col_cam1:
    frame_placeholder_top = st.empty()
with col_cam2:
    frame_placeholder_bottom = st.empty()
    
movement_placeholder = st.empty()

if not st.session_state.run_detection:
    idle_status_updater()
    frame_placeholder_top.info("Click Start to view feed")


# ==================== 8. Processing Loop ====================
if start_click:
    if model and rf_model:
        if st.session_state.camera_connected and st.session_state.stream_url:
            st.session_state.run_detection = True
            st.session_state.person_count = 0
            st.session_state.oci_calculator = OCICalculator()
            st.session_state.system_start_time = time.time()
            st.rerun()
        else:
            st.warning("รอสักครู่... ระบบกำลังสแกนหา IP กล้องในเครือข่าย")

if stop_click:
    st.session_state.run_detection = False
    st.session_state.system_start_time = None
    st.rerun()

if model and rf_model and st.session_state.run_detection:
    stream_url = st.session_state.stream_url
    
    if 'cap_thread' not in st.session_state or st.session_state.cap_thread is None:
        st.session_state.cap_thread = VideoCaptureThread(stream_url)
        time.sleep(1.5)

    cap = st.session_state.cap_thread

    last_process_time = 0
    last_ai_time = 0
    last_gc_time = time.time()
    last_metrics_update_time = time.time()

    GC_INTERVAL = 10
    AI_INTERVAL = 60
    MIN_FRAME_INTERVAL = 1.0 / CONFIG["TARGET_FPS"] 
    UI_UPDATE_INTERVAL = 1 # older parameter is 1

    frame_count = 0
    last_frame_time = time.time() 
    current_fps = 0.0             

    try:
        while st.session_state.run_detection:
            frame_top_display = None
            frame_bottom_display = None
            buffer_top = None
            buffer_bot = None
            input_df = None
            
            current_time = time.time()
            
            if (current_time - last_process_time) < MIN_FRAME_INTERVAL:
                time.sleep(0.001)
                continue
            
            last_process_time = current_time
            
            ret, frame = cap.read()
            
            if not ret or frame is None:
                time.sleep(0.01)
                continue
            
            if frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
                continue
            
            try:
                resized_frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT_TOTAL), interpolation=cv2.INTER_LINEAR)
            except Exception as e:
                continue
            
            try:
                annotated_frame_full, person_count, coverage, current_centroids = \
                    process_frame_with_custom_tracker(resized_frame, model)
                
                now = time.time()
                if (now - last_frame_time) > 0:
                    current_fps = 1.0 / (now - last_frame_time)
                else:
                    current_fps = 0.0
                last_frame_time = now

            except Exception as e:
                print(f"Tracking error: {e}")
                continue
            
            oci_result = st.session_state.oci_calculator.update(person_count, current_centroids)
            st.session_state.oci_metrics = oci_result

            smoothed_count = oci_result["Smoothed_People"]
            
            st.session_state.person_count = smoothed_count
            st.session_state.smoothed_person_count = smoothed_count
            
            raw_move_pixels = oci_result.get('Raw_Move', 0.0)
            
            
            # Auto Close Logic
            if smoothed_count > 0:
                if st.session_state.target_temp_selector == "Close":
                    st.toast("Detect Person: Auto Resume")
                    st.session_state["_force_auto_target_temp"] = True
                    st.session_state.last_sent_cmd = None
                    st.session_state.no_person_start_ts = None
                    st.rerun()
                st.session_state.no_person_start_ts = None
            else:
                if st.session_state.no_person_start_ts is None:
                    st.session_state.no_person_start_ts = current_time
                else:
                    elapsed = current_time - st.session_state.no_person_start_ts
                    if elapsed >= st.session_state.no_person_required_secs:
                        if st.session_state.rm4_connected and st.session_state.last_sent_cmd != "Close":
                            send_ac_command(st.session_state.rm4_ip, DEVICE_MAC, "Close")
                            st.toast("No Person: Auto Close")
                            st.session_state.last_sent_cmd = "Close"
                            st.session_state["_force_close_target_temp"] = True
                            st.session_state.no_person_start_ts = None
                            st.rerun()
            
            # AI Prediction Logic
            if st.session_state.target_temp_selector == "Auto" and smoothed_count > 0:
                if (current_time - last_ai_time) >= AI_INTERVAL:
                    t_in = sensor.temperature_indoor
                    h_in = sensor.humidity_indoor
                    if t_in and h_in:
                        oci_val = oci_result.get('OCI', 0.0)
                        delta_oci = oci_result.get('Delta_OCI', 0.0)
                        input_df = pd.DataFrame([[oci_val, delta_oci, t_in, h_in]],
                                                columns=['OCI', 'delta_OCI', 'temp_indoor', 'humi_indoor'])
                        
                        pred_class = rf_model.predict(input_df.values)[0]
                        class_map = {0: 28, 1: 25, 2: 23}
                        target_t = class_map.get(pred_class, 25)

                        st.session_state.ai_decision = f"{target_t}°C"
                        st.session_state.ai_class = f"Class: {pred_class}"
                        
                        if st.session_state.rm4_connected and st.session_state.last_sent_cmd != target_t:
                            send_ac_command(st.session_state.rm4_ip, DEVICE_MAC, target_t)
                            st.toast(f"AI Adjust: {target_t}°C")
                            st.session_state.last_sent_cmd = target_t
                        
                        last_ai_time = current_time
            
            # Display Update
            frame_count += 1
            if frame_count % UI_UPDATE_INTERVAL == 0:
                with movement_placeholder.container():
                    st.metric("Movement (Pixels)", f"{raw_move_pixels:.1f} px", delta=None)
                frame_top_display = annotated_frame_full[0:360, :]
                frame_bottom_display = annotated_frame_full[360:720, :]
                
                if frame_top_display is not None and frame_top_display.size > 0:
                    cv2.putText(frame_top_display, f"P: {smoothed_count}", (10, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame_top_display, f"FPS: {current_fps:.1f}", (10, 80),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    
                    success, buffer_top = cv2.imencode('.jpg', frame_top_display, [cv2.IMWRITE_JPEG_QUALITY, 60]) # older quality is 70
                    if success:
                        frame_placeholder_top.image(buffer_top.tobytes(),
                                                   caption="Top lens",
                                                   width='stretch')
                
                if frame_bottom_display is not None and frame_bottom_display.size > 0:
                    success_bot, buffer_bot = cv2.imencode('.jpg', frame_bottom_display, [cv2.IMWRITE_JPEG_QUALITY, 60]) # older quality is 70 
                    if success_bot:
                        frame_placeholder_bottom.image(buffer_bot.tobytes(),
                                                      caption="Bottom lens",
                                                      width='stretch')
            if current_time - last_metrics_update_time >= 1.0:
                update_metrics_display(metrics_placeholder, sidebar_status_placeholder) 
                last_metrics_update_time = current_time
            
            if (current_time - last_gc_time) >= GC_INTERVAL:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                last_gc_time = current_time

            if frame_count % 10 == 0:
                time.sleep(0.001) 

    except Exception as e:
        st.error(f"Stream Error: {e}")
    finally:
        if st.session_state.cap_thread:
            st.session_state.cap_thread.release()
            st.session_state.cap_thread = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()