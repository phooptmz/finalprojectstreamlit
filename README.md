# AI Smart Air Conditioning System

ระบบควบคุมเครื่องปรับอากาศอัจฉริยะแบบเรียลไทม์ ทำงานบน Web Dashboard ผ่าน Streamlit ระบบนี้ผสานรวมเทคโนโลยี **Computer Vision (YOLO)** สำหรับตรวจจับบุคคลและการเคลื่อนไหว, **Machine Learning (LightGBM)** สำหรับทำนายอุณหภูมิที่เหมาะสม และเชื่อมต่อกับฮาร์ดแวร์ **Broadlink RM4** เพื่ออ่านค่าเซ็นเซอร์และสั่งการเครื่องปรับอากาศอัตโนมัติ

## Key Features

* **Real-Time Dual-Lens Tracking** ตรวจจับและติดตามบุคคล (Person Tracking) ผ่านกล้อง IP Camera แบบเลนส์คู่ (Top/Bottom) ด้วยโมเดล YOLO (v8 หรือ v11)
* **Occupancy in context index (OCI)** อัลกอริทึมคำนวณดัชนีความสบายในห้องแบบไดนามิก โดยพิจารณาจาก จำนวนคน, ระดับการเคลื่อนไหว และระยะเวลาที่อยู่ในห้อง
* **AI Temperature Prediction** ใช้โมเดล LightGBM วิเคราะห์ค่า OCI ร่วมกับอุณหภูมิและความชื้นปัจจุบัน เพื่อปรับแอร์ไปที่ระดับที่เหมาะสมที่สุด (23°C, 25°C หรือ 28°C)
* **Auto-Close / Auto-Resume** ระบบปิดแอร์อัตโนมัติเมื่อไม่มีคนอยู่ในห้องตามระยะเวลาที่กำหนด (ประหยัดพลังงาน) และเปิดกลับเป็นโหมด Auto ทันทีเมื่อมีคนเดินเข้ามา
* **Hardware Integration** สแกนหาและเชื่อมต่อกับฮาร์ดแวร์ Broadlink ในเครือข่ายอัตโนมัติ เพื่อส่งสัญญาณ IR (Infrared) ไปยังแอร์โดยไม่ต้องใช้มนุษย์ควบคุม

## System Workflow

ระบบมีการทำงานแบบ Multi-threading เพื่อไม่ให้ UI สะดุด โดยแบ่งการทำงานออกเป็นส่วนต่างๆ ดังนี้

1. **Background Device Scanner** สแกนหา IP Camera (พอร์ต RTSP) และ Broadlink RM4 ในเครือข่ายวงแลน (192.168.1.xxx) เมื่อพบอุปกรณ์ ระบบจะทำการเชื่อมต่อและดึงข้อมูลอุณหภูมิ/ความชื้นเริ่มต้นทันที
2. **Video Capture & Processing** รับภาพจากกล้องผ่านโปรโตคอล RTSP โดยใช้ Thread แยกเพื่อลด Latency ภาพจะถูกย่อขนาดและส่งเข้าโมเดล YOLO เพื่อตรวจจับคน พร้อม Custom Tracker ที่จำแนกการเดินข้ามระหว่าง "เลนส์บน" และ "เลนส์ล่าง" ของกล้อง เพื่อไม่ให้เกิดการนับจำนวนคนซ้ำ
3. **OCI Calculation** คำนวณ N (จำนวนคนเฉลี่ย), A (ระยะกระจัด/การเคลื่อนไหว), และ P (การมีอยู่ของคน) นำมาถ่วงน้ำหนักออกมาเป็นค่า **OCI** และ **Delta OCI**
4. **AI Decision Engine** ในโหมด "Auto" ทุกๆ 60 วินาที ระบบจะนำค่า [OCI, Delta_OCI, อุณหภูมิในห้อง, ความชื้นในห้อง] ส่งเข้าโมเดล LightGBM เพื่อทำนายคลาสอุณหภูมิเป้าหมาย (28°C, 25°C, 23°C)
5. **IR Command Execution** ส่งคำสั่งควบคุมผ่าน API ของ Broadlink RM4 ไปยังแอร์ และมีระบบ Auto-Close Delay หากไม่พบคนต่อเนื่องตามเวลาที่กำหนด ระบบจะยิงคำสั่ง Close ทันที

## Prerequisites

* **Python** 3.9 หรือใหม่กว่า
* **Hardware**
  * กล้อง IP Camera ที่รองรับ RTSP (Dual-lens)
  * อุปกรณ์ Broadlink RM4 Pro/Mini
  * เครื่องปรับอากาศที่รองรับการสั่งงานด้วยอินฟราเรด (IR)
* **Model & Data Files**
  * โมเดล YOLO yolov8n.pt หรือ yolo11n.pt (และไฟล์ .onnx หากตั้งค่า USE_ONNX = True)
  * โมเดล AI LightGBM_model.joblib
  * โฟลเดอร์ ircode ที่บรรจุไฟล์คำสั่ง .bin (เช่น 23.bin, 25.bin, close.bin)

## Getting Started

**1. ติดตั้ง Dependencies**
```bash
pip install -r requirements.txt
