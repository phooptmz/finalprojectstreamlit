import broadlink
import time
DEVICE_IP = "192.168.1.243"
DEVICE_MAC = bytes.fromhex("E87072ABC3E9")
DEVTYPE = 0x520c
def learn_ir_code(filename="ir_code.bin"):
    try:
        print("กำลังเชื่อมต่ออุปกรณ์ ...")
        device = broadlink.rm4(host=(DEVICE_IP, 80), mac=DEVICE_MAC, devtype=DEVTYPE)
        device.auth()
        print("🟡 กรุณากดปุ่มบนรีโมทที่ต้องการเรียนรู้ (ภายใน 5 วินาที)... : ", filename)
        device.enter_learning()
        time.sleep(5)  # ให้เวลากดปุ่มบนรีโมท
        print("⏳ กำลังตรวจสอบว่ามี IR code หรือไม่ ...")
        ir_code = device.check_data()
        if ir_code:
            print("✅ ได้ IR code แล้ว:", ir_code.hex())
            with open(filename, "wb") as f:
                f.write(ir_code)
            print(f"บันทึกไฟล์เป็น: {filename}")
        else:
            print("❌ ไม่พบ IR code ที่เรียนรู้ กรุณาลองใหม่")
    except Exception as e:
        print("เกิดข้อผิดพลาด:", e)
if __name__ == "__main__":
    learn_ir_code("low fan.bin")
