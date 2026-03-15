import broadlink
import time
DEVICE_IP = "192.168.1.243"            # ← เปลี่ยนเป็น IP ของคุณ
DEVICE_MAC = bytes.fromhex("E87072ABC3E9")  # ← เปลี่ยนเป็น MAC ของคุณ
DEVTYPE = 0x520c  # ส่วนมาก RM4 mini ใช้ค่านี้ ถ้า error เปลี่ยน devtype
def send_ir_command(filename):
    with open(filename, "rb") as f:
        ir_code = f.read()
    try:
        device = broadlink.rm4(host=(DEVICE_IP, 80), mac=DEVICE_MAC, devtype=DEVTYPE)
        device.auth()
        device.send_data(ir_code)
        print(f"✅ ส่ง IR code จาก {filename} เรียบร้อย")
    except Exception as e:
        print("❌ ส่ง IR ไม่สำเร็จ:", e)
if __name__ == "__main__":
    while True:
        send_ir_command("fast fan.bin")

