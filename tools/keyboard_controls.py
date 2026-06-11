import serial
import time

PORT = "/dev/ttyUSB0"   # maybe /dev/ttyACM0 on Linux, COM3 on Windows
BAUD = 115200

ser = serial.Serial(PORT, BAUD, timeout=0.1)
time.sleep(2)

def send(cmd):
    print(">", cmd)
    ser.write((cmd + "\n").encode())

send("ENABLE")

try:
    while True:
        key = input("Command w/a/s/d/x/q: ").strip()

        if key == "w":
            send("CMD_VEL 0.25 0.0")
        elif key == "s":
            send("CMD_VEL -0.25 0.0")
        elif key == "a":
            send("CMD_VEL 0.0 0.25")
        elif key == "d":
            send("CMD_VEL 0.0 -0.25")
        elif key == "x":
            send("STOP")
        elif key == "q":
            send("DISABLE")
            break
        elif key == "state":
            send("GET_STATE")
        else:
            print("unknown command")

        time.sleep(0.05)

finally:
    send("DISABLE")
    ser.close()