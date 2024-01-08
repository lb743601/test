from periphery import Serial
import time
# Open /dev/ttyUSB0 with baudrate 115200, and defaults of 8N1, no flow control
serial = Serial("/dev/ttyS0",115200,8)
while True:
    data = bytes([0b11111111])
    serial.write(data)
    time.sleep(0.2)

# Read up to 128 bytes with 500ms timeout
#buf = serial.read(128, 0.5)
#print("read {:d} bytes: _{:s}_".format(len(buf), buf))

serial.close()

