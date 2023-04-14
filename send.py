import serial
from threading import Thread

port = '/dev/ttyUSB0'
baud = 57600

ser = serial.Serial(port = port, baudrate = baud)

command_queue = []

def read_serial(s):
    print("Reading from " + s.name)
    while True:
        if s.inWaiting() > 0:
            line = s.readline().decode()
	    print(line)
        if len(command_queue) > 0:
            for cmd in command_queue:
                ser.write(cmd)


def clamp(val, low, high):
    if val > high:
        return high
    if val < low:
        return low
    return val


def speedToByteStr(speed_val):
    return int16ToByteStr(clamp(speed_val, -500, 500))
def radiusToByteStr(radius_val):
    return int16ToByteStr(clamp(radius_val, -2000, 2000))


def int16ToByteStr(val):
    
    first_byte = 0
    second_byte = 0

    if val < 0:
        val = 2**16 + val
    #    first_byte = 255 # FF
    #    second_byte = speed_val - (255 << 8)
    #else: 
    first_byte = val >> 8
    second_byte = val - (first_byte << 8)
    
    return str(first_byte) + " " + str(second_byte)
        

speed = 100
radius = 500

while True:
    line = raw_input("Enter serial command or 'quit': ")
    if line == 'q' or line == 'quit' or line == 'exit':
        break
    parts = line.split(" ")
    cmd = line
    if line == "init":
        cmd = "128 131"
    if parts[0] == "speed" or parts[0] == "sp":
        speed = int(parts[1])
        print("Set speed to " + str(speed) + " millimeters/s") 
        continue
    if parts[0] == "radius" or parts[0] == "r":
        radius = int(parts[1])
        print("Set radius to " + str(radius) + " millimeters") 
        continue

    if line == "stop" or line == "s":
        cmd = "137 0 0 0 0"
    if line == "forward" or line == "f":
        cmd = "137 " + speedToByteStr(speed) + " 128 0" # 128 means straight
    if line == "back" or line == "b":
        cmd = "137 " + speedToByteStr(-speed) + " 128 0"
    if line == "right" or line == "r":
        cmd = "137 " + speedToByteStr(speed) + " 255 255" # 0xFFFF in last 2 mean turn in place clockwise
    if line == "left" or line == "l":
        cmd = "137 " + speedToByteStr(speed) + " 0 1" # 0x0001 means turn in place counterclockwise
    if line == "fr":
        cmd = "137 " + speedToByteStr(speed) + " " + radiusToByteStr(-radius)
    if line == "fl":
        cmd = "137 " + speedToByteStr(speed) + " " + radiusToByteStr(radius)
    if line == "br":
        cmd = "137 " + speedToByteStr(-speed) + " " + radiusToByteStr(-radius)
    if line == "bl":
        cmd = "137 " + speedToByteStr(-speed) + " " + radiusToByteStr(radius)
    
    parts = cmd.split(" ")
    print("  Parsing " + cmd)
    try:
        cmd_bytes = bytearray([int(p) for p in parts])
    except Exception,e:
        print(str(e))
        print("  Error: " + p + " is not a number")
        continue
    print("  Sending " + cmd)
    ser.write(cmd_bytes)
    
#t = Thread(target=read_serial, args=(ser,))
#t.start()
#read_serial(ser)
#ser.write(bytearray([128, 132, 139, 2, 0, 0]))
#ser.write(chr(128))
#ser.write(bytearray([128, 131]))
#ser.write(chr(131))
ser.close()
