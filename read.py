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
            line = s.readline()#.decode()
	    print(line)
        if len(command_queue) > 0:
            for cmd in command_queue:
                ser.write(cmd)

#t = Thread(target=read_serial, args=(ser,))
#t.start()
read_serial(ser)
#ser.write(bytearray([128, 132, 139, 2, 0, 0]))
#ser.write(chr(131))
#ser.close()
