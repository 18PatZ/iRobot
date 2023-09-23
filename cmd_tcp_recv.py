import serial
import socket

serial_port = '/dev/ttyUSB0'
baud = 57600

HOST = "0.0.0.0" # Standard loopback interface address (localhost)
PORT = 6666


ser = serial.Serial(port = serial_port, baudrate = baud)
print("Serial connection open.")


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

s.bind((HOST, PORT))
print("Server bound to " + HOST + ":" + str(PORT))
s.listen()
print("Listening for inbound connections")
conn, addr = s.accept()

#with conn:
print("Connection established from " + str(addr))
while True:
    data = conn.recv(1)

    if len(data) == 0:
        raise RuntimeError("Socket connection broken")

    cmd_len = data[0]
    
    print("Reading {cmd_len}-byte command...")

    cmd = conn.recv(cmd_len)

    if len(data) == 0:
        raise RuntimeError("Socket connection broken")

    print("  Command received: ", list(cmd))
    ser.write(cmd)

conn.close()
s.close()
ser.close()
