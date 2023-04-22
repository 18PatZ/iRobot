import socket

HOST = "10.224.112.177" # Standard loopback interface address (localhost)
PORT = 6666

s = socket.socket()
print("Connecting to " + HOST + ":" + str(PORT) + "...")
s.connect((HOST, PORT))
print("Connected!")

while True:
    line = input("Enter something to send: ")

    s.send(line.encode())

    data = s.recv(1024)
    received = data.decode()
    print("Received: " + received)

    if line == "exit":
        s.close()
        break
