import socket

HOST = "0.0.0.0" # Standard loopback interface address (localhost)
PORT = 6666

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
    line = input("Enter something to send: ")

    conn.send(line.encode())

    data = conn.recv(1024)
    received = data
    # received = data.decode()
    print("Received: " + received)

    if line == "exit":
        conn.close()
        s.close()
        break