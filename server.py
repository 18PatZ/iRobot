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
    data = conn.recv(1024)
    # if not data:
    #     break
    received = data.decode()
    print("Received: " + received)

    to_send = ""
    for i in range(5000):
        to_send += received
    # conn.send(("ping " + received).encode())
    print("sending back", len(to_send),"bytes")
    conn.send(len(to_send).to_bytes(2, 'little', signed=False))
    conn.send(to_send.encode())

    if received == "exit":
        conn.close()
        s.close()
        break
