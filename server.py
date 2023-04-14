import socket

HOST = "127.0.0.1" # Standard loopback interface address (localhost)
PORT = 6666

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

s.bind((HOST, PORT))
print("Server bound to " + HOST + ":" + str(PORT))
s.listen()
print("Listening for inbound connections")
conn, addr = s.accept()

#with conn:
print(f"Connection established from {addr}")
while True:
    data = conn.recv(1024)
    # if not data:
    #     break
    received = data.decode()
    print("Received: " + received)
    conn.send(("ping " + received).encode())

    if received == "exit":
        conn.close()
        s.close()
        break