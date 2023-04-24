import socket
from threading import Thread
import time
import json
from socketUtil import *

HOST = "0.0.0.0" # bind to all interfaces
PORT = 6666

PLANNER_HOST = "127.0.0.1"
PLANNER_PORT = 6667

send_grid = False
grid = None
plan = None


def plannerInterface():
    global send_grid
    global grid
    global plan

    planner = socket.socket()
    print("Connecting to planner at " + PLANNER_HOST + ":" + str(PLANNER_PORT) + "...")
    planner.connect((PLANNER_HOST, PLANNER_PORT))
    print("Connected!")

    while not send_grid:
        time.sleep(1)

    send_grid = False

    grid_json = json.dumps(grid)

    print("Sending " + str(len(grid)) + "x" + str(len(grid[0])) + " grid to planner...")
    
    planner.send("PLAN".encode())
    sendMessage(planner, grid_json)

    print("Sent! Waiting for plan...")
    plan_json = receiveMessage(planner)
    plan = json.loads(plan_json)

    #to_send = {"Policy": policyToJsonFriendly(policy), "Schedule": schedule}
    policy = jsonFriendlyToPolicy(plan["Policy"])
    schedule = plan["Schedule"]
    print("Received policy:", policy)
    print("Received schedule:", schedule)
    




thread = Thread(target=plannerInterface, args=[])
thread.start()


### dummy for when camera is ready
grid = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2], [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
send_grid = True 
###



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

thread.join()