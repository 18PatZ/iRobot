import socket
import json
from threading import Thread

from psomdp.mdp import *
from socketUtil import *

do_exit = False

def makeMDP(grid, discount = math.sqrt(0.99)):
    goalActionReward = 10000
    noopReward = 0#-1
    wallPenalty = -300000
    movePenalty = -1

    moveProb = 0.9

    start_state = (1, 2)

    mdp = createMDP(grid, goalActionReward, noopReward, wallPenalty, movePenalty, moveProb)
    return grid, mdp, discount, start_state


def planForGrid(grid):
    grid, mdp, discount, start_state = makeMDP(grid)


    discount_checkin = discount

    target_state = findTargetState(grid)

    checkin_period = 3
    schedule = [2, checkin_period]

    checkin_periods = [2, checkin_period]

    scaling_factor = 9.69/1.47e6 # y / x
    midpoints = [0.2, 0.4, 0.6, 0.8]
    midpoints = [getAdjustedAlphaValue(m, scaling_factor) for m in midpoints]

    # compMDP = createCompositeMDP(mdp, discount, checkin_period)
    # discount_t = pow(discount, checkin_period)

    # policy, values = linearProgrammingSolve(grid, compMDP, discount_t, restricted_action_set = None)

    sched, compMDPs, greedyCompMDPs = solveSchedulePolicies(grid, mdp, discount, discount_checkin, start_state, target_state, 
        checkin_periods, schedule, midpoints)

    policies = sched.policies_exec#[-1]
    values = sched.pi_exec_data[0]

    k = schedule[-1]
    compMDP = compMDPs[k]
    # print(policy)

    return policies, sched.strides



# grid, mdp, discount, start_state = paper2An(3)#corridorTwoCadence(n1=3, n2=6, cadence1=2, cadence2=3)
# grid = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2], [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]


listener = None
plan = None

def sendPlan(conn, plan):
    (policy, schedule) = plan
    to_send = {"Policy": policyToJsonFriendly(policy), "Schedule": schedule}
    to_send_json = json.dumps(to_send)

    sendMessage(conn, to_send_json)

    print("Sent plan!")


def handleConnection(conn):
    global listener
    global do_exit
    global plan
    # draw(grid, compMDP, values, policy, True, False, "output/policy-"+str(k)+"-lp")

    while True:
        # conn.send("READY".encode())

        data = conn.recv(4)
        received = data.decode()

        if received == "SUBS":
            listener = lambda plan: sendPlan(conn, plan)
            print("Agent is waiting for plan...")

            if plan is not None:
                listener(plan)


        elif received == "PLAN":
            print("Observer is sending grid...")
            grid_json = receiveMessage(conn)
            grid = json.loads(grid_json)
            print(f"Received {len(grid)}x{len(grid[0])} grid from observer:", grid)
            print("Planning...")

            plan = planForGrid(grid)

            print("Plan complete.")
            print("Policy:", plan[0])
            print("Schedule:", plan[1])

            sendPlan(conn, plan)

            if listener is not None:
                listener(plan)

            

        # received = data.decode()
        # print("Received: " + received)

        if received == "exit":
            conn.close()
            do_exit = True
            return


HOST = "0.0.0.0" # Standard loopback interface address (localhost)
PORT = 6667

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

s.bind((HOST, PORT))
print("Server bound to " + HOST + ":" + str(PORT))
s.listen()
print("Listening for inbound connections")



conn, addr = s.accept()
print("Connection 1 established from " + str(addr))

thread = Thread(target=handleConnection, args=(conn,))
thread.start()

conn2, addr2 = s.accept()
print("Connection 2 established from " + str(addr2))

handleConnection(conn2)

thread.join()

s.close()
