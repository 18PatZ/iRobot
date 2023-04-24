import socket

from psomdp.mdp import *


def makeMDP(grid, discount = math.sqrt(0.99)):
    goalActionReward = 10000
    noopReward = 0#-1
    wallPenalty = -300000
    movePenalty = -1

    moveProb = 0.9

    start_state = (1, 2)

    mdp = createMDP(grid, goalActionReward, noopReward, wallPenalty, movePenalty, moveProb)
    return grid, mdp, discount, start_state


# grid, mdp, discount, start_state = paper2An(3)#corridorTwoCadence(n1=3, n2=6, cadence1=2, cadence2=3)



grid = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2], [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
grid, mdp, discount, start_state = makeMDP(grid)


discount_checkin = discount

target_state = findTargetState(grid)

checkin_period = 2
schedule = [checkin_period]

checkin_periods = [checkin_period]

scaling_factor = 9.69/1.47e6 # y / x
midpoints = [0.2, 0.4, 0.6, 0.8]
midpoints = [getAdjustedAlphaValue(m, scaling_factor) for m in midpoints]

# compMDP = createCompositeMDP(mdp, discount, checkin_period)
# discount_t = pow(discount, checkin_period)

# policy, values = linearProgrammingSolve(grid, compMDP, discount_t, restricted_action_set = None)

sched, compMDPs, greedyCompMDPs = solveSchedulePolicies(grid, mdp, discount, discount_checkin, start_state, target_state, 
    checkin_periods, schedule, midpoints)

policy = sched.policies_exec[-1]
values = sched.pi_exec_data[0]

k = schedule[-1]
compMDP = compMDPs[k]
# print(policy)

draw(grid, compMDP, values, policy, True, False, "output/policy-"+str(k)+"-lp")

# HOST = "0.0.0.0" # Standard loopback interface address (localhost)
# PORT = 6666

# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# s.bind((HOST, PORT))
# print("Server bound to " + HOST + ":" + str(PORT))
# s.listen()
# print("Listening for inbound connections")
# conn, addr = s.accept()

# #with conn:
# print("Connection established from " + str(addr))

# while True:
#     line = input("Enter something to send: ")

#     conn.send(line.encode())

#     data = conn.recv(1024)
#     received = data
#     # received = data.decode()
#     print("Received: " + received)

#     if line == "exit":
#         conn.close()
#         s.close()
#         break