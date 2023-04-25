import codecs

def stateTupleToStr(tup):
    return str(tup[0]) + "-" + str(tup[1])

def strToStateTuple(stateStr):
    spl = stateStr.split("-")
    return (int(spl[0]), int(spl[1]))

def policyToJsonFriendly(policies):
    return [{stateTupleToStr(state): list(policy[state]) for state in policy} for policy in policies]

def jsonFriendlyToPolicy(policies):
    # return {strToStateTuple(state): tuple(policy[state]) for state in policy}
    return [{strToStateTuple(state): policy[state] for state in policy} for policy in policies]

def sendMessage(sock, message):
    sock.send(len(message).to_bytes(2, 'big', signed=False))
    sock.send(message.encode())

def receiveMessage(sock):
    data = sock.recv(2)
    # expect = int.from_bytes(data, 'little', signed=False)
    expect = int(codecs.encode(data, 'hex'), 16)
    print("Waiting for", expect, "bytes")

    received = ""
    while len(received) < expect:
        data = sock.recv(1024)
        received += data.decode()
    # print("Received: " + received)
    print("Received", len(received),"bytes")

    return received