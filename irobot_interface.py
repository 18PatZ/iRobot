import serial
from threading import Thread
import socket

NOTES = {
    "C": 60,
    "D": 62,
    "E": 64,
    "F": 65,
    "G": 67,
    "A": 69
}

class iRobotInterface:

    def __init__(self, 
            use_serial = True, serial_port = '/dev/ttyUSB0', baud = 57600, 
            use_TCP = False, tcp_ip = '10.228.106.75', tcp_port = 6666):

        self.command_queue = []

        self.use_serial = use_serial
        self.use_TCP = use_TCP
        
        if use_serial:
            self.ser = serial.Serial(port = serial_port, baudrate = baud)

        if use_TCP:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            
            print(f"Connecting to TCP {tcp_ip}:{tcp_port}...")
            self.sock.connect((tcp_ip, tcp_port))

            print("Connected to TCP receiver.")

        print("Ready to send commands.")



    def read_serial(self, s):
        print("Reading from " + s.name)
        while True:
            if s.inWaiting() > 0:
                line = s.readline().decode()
            print(line)
            if len(self.command_queue) > 0:
                for cmd in self.command_queue:
                    self.ser.write(cmd)


    def write_commands(self):
        for command in self.command_queue:
            cmd = bytearray(command)

            if self.use_serial:
                self.ser.write(cmd)
            
            if self.use_TCP:
                # transmit byte with length of command (in bytes)
                if self.sock.send(bytearray([len(cmd)])) == 0:
                    raise RuntimeError("Socket connection broken")
                if self.sock.send(cmd) == 0:
                    raise RuntimeError("Socket connection broken")

        del self.command_queue[:]


    def send_command(self, command, write_queue=True, isStr=False):
        cmd = command
        if isStr:
            parts = cmd.split(" ")
            try:
                cmd = [int(p) for p in parts]
            except Exception as e:
                print(str(e))
                print("  Error: " + p + " is not a number")
        
        self.command_queue.append(cmd)
        if write_queue:
            self.write_commands()


    def close(self):
        self.ser.close()




    def clamp(self, val, low, high):
        if val > high:
            return high
        if val < low:
            return low
        return val


    def int16ToBytes(self, val):
        
        first_byte = 0
        second_byte = 0

        if val < 0:
            val = 2**16 + val
        #    first_byte = 255 # FF
        #    second_byte = speed_val - (255 << 8)
        #else: 
        first_byte = val >> 8
        second_byte = val - (first_byte << 8)
        
        return first_byte, second_byte


    def speedToBytes(self, speed_val):
        return self.int16ToBytes(int(self.clamp(speed_val, -500, 500)))
    def radiusToBytes(self, radius_val):
        return self.int16ToBytes(int(self.clamp(radius_val, -2000, 2000)))


    def start_mode(self):
        # Opcode 128: Start
        self.send_command([128])

    def passive_mode(self):
        # Opcode 128: Start, changes any mode to passive

        # Upon sending the Start command or any one of the demo
        # commands (which also starts the specific demo, e.g., Spot
        # Cover, Cover, Cover and Dock, or Demo), the OI enters
        # into Passive mode. When the OI is in Passive mode, you
        # can request and receive sensor data using any of the
        # sensors commands, but you cannot change the current
        # command parameters for the actuators (motors, speaker,
        # lights, low side drivers, digital outputs) to something else.
        # To change how one of the actuators operates, you must
        # switch from Passive mode to Full mode or Safe mode.
        # While in Passive mode, you can read Create’s sensors,
        # watch Create perform any one of its ten built-in demos,
        # and charge the battery.
        print("Passive mode")
        self.send_command([128])

    def safe_mode(self):
        # Opcode 131: Safe
        # This command puts the OI into Safe mode, enabling user
        # control of Create. It turns off all LEDs. The OI can be in
        # Passive, Safe, or Full mode to accept this command.

        # When you send a Safe command to the OI, Create enters into
        # Safe mode. Safe mode gives you full control of Create, with
        # the exception of the following safety-related conditions:
        # • Detection of a cliff while moving forward (or moving
        # backward with a small turning radius, less than one robot
        # radius).
        # • Detection of a wheel drop (on any wheel).
        # • Charger plugged in and powered.
        # Should one of the above safety-related conditions occur
        # while the OI is in Safe mode, Create stops all motors and
        # reverts to the Passive mode.
        # If no commands are sent to the OI when in Safe mode, Create
        # waits with all motors and LEDs off and does not respond to
        # Play or Advance button presses or other sensor input.
        # Note that charging terminates when you enter Safe Mode.
        print("Safe mode")
        self.send_command([131])

    def full_mode(self):
        # Opcode 132: Full
        # This command gives you complete control over Create
        # by putting the OI into Full mode, and turning off the cliff,
        # wheel-drop and internal charger safety features. That is, in
        # Full mode, Create executes any command that you send
        # it, even if the internal charger is plugged in, or the robot
        # senses a cliff or wheel drop.

        # When you send a Full command to the OI, Create enters
        # into Full mode. Full mode gives you complete control over
        # Create, all of its actuators, and all of the safety-related
        # conditions that are restricted when the OI is in Safe mode,
        # as Full mode shuts off the cliff, wheel-drop and internal
        # charger safety features. To put the OI back into Safe mode,
        # you must send the Safe command.
        # If no commands are sent to the OI when in Full mode, Create
        # waits with all motors and LEDs off and does not respond to
        # Play or Advance button presses or other sensor input.
        # Note that charging terminates when you enter Full Mode.
        print("Full mode")
        self.send_command([132])

    def drive(self, speed = 100, turn_radius = None):
        # Drive takes four data bytes, interpreted as two 16-bit signed values using two’s complement. 
        # The first two bytes specify the average
        # velocity of the drive wheels in millimeters per second
        # (mm/s), with the high byte being sent first. 
        # The next two bytes specify the radius in millimeters at which Create will
        # turn. The longer radii make Create drive straighter, while
        # the shorter radii make Create turn more. The radius is
        # measured from the center of the turning circle to the center
        # of Create. A Drive command with a positive velocity and a
        # positive radius makes Create drive forward while turning
        # toward the left. A negative radius makes Create turn toward
        # the right. Special cases for the radius make Create turn
        # in place or drive straight, as specified below. 
        # A negative velocity makes Create drive backward.

        # Velocity: -500 to 500 mm/s
        # Radius: -2000 to 2000 mm

        # Straight = 32768 or 32767 = hex 8000 or 7FFF

        print("Drive " + str(speed) + ("" if turn_radius is None else (" " + str(turn_radius))))
        # Opcode 137: Drive
        velocity_high_byte, velocity_low_byte = self.speedToBytes(speed)
        if turn_radius is None:
            # Straight = 32768 or 32767 = hex 8000 or 7FFF indicates straight motion
            self.send_command([137, velocity_high_byte, velocity_low_byte, 128, 0])
        else:
            # positive radius goes left for some reason, flip to right
            radius_high_byte, radius_low_byte = self.radiusToBytes(-turn_radius)
            self.send_command([137, velocity_high_byte, velocity_low_byte, radius_high_byte, radius_low_byte])

    def stop(self):
        print("Stop")
        self.send_command([137, 0, 0, 0, 0])

    def turn(self, speed = 100):
        print("Turn " + str(speed))
        # Opcode 137: Drive
        # Turn in place clockwise = 0xFFFF
        # Turn in place counter-clockwise = 0x0001
        # Works with negative speed as well

        velocity_high_byte, velocity_low_byte = self.speedToBytes(speed)
        radius_high_byte, radius_low_byte = (255, 255)
        self.send_command([137, velocity_high_byte, velocity_low_byte, radius_high_byte, radius_low_byte])

    def register_beep(self, id, notes):
        print("Register beep", id)
        # Opcode 140: Song
        cmd = [140, id, len(notes)]
        for note in notes:
            cmd.append(note[0], note[1])
        self.send_command(cmd)

    def register_beeps(self):
        print("Register beeps")
        # Opcode 140: Song
        self.send_command([140, 0, 4, 60, 12, 64, 12, 67, 12, 72, 36])
        #self.send_command([140, 1, 2, 72, 12, 72, 36])
        self.send_command([140, 1, 2, 55, 12, 55, 36])
        self.send_command([140, 2, 2, 60, 12, 60, 36])
        self.send_command([140, 3, 1, 64, 36])
        self.send_command([140, 4, 2, 67, 12, 67, 36])
        self.send_command([140, 5, 2, 72, 12, 72, 36])
        self.send_command([140, 6, 5, 60, 12, 64, 12, 67, 12, 72, 12, 72, 36])
        self.send_command([140, 7, 7,
            NOTES["C"], 12, NOTES["C"], 12, 
            NOTES["G"], 12, NOTES["G"], 12, 
            NOTES["A"], 12, NOTES["A"], 12, 
            NOTES["G"], 36])
        #self.send_command([140 0 4 62 12 66 12 69 12 74 36])

    def beep(self, num):
        print("Beep " + str(num))
        # Opcode 141: Play Song
        # Need to register with Opcode 140 first!
        self.send_command([141, num])