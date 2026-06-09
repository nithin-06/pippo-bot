import socket

HOST = "0.0.0.0"
PORT = 5000

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server.bind((HOST, PORT))

server.listen(1)

print("Waiting for robot connection...")

conn, addr = server.accept()

print("Robot connected:", addr)

while True:

    print("\nCommands:")
    print("1 Forward")
    print("2 Back")
    print("3 Left")
    print("4 Right")
    print("5 Stop")
    print("6 Red LED")
    print("7 Servo")

    choice = input("Enter: ")

    if choice == "1":
        cmd = "CMD_MOTOR#2000#2000#\n"

    elif choice == "2":
        cmd = "CMD_MOTOR#-2000#-2000#\n"

    elif choice == "3":
        cmd = "CMD_MOTOR#-1500#1500#\n"

    elif choice == "4":
        cmd = "CMD_MOTOR#1500#-1500#\n"

    elif choice == "5":
        cmd = "CMD_MOTOR#0#0#\n"

    elif choice == "6":
        cmd = "CMD_LED#255#0#0#\n"

    elif choice == "7":
        cmd = "CMD_SERVO#0#90#\n"

    else:
        continue

    conn.send((cmd + "\n").encode())