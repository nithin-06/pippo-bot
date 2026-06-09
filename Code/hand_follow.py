"""
hand_following_robot.py  —  Runs on the Raspberry Pi  (CLIENT)
===============================================================
Architecture (FLIPPED from original Freenove):
  Pi  → connects TO laptop on port 5003  (CMD channel, Pi receives commands)
  Pi  → connects TO laptop on port 8003  (VIDEO channel, Pi sends frames)

The Pi handles ONLY hardware:
  • Streams JPEG frames to laptop
  • Receives CMD_MOTOR commands and drives motors
  • No vision / no AI — all of that runs on the laptop

Usage:
    python hand_following_robot.py <LAPTOP_IP>
    python hand_following_robot.py 10.42.0.1
"""

from __future__ import annotations   # allows  X | Y  type hints on Python 3.7–3.9

import socket
import struct
import threading
import time
import sys

# ── Hardware (all available on the Pi)
from camera  import Camera
from motor   import tankMotor
from message import MessageParser

# ── Command constants
# The Server folder contains  command.py  (lowercase) → class Command (instance attrs)
# The Client folder contains  Command.py  (uppercase) → class COMMAND (class attrs)
# This block handles both so the file works from either location.
try:
    from Command import COMMAND          # Client-side: Command.py
    CMD_MOTOR = COMMAND.CMD_MOTOR
    CMD_LED   = COMMAND.CMD_LED
except ModuleNotFoundError:
    try:
        from command import Command      # Server-side: command.py
        _c        = Command()
        CMD_MOTOR = _c.CMD_MOTOR
        CMD_LED   = _c.CMD_LED
    except ModuleNotFoundError:
        # Hardcoded fallback — never wrong regardless of which file is present
        CMD_MOTOR = "CMD_MOTOR"
        CMD_LED   = "CMD_LED"


# ─────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────
DEFAULT_LAPTOP_IP = "10.42.0.1"   # ← change to your laptop's IP
CMD_PORT          = 5003           # laptop listens here for commands
VIDEO_PORT        = 8003           # laptop listens here for video


# ─────────────────────────────────────────────────────────────
#  HandFollowRobot
# ─────────────────────────────────────────────────────────────

class HandFollowRobot:
    """
    Raspberry Pi side of the hand-following system.
    Connects to the laptop server, streams video, and obeys motor commands.
    """

    def __init__(self, laptop_ip: str):
        self.laptop_ip = laptop_ip
        self.running   = False

        print("[ROBOT] Initialising hardware…")
        self.motor   = tankMotor()
        self.camera  = Camera(stream_size=(400, 300))   # match original Main.py
        self.parser  = MessageParser()

        self.cmd_sock   = None
        self.video_sock = None

        # Safety: motors stop at startup
        self.motor.setMotorModel(0, 0)

    # ──────────────────────────────────────────
    #  Connection
    # ──────────────────────────────────────────

    def _connect(self, retries: int = 15) -> bool:
        """
        Try to connect both sockets to the laptop server.
        Returns True on success.

        IMPORTANT: Start hand_following_server.py on the LAPTOP first,
        then run this script on the Pi.
        """
        for attempt in range(1, retries + 1):
            print(f"[ROBOT] Connecting to {self.laptop_ip}  "
                  f"(attempt {attempt}/{retries})…")

            # Initialise to None BEFORE the try block so the except
            # clause can safely reference them even if an error occurs
            # before either socket is created.
            cmd = None
            vid = None

            try:
                # ── CMD socket  (Pi ← Laptop: receives motor commands)
                cmd = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                cmd.settimeout(5)
                cmd.connect((self.laptop_ip, CMD_PORT))
                cmd.settimeout(None)

                # ── VIDEO socket  (Pi → Laptop: sends JPEG frames)
                vid = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                vid.settimeout(5)
                vid.connect((self.laptop_ip, VIDEO_PORT))
                vid.settimeout(None)

                self.cmd_sock   = cmd
                self.video_sock = vid
                print(f"[ROBOT] Connected — CMD:{CMD_PORT}  VIDEO:{VIDEO_PORT}")
                return True

            except ConnectionRefusedError:
                # Server not up yet — give a clear message
                print(f"[ROBOT] Connection refused — "
                      f"is hand_following_server.py running on {self.laptop_ip}?")

            except Exception as exc:
                print(f"[ROBOT] Failed: {exc}")

            finally:
                # Only close sockets that were actually created AND
                # the connection failed (i.e. we didn't store them)
                if self.cmd_sock is None:
                    for s in (cmd, vid):
                        if s is not None:
                            try:
                                s.close()
                            except Exception:
                                pass

            time.sleep(2)

        return False

    # ──────────────────────────────────────────
    #  Start
    # ──────────────────────────────────────────

    def start(self):
        if not self._connect():
            print("[ROBOT] Could not reach laptop server. "
                  "Make sure hand_following_server.py is running first.")
            return

        self.running = True

        # Start video streaming in background
        threading.Thread(
            target=self._stream_video,
            daemon=True,
            name="video-stream"
        ).start()

        print("[ROBOT] Streaming started. Waiting for hand commands…\n")

        # Block on command receive (main thread)
        self._receive_commands()

    # ──────────────────────────────────────────
    #  Video streaming  (Pi → Laptop)
    # ──────────────────────────────────────────

    def _stream_video(self):
        """
        Continuously read JPEG frames from the Pi camera and send them
        to the laptop with a 4-byte little-endian length header.

        Frame format (matches original main.py threading_video_send):
            [4 bytes LE uint32 = frame length] [JPEG bytes]
        """
        try:
            self.camera.start_stream()
            print("[ROBOT] Camera stream active")

            while self.running:
                frame = self.camera.get_frame()
                if frame is None:
                    time.sleep(0.005)
                    continue

                header = struct.pack('<I', len(frame))   # little-endian, matches Video.py
                try:
                    self.video_sock.sendall(header + frame)
                except BrokenPipeError:
                    print("[ROBOT] Video connection lost")
                    break
                except Exception as exc:
                    if self.running:
                        print(f"[ROBOT] Video send error: {exc}")
                    break

        except Exception as exc:
            print(f"[ROBOT] Camera error: {exc}")
        finally:
            self.camera.stop_stream()
            print("[ROBOT] Camera stream stopped")

    # ──────────────────────────────────────────
    #  Command receive  (Pi ← Laptop)
    # ──────────────────────────────────────────

    def _receive_commands(self):
        """
        Receive newline-terminated commands from the laptop and execute them.
        Runs in the main thread; blocks until disconnected.

        Expected format:  CMD_MOTOR#left#right\n
        """
        buf = ""
        while self.running:
            try:
                data = self.cmd_sock.recv(1024).decode("utf-8")
                if not data:
                    print("[ROBOT] Laptop disconnected")
                    break

                buf += data
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if line:
                        self._execute(line)

            except Exception as exc:
                if self.running:
                    print(f"[ROBOT] Command receive error: {exc}")
                break

        # Safety stop
        self.motor.setMotorModel(0, 0)
        print("[ROBOT] Motors stopped (connection closed)")

    # ──────────────────────────────────────────
    #  Command execution
    # ──────────────────────────────────────────

    def _execute(self, cmd_str: str):
        """
        Parse one command string and apply it to the hardware.
        Uses the existing MessageParser + COMMAND classes (unchanged).
        """
        self.parser.clearParameters()
        self.parser.parser(cmd_str)

        cmd_type = self.parser.commandString

        if cmd_type == CMD_MOTOR:
            if len(self.parser.intParameter) >= 2:
                left  = self.parser.intParameter[0]
                right = self.parser.intParameter[1]
                self.motor.setMotorModel(left, right)
                # Debug (uncomment if needed):
                # print(f"[ROBOT] Motor ← L={left:+5d}  R={right:+5d}")

        elif cmd_type == CMD_LED:
            pass   # extend here if you want LED feedback during follow

        else:
            print(f"[ROBOT] Unhandled command: {cmd_str}")

    # ──────────────────────────────────────────
    #  Cleanup
    # ──────────────────────────────────────────

    def stop(self):
        self.running = False
        self.motor.setMotorModel(0, 0)
        self.motor.close()
        self.camera.close()
        for s in (self.cmd_sock, self.video_sock):
            try:
                if s:
                    s.close()
            except Exception:
                pass
        print("[ROBOT] Shutdown complete")


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    laptop_ip = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LAPTOP_IP
    print(f"[ROBOT] Targeting laptop at {laptop_ip}")

    robot = HandFollowRobot(laptop_ip=laptop_ip)
    try:
        robot.start()
    except KeyboardInterrupt:
        print("\n[ROBOT] Interrupted by user")
    finally:
        robot.stop()