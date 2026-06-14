from __future__ import annotations
import io
import os
import socket
import struct
import subprocess
import sys
import threading
import time
import wave

DEFAULT_LAPTOP_IP = "10.42.0.233"   
CMD_PORT          = 5003
VIDEO_PORT        = 8003
EXPR_PORT         = 5010          
TOUCH_GPIO_PIN    = 17             
CAMERA_WIDTH      = 400
CAMERA_HEIGHT     = 300
EXPR_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "expression.py")


try:
    from motor import tankMotor
    _motor = tankMotor()
    MOTOR_OK = True
    print("[OK] motor")
except Exception as _e:
    _motor = None
    MOTOR_OK = False
    print(f"[WARN] motor unavailable ({_e})")

try:
    from picamera2 import Picamera2
    from picamera2.encoders import JpegEncoder
    from picamera2.outputs import FileOutput
    from libcamera import Transform
    CAMERA_OK = True
    print("[OK] picamera2")
except Exception as _e:
    Picamera2 = None
    CAMERA_OK = False
    print(f"[WARN] picamera2 unavailable ({_e})")

try:
    from gpiozero import Button as _Button
    _touch = _Button(TOUCH_GPIO_PIN, pull_up=False, bounce_time=0.25)
    TOUCH_OK = True
    print(f"[OK] TTP223 touch sensor on GPIO {TOUCH_GPIO_PIN}")
except Exception as _e:
    _touch = None
    TOUCH_OK = False
    print(f"[WARN] Touch sensor unavailable on GPIO {TOUCH_GPIO_PIN} ({_e})")

_led = None
LED_OK = False
print("[SKIP] LED disabled (params.json not configured)")

import subprocess as _subprocess
TTS_OK = True
print("[OK] aplay (robot speaker — no install needed)")


class StreamBuffer(io.BufferedIOBase):

    def __init__(self):
        super().__init__()
        self._frame = None
        self._cond  = threading.Condition()

    def write(self, data: bytes) -> int:
        with self._cond:
            self._frame = bytes(data)
            self._cond.notify_all()
        return len(data)

    def get_frame(self, timeout: float = 1.0) -> bytes | None:
        with self._cond:
            self._cond.wait(timeout)
            return self._frame


class PippoRobot:

    def __init__(self, laptop_ip: str):
        self._laptop_ip  = laptop_ip
        self._cmd_sock   = None
        self._vid_sock   = None
        self._running    = False
        self._send_lock  = threading.Lock()
        self._buf        = StreamBuffer()
        self._camera     = None
        self._expr_proc  = None
        self._tts_queue  = None
        self._tts_thread = None


    def start(self):
        self._running = True
        self._start_expression_screen()
        if not self._connect():
            print("[ROBOT] Could not connect to laptop. Check the IP and that")
            print("        pippo_server.py is running FIRST on the laptop.")
            self.stop()
            return

        self._start_camera()
        threading.Thread(target=self._stream_video,  daemon=True).start()
        threading.Thread(target=self._recv_commands, daemon=True).start()
        print("\n[ROBOT] Running!  Press Ctrl-C to stop.\n")
        try:
            while self._running:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self):
        print("[ROBOT] Shutting down …")
        self._running = False
        if MOTOR_OK and _motor is not None:
            try:
                _motor.setMotorModel(0, 0)
                _motor.close()
            except Exception:
                pass

        if self._camera is not None:
            try:
                self._camera.stop()
                self._camera.close()
            except Exception:
                pass

        for sock in [self._cmd_sock, self._vid_sock]:
            try:
                if sock:
                    sock.close()
            except Exception:
                pass

        if self._expr_proc is not None:
            try:
                self._expr_proc.terminate()
            except Exception:
                pass

        print("[ROBOT] Shutdown complete")


    def _start_expression_screen(self):
        if not os.path.exists(EXPR_SCRIPT):
            print(f"[EXPR] expression_screen.py not found at:\n       {EXPR_SCRIPT}")
            print("[EXPR] Skipping — run it manually if needed.")
            return
        try:
            env = dict(os.environ)
            env.setdefault('DISPLAY', ':0')
            self._expr_proc = subprocess.Popen(
                [sys.executable, EXPR_SCRIPT, '--full'],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"[EXPR] expression_screen.py started (PID {self._expr_proc.pid})")
            time.sleep(2.5)
        except Exception as _e:
            print(f"[EXPR] Failed to start expression_screen.py: {_e}")
            self._expr_proc = None


    def _play_audio(self, wav_bytes: bytes):
        try:
            import tempfile, os
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                f.write(wav_bytes)
                tmp = f.name
            _subprocess.run(['aplay', '-q', tmp], check=False)
            os.unlink(tmp)
        except Exception as _e:
            print(f"[SPEAK] aplay error: {_e}")


    def _connect(self) -> bool:
        max_attempts = 25
        for attempt in range(1, max_attempts + 1):
            print(f"[CONN] Attempt {attempt}/{max_attempts} → {self._laptop_ip} …")
            cmd_s = None
            vid_s = None
            try:
                cmd_s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                cmd_s.settimeout(5)
                cmd_s.connect((self._laptop_ip, CMD_PORT))

                vid_s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                vid_s.settimeout(5)
                vid_s.connect((self._laptop_ip, VIDEO_PORT))

                cmd_s.settimeout(None)
                vid_s.settimeout(None)

                self._cmd_sock = cmd_s
                self._vid_sock = vid_s
                print(f"[CONN] Connected to laptop at {self._laptop_ip} ✓")
                return True

            except Exception as _e:
                for s in [cmd_s, vid_s]:
                    if s:
                        try:
                            s.close()
                        except Exception:
                            pass
                msg = str(_e)
                if 'Connection refused' in msg:
                    print(f"[CONN] Refused — is pippo_server.py running on the laptop?")
                else:
                    print(f"[CONN] Failed: {msg}")
                time.sleep(2)

        return False


    def _start_camera(self):
        if not CAMERA_OK or Picamera2 is None:
            print("[WARN] Camera not available — video stream disabled")
            return
        try:
            self._camera = Picamera2()
            config = self._camera.create_video_configuration(
                main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT)},
                transform=Transform(hflip=1, vflip=1),
            )
            self._camera.configure(config)
            encoder = JpegEncoder(q=70)
            output  = FileOutput(self._buf)
            self._camera.start_recording(encoder, output)
            print(f"[CAMERA] Streaming {CAMERA_WIDTH}×{CAMERA_HEIGHT}")
        except Exception as _e:
            print(f"[CAMERA] Failed to start: {_e}")
            self._camera = None

    def _stream_video(self):
        if self._vid_sock is None:
            return
        while self._running:
            frame = self._buf.get_frame(timeout=1.0)
            if frame is None:
                continue
            try:
                header = struct.pack('<I', len(frame))
                self._vid_sock.sendall(header + frame)
            except Exception as _e:
                if self._running:
                    print(f"[VIDEO] Send error: {_e}")
                self._running = False
                break
        print("[VIDEO] Streamer stopped")


    def _recv_commands(self):
        if self._cmd_sock is None:
            return
        buf = ''
        while self._running:
            try:
                chunk = self._cmd_sock.recv(512).decode('utf-8', errors='replace')
                if not chunk:
                    print("[CMD] Laptop disconnected")
                    self._running = False
                    break
                buf += chunk
                while '\n' in buf:
                    line, buf = buf.split('\n', 1)
                    self._route(line.strip())
            except Exception as _e:
                if self._running:
                    print(f"[CMD] Recv error: {_e}")
                self._running = False
                break
        print("[CMD] Receiver stopped")

    def _route(self, cmd: str):
        if not cmd:
            return
        parts = cmd.split('#')
        key   = parts[0]

        if key == 'CMD_MOTOR':
            try:
                left  = max(-4095, min(4095, int(float(parts[1]))))
                right = max(-4095, min(4095, int(float(parts[2]))))
                if MOTOR_OK and _motor is not None:
                    _motor.setMotorModel(left, right)
            except (IndexError, ValueError) as _e:
                print(f"[CMD] Bad CMD_MOTOR: {cmd}  ({_e})")

        elif key == 'CMD_EXPRESSION':
            expr = parts[1].strip() if len(parts) > 1 else 'neutral'
            self._forward_expression(expr)

        elif key == 'CMD_LED':
            try:
                self._set_led(int(parts[1]))
            except (IndexError, ValueError):
                pass

        elif key == 'CMD_SPEAK':
            pass

        elif key == 'CMD_AUDIO':
            try:
                import base64
                wav_bytes = base64.b64decode(parts[1].strip())
                print(f"[SPEAK] Playing audio ({len(wav_bytes)} bytes)")
                threading.Thread(
                    target=self._play_audio, args=(wav_bytes,), daemon=True
                ).start()
            except Exception as _e:
                print(f"[SPEAK] CMD_AUDIO error: {_e}")

        else:
            print(f"[CMD] Unknown: {cmd}")


    def _forward_expression(self, expr: str):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1.0)
            s.connect(('127.0.0.1', EXPR_PORT))
            s.sendall(f"CMD_EXPRESSION#{expr}\n".encode('utf-8'))
            s.close()
        except Exception:
            pass


    def _set_led(self, mode: int):
        if not LED_OK or _led is None:
            return
        try:
            if mode == 0:
                _led.colorWipe((0, 0, 0), 10)
            elif mode == 1:
                _led.colorWipe((0, 0, 255), 10)     
            elif mode == 2:
                _led.colorWipe((0, 255, 0), 10)     
            elif mode == 3:
                _led.colorWipe((255, 0, 0), 10)     
            elif mode == 4:
                _led.theaterChaseRainbow()
        except Exception as _e:
            print(f"[LED] Error: {_e}")


    def _on_touch(self):
        print("[TOUCH] Sensor pressed — sending CMD_TOUCH to laptop")
        self._send_to_laptop('CMD_TOUCH#1')

    def _send_to_laptop(self, cmd: str):
        with self._send_lock:
            if self._cmd_sock is None:
                return
            try:
                if not cmd.endswith('\n'):
                    cmd += '\n'
                self._cmd_sock.sendall(cmd.encode('utf-8'))
            except Exception as _e:
                print(f"[CMD] Send error: {_e}")



def main():
    ip = DEFAULT_LAPTOP_IP
    if len(sys.argv) > 1:
        ip = sys.argv[1]

    print(f"[ROBOT] Pippo-bot Robot Client")
    print(f"[ROBOT] Laptop IP → {ip}")
    print(f"[ROBOT] CMD port  → {CMD_PORT}")
    print(f"[ROBOT] VIDEO port→ {VIDEO_PORT}\n")

    robot = PippoRobot(ip)
    robot.start()


if __name__ == '__main__':
    main()