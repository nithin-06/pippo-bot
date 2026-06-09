"""
hand_following_server.py  —  Runs on the LAPTOP  (SERVER)
==========================================================
Architecture (FLIPPED from original Freenove):
  Laptop listens on port 5003  — sends CMD_MOTOR commands to Pi
  Laptop listens on port 8003  — receives JPEG video from Pi

All heavy computation happens here:
  • MediaPipe hand landmark detection
  • Incremental PID controller (reuses PID.py from the project)
  • Decision logic: turning  + approach/retreat
  • Live annotated OpenCV display

Ports match the original Freenove constants so no other files need editing.

Install on laptop:
    pip install opencv-python mediapipe numpy

Usage:
    python hand_following_server.py
Press  Q  in the video window to quit.
"""

from __future__ import annotations   # allows  X | Y  type hints on Python 3.7–3.9

import cv2
import socket
import struct
import threading
import time
import queue
import sys
import os
import urllib.request
import numpy as np

# ── Reuse the existing PID module from the project
try:
    from PID import Incremental_PID
    print("[INFO] PID.py loaded from project")
except ImportError:
    # Embedded fallback — identical logic to PID.py
    class Incremental_PID:
        """Incremental PID  (identical to the project's PID.py)."""
        def __init__(self, P=0.0, I=0.0, D=0.0):
            self.setPoint      = 0.0
            self.Kp            = P
            self.Ki            = I
            self.Kd            = D
            self.last_error    = 0.0
            self.P_error       = 0.0
            self.I_error       = 0.0
            self.D_error       = 0.0
            self.I_saturation  = 10.0
            self.output        = 0.0

        def PID_compute(self, feedback_val):
            error        = self.setPoint - feedback_val
            self.P_error = self.Kp  * error
            self.I_error += error
            self.D_error = self.Kd * (error - self.last_error)
            self.I_error = max(-self.I_saturation,
                               min(self.I_saturation, self.I_error))
            self.output      = self.P_error + (self.Ki * self.I_error) + self.D_error
            self.last_error  = error
            return -self.output

        def setKp(self, v): self.Kp = v
        def setKi(self, v): self.Ki = v
        def setKd(self, v): self.Kd = v

    print("[INFO] Using embedded PID fallback")

# ── Command constants (mirrors Command.py)
try:
    from Command import COMMAND
    CMD_MOTOR = COMMAND.CMD_MOTOR
except ImportError:
    CMD_MOTOR = "CMD_MOTOR"

# ── MediaPipe setup — supports 0.10+ (Tasks API) and older (solutions API)
import mediapipe as mp

_USE_TASKS_API = False
try:
    from mediapipe.tasks.python import vision as _mp_vision
    from mediapipe.tasks.python import BaseOptions as _BaseOptions   # correct location
    from mediapipe.tasks.python.vision import (
        HandLandmarker,
        HandLandmarkerOptions,
        HandLandmarkerResult,
        RunningMode,
    )
    _USE_TASKS_API = True
    print("[INFO] MediaPipe Tasks API (>=0.10)")
except ImportError:
    try:
        _mp_hands = mp.solutions.hands
        _mp_draw  = mp.solutions.drawing_utils
        print("[INFO] MediaPipe Solutions API (<0.10)")
    except AttributeError:
        print("[ERROR] Cannot import MediaPipe hand tracking.")
        print("        Run:  pip install mediapipe==0.9.3")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────
#  Config — tune these values
# ─────────────────────────────────────────────────────────────

CMD_PORT    = 5003          # Pi will connect here to receive motor commands
VIDEO_PORT  = 8003          # Pi will connect here to send video

# PID gains — same defaults as Main.py (Incremental_PID(1, 0, 0.0025))
PID_KP = 1.0
PID_KI = 0.0
PID_KD = 0.0025

# Motor speeds
MOTOR_BASE_SPEED    = 1000   # forward speed when hand is centred  (0–4095)
MOTOR_MAX_TURN      = 2000   # maximum turning correction from PID
MOTOR_APPROACH      = 1200   # base speed when hand is far  (small in frame)
MOTOR_RETREAT       = -800   # base speed when hand is too close (large in frame)

# Hand-size thresholds (bounding-box area in pixels, 400×300 stream)
AREA_TOO_CLOSE      = 18000  # hand fills too much frame → reverse
AREA_TOO_FAR        = 3000   # hand is tiny → increase forward speed
AREA_IDEAL_MIN      = 5000   # comfortable "follow" zone
AREA_IDEAL_MAX      = 14000

# Dead-zone: if hand is within ±DEAD_ZONE_FRAC of centre, don't turn
DEAD_ZONE_FRAC      = 0.07   # 7 % of half-width on each side

# MediaPipe model (auto-downloaded on first run)
MODEL_PATH  = "hand_landmarker.task"
MODEL_URL   = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# Hand landmark index — middle-finger MCP (landmark 9) gives stable X position
LM_TRACK = 9

# Skeleton connections for manual drawing (Tasks API)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]


# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────

def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
    """Receive exactly n bytes; returns None if the connection closes."""
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def _ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    print(f"[INFO] Downloading hand_landmarker model (~1 MB)…")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"[INFO] Saved → {MODEL_PATH}")
    except Exception as exc:
        print(f"[ERROR] Download failed: {exc}\n"
              f"        Get it manually from:\n        {MODEL_URL}")
        sys.exit(1)


def _draw_landmarks(frame: np.ndarray, lms, h: int, w: int):
    """Draw skeleton on frame; return list of (px, py) for all 21 landmarks."""
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
    for s, e in HAND_CONNECTIONS:
        cv2.line(frame, pts[s], pts[e], (80, 220, 80), 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
    return pts


# ─────────────────────────────────────────────────────────────
#  Hand detector — unified wrapper over both API versions
# ─────────────────────────────────────────────────────────────

class HandDetector:
    def __init__(self):
        if _USE_TASKS_API:
            _ensure_model()
            opts = HandLandmarkerOptions(
                base_options=_BaseOptions(          # imported from mediapipe.tasks.python
                    model_asset_path=MODEL_PATH),
                running_mode=RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=0.7,
                min_hand_presence_confidence=0.7,
                min_tracking_confidence=0.7,
            )
            self._det   = HandLandmarker.create_from_options(opts)
            self._ts_ms = 0
        else:
            self._det = _mp_hands.Hands(
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7,
            )

    def detect(self, bgr: np.ndarray):
        """
        Returns (landmarks_or_None, annotated_frame, pixel_points).
        landmarks is a list of objects with .x .y  (normalised 0–1).
        pixel_points is a list of (px, py) for all 21 landmarks.
        """
        h, w   = bgr.shape[:2]
        canvas = bgr.copy()

        if _USE_TASKS_API:
            rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            self._ts_ms += 33
            result: HandLandmarkerResult = self._det.detect_for_video(
                mp_img, self._ts_ms)
            if result.hand_landmarks:
                lms = result.hand_landmarks[0]
                pts = _draw_landmarks(canvas, lms, h, w)
                return lms, canvas, pts
            return None, canvas, []
        else:
            rgb     = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            results = self._det.process(rgb)
            if results.multi_hand_landmarks:
                hl  = results.multi_hand_landmarks[0]
                _mp_draw.draw_landmarks(canvas, hl, _mp_hands.HAND_CONNECTIONS)
                lms = hl.landmark
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
                return lms, canvas, pts
            return None, canvas, []

    def close(self):
        try:
            self._det.close()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────
#  Hand Following Server
# ─────────────────────────────────────────────────────────────

class HandFollowingServer:
    """
    1. Waits for the Pi to connect on two ports (video + cmd).
    2. Receives JPEG frames, runs hand detection, computes PID.
    3. Sends CMD_MOTOR commands back to the Pi.
    4. Displays annotated feed in an OpenCV window.
    """

    def __init__(self):
        self.cmd_conn   = None    # socket to send motor commands to Pi
        self.video_conn = None    # socket to receive video from Pi
        self.running    = False

        # Thread-safe frame queue: laptop display ← video recv thread
        self._frame_q: queue.Queue = queue.Queue(maxsize=3)

        # PID for horizontal turning
        # Same initialisation as Main.py: Incremental_PID(1, 0, 0.0025)
        self.pid = Incremental_PID(PID_KP, PID_KI, PID_KD)

        # Latest motor values (for HUD display)
        self._last_left  = 0
        self._last_right = 0
        self._last_cmd   = ""

        # FPS tracking
        self._fps_count = 0
        self._fps_t0    = time.time()
        self._fps       = 0.0

    # ──────────────────────────────────────────
    #  Server start
    # ──────────────────────────────────────────

    def start(self):
        print(f"\n{'='*54}")
        print(f"  Hand Following Server")
        print(f"  CMD  port : {CMD_PORT}   (sends motor commands to Pi)")
        print(f"  VIDEO port: {VIDEO_PORT}  (receives camera frames from Pi)")
        print(f"{'='*54}")
        print("  Waiting for robot to connect…\n")

        # Accept both connections (could arrive in any order)
        cmd_thread   = threading.Thread(target=self._accept_cmd,   daemon=True)
        video_thread = threading.Thread(target=self._accept_video, daemon=True)
        cmd_thread.start()
        video_thread.start()

        # Block until both are connected
        while self.cmd_conn is None or self.video_conn is None:
            time.sleep(0.1)

        print("[SERVER] Both connections established — starting hand following!\n")
        self.running = True

        # Start video receive thread
        threading.Thread(
            target=self._recv_video_loop,
            daemon=True,
            name="vid-recv"
        ).start()

        # Main thread: process frames + display
        self._main_loop()

    # ──────────────────────────────────────────
    #  Accept helpers
    # ──────────────────────────────────────────

    def _accept_cmd(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("", CMD_PORT))
        srv.listen(1)
        print(f"[CMD]   Listening on :{CMD_PORT}")
        conn, addr = srv.accept()
        print(f"[CMD]   Robot connected from {addr[0]}:{addr[1]}")
        self.cmd_conn = conn
        srv.close()

    def _accept_video(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("", VIDEO_PORT))
        srv.listen(1)
        print(f"[VIDEO] Listening on :{VIDEO_PORT}")
        conn, addr = srv.accept()
        print(f"[VIDEO] Robot connected from {addr[0]}:{addr[1]}")
        self.video_conn = conn
        srv.close()

    # ──────────────────────────────────────────
    #  Video receive thread
    # ──────────────────────────────────────────

    def _recv_video_loop(self):
        """
        Read length-prefixed JPEG frames from the Pi.
        Frame format (matches hand_following_robot.py and original main.py):
            [4 bytes little-endian uint32 = length] [JPEG bytes…]
        """
        while self.running:
            # Read 4-byte header
            raw_len = _recv_exact(self.video_conn, 4)
            if raw_len is None:
                print("[VIDEO] Connection closed by robot")
                break

            (frame_len,) = struct.unpack('<I', raw_len)
            if frame_len == 0 or frame_len > 10_000_000:
                continue  # skip corrupt packet

            raw_frame = _recv_exact(self.video_conn, frame_len)
            if raw_frame is None:
                break

            bgr = cv2.imdecode(
                np.frombuffer(raw_frame, dtype=np.uint8),
                cv2.IMREAD_COLOR
            )
            if bgr is None:
                continue

            # Drop oldest frame if queue is full (keep latency low)
            if self._frame_q.full():
                try:
                    self._frame_q.get_nowait()
                except queue.Empty:
                    pass
            self._frame_q.put(bgr)

        self.running = False

    # ──────────────────────────────────────────
    #  Main loop — hand detection + PID + display
    # ──────────────────────────────────────────

    def _main_loop(self):
        """
        Runs in the main thread so cv2.imshow works on all platforms.
        1. Pull a frame from the queue.
        2. Run MediaPipe hand detection.
        3. Compute PID-based motor commands.
        4. Send command to Pi via cmd_conn.
        5. Annotate and display frame.
        """
        detector = HandDetector()

        # PID setpoint = horizontal centre of the frame
        # Will be updated once we know frame dimensions
        frame_cx = None

        print("[SERVER] Press  Q  in the video window to stop.\n")

        while self.running:
            # ── Get next frame ────────────────────────────────────────
            try:
                bgr = self._frame_q.get(timeout=0.5)
            except queue.Empty:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            h, w = bgr.shape[:2]

            # Initialise PID setpoint to frame centre (once)
            if frame_cx is None:
                frame_cx        = w // 2
                self.pid.setPoint = float(frame_cx)
                print(f"[PID]   setPoint = {frame_cx}  (frame {w}×{h})")

            # ── MediaPipe hand detection ──────────────────────────────
            lms, canvas, pts = detector.detect(bgr)

            # ── Motor command calculation ─────────────────────────────
            if lms is not None and pts:
                hand_x = pts[LM_TRACK][0]   # pixel X of landmark 9

                # ── Bounding box → estimate hand size (approach/retreat)
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                bbox_w = max(xs) - min(xs)
                bbox_h = max(ys) - min(ys)
                hand_area = bbox_w * bbox_h

                # ── Base speed based on hand distance ─────────────────
                if hand_area > AREA_TOO_CLOSE:
                    base = MOTOR_RETREAT          # too close → back up
                    dist_label = "TOO CLOSE"
                    dist_color = (0, 0, 255)
                elif hand_area < AREA_TOO_FAR:
                    base = MOTOR_APPROACH         # too far → go faster
                    dist_label = "TOO FAR"
                    dist_color = (255, 165, 0)
                else:
                    base = MOTOR_BASE_SPEED       # comfortable zone
                    dist_label = "FOLLOW"
                    dist_color = (0, 255, 128)

                # ── PID turning correction ────────────────────────────
                dead_zone = int(w * DEAD_ZONE_FRAC)
                offset    = hand_x - frame_cx

                if abs(offset) < dead_zone:
                    correction = 0              # dead zone: go straight
                else:
                    # PID_compute(feedback):
                    #   error = setPoint − feedback = centre − hand_x
                    #   When hand is RIGHT: error < 0 → output < 0 → returns +value
                    #   left gets +correction, right gets −correction → turns RIGHT ✓
                    raw = self.pid.PID_compute(float(hand_x))
                    correction = int(max(-MOTOR_MAX_TURN,
                                         min(MOTOR_MAX_TURN, raw)))

                left  = max(-4095, min(4095, base + correction))
                right = max(-4095, min(4095, base - correction))

                # ── Determine direction label ─────────────────────────
                if abs(offset) < dead_zone:
                    turn_label  = "STRAIGHT"
                    turn_color  = (255, 255, 255)
                elif offset > 0:
                    turn_label = "TURN RIGHT"
                    turn_color = (0, 200, 255)
                else:
                    turn_label = "TURN LEFT"
                    turn_color = (0, 200, 255)

                cmd_str = f"{CMD_MOTOR}#{left}#{right}"

            else:
                # ── No hand detected → stop ───────────────────────────
                left, right        = 0, 0
                cmd_str            = f"{CMD_MOTOR}#0#0"
                dist_label         = "NO HAND"
                dist_color         = (200, 200, 200)
                turn_label         = ""
                turn_color         = (200, 200, 200)
                self.pid.last_error = 0.0   # reset PID integral drift
                self.pid.I_error    = 0.0

            # ── Send command (only when it changes) ───────────────────
            if cmd_str != self._last_cmd:
                self._send_motor(left, right)
                self._last_cmd   = cmd_str
                self._last_left  = left
                self._last_right = right

            # ── FPS ───────────────────────────────────────────────────
            self._fps_count += 1
            now = time.time()
            if now - self._fps_t0 >= 1.0:
                self._fps       = self._fps_count / (now - self._fps_t0)
                self._fps_count = 0
                self._fps_t0    = now

            # ── Annotate display ──────────────────────────────────────
            canvas = self._draw_hud(
                canvas, h, w, frame_cx,
                left, right,
                dist_label, dist_color,
                turn_label, turn_color,
                lms is not None
            )

            cv2.imshow("Hand Following — Tank Robot", canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[SERVER] Q pressed — stopping")
                break

        # ── Cleanup ───────────────────────────────────────────────────
        self._send_motor(0, 0)
        detector.close()
        cv2.destroyAllWindows()
        self.running = False
        print("[SERVER] Stopped.")

    # ──────────────────────────────────────────
    #  Command sending
    # ──────────────────────────────────────────

    def _send_motor(self, left: int, right: int):
        """
        Send  CMD_MOTOR#left#right\n  to the robot.
        Matches the wire format used everywhere in the project.
        """
        msg = f"{CMD_MOTOR}#{left}#{right}\n"
        try:
            self.cmd_conn.sendall(msg.encode("utf-8"))
        except Exception as exc:
            print(f"[CMD] Send error: {exc}")
            self.running = False

    # ──────────────────────────────────────────
    #  HUD drawing
    # ──────────────────────────────────────────

    def _draw_hud(self, canvas, h, w, cx,
                  left, right,
                  dist_label, dist_color,
                  turn_label, turn_color,
                  hand_found: bool) -> np.ndarray:
        """Draw all on-screen information onto the frame."""

        # ── Centre crosshair ─────────────────────────────────────────
        cv2.line(canvas, (cx, 0),   (cx, h),   (60, 60, 60), 1)
        cv2.line(canvas, (0, h//2), (w, h//2), (60, 60, 60), 1)

        # ── Dead-zone markers ─────────────────────────────────────────
        dz = int(w * DEAD_ZONE_FRAC)
        cv2.line(canvas, (cx - dz, 0), (cx - dz, h), (80, 80, 80), 1)
        cv2.line(canvas, (cx + dz, 0), (cx + dz, h), (80, 80, 80), 1)

        # ── Top bar ───────────────────────────────────────────────────
        bar_h = 58
        cv2.rectangle(canvas, (0, 0), (w, bar_h), (30, 30, 30), -1)

        # FPS
        cv2.putText(canvas, f"FPS {self._fps:.0f}",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (160, 160, 160), 1)

        # Motors
        l_bar = int(abs(left)  / 4095 * 60)
        r_bar = int(abs(right) / 4095 * 60)
        l_col = (0, 255, 100) if left  >= 0 else (0, 100, 255)
        r_col = (0, 255, 100) if right >= 0 else (0, 100, 255)
        cv2.rectangle(canvas, (w - 130, 6), (w - 130 + l_bar, 22), l_col, -1)
        cv2.rectangle(canvas, (w - 130, 26), (w - 130 + r_bar, 42), r_col, -1)
        cv2.putText(canvas, f"L {left:+5d}",
                    (w - 130, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, l_col, 1)
        cv2.putText(canvas, f"R {right:+5d}",
                    (w - 130, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, r_col, 1)

        # Distance label
        cv2.putText(canvas, dist_label,
                    (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    dist_color, 2)

        # Turn direction label
        if turn_label:
            cv2.putText(canvas, turn_label,
                        (w // 2 - 60, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, turn_color, 2)

        # ── Bottom bar ────────────────────────────────────────────────
        cv2.rectangle(canvas, (0, h - 22), (w, h), (30, 30, 30), -1)
        status = "TRACKING" if hand_found else "SEARCHING…"
        col    = (0, 255, 128) if hand_found else (0, 130, 255)
        cv2.putText(canvas, status,
                    (8, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        cv2.putText(canvas, "Q = quit",
                    (w - 68, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (120, 120, 120), 1)

        return canvas


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    server = HandFollowingServer()
    try:
        server.start()
    except KeyboardInterrupt:
        print("\n[SERVER] Interrupted")
        server._send_motor(0, 0)