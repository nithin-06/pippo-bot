"""
expression_screen.py  —  Runs on the Raspberry Pi
===================================================
Renders an animated robot face on the 5" 800×480 HDMI display.

Expressions:  neutral | happy | sad | surprised | angry | excited | thinking | blushing
Features:     smooth transitions · auto-blinking · particle effects · TCP command receiver

Receive commands from laptop:
    CMD_EXPRESSION#happy\n          — change expression
    CMD_EXPRESSION#angry\n
    ...

TCP port: 5010  (laptop sends, Pi receives)

Keyboard shortcuts (standalone testing):
    1=neutral   2=happy    3=sad       4=surprised
    5=angry     6=excited  7=thinking  8=blushing
    ESC = quit  F = toggle fullscreen

Run:
    python expression_screen.py           # windowed
    python expression_screen.py --full    # fullscreen (on Pi with HDMI display)
"""

from __future__ import annotations
import pygame
import socket
import threading
import math
import time
import random
import sys
import os

# ─────────────────────────────────────────────────────────────
#  Display & timing
# ─────────────────────────────────────────────────────────────
WIDTH, HEIGHT = 800, 480
FPS           = 60
EXPR_PORT     = 5010
CMD_EXPRESSION = "CMD_EXPRESSION"

# ─────────────────────────────────────────────────────────────
#  Colour palette — dark robot aesthetic
# ─────────────────────────────────────────────────────────────
C_BG         = ( 10,  14,  20)
C_FACE       = ( 18,  24,  36)
C_FACE_RIM   = ( 35,  48,  72)
C_EYE        = (  0, 255, 200)
C_EYE_DIM    = (  0, 110,  90)
C_MOUTH      = (  0, 255, 200)
C_BLUSH      = (255,  80, 110)
C_TEAR       = ( 80, 160, 255)
C_ANGRY      = (255,  65,  50)
C_THINK      = (160, 140, 255)
C_SPARKLE    = (255, 240, 100)
C_PANEL_TEXT = ( 45,  65,  85)
C_SCANLINE   = (  0, 255, 200)     # very faint, for scanline overlay

# ─────────────────────────────────────────────────────────────
#  Face layout (all coordinates for 800×480)
# ─────────────────────────────────────────────────────────────
FACE_CX, FACE_CY = 400, 240
FACE_W,  FACE_H  = 560, 420

EYE_L_CX, EYE_R_CX = 265, 535
EYE_CY   = 205
EYE_W    = 145
EYE_H    = 88

MOUTH_CX, MOUTH_CY = 400, 348
MOUTH_W             = 210
MOUTH_H             = 65

BROW_Y_OFFSET = 55      # above eye centre
CHEEK_L  = (195, 270)
CHEEK_R  = (605, 270)
CHEEK_R2 = 55

# ─────────────────────────────────────────────────────────────
#  Expression parameter table
#  All values lerp smoothly between expressions.
#  eye_h    : eye height scale  (0=closed → 1=normal → 1.4=wide)
#  l/r_tl   : top-left corner y-offset of left/right eye  (+ve = down)
#  l/r_tr   : top-right corner y-offset of left/right eye
#  squint   : 0=draw as rect, 1=draw as curved line (happy squint)
#  mouth    : 'flat'|'smile'|'grin'|'frown'|'frown2'|'open'|'smirk'
#  m_w      : mouth width scale
#  m_h      : mouth height scale (for open/grin)
#  blush    : draw cheek blush
#  tears    : draw teardrop
#  red_eyes : eye colour → angry red
#  sparkle  : draw star sparkles in eyes
# ─────────────────────────────────────────────────────────────
EXPR = {
    'neutral': {
        'eye_h':1.00, 'squint':0,
        'l_tl': 0,'l_tr': 0, 'r_tl': 0,'r_tr': 0,
        'mouth':'flat',   'm_w':0.70, 'm_h':0.15,
        'blush':False,'tears':False,'red_eyes':False,'sparkle':False,
    },
    'happy': {
        'eye_h':0.55, 'squint':1,
        'l_tl':-10,'l_tr':-10, 'r_tl':-10,'r_tr':-10,
        'mouth':'smile',  'm_w':0.95, 'm_h':0.65,
        'blush':False,'tears':False,'red_eyes':False,'sparkle':False,
    },
    'blushing': {
        'eye_h':0.55, 'squint':1,
        'l_tl':-10,'l_tr':-10, 'r_tl':-10,'r_tr':-10,
        'mouth':'smile',  'm_w':0.85, 'm_h':0.55,
        'blush':True, 'tears':False,'red_eyes':False,'sparkle':False,
    },
    'sad': {
        'eye_h':0.72, 'squint':0,
        'l_tl':14,'l_tr':-8,  'r_tl':-8,'r_tr':14,  # outer corners droop
        'mouth':'frown',  'm_w':0.75, 'm_h':0.40,
        'blush':False,'tears':True, 'red_eyes':False,'sparkle':False,
    },
    'surprised': {
        'eye_h':1.42, 'squint':0,
        'l_tl': 0,'l_tr': 0, 'r_tl': 0,'r_tr': 0,
        'mouth':'open',   'm_w':0.52, 'm_h':0.92,
        'blush':False,'tears':False,'red_eyes':False,'sparkle':False,
    },
    'angry': {
        'eye_h':0.68, 'squint':0,
        'l_tl':-18,'l_tr':12, 'r_tl':12,'r_tr':-18,  # inner corners pull down
        'mouth':'frown2', 'm_w':0.70, 'm_h':0.35,
        'blush':False,'tears':False,'red_eyes':True, 'sparkle':False,
    },
    'excited': {
        'eye_h':1.22, 'squint':0,
        'l_tl':-8,'l_tr':-8, 'r_tl':-8,'r_tr':-8,
        'mouth':'grin',   'm_w':1.00, 'm_h':0.80,
        'blush':False,'tears':False,'red_eyes':False,'sparkle':True,
    },
    'thinking': {
        'eye_h':0.78, 'squint':0,
        'l_tl': 6,'l_tr':-4, 'r_tl': 0,'r_tr': 0,  # left eye squints
        'mouth':'smirk',  'm_w':0.48, 'm_h':0.28,
        'blush':False,'tears':False,'red_eyes':False,'sparkle':False,
    },
}

EXPR_KEYS = list(EXPR.keys())


# ─────────────────────────────────────────────────────────────
#  Smooth lerp helpers
# ─────────────────────────────────────────────────────────────

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def lerp_color(c1, c2, t):
    return tuple(int(lerp(c1[i], c2[i], t)) for i in range(3))


# ─────────────────────────────────────────────────────────────
#  Drawing helpers
# ─────────────────────────────────────────────────────────────

def draw_rounded_rect(surf, color, rect, radius=20, width=0):
    """Draw a rect with fully rounded corners."""
    x, y, w, h = rect
    r = min(radius, h // 2, w // 2)
    pygame.draw.rect(surf, color, (x + r, y, w - 2*r, h), width)
    pygame.draw.rect(surf, color, (x, y + r, w, h - 2*r), width)
    for cx, cy in [(x+r, y+r), (x+w-r, y+r), (x+r, y+h-r), (x+w-r, y+h-r)]:
        pygame.draw.circle(surf, color, (cx, cy), r, width)


def draw_glow(surf, color, points_or_center, radius=0, width=3, alpha=40):
    """Draw a soft glow layer on a temp surface."""
    temp = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    glow_col = (*color, alpha)
    for expand in (6, 4, 2):
        if radius:
            pygame.draw.circle(temp, (*color, alpha // 2), points_or_center, radius + expand, width + expand)
        else:
            inflated = [(p[0] + (1 if p[0] > WIDTH//2 else -1) * expand,
                         p[1] + (1 if p[1] > HEIGHT//2 else -1) * expand)
                        for p in points_or_center]
    surf.blit(temp, (0, 0))


def draw_eye_trapezoid(surf, cx, cy, w, h, tl_off, tr_off, color, outline_w=3):
    """Draw eye as a trapezoid (tilted rectangle) with glow outline."""
    hw, hh = w // 2, h // 2
    pts = [
        (cx - hw, cy - hh + int(tl_off)),
        (cx + hw, cy - hh + int(tr_off)),
        (cx + hw, cy + hh),
        (cx - hw, cy + hh),
    ]
    # glow: slightly enlarged, low alpha
    temp = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    big = [
        (cx - hw - 3, cy - hh + int(tl_off) - 3),
        (cx + hw + 3, cy - hh + int(tr_off) - 3),
        (cx + hw + 3, cy + hh + 3),
        (cx - hw - 3, cy + hh + 3),
    ]
    pygame.draw.polygon(temp, (*color, 30), big)
    pygame.draw.polygon(temp, (*color, 15), [
        (cx - hw - 6, cy - hh + int(tl_off) - 6),
        (cx + hw + 6, cy - hh + int(tr_off) - 6),
        (cx + hw + 6, cy + hh + 6),
        (cx - hw - 6, cy + hh + 6),
    ])
    surf.blit(temp, (0, 0))
    pygame.draw.polygon(surf, color, pts)
    pygame.draw.polygon(surf, color, pts, outline_w)


def draw_eye_squint(surf, cx, cy, w, color, outline_w=4):
    """Draw a happy/squinting eye as a thick downward arc."""
    rect = pygame.Rect(cx - w//2, cy - 18, w, 36)
    pygame.draw.arc(surf, color, rect, math.radians(200), math.radians(340), outline_w)
    pygame.draw.arc(surf, color, rect.inflate(6, 6), math.radians(200), math.radians(340), 2)


def draw_mouth_flat(surf, cx, cy, w, color, line_w=4):
    hw = int(w * MOUTH_W // 2)
    pygame.draw.line(surf, color, (cx - hw, cy), (cx + hw, cy), line_w)


def draw_mouth_arc(surf, cx, cy, w, h, color, smile=True, line_w=5):
    """smile=True → upward arc (smile); smile=False → downward arc (frown)."""
    hw = int(w / 2)
    hh = int(h / 2)
    rect = pygame.Rect(cx - hw, cy - hh, hw * 2, hh * 2)
    if smile:
        pygame.draw.arc(surf, color, rect, math.radians(195), math.radians(345), line_w)
        pygame.draw.arc(surf, color, rect.inflate(4, 4), math.radians(196), math.radians(344), 2)
    else:
        pygame.draw.arc(surf, color, rect, math.radians(15), math.radians(165), line_w)
        pygame.draw.arc(surf, color, rect.inflate(4, 4), math.radians(16), math.radians(164), 2)


def draw_mouth_grin(surf, cx, cy, w, h, color, line_w=5):
    """Very wide smile with teeth-like inner arc."""
    hw = int(w / 2)
    hh = int(h / 2)
    rect  = pygame.Rect(cx - hw, cy - hh, hw * 2, hh * 2)
    inner = pygame.Rect(cx - hw + 12, cy - hh // 2, (hw - 12) * 2, hh)
    pygame.draw.arc(surf, color, rect,  math.radians(195), math.radians(345), line_w)
    pygame.draw.arc(surf, color, inner, math.radians(200), math.radians(340), line_w - 1)


def draw_mouth_open(surf, cx, cy, w, h, color, line_w=4):
    """Open O mouth (ellipse outline)."""
    hw = int(w / 2)
    hh = int(h / 2)
    pygame.draw.ellipse(surf, color, (cx - hw, cy - hh, hw * 2, hh * 2), line_w)
    temp = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    pygame.draw.ellipse(temp, (*color, 25), (cx - hw - 4, cy - hh - 4, (hw + 4) * 2, (hh + 4) * 2))
    surf.blit(temp, (0, 0))


def draw_mouth_smirk(surf, cx, cy, w, h, color, line_w=4):
    """Smirk — short arc on the right side only."""
    hw = int(w / 2)
    hh = int(h / 2)
    rect = pygame.Rect(cx, cy - hh, hw * 2, hh * 2)
    pygame.draw.arc(surf, color, rect, math.radians(200), math.radians(340), line_w)


def draw_mouth_frown2(surf, cx, cy, w, h, color, line_w=5):
    """Tight angry frown — narrower arc."""
    hw = int(w / 2)
    hh = int(h // 3)
    rect = pygame.Rect(cx - hw, cy - hh, hw * 2, hh * 2)
    pygame.draw.arc(surf, color, rect, math.radians(20), math.radians(160), line_w)


def draw_tear(surf, cx, cy, progress, color):
    """Animate a teardrop falling."""
    if progress <= 0:
        return
    drop_y = cy + int(progress * 80)
    r = 6
    pygame.draw.circle(surf, color, (cx, drop_y), r)
    pts = [(cx - r, drop_y), (cx + r, drop_y), (cx, drop_y - 18)]
    pygame.draw.polygon(surf, color, pts)


def draw_blush(surf, cx, cy, radius, alpha=60):
    temp = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    for dr in range(radius, 0, -4):
        a = max(0, int(alpha * (1 - dr / radius)))
        pygame.draw.circle(temp, (*C_BLUSH, a), (cx, cy), dr)
    surf.blit(temp, (0, 0))


def draw_sparkle(surf, cx, cy, size, angle, color):
    """Draw a 4-point star."""
    pts = []
    for i in range(8):
        r = size if i % 2 == 0 else size // 3
        a = angle + i * math.pi / 4
        pts.append((cx + int(r * math.cos(a)), cy + int(r * math.sin(a))))
    pygame.draw.polygon(surf, color, pts)


def draw_thinking_dots(surf, cx, cy, frame, color):
    """Three bouncing dots for thinking animation."""
    for i in range(3):
        offset = math.sin((frame * 0.08) + i * 1.2) * 6
        pygame.draw.circle(surf, color,
                           (cx - 30 + i * 30, int(cy + offset)), 6)


# ─────────────────────────────────────────────────────────────
#  Animated state (lerped values)
# ─────────────────────────────────────────────────────────────

class FaceState:
    """Holds all currently-displayed visual parameters, smoothly lerped."""

    LERP_SPEED = 0.08   # per frame (lower = slower transition)

    def __init__(self):
        start = EXPR['neutral']
        self.eye_h  = float(start['eye_h'])
        self.squint = float(start['squint'])
        self.l_tl   = float(start['l_tl'])
        self.l_tr   = float(start['l_tr'])
        self.r_tl   = float(start['r_tl'])
        self.r_tr   = float(start['r_tr'])
        self.m_w    = float(start['m_w'])
        self.m_h    = float(start['m_h'])

        self.mouth_type   = start['mouth']
        self.blush        = start['blush']
        self.tears        = start['tears']
        self.red_eyes     = start['red_eyes']
        self.sparkle      = start['sparkle']

        # target values
        self._target = dict(start)

        # blinking
        self.blink_h       = 1.0   # 1=open, 0=closed
        self._blink_frame  = 0
        self._next_blink   = random.randint(120, 300)

        # tear drop animation
        self.tear_progress = 0.0

        # sparkle rotation
        self.sparkle_angle = 0.0

        # thinking dots
        self.frame = 0

    def set_expression(self, name: str):
        if name in EXPR:
            self._target = dict(EXPR[name])

    def update(self):
        t = self.LERP_SPEED
        tgt = self._target

        self.eye_h  = lerp(self.eye_h,  tgt['eye_h'],  t)
        self.squint = lerp(self.squint, float(tgt['squint']), t * 1.5)
        self.l_tl   = lerp(self.l_tl,  tgt['l_tl'],   t)
        self.l_tr   = lerp(self.l_tr,  tgt['l_tr'],   t)
        self.r_tl   = lerp(self.r_tl,  tgt['r_tl'],   t)
        self.r_tr   = lerp(self.r_tr,  tgt['r_tr'],   t)
        self.m_w    = lerp(self.m_w,   tgt['m_w'],    t)
        self.m_h    = lerp(self.m_h,   tgt['m_h'],    t)

        # Snap bool flags after halfway
        snap = t * 5
        self.mouth_type = tgt['mouth']
        self.blush      = tgt['blush']
        self.tears      = tgt['tears']
        self.red_eyes   = tgt['red_eyes']
        self.sparkle    = tgt['sparkle']

        # Blink
        self._blink_frame += 1
        if self._blink_frame >= self._next_blink:
            progress = self._blink_frame - self._next_blink
            if progress < 5:
                self.blink_h = max(0.0, 1.0 - progress * 0.22)
            elif progress < 10:
                self.blink_h = min(1.0, (progress - 4) * 0.22)
            else:
                self.blink_h = 1.0
                self._blink_frame = 0
                self._next_blink = random.randint(120, 300)

        # Tear drop
        if self.tears:
            self.tear_progress = min(1.0, self.tear_progress + 0.008)
            if self.tear_progress >= 1.0:
                self.tear_progress = 0.0
        else:
            self.tear_progress = 0.0

        # Sparkle
        self.sparkle_angle += 0.04

        self.frame += 1


# ─────────────────────────────────────────────────────────────
#  Main face renderer
# ─────────────────────────────────────────────────────────────

class ExpressionFace:

    def __init__(self, fullscreen: bool = False):
        pygame.init()
        flags = pygame.FULLSCREEN | pygame.NOFRAME if fullscreen else 0
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT), flags)
        pygame.display.set_caption("Pippo-bot")
        pygame.mouse.set_visible(False)
        self.clock  = pygame.time.Clock()
        self.state  = FaceState()
        self.font   = pygame.font.SysFont("monospace", 16)
        self.running = True

        # TCP command listener
        self._cmd_queue = []
        self._queue_lock = threading.Lock()
        threading.Thread(target=self._tcp_listener, daemon=True).start()

        # Scanline overlay surface (created once)
        self._scanlines = self._make_scanlines()

        self._current_expr = 'neutral'

    # ──────────────────────────────────────────
    #  TCP listener
    # ──────────────────────────────────────────

    def _tcp_listener(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("", EXPR_PORT))
        srv.listen(5)
        print(f"[EXPR] Listening for commands on port {EXPR_PORT}")
        while True:
            try:
                conn, addr = srv.accept()
                threading.Thread(target=self._handle_client,
                                 args=(conn,), daemon=True).start()
            except Exception:
                pass

    def _handle_client(self, conn):
        buf = ""
        try:
            while True:
                data = conn.recv(256).decode("utf-8")
                if not data:
                    break
                buf += data
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if line.startswith(CMD_EXPRESSION + "#"):
                        expr = line.split("#")[1].lower().strip()
                        with self._queue_lock:
                            self._cmd_queue.append(expr)
                        print(f"[EXPR] → {expr}")
        except Exception:
            pass
        finally:
            conn.close()

    # ──────────────────────────────────────────
    #  Scanline overlay
    # ──────────────────────────────────────────

    def _make_scanlines(self):
        surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        for y in range(0, HEIGHT, 3):
            pygame.draw.line(surf, (0, 0, 0, 18), (0, y), (WIDTH, y))
        return surf

    # ──────────────────────────────────────────
    #  Drawing
    # ──────────────────────────────────────────

    def _draw_frame(self):
        s    = self.state
        surf = self.screen

        # Background
        surf.fill(C_BG)

        # Face oval
        draw_rounded_rect(surf, C_FACE,
                          (FACE_CX - FACE_W//2, FACE_CY - FACE_H//2, FACE_W, FACE_H),
                          radius=70)
        draw_rounded_rect(surf, C_FACE_RIM,
                          (FACE_CX - FACE_W//2, FACE_CY - FACE_H//2, FACE_W, FACE_H),
                          radius=70, width=2)

        eye_color = C_ANGRY if s.red_eyes else C_EYE

        # Effective eye height with blink
        eff_h = int(EYE_H * s.eye_h * s.blink_h)
        eff_h = max(2, eff_h)

        # ── EYES ──────────────────────────────────────────────
        for cx, tl, tr in [
            (EYE_L_CX, s.l_tl, s.l_tr),
            (EYE_R_CX, s.r_tl, s.r_tr),
        ]:
            if s.squint > 0.5:
                draw_eye_squint(surf, cx, EYE_CY, EYE_W, eye_color)
            else:
                blink_tl = tl * s.blink_h
                blink_tr = tr * s.blink_h
                draw_eye_trapezoid(surf, cx, EYE_CY, EYE_W, eff_h,
                                   blink_tl, blink_tr, eye_color)

            # Inner pupil glow dot
            if s.squint <= 0.5 and eff_h > 10:
                pygame.draw.circle(surf, C_BG,
                                   (cx, EYE_CY), max(4, eff_h // 5))
                pygame.draw.circle(surf, eye_color,
                                   (cx, EYE_CY), max(2, eff_h // 8))

            # Sparkle (excited)
            if s.sparkle:
                draw_sparkle(surf, cx - 30, EYE_CY - 35, 10,
                             s.sparkle_angle, C_SPARKLE)
                draw_sparkle(surf, cx + 30, EYE_CY - 38, 7,
                             s.sparkle_angle + 0.8, C_SPARKLE)

        # ── MOUTH ─────────────────────────────────────────────
        mw = int(MOUTH_W * s.m_w)
        mh = int(MOUTH_H * s.m_h)
        mt = s.mouth_type

        if mt == 'flat':
            pygame.draw.line(surf, C_MOUTH,
                             (MOUTH_CX - mw//2, MOUTH_CY),
                             (MOUTH_CX + mw//2, MOUTH_CY), 4)
        elif mt == 'smile':
            draw_mouth_arc(surf, MOUTH_CX, MOUTH_CY, mw, mh, C_MOUTH, smile=True)
        elif mt == 'grin':
            draw_mouth_grin(surf, MOUTH_CX, MOUTH_CY, mw, mh, C_MOUTH)
        elif mt == 'frown':
            draw_mouth_arc(surf, MOUTH_CX, MOUTH_CY, mw, mh, C_MOUTH, smile=False)
        elif mt == 'frown2':
            draw_mouth_frown2(surf, MOUTH_CX, MOUTH_CY, mw, mh, C_ANGRY)
        elif mt == 'open':
            draw_mouth_open(surf, MOUTH_CX, MOUTH_CY, mw, mh, C_MOUTH)
        elif mt == 'smirk':
            draw_mouth_smirk(surf, MOUTH_CX, MOUTH_CY, mw, mh, C_THINK)

        # ── EXTRAS ────────────────────────────────────────────
        if s.blush:
            draw_blush(surf, *CHEEK_L, CHEEK_R2, alpha=55)
            draw_blush(surf, *CHEEK_R, CHEEK_R2, alpha=55)

        if s.tears:
            draw_tear(surf, EYE_L_CX + 20, EYE_CY + 50,
                      s.tear_progress, C_TEAR)

        if mt == 'smirk':
            draw_thinking_dots(surf, MOUTH_CX + 90, MOUTH_CY - 50,
                               s.frame, C_THINK)

        # ── HUD text ──────────────────────────────────────────
        label = self.font.render(
            f"PIPPO-BOT  ·  {self._current_expr.upper()}",
            True, C_PANEL_TEXT)
        surf.blit(label, (WIDTH - label.get_width() - 16, HEIGHT - 24))

        # Scanlines overlay
        surf.blit(self._scanlines, (0, 0))

    # ──────────────────────────────────────────
    #  Main loop
    # ──────────────────────────────────────────

    def run(self):
        print("[EXPR] Running — keyboard: 1-8 to change expression, ESC to quit")
        while self.running:
            # ── Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    self._handle_key(event.key)

            # ── Pull TCP commands
            with self._queue_lock:
                cmds = self._cmd_queue[:]
                self._cmd_queue.clear()
            for expr in cmds:
                self._set_expression(expr)

            # ── Update + draw
            self.state.update()
            self._draw_frame()
            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()

    def _handle_key(self, key):
        mapping = {
            pygame.K_1: 'neutral',
            pygame.K_2: 'happy',
            pygame.K_3: 'sad',
            pygame.K_4: 'surprised',
            pygame.K_5: 'angry',
            pygame.K_6: 'excited',
            pygame.K_7: 'thinking',
            pygame.K_8: 'blushing',
            pygame.K_ESCAPE: None,
        }
        if key == pygame.K_ESCAPE:
            self.running = False
        elif key in mapping:
            self._set_expression(mapping[key])

    def _set_expression(self, name: str):
        if name in EXPR:
            self.state.set_expression(name)
            self._current_expr = name
            print(f"[EXPR] Expression → {name}")
        else:
            print(f"[EXPR] Unknown expression: {name}")


# ─────────────────────────────────────────────────────────────
#  Send helper (call from laptop or any other module)
# ─────────────────────────────────────────────────────────────

def send_expression(pi_ip: str, expression: str, port: int = EXPR_PORT):
    """
    Utility: send an expression command to the Pi from the laptop.
    Usage:
        from expression_screen import send_expression
        send_expression("10.42.0.212", "happy")
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        s.connect((pi_ip, port))
        s.sendall(f"{CMD_EXPRESSION}#{expression}\n".encode())
        s.close()
    except Exception as exc:
        print(f"[EXPR] Send failed: {exc}")


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Ensure X display is available on Pi
    if "DISPLAY" not in os.environ:
        os.environ["DISPLAY"] = ":0"

    fullscreen = "--full" in sys.argv or "-f" in sys.argv
    face = ExpressionFace(fullscreen=fullscreen)
    face.run()