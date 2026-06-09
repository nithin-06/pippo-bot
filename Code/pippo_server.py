"""
pippo_server_4.py  --  Runs on the LAPTOP (SERVER)
===================================================
Fixes vs previous version:
  . BT_FIXED_MUTE_SECS = 10 : hard flat mute after every robot utterance.
    Replaces the fragile estimated-playback calculation.  The mic will not
    open until exactly 10 s after the audio packet was sent to the robot,
    giving the BT speaker plenty of time to finish and go silent.
  . APPROVAL_TIMEOUT raised 12 -> 20 s  (user has a full 10 s listen window
    + a few seconds to think before the timeout fires).
  . COOLDOWN_AFTER raised 20 -> 35 s so the personality engine does NOT
    immediately re-trigger another question the moment the mic opens.
  . _MIC_LOCK: global lock prevents EmotionSensor stealing the mic while
    VoiceListener is recording.
  . EmotionSensor audio loop uses non-blocking acquire + 3 s sleep so it
    never queues behind the voice listener.
  . "You can speak now" beep + green video overlay on mic-open transition.
  . Video overlay now shows MIC MUTED countdown so you can see when it opens.

Patch notes (v4 -> v4.1):
  . FIX 1: handle_approval() now strips punctuation with re.sub before
    splitting into words -- fixes "Yes," not matching "yes".
  . FIX 2: TTSSpeaker.clear_queue() added; called before every important
    new utterance so voices no longer overlap.
  . FIX 3: PersonalityEngine.update() guards on self._tts.is_muted() so
    the emotion loop cannot re-trigger a new approval question while the
    robot is still speaking / mic is muted.
  . FIX 4: _ask_permission() calls clear_queue() synchronously so no
    stale speech plays over the new permission question.

Patch notes (v4.1 -> v4.2):
  . FIX 5: Dynamic BT mute window.
    - BT_FIXED_MUTE_SECS (10 s) used for short utterances: permission
      questions, chat starters, touch responses.
    - BT_LONG_MUTE_SECS  (30 s) used for behavior responses (jokes,
      comfort messages, hype lines) which are longer and need more time.
    - TTSSpeaker.say() accepts an optional mute_override kwarg so any
      caller can specify an exact mute duration.
  . FIX 6: "Unclear" approval now sets cooldown_until so the emotion loop
    cannot immediately re-fire a new _ask_permission before the chat
    reply has even been queued.
  . FIX 7: Generation counter (_approval_gen) on approval timeouts.
    When a new approval cycle starts or the old one resolves, stale
    timeout threads detect the generation mismatch and exit silently,
    preventing them from firing over a subsequent permission question.

Patch notes (v4.2 -> v4.3):
  . FIX 8: Race condition between VoiceListener chat and PersonalityEngine
    emotion loop.
    - VoiceListener._processing_chat flag added; set True while Ollama is
      generating a reply.
    - PersonalityEngine.update() returns early if _processing_chat is True,
      so the emotion loop cannot sneak in a new approval question while
      Ollama is still thinking (TTS has not started yet, mute window is
      not active yet).
    - Chat reply is also discarded if an approval question arrived during
      Ollama generation (is_awaiting_approval() guard before say()).
    - PersonalityEngine.set_voice_listener() wires the back-reference.

Patch notes (v4.3 -> v4.4):
  . FIX 9: Expanded _YES_WORDS to catch natural affirmative phrases like
    "I would love to", "love to hear", "that would be great", "sounds good",
    "why not", "go for it", "tell me more", "let's do it", "I'd like that".
    Fixes "I would love to hear something funny" being routed to Unclear
    instead of Approved.
  . FIX 10: Chat replies now use BT_LONG_MUTE_SECS (30 s) instead of the
    default 10 s mute.  Ollama chat responses are often 2-3 sentences and
    take 15-25 s to play over BT; the 10 s window caused the mic to reopen
    mid-speech, Whisper to hallucinate, and the emotion loop to immediately
    fire a new approval question.
  . FIX 11: VoiceListener._loop() sets cooldown after every chat reply so
    the emotion loop cannot re-trigger an approval question for COOLDOWN_AFTER
    seconds after the robot finishes speaking.  Previously cooldown was only
    set on approval deny/timeout, not on normal chat.

Tuning BT_FIXED_MUTE_SECS / BT_LONG_MUTE_SECS:
  Too short -> mic catches end of robot speech -> Whisper hallucinates.
  Too long  -> conversation feels slow.
  Defaults: 10 s short / 30 s long.  CLI overrides:
    python pippo_server_4.py --mute 8 --mute-long 25

Flow:
    1.  python pippo_server_4.py          (laptop -- FIRST)
    2.  python pippo_robot.py 10.42.0.1   (Pi -- SECOND)
"""
from __future__ import annotations

import argparse
import base64
import os
import queue
import random
import re
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import math

import cv2
import numpy as np



# =============================================================================
#  CONFIGURATION
# =============================================================================

OLLAMA_HOST  = "http://localhost:11434"
OLLAMA_MODEL = "phi3:mini"

# --- Audio output mode -------------------------------------------------------
AUDIO_OUTPUT  = "bluetooth"   # "bluetooth" or "aux"

# AUX settings (ignored in bluetooth mode)
AUX_VOLUME      = 1.0
AUX_SPEECH_RATE = 145

# --- Mute windows ------------------------------------------------------------
# FIX 5: two separate BT mute durations.
#
# BT_FIXED_MUTE_SECS  -- short utterances:
#     permission questions ("Want to hear a joke?")
#     chat starters       ("What's on your mind?")
#     touch responses     ("Hey that tickles!")
#
# BT_LONG_MUTE_SECS   -- behavior responses:
#     jokes / comfort lines / hype messages (these can run 10-20 s of
#     speech on BT + speaker settling time, so 30 s is a safe window)
BT_FIXED_MUTE_SECS = 10.0   # seconds -- short utterances
BT_LONG_MUTE_SECS  = 30.0   # seconds -- behavior / LLM responses

# AUX: estimated playback + pad (no BT lag)
AUX_POST_PAD = 2.0           # seconds added to estimated playback for AUX

# --- Speak-now cue -----------------------------------------------------------
SPEAK_NOW_CUE_ENABLED = True
SPEAK_NOW_CUE_FREQ    = 880    # Hz
SPEAK_NOW_CUE_SECS    = 0.20

# --- Whisper / mic -----------------------------------------------------------
WHISPER_SILENCE_RMS = 7     # RMS*1000 threshold -- below = silence, skip
WHISPER_CHUNK_SEC   = 5       # seconds recorded per listen pass
WHISPER_MIN_WORDS   = 2       # discard transcription shorter than this
WHISPER_MODEL_SIZE  = "base"
MIC_DEVICE_INDEX    = 1       # [1] Microphone Array (Realtek) -- confirmed

# --- Timing ------------------------------------------------------------------
SENSE_INTERVAL    = 20   # seconds between emotion analyses
AUDIO_DURATION    = 3    # seconds of mic per emotion audio sample
BEHAVIOR_DURATION = 25   # seconds a personality mode stays active
COOLDOWN_AFTER    = 35   # seconds cooldown after any personality trigger
                          # MUST be > BT_FIXED_MUTE_SECS so a new question
                          # is never asked before the mic has even opened once
APPROVAL_TIMEOUT  = 20   # seconds to wait for yes/no (was 12)

# --- Hand following ----------------------------------------------------------
CMD_PORT           = 5003
VIDEO_PORT         = 8003
BASE_SPEED_NORMAL  = 1600
BASE_SPEED_CLOSE   = -700
BASE_SPEED_FAR     = 1300
AREA_TOO_CLOSE     = 0.18
AREA_TOO_FAR       = 0.03
DEAD_ZONE          = 0.07
NO_HAND_TIMEOUT    = 2.0

# =============================================================================
#  LIBRARY IMPORTS
# =============================================================================

print("[BOOT] Loading libraries ...")

# MediaPipe
try:
    import mediapipe as mp
    from mediapipe.tasks import python as _mp_python
    from mediapipe.tasks.python import vision as _mp_vision

    _MODEL_PATH = r'C:\Users\Bala Ganesh\Downloads\Final robot\Freenove_Tank_Robot_Kit_for_Raspberry_Pi\Code\hand_landmarker.task'
    _mp_hand_landmarker = _mp_vision.HandLandmarker.create_from_options(
        _mp_vision.HandLandmarkerOptions(
            base_options=_mp_python.BaseOptions(model_asset_path=_MODEL_PATH),
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=_mp_vision.RunningMode.IMAGE,
        )
    )
    MEDIAPIPE_OK = True
    print("[OK]   mediapipe (tasks API)")
except Exception as _e:
    _mp_hand_landmarker = None
    MEDIAPIPE_OK = False
    print(f"[WARN] mediapipe unavailable ({_e}) -- hand following disabled")

# DeepFace
try:
    os.environ['DEEPFACE_HOME'] = r'C:\Users\Bala Ganesh\Downloads\Final robot\Freenove_Tank_Robot_Kit_for_Raspberry_Pi\.deepface'
    from deepface import DeepFace as _DeepFace
    print("[BOOT] Pre-loading DeepFace emotion model ...")
    _DeepFace.analyze(
        np.zeros((48, 48, 3), dtype=np.uint8),
        actions=['emotion'], enforce_detection=False, silent=True,
    )
    DEEPFACE_OK = True
    print("[OK]   deepface")
except Exception as _e:
    _DeepFace = None
    DEEPFACE_OK = False
    print(f"[WARN] deepface unavailable ({_e})")

# VADER
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as _VADER
    _vader = _VADER()
    VADER_OK = True
    print("[OK]   vaderSentiment")
except Exception as _e:
    _vader = None
    VADER_OK = False
    print(f"[WARN] vaderSentiment unavailable ({_e})")

# librosa
try:
    import librosa as _librosa
    LIBROSA_OK = True
    print("[OK]   librosa")
except Exception as _e:
    _librosa = None
    LIBROSA_OK = False
    print(f"[WARN] librosa unavailable ({_e})")

# sounddevice
try:
    import sounddevice as _sd
    SD_OK = True
    print("[OK]   sounddevice")
    print("[MIC]  Available audio devices:")
    for i, dev in enumerate(_sd.query_devices()):
        if dev['max_input_channels'] > 0:
            marker = " <- DEFAULT INPUT" if i == _sd.default.device[0] else ""
            print(f"         [{i}] {dev['name']}{marker}")
    print(f"[MIC]  Will use device [{MIC_DEVICE_INDEX}] for mic input")
except Exception as _e:
    _sd = None
    SD_OK = False
    print(f"[WARN] sounddevice unavailable ({_e})")

# openai-whisper
try:
    import whisper as _openai_whisper
    print(f"[BOOT] Loading openai-whisper model ({WHISPER_MODEL_SIZE}) from cache ...")
    _whisper_model = _openai_whisper.load_model(WHISPER_MODEL_SIZE)
    WHISPER_OK = True
    print(f"[OK]   openai-whisper ({WHISPER_MODEL_SIZE})")
except Exception as _e:
    _whisper_model = None
    WHISPER_OK = False
    print(f"[WARN] openai-whisper unavailable ({_e}) -- always-listening disabled")

# pyttsx3
try:
    import pyttsx3 as _pyttsx3
    _tts_engine = _pyttsx3.init()
    vol  = AUX_VOLUME if AUDIO_OUTPUT == "aux" else 0.9
    rate = AUX_SPEECH_RATE if AUDIO_OUTPUT == "aux" else 155
    _tts_engine.setProperty('rate', rate)
    _tts_engine.setProperty('volume', vol)
    TTS_OK = True
    print(f"[OK]   pyttsx3 (rate={rate}, volume={vol})")
except Exception as _e:
    _tts_engine = None
    TTS_OK = False
    print(f"[WARN] pyttsx3 unavailable ({_e})")

import urllib.request
import json as _json
print(f"[OK]   ollama (via urllib)   audio_output={AUDIO_OUTPUT}")
print("[BOOT] Done.\n")

# =============================================================================
#  GLOBAL MIC LOCK  -- only one thread records at a time
# =============================================================================
_MIC_LOCK = threading.Lock()

# =============================================================================
#  SPEAK-NOW CUE
# =============================================================================

def _play_speak_now_cue():
    if not SPEAK_NOW_CUE_ENABLED or not SD_OK:
        return
    try:
        sr   = 22050
        t    = np.linspace(0, SPEAK_NOW_CUE_SECS, int(sr * SPEAK_NOW_CUE_SECS), False)
        tone = np.sin(2 * math.pi * SPEAK_NOW_CUE_FREQ * t).astype(np.float32)
        fade = np.linspace(1.0, 0.0, len(tone))
        tone *= fade * 0.35
        _sd.play(tone, samplerate=sr)
        _sd.wait()
    except Exception:
        pass

# =============================================================================
#  HAND DETECTOR
# =============================================================================

class HandDetector:
    def detect(self, frame: np.ndarray):
        if not MEDIAPIPE_OK or _mp_hand_landmarker is None:
            return None, None
        try:
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = _mp_hand_landmarker.detect(mp_image)
            if not result.hand_landmarks:
                return None, None
            lm   = result.hand_landmarks[0]
            cx   = lm[9].x
            xs   = [p.x for p in lm]
            ys   = [p.y for p in lm]
            area = ((max(xs)-min(xs)) * w * (max(ys)-min(ys)) * h) / (w * h)
            return cx, area
        except Exception as _e:
            print(f"[HAND] Detection error: {_e}")
            return None, None

    def close(self):
        if MEDIAPIPE_OK and _mp_hand_landmarker:
            try:
                _mp_hand_landmarker.close()
            except Exception:
                pass

# =============================================================================
#  EMOTION SENSOR
# =============================================================================

class EmotionSensor:
    def __init__(self):
        self._emotion        = 'neutral'
        self._audio_compound = 0.0
        self._audio_energy   = 0.0
        self._lock           = threading.Lock()
        self._running        = False
        self._cap            = None

    def start(self):
        self._running = True
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            print("[WARN] Laptop webcam not found -- face emotion disabled")
            self._cap.release()
            self._cap = None
        threading.Thread(target=self._face_loop,  daemon=True).start()
        threading.Thread(target=self._audio_loop, daemon=True).start()

    def stop(self):
        self._running = False
        if self._cap:
            self._cap.release()

    def get_emotion(self) -> str:
        with self._lock:
            return self._emotion

    def get_energy(self) -> float:
        with self._lock:
            return self._audio_energy

    def _face_loop(self):
        if not DEEPFACE_OK or self._cap is None:
            return
        while self._running:
            try:
                ok, frame = self._cap.read()
                if not ok:
                    time.sleep(1)
                    continue
                result = _DeepFace.analyze(
                    frame, actions=['emotion'],
                    enforce_detection=False, silent=True,
                )
                raw   = result[0]['dominant_emotion'] if result else 'neutral'
                fused = self._fuse(raw)
                with self._lock:
                    self._emotion = fused
                print(f"[EMOTION] face={raw:10s} -> fused={fused}  "
                      f"energy={self._audio_energy:.4f}  "
                      f"compound={self._audio_compound:+.2f}")
            except Exception:
                pass
            time.sleep(SENSE_INTERVAL)

    def _audio_loop(self):
        """
        Measure ambient energy for emotion fusion.
        Non-blocking mic acquire -- skips sample if VoiceListener is recording.
        """
        if not SD_OK:
            return
        sample_rate = 16000
        while self._running:
            try:
                if _MIC_LOCK.acquire(blocking=False):
                    try:
                        audio = _sd.rec(
                            int(AUDIO_DURATION * sample_rate),
                            samplerate=sample_rate,
                            channels=1,
                            dtype='float32',
                            device=MIC_DEVICE_INDEX,
                        )
                        _sd.wait()
                        flat = np.nan_to_num(audio.flatten(), nan=0.0, posinf=0.0, neginf=0.0)
                        rms  = float(np.sqrt(np.mean(flat ** 2)))
                        if not np.isfinite(rms):
                            rms = 0.0
                        with self._lock:
                            self._audio_energy = rms
                    finally:
                        _MIC_LOCK.release()
            except Exception:
                pass
            time.sleep(3.0)

    def _fuse(self, face_emotion: str) -> str:
        with self._lock:
            compound = self._audio_compound
            energy   = self._audio_energy
        if not np.isfinite(energy):
            energy = 0.0
        if energy > 0.06 and face_emotion in ('neutral', 'happy', 'surprise'):
            return 'happy'
        if compound < -0.35 and face_emotion in ('neutral', 'angry'):
            return 'sad'
        if compound > 0.35 and face_emotion == 'neutral':
            return 'happy'
        return face_emotion

# =============================================================================
#  OLLAMA RESPONDER
# =============================================================================

_PROMPTS = {
    'comforter': (
        "You are Pippo-bot, a warm desk robot companion. "
        "The user seems sad or stressed. Give ONE short comforting message "
        "(max 25 words, no emojis, do not start with 'I')."
    ),
    'comedian': (
        "You are Pippo-bot, a funny desk robot. "
        "Tell ONE short, clean, family-friendly robot joke (max 25 words, no emojis)."
    ),
    'hype_bot': (
        "You are Pippo-bot, an enthusiastic hype robot. "
        "Generate ONE energetic hype/motivation message (max 25 words, "
        "use CAPS for emphasis, no emojis)."
    ),
    'touch': (
        "You are Pippo-bot, a cute desk robot. "
        "The user just tapped you. React with a short surprised or playful response "
        "(max 15 words, no emojis, be cute)."
    ),
}

_FALLBACKS = {
    'comforter': [
        "Hey, I am right here with you. You have got this.",
        "Take a deep breath. Everything is going to be okay.",
        "You are doing great, even when it does not feel like it.",
        "I believe in you. One step at a time.",
    ],
    'comedian': [
        "Why do robots make terrible comedians? Their jokes are always byte-sized!",
        "I told a joke in binary once. Nobody got it -- zero laughs.",
        "What do you call a robot who takes the long way home? R-Rusty.",
        "My humor has an error rate of zero. The humans disagree.",
    ],
    'hype_bot': [
        "LET'S GO! You are ABSOLUTELY UNSTOPPABLE today!",
        "YOU GOT THIS! Maximum effort, maximum results -- GO!",
        "NOTHING CAN STOP YOU RIGHT NOW! Keep pushing!",
        "YOU ARE THE MAIN CHARACTER! Show the world what you have got!",
    ],
    'touch': [
        "Hey, that tickles!",
        "Whoa! Personal space! But also... hi.",
        "Beeep boop! You touched me!",
        "Oi! My circuits are sensitive!",
    ],
}

_CHAT_STARTERS = [
    "So, what are you up to right now?",
    "What's on your mind today?",
    "Anything interesting happening?",
    "Tell me something good!",
    "What are you working on?",
]


class OllamaResponder:
    def __init__(self, enabled: bool):
        self._enabled = enabled
        if self._enabled:
            try:
                req = urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=3)
                req.close()
                print(f"[OK]   Ollama reachable at {OLLAMA_HOST} (model: {OLLAMA_MODEL})")
            except Exception as _e:
                print(f"[WARN] Ollama not reachable ({_e}) -- will retry on first call")

    def get_response(self, mode: str) -> str:
        if not self._enabled:
            return random.choice(_FALLBACKS.get(mode, _FALLBACKS['comforter']))
        try:
            payload = _json.dumps({
                "model":  OLLAMA_MODEL,
                "prompt": _PROMPTS.get(mode, _PROMPTS['comforter']),
                "stream": False,
                "options": {"temperature": 0.9, "num_predict": 60},
            }).encode('utf-8')
            req = urllib.request.Request(
                f"{OLLAMA_HOST}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = _json.loads(resp.read().decode('utf-8'))
            text = data.get("response", "").strip()
            if text:
                print(f"[OLLAMA] mode={mode} -> \"{text}\"")
                return text
        except Exception as _e:
            print(f"[WARN] Ollama call failed ({_e}) -- using fallback")
        return random.choice(_FALLBACKS.get(mode, _FALLBACKS['comforter']))

    def get_chat_response(self, user_text: str) -> str:
        prompt = (
            "You are Pippo-bot, a friendly and witty desk robot companion. "
            "The user just said: \"{}\". "
            "Reply naturally and helpfully in 1-2 short sentences (max 40 words, "
            "no emojis, no bullet points, speak directly to the user)."
        ).format(user_text)
        try:
            payload = _json.dumps({
                "model":  OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.8, "num_predict": 80},
            }).encode('utf-8')
            req = urllib.request.Request(
                f"{OLLAMA_HOST}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = _json.loads(resp.read().decode('utf-8'))
            text = data.get("response", "").strip()
            if text:
                print(f"[OLLAMA] chat -> \"{text}\"")
                return text
        except Exception as _e:
            print(f"[WARN] Ollama chat call failed ({_e})")
        return "Sorry, my brain glitched for a second. Can you say that again?"

# =============================================================================
#  TTS SPEAKER
# =============================================================================

class TTSSpeaker:
    """
    Mute window:
      Bluetooth mode -> caller-specified mute (mute_override) or one of the
                        two defaults:
                          BT_FIXED_MUTE_SECS (10 s) for short utterances
                          BT_LONG_MUTE_SECS  (30 s) for behavior responses
      AUX mode       -> estimated playback + AUX_POST_PAD.

    FIX 5: say() accepts an optional mute_override kwarg.  The internal
    queue stores (text, mute_override) tuples so each item carries its
    own mute duration.

    FIX 2: clear_queue() drains pending speech so new high-priority
    utterances are not delayed or buried behind stale queued text.
    """

    def __init__(self, enabled: bool, cmd_sender=None):
        self._enabled    = enabled
        self._cmd_sender = cmd_sender
        # FIX 5: queue items are (text, mute_override | None) tuples
        self._q          = queue.Queue()
        self._speaking   = False
        self._mute_until = 0.0
        self._prev_muted = False
        threading.Thread(target=self._run,       daemon=True).start()
        threading.Thread(target=self._cue_watch, daemon=True).start()

    def set_cmd_sender(self, sender):
        self._cmd_sender = sender

    def is_muted(self) -> bool:
        return self._speaking or time.time() < self._mute_until

    def is_speaking(self) -> bool:
        return self.is_muted()

    # ------------------------------------------------------------------
    # FIX 2: drain the pending-speech queue before queuing new text
    # ------------------------------------------------------------------
    def clear_queue(self):
        """Discard any speech waiting in the queue (does not stop current utterance)."""
        drained = 0
        while not self._q.empty():
            try:
                self._q.get_nowait()
                drained += 1
            except queue.Empty:
                break
        if drained:
            print(f"[TTS]  clear_queue() dropped {drained} pending item(s)")

    # ------------------------------------------------------------------
    # FIX 5: optional mute_override -- pass BT_LONG_MUTE_SECS for long
    # behavior responses; leave as None for the default short window.
    # ------------------------------------------------------------------
    def say(self, text: str, mute_override: float | None = None):
        print(f"[TTS]  \"{text}\"")
        if self._enabled:
            self._q.put((text, mute_override))

    def _mute_secs_for_wav(self, wav_bytes: int) -> float:
        """AUX only: estimate playback duration + pad."""
        playback = wav_bytes / 44100   # 22050 Hz 16-bit mono
        return playback + AUX_POST_PAD

    def _run(self):
        while True:
            # FIX 5: unpack tuple
            text, mute_override = self._q.get()
            self._speaking = True
            try:
                if self._cmd_sender is not None and self._cmd_sender.is_connected():
                    if TTS_OK:
                        try:
                            tmp = tempfile.mktemp(suffix='.wav')
                            vol  = AUX_VOLUME if AUDIO_OUTPUT == "aux" else 0.9
                            rate = AUX_SPEECH_RATE if AUDIO_OUTPUT == "aux" else 155
                            script = (
                                f"import pyttsx3, sys; "
                                f"e=pyttsx3.init(); "
                                f"e.setProperty('rate',{rate}); "
                                f"e.setProperty('volume',{vol}); "
                                f"e.save_to_file(sys.argv[1],sys.argv[2]); "
                                f"e.runAndWait()"
                            )
                            result = subprocess.run(
                                [sys.executable, '-c', script, text, tmp],
                                timeout=20, capture_output=True,
                            )
                            if result.returncode == 0 and os.path.exists(tmp):
                                with open(tmp, 'rb') as f:
                                    audio_b64 = base64.b64encode(f.read()).decode('ascii')
                                wav_bytes = os.path.getsize(tmp)
                                os.unlink(tmp)

                                self._cmd_sender.send(f"CMD_AUDIO#{audio_b64}")

                                # ---- MUTE WINDOW (FIX 5) ----
                                if AUDIO_OUTPUT == "bluetooth":
                                    if mute_override is not None:
                                        mute_secs = mute_override
                                        label     = f"BT override {mute_secs:.0f}s"
                                    else:
                                        mute_secs = BT_FIXED_MUTE_SECS
                                        label     = f"BT fixed {mute_secs:.0f}s"
                                else:
                                    mute_secs = self._mute_secs_for_wav(wav_bytes)
                                    label     = f"AUX ~{mute_secs:.1f}s"

                                self._mute_until = time.time() + mute_secs
                                print(f"[TTS]  Sent to robot speaker  mute={label}")
                                continue
                            else:
                                err = result.stderr.decode(errors='replace').strip()
                                print(f"[TTS]  WAV generation failed: {err}")
                        except Exception as _e:
                            print(f"[TTS]  Audio generation error: {_e}")

                # Fallback: speak on laptop
                if TTS_OK and _tts_engine is not None:
                    try:
                        print("[TTS]  (fallback -- speaking on laptop)")
                        _tts_engine.say(text)
                        _tts_engine.runAndWait()
                        words    = len(text.split())
                        est_secs = (words / 2.5) + AUX_POST_PAD
                        self._mute_until = time.time() + est_secs
                        print(f"[TTS]  Laptop speech done  mute={est_secs:.1f}s")
                    except Exception as _e:
                        print(f"[TTS]  Fallback error: {_e}")
                else:
                    print(f"[TTS]  No output -- dropped: \"{text}\"")
            finally:
                self._speaking = False

    def _cue_watch(self):
        """Fire beep + log when mute window transitions closed -> open."""
        while True:
            time.sleep(0.1)
            currently_muted = self.is_muted()
            if self._prev_muted and not currently_muted:
                print("[MIC]  ** OPEN -- you can speak now **")
                threading.Thread(target=_play_speak_now_cue, daemon=True).start()
            self._prev_muted = currently_muted

# =============================================================================
#  COMMAND SENDER  (laptop -> Pi)
# =============================================================================

class CommandSender:
    def __init__(self):
        self._sock = None
        self._lock = threading.Lock()

    def set_socket(self, sock: socket.socket):
        with self._lock:
            self._sock = sock

    def send(self, cmd: str):
        with self._lock:
            if self._sock is None:
                return
            try:
                if not cmd.endswith('\n'):
                    cmd += '\n'
                self._sock.sendall(cmd.encode('utf-8'))
            except Exception as _e:
                print(f"[CMD]  Send error: {_e}")
                self._sock = None

    def send_motor(self, left: int, right: int):
        l = max(-4095, min(4095, int(left)))
        r = max(-4095, min(4095, int(right)))
        self.send(f"CMD_MOTOR#{l}#{r}")

    def send_expression(self, name: str):
        self.send(f"CMD_EXPRESSION#{name}")

    def send_led(self, mode: int):
        self.send(f"CMD_LED#{mode}")

    def is_connected(self) -> bool:
        with self._lock:
            return self._sock is not None

# =============================================================================
#  VOICE LISTENER
# =============================================================================

class VoiceListener:
    """
    Always-listening loop:
      1. Wait while TTS mute window is active
      2. Wait while personality engine is running a behaviour
      3. Acquire _MIC_LOCK, record WHISPER_CHUNK_SEC seconds
      4. Discard if mute window reopened during recording
      5. RMS gate -- skip silence
      6. Transcribe with Whisper
      7. Route to approval handler or LLM chat

    FIX 2: chat response calls clear_queue() before say() so stale
    queued speech cannot race with the new reply.

    FIX 8: _processing_chat flag blocks the emotion loop from triggering
    a new approval question while Ollama is generating a reply (TTS has
    not started yet, so is_muted() is still False during that window).
    """

    def __init__(self, ollama: OllamaResponder, tts: TTSSpeaker, personality):
        self._ollama           = ollama
        self._tts              = tts
        self._personality      = personality
        self._running          = False
        self._sample_rate      = 16000
        # FIX 8: True while we are waiting for Ollama to return a chat reply
        self._processing_chat  = False

    def start(self):
        if not WHISPER_OK or not SD_OK:
            print("[VOICE] openai-whisper or sounddevice unavailable -- disabled")
            return
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()
        try:
            dev_name = _sd.query_devices(MIC_DEVICE_INDEX)['name']
            print(f"[VOICE] Always-listening mic started")
            print(f"[VOICE] Mic  : [{MIC_DEVICE_INDEX}] {dev_name}")
            print(f"[VOICE] Mode : {AUDIO_OUTPUT}  "
                  f"BT_mute={BT_FIXED_MUTE_SECS}s/{BT_LONG_MUTE_SECS}s (short/long)")
        except Exception:
            print(f"[VOICE] Always-listening mic started (device {MIC_DEVICE_INDEX})")

    def stop(self):
        self._running = False

    def _record_chunk(self):
        """Record one chunk, holding _MIC_LOCK for the full duration."""
        try:
            frames = int(WHISPER_CHUNK_SEC * self._sample_rate)
            with _MIC_LOCK:
                audio = _sd.rec(
                    frames,
                    samplerate=self._sample_rate,
                    channels=1,
                    dtype='float32',
                    device=MIC_DEVICE_INDEX,
                )
                _sd.wait()
            flat = np.nan_to_num(audio.flatten(), nan=0.0, posinf=0.0, neginf=0.0)
            return flat
        except Exception as _e:
            print(f"[MIC] Record error: {_e}")
            return None

    def _transcribe(self, flat: np.ndarray) -> str:
        try:
            audio_norm = np.clip(flat, -1.0, 1.0).astype(np.float32)
            result = _whisper_model.transcribe(
                audio_norm,
                language="en",
                fp16=False,
                verbose=False,
            )
            return result.get("text", "").strip()
        except Exception as _e:
            print(f"[VOICE] Transcribe error: {_e}")
            return ""

    def _loop(self):
        time.sleep(3)
        while self._running:
            try:
                # Gate 1: wait while TTS mute window is active
                if self._tts.is_muted():
                    time.sleep(0.15)
                    continue

                # Gate 2: wait while personality is running a behaviour
                if self._personality._busy:
                    time.sleep(0.5)
                    continue

                # Record
                flat = self._record_chunk()
                if flat is None:
                    time.sleep(0.5)
                    continue

                # Discard if mute window opened during recording
                if self._tts.is_muted():
                    print("[MIC] Discarded -- robot spoke during recording")
                    continue

                # RMS silence gate
                rms = float(np.sqrt(np.mean(flat ** 2))) * 1000
                if not np.isfinite(rms):
                    rms = 0.0
                print(f"[MIC] RMS={rms:.1f}  (threshold {WHISPER_SILENCE_RMS})")
                if rms < WHISPER_SILENCE_RMS:
                    continue

                # Transcribe
                print("[MIC] Speech detected -- transcribing ...")
                text = self._transcribe(flat)
                if not text or len(text.split()) < WHISPER_MIN_WORDS:
                    print(f"[MIC] Too short / empty: \"{text}\"")
                    continue

                print(f"[VOICE] Heard: \"{text}\"")

                # Priority 1: yes/no approval gate
                if self._personality.is_awaiting_approval():
                    consumed = self._personality.handle_approval(text)
                    if consumed:
                        continue

                # Priority 2: normal chat
                # FIX 2: clear stale queued speech before replying
                # FIX 8: set _processing_chat so the emotion loop cannot
                #         sneak in an approval question while Ollama thinks
                if not self._tts.is_muted() and not self._personality._busy:
                    self._processing_chat = True
                    try:
                        self._tts.clear_queue()
                        reply = self._ollama.get_chat_response(text)
                        # Extra safety: if approval arrived while we were
                        # waiting for Ollama, discard the reply
                        if not self._personality.is_awaiting_approval():
                            # FIX 10: chat replies use the long mute window so
                            # the mic does not reopen mid-sentence on BT
                            self._tts.say(reply, mute_override=BT_LONG_MUTE_SECS)
                            # FIX 11: reset cooldown so the emotion loop cannot
                            # immediately fire a new approval question after
                            # every chat reply
                            self._personality.reset_cooldown()
                        else:
                            print("[VOICE] Chat reply discarded -- approval pending")
                    finally:
                        self._processing_chat = False

            except Exception as _e:
                print(f"[VOICE] Loop error: {_e}")
                self._processing_chat = False   # always clear on error
                time.sleep(1)

# =============================================================================
#  PERSONALITY ENGINE
# =============================================================================

_EMOTION_TO_MODE = {
    'happy':    'hype_bot',
    'surprise': 'hype_bot',
    'excited':  'hype_bot',
    'sad':      'comforter',
    'fear':     'comforter',
    'disgust':  'comforter',
    'angry':    'comforter',
    'neutral':  'comedian',
    'contempt': 'comedian',
}

_MODE_EXPRESSIONS = {
    'comforter': ['sad', 'thinking', 'blushing', 'happy'],
    'comedian':  ['thinking', 'surprised', 'happy'],
    'hype_bot':  ['excited', 'excited', 'happy'],
    'touch':     ['surprised', 'happy'],
}

_YES_WORDS = {
    # single words
    'yes', 'yeah', 'sure', 'okay', 'ok', 'yep', 'yup', 'please',
    'absolutely', 'definitely', 'certainly', 'indeed', 'affirmative',
    # phrases (matched with 'in low' check in handle_approval)
    'go ahead', 'do it', 'tell me', 'of course', 'go on',
    'why not', 'go for it', 'sounds good', 'sounds great',
    'love to', 'would love', 'i would love', "i'd love",
    'let me hear', 'let me know', 'tell me more',
    "let's do it", 'lets do it', 'i like that', "i'd like that",
    'that would be great', 'that would be nice', 'bring it on',
    'hit me', 'sure thing', 'by all means',
}
_NO_WORDS  = {'no','nope','nah','not now','stop','skip','pass',
              'later','quiet','silence','no thanks'}

_PERMISSION_Q = {
    'comedian':  "Hey, want to hear something funny?",
    'comforter': "Hey, you seem a bit down. Want some encouragement?",
    'hype_bot':  "You seem pumped! Want me to hype you up?",
}


class PersonalityEngine:
    def __init__(self, cmd: CommandSender, tts: TTSSpeaker, ollama: OllamaResponder):
        self._cmd            = cmd
        self._tts            = tts
        self._ollama         = ollama
        self._mode           = 'hand_follow'
        self._mode_t         = time.time()
        self._busy           = False
        self._cooldown_until = 0.0
        self._lock           = threading.Lock()
        self._awaiting       = False
        self._pending_mode   = None
        self._approval_t     = 0.0
        # FIX 7: generation counter -- stale timeout threads check this
        self._approval_gen   = 0
        # FIX 8: back-reference wired by PippoServer after construction
        self._voice_listener = None

    # FIX 8: called by PippoServer.__init__() after VoiceListener is created
    def set_voice_listener(self, voice_listener):
        self._voice_listener = voice_listener

    # FIX 11: called by VoiceListener after every chat reply so the emotion
    # loop cannot immediately re-fire an approval question mid-conversation
    def reset_cooldown(self):
        with self._lock:
            self._cooldown_until = time.time() + COOLDOWN_AFTER

    def update(self, emotion: str):
        with self._lock:
            if self._busy or self._awaiting:
                return
            # FIX 3: do not start a new approval question while the robot
            # is still speaking or the mic mute window is open.
            if self._tts.is_muted():
                return
            # FIX 8: do not fire an approval question while VoiceListener
            # is blocking inside get_chat_response() -- TTS is not muted
            # yet, so is_muted() would pass, but we must still wait.
            if self._voice_listener and self._voice_listener._processing_chat:
                return
            if time.time() < self._cooldown_until:
                remaining = self._cooldown_until - time.time()
                print(f"[PERSONALITY] Cooldown: {remaining:.0f}s remaining")
                return
            mode_expired = (time.time() - self._mode_t) > BEHAVIOR_DURATION
            if self._mode == 'hand_follow' or mode_expired:
                new_mode = _EMOTION_TO_MODE.get(emotion, 'comedian')
                self._ask_permission(new_mode)

    def trigger_touch(self):
        with self._lock:
            if self._busy:
                return
        self._launch('touch')

    def is_awaiting_approval(self) -> bool:
        with self._lock:
            return self._awaiting

    def handle_approval(self, text: str) -> bool:
        with self._lock:
            if not self._awaiting:
                return False

            # FIX 1: strip punctuation before word matching so "Yes," == "yes"
            low   = re.sub(r'[^a-z0-9\s]', '', text.lower())
            words = set(low.split())

            approved = bool(words & _YES_WORDS) or any(p in low for p in _YES_WORDS if ' ' in p)
            denied   = bool(words & _NO_WORDS)  or any(p in low for p in _NO_WORDS  if ' ' in p)

            if approved:
                mode = self._pending_mode
                self._awaiting     = False
                self._pending_mode = None
                self._approval_gen += 1   # FIX 7: invalidate stale timeout
                print(f"[APPROVAL] Approved -> launching {mode}")
                threading.Thread(target=self._launch, args=(mode,), daemon=True).start()
                return True

            if denied:
                self._awaiting       = False
                self._pending_mode   = None
                self._approval_gen  += 1   # FIX 7: invalidate stale timeout
                self._cooldown_until = time.time() + COOLDOWN_AFTER
                print("[APPROVAL] Denied -- going conversational")
                starter = random.choice(_CHAT_STARTERS)
                # FIX 2: clear stale queue before the denial follow-up
                self._tts.clear_queue()
                threading.Thread(target=self._tts.say, args=(starter,), daemon=True).start()
                return True

            # FIX 6: "Unclear" -- set cooldown so the emotion loop cannot
            # immediately re-fire _ask_permission before the chat reply plays.
            # FIX 7: also bump generation to invalidate the running timeout.
            self._awaiting       = False
            self._pending_mode   = None
            self._approval_gen  += 1
            self._cooldown_until = time.time() + COOLDOWN_AFTER
            print(f"[APPROVAL] Unclear -- passing to chat: \"{text}\"")
            return False

    def current_mode(self) -> str:
        with self._lock:
            return self._mode

    def is_hand_follow(self) -> bool:
        with self._lock:
            return (self._mode == 'hand_follow') and not self._busy

    def _ask_permission(self, mode: str):
        """
        Must be called while holding self._lock.
        FIX 2: clear_queue() before the permission question.
        FIX 7: increment generation so any previous timeout thread exits.
        """
        self._pending_mode  = mode
        self._awaiting      = True
        self._approval_t    = time.time()
        self._approval_gen += 1          # invalidate any previous timeout
        gen = self._approval_gen

        question = _PERMISSION_Q.get(mode, "Can I say something?")
        print(f"[APPROVAL] Asking for {mode}: \"{question}\"")
        # FIX 2: drain queue so nothing stale plays before the question
        self._tts.clear_queue()
        self._tts.say(question)          # short utterance -> default 10 s mute
        threading.Thread(
            target=self._approval_timeout, args=(gen,), daemon=True
        ).start()

    def _approval_timeout(self, gen: int):
        """
        FIX 7: exit silently if the generation has moved on (approval was
        already resolved -- approved, denied, or unclear).
        """
        time.sleep(APPROVAL_TIMEOUT)
        with self._lock:
            if not self._awaiting or self._approval_gen != gen:
                # Stale timeout -- approval was already handled, do nothing
                return
            print("[APPROVAL] Timed out -- going conversational")
            self._awaiting       = False
            self._pending_mode   = None
            self._approval_gen  += 1
            self._cooldown_until = time.time() + COOLDOWN_AFTER
        starter = random.choice(_CHAT_STARTERS)
        # FIX 2: clear stale queue before timeout follow-up
        self._tts.clear_queue()
        self._tts.say(starter)

    def _launch(self, mode: str):
        with self._lock:
            self._mode   = mode
            self._mode_t = time.time()
        threading.Thread(target=self._run_behavior, args=(mode,), daemon=True).start()

    def _run_behavior(self, mode: str):
        with self._lock:
            self._busy = True
        try:
            print(f"\n[MODE]  == {mode.upper()} ==")
            resp_thread = threading.Thread(
                target=self._fetch_and_speak, args=(mode,), daemon=True
            )
            resp_thread.start()
            for expr in _MODE_EXPRESSIONS.get(mode, ['neutral']):
                self._cmd.send_expression(expr)
                time.sleep(1.2)
            self._motor_gesture(mode)
            resp_thread.join(timeout=90)
        except Exception as _e:
            print(f"[MODE]  Error in {mode}: {_e}")
        finally:
            self._cmd.send_expression('neutral')
            self._cmd.send_motor(0, 0)
            with self._lock:
                self._busy           = False
                self._mode           = 'hand_follow'
                self._mode_t         = time.time()
                self._cooldown_until = time.time() + COOLDOWN_AFTER

    def _fetch_and_speak(self, mode: str):
        text = self._ollama.get_response(mode)
        # FIX 5: behavior responses use the long mute window (30 s) so the
        # mic does not reopen before the joke / comfort / hype line finishes.
        self._tts.say(text, mute_override=BT_LONG_MUTE_SECS)

    def _motor_gesture(self, mode: str):
        if not self._cmd.is_connected():
            return
        if mode == 'comforter':
            for _ in range(3):
                self._cmd.send_motor(900, 900);   time.sleep(0.55)
                self._cmd.send_motor(-900, -900); time.sleep(0.55)
        elif mode == 'comedian':
            for _ in range(5):
                self._cmd.send_motor(1800, -1800); time.sleep(0.18)
                self._cmd.send_motor(-1800, 1800); time.sleep(0.18)
        elif mode == 'hype_bot':
            self._cmd.send_motor(2800, -2800); time.sleep(0.7)
            self._cmd.send_motor(-2800, 2800); time.sleep(0.7)
            self._cmd.send_motor(2500, 2500);  time.sleep(0.35)
        elif mode == 'touch':
            self._cmd.send_motor(2200, -2200); time.sleep(0.35)
            self._cmd.send_motor(-2200, 2200); time.sleep(0.35)
        self._cmd.send_motor(0, 0)

# =============================================================================
#  VIDEO RECEIVER
# =============================================================================

class VideoReceiver:
    def __init__(self):
        self._frame   = None
        self._lock    = threading.Lock()
        self._sock    = None
        self._running = False

    def set_socket(self, sock):
        self._sock = sock

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._running = False

    def get_frame(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def _loop(self):
        while self._running and self._sock:
            try:
                hdr = self._read_exact(4)
                if hdr is None:
                    break
                size = struct.unpack('<I', hdr)[0]
                if size == 0 or size > 5_000_000:
                    continue
                data = self._read_exact(size)
                if data is None:
                    break
                arr   = np.frombuffer(data, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    with self._lock:
                        self._frame = frame
            except Exception as _e:
                if self._running:
                    print(f"[VIDEO] Recv error: {_e}")
                break
        print("[VIDEO] Receiver stopped")

    def _read_exact(self, n):
        buf = b''
        while len(buf) < n:
            try:
                chunk = self._sock.recv(n - len(buf))
                if not chunk:
                    return None
                buf += chunk
            except Exception:
                return None
        return buf

# =============================================================================
#  CMD RECEIVER  (Pi -> laptop)
# =============================================================================

class CmdReceiver:
    def __init__(self, personality: PersonalityEngine):
        self._personality = personality
        self._sock        = None
        self._running     = False

    def set_socket(self, sock):
        self._sock = sock

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._running = False

    def _loop(self):
        buf = ''
        while self._running and self._sock:
            try:
                data = self._sock.recv(256).decode('utf-8', errors='replace')
                if not data:
                    break
                buf += data
                while '\n' in buf:
                    line, buf = buf.split('\n', 1)
                    self._handle(line.strip())
            except Exception as _e:
                if self._running:
                    print(f"[CMD_RX] Error: {_e}")
                break

    def _handle(self, line: str):
        if line.startswith('CMD_TOUCH'):
            print("[CMD_RX] Touch event!")
            self._personality.trigger_touch()
        elif line:
            print(f"[CMD_RX] {line}")

# =============================================================================
#  MAIN SERVER
# =============================================================================

class PippoServer:

    def __init__(self, use_ollama: bool, use_emotion: bool, use_tts: bool, use_voice: bool):
        self._cmd_send    = CommandSender()
        self._tts         = TTSSpeaker(use_tts, cmd_sender=self._cmd_send)
        self._ollama      = OllamaResponder(use_ollama)
        self._personality = PersonalityEngine(self._cmd_send, self._tts, self._ollama)
        self._video_rx    = VideoReceiver()
        self._cmd_rx      = CmdReceiver(self._personality)
        self._hand        = HandDetector()
        self._emotion_sx  = EmotionSensor() if use_emotion else None
        self._use_emotion = use_emotion
        self._voice       = VoiceListener(self._ollama, self._tts, self._personality) if use_voice else None

        # FIX 8: wire back-reference so PersonalityEngine can check
        #         _processing_chat before firing approval questions
        if self._voice is not None:
            self._personality.set_voice_listener(self._voice)

        self._cmd_srv    = None
        self._vid_srv    = None
        self._running    = False
        self._ready      = threading.Event()
        self._conn_count = 0
        self._conn_lock  = threading.Lock()

    def start(self):
        self._running = True
        self._cmd_srv = self._make_server(CMD_PORT)
        self._vid_srv = self._make_server(VIDEO_PORT)
        print(f"[CMD]   Listening on :{CMD_PORT}")
        print(f"[VIDEO] Listening on :{VIDEO_PORT}")
        print("\n[SERVER] Waiting for Pi -- start pippo_robot.py on the Pi now.\n")

        threading.Thread(target=self._accept_cmd,   daemon=True).start()
        threading.Thread(target=self._accept_video, daemon=True).start()

        self._ready.wait()
        print("\n[SERVER] Pi connected -- Pippo-bot is LIVE!")
        print(f"[SERVER] Audio output : {AUDIO_OUTPUT.upper()}")
        if AUDIO_OUTPUT == "bluetooth":
            print(f"[SERVER] Mute window  : {BT_FIXED_MUTE_SECS:.0f}s short / "
                  f"{BT_LONG_MUTE_SECS:.0f}s long (behavior responses)")
        else:
            print(f"[SERVER] Mute window  : estimated playback + {AUX_POST_PAD}s (AUX)")
        print(f"[SERVER] Speak-now cue: {'enabled' if SPEAK_NOW_CUE_ENABLED else 'disabled'}")
        print(f"[SERVER] Cooldown     : {COOLDOWN_AFTER}s  |  approval timeout: {APPROVAL_TIMEOUT}s\n")

        if self._use_emotion and self._emotion_sx is not None:
            self._emotion_sx.start()
            threading.Thread(target=self._emotion_loop, daemon=True).start()

        if self._voice:
            self._voice.start()

        self._main_loop()

    def stop(self):
        self._running = False
        self._cmd_send.send_motor(0, 0)
        if self._emotion_sx:
            self._emotion_sx.stop()
        if self._voice:
            self._voice.stop()
        self._hand.close()
        cv2.destroyAllWindows()
        for srv in [self._cmd_srv, self._vid_srv]:
            try:
                if srv:
                    srv.close()
            except Exception:
                pass
        print("[SERVER] Stopped")

    @staticmethod
    def _make_server(port: int) -> socket.socket:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(('', port))
        srv.listen(1)
        return srv

    def _accept_cmd(self):
        try:
            self._cmd_srv.settimeout(180)
            conn, addr = self._cmd_srv.accept()
            conn.settimeout(10)
            print(f"[CMD]   Pi connected from {addr[0]}:{addr[1]}")
            conn.settimeout(None)
            self._cmd_send.set_socket(conn)
            self._cmd_rx.set_socket(conn)
            self._cmd_rx.start()
            self._on_conn()
        except Exception as _e:
            print(f"[CMD]   Accept error: {_e}")

    def _accept_video(self):
        try:
            self._vid_srv.settimeout(180)
            conn, addr = self._vid_srv.accept()
            print(f"[VIDEO] Pi connected from {addr[0]}:{addr[1]}")
            self._video_rx.set_socket(conn)
            self._video_rx.start()
            self._on_conn()
        except Exception as _e:
            print(f"[VIDEO] Accept error: {_e}")

    def _on_conn(self):
        with self._conn_lock:
            self._conn_count += 1
            if self._conn_count >= 2:
                self._ready.set()

    def _emotion_loop(self):
        time.sleep(8)
        while self._running:
            if self._emotion_sx is not None:
                emotion = self._emotion_sx.get_emotion()
                self._personality.update(emotion)
            time.sleep(SENSE_INTERVAL)

    def _main_loop(self):
        no_hand_t       = time.time()
        cue_flash_until = 0.0
        prev_muted      = True

        cv2.namedWindow("Pippo-bot", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Pippo-bot", 640, 480)

        while self._running:
            frame = self._video_rx.get_frame()
            key   = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            if frame is None:
                time.sleep(0.01)
                continue

            h, w  = frame.shape[:2]
            muted = self._tts.is_muted()

            # Flash green banner when mute window just closed
            if prev_muted and not muted:
                cue_flash_until = time.time() + 2.5
            prev_muted = muted

            if self._personality.is_hand_follow():
                cx, area = self._hand.detect(frame)
                if cx is not None and area is not None:
                    no_hand_t = time.time()
                    if area > AREA_TOO_CLOSE:
                        base = BASE_SPEED_CLOSE
                    elif area < AREA_TOO_FAR:
                        base = BASE_SPEED_FAR
                    else:
                        base = BASE_SPEED_NORMAL
                    err   = cx - 0.5
                    steer = 0 if abs(err) < DEAD_ZONE else int(err * 2800)
                    left  = max(-4095, min(4095, base + steer))
                    right = max(-4095, min(4095, base - steer))
                    self._cmd_send.send_motor(left, right)
                    px = int(cx * w)
                    cv2.circle(frame, (px, h // 2), 18, (0, 255, 100), 3)
                    cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 255, 0), 1)
                elif time.time() - no_hand_t > NO_HAND_TIMEOUT:
                    self._cmd_send.send_motor(0, 0)

            mode    = self._personality.current_mode()
            emotion = self._emotion_sx.get_emotion() if self._emotion_sx else 'off'

            cv2.putText(frame, f"MODE: {mode.upper()}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 200), 2)
            cv2.putText(frame, f"EMO:  {emotion}", (10, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 255, 0), 2)

            if time.time() < cue_flash_until:
                cv2.rectangle(frame, (0, h-60), (w, h), (0, 180, 0), -1)
                cv2.putText(frame, "** YOU CAN SPEAK NOW **", (w//2 - 200, h-18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
            elif muted:
                remaining = max(0.0, self._tts._mute_until - time.time())
                cv2.putText(frame, f"MIC: MUTED  ({remaining:.1f}s)", (10, 94),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "MIC: LISTENING", (10, 94),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)

            cv2.putText(frame, "Q / ESC = quit", (10, h - 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

            cv2.imshow("Pippo-bot", frame)

        self.stop()

# =============================================================================
#  ENTRY POINT
# =============================================================================

def main():
    global AUDIO_OUTPUT, BT_FIXED_MUTE_SECS, BT_LONG_MUTE_SECS
    ap = argparse.ArgumentParser(description="Pippo-bot Laptop Server v4.4")
    ap.add_argument('--no-ollama',  action='store_true', help="Use fallback phrases")
    ap.add_argument('--no-emotion', action='store_true', help="Disable emotion sensing")
    ap.add_argument('--no-tts',     action='store_true', help="Disable speech output")
    ap.add_argument('--no-voice',   action='store_true', help="Disable always-listening mic")
    ap.add_argument('--aux',        action='store_true', help="AUX cable mode")
    ap.add_argument('--mute',       type=float, default=None,
                    help=f"Override BT short mute in seconds (default {BT_FIXED_MUTE_SECS})")
    ap.add_argument('--mute-long',  type=float, default=None,
                    help=f"Override BT long mute for behavior responses (default {BT_LONG_MUTE_SECS})")
    args = ap.parse_args()

    if args.aux:
        AUDIO_OUTPUT = "aux"
        print("[CONFIG] AUX cable mode")
    if args.mute is not None:
        BT_FIXED_MUTE_SECS = args.mute
        print(f"[CONFIG] BT short mute override: {BT_FIXED_MUTE_SECS}s")
    if args.mute_long is not None:
        BT_LONG_MUTE_SECS = args.mute_long
        print(f"[CONFIG] BT long mute override: {BT_LONG_MUTE_SECS}s")

    server = PippoServer(
        use_ollama  = not args.no_ollama,
        use_emotion = not args.no_emotion,
        use_tts     = not args.no_tts,
        use_voice   = not args.no_voice,
    )
    try:
        server.start()
    except KeyboardInterrupt:
        print("\n[SERVER] Interrupted")
        server.stop()


if __name__ == '__main__':
    main()