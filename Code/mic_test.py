import sounddevice as sd
import numpy as np
import io, wave
from faster_whisper import WhisperModel

model = WhisperModel('tiny', device='cpu', compute_type='int8')
sr = 16000

print('Speak now for 4 seconds...')
audio = sd.rec(int(4*sr), samplerate=sr, channels=1, dtype='float32', device=1)
sd.wait()

flat = audio.flatten()
pcm = (np.clip(flat, -1.0, 1.0) * 32767).astype(np.int16)

buf = io.BytesIO()
with wave.open(buf, 'wb') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sr)
    wf.writeframes(pcm.tobytes())
buf.seek(0)

segments, _ = model.transcribe(buf, language='en', beam_size=3, vad_filter=True)
text = ' '.join(s.text.strip() for s in segments).strip()
print(f'Transcribed: "{text}"')