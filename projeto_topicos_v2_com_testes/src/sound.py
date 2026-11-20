import wave
import struct
import math
import io
import base64

def generate_alert_sound_b64(freq=880, duration=0.20, volume=0.35):
    """
    Gera um som de bipe de alerta de áudio (formato WAV codificado em Base64).
    100% nativo em Python, não depende de arquivos externos ou internet.
    """
    sample_rate = 44100
    n_samples = int(sample_rate * duration)
    
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wav_file:
        wav_file.setnchannels(1)        # Mono
        wav_file.setsampwidth(2)       # 16-bit
        wav_file.setframerate(sample_rate)
        
        for i in range(n_samples):
            # Onda senoidal com decaimento suave no final
            decay = 1.0 - (i / n_samples) * 0.4
            sample = int(32767 * volume * decay * math.sin(2 * math.pi * freq * i / sample_rate))
            wav_file.writeframes(struct.pack('<h', sample))
            
    return base64.b64encode(buf.getvalue()).decode()

ALERT_BEEP_B64 = generate_alert_sound_b64(freq=850, duration=0.22, volume=0.40)
CRITICAL_SIREN_B64 = generate_alert_sound_b64(freq=1050, duration=0.30, volume=0.50)
