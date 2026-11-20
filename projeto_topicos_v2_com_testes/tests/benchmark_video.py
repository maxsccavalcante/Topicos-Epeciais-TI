"""
Roda o detector num vídeo com as configurações padrão do app.py
(pulo de 3 frames, largura máxima 480 px, inferência em 384 px) e imprime um resumo.
Também salva tests/output/bus_anotado.jpg (imagem de exemplo com pessoas reais).

Uso, a partir da pasta do projeto:
    python tests/benchmark_video.py [caminho_do_video]
"""
import os
import sys
import time

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
os.chdir(ROOT)
from src.detector import SafetyVisionDetector  # noqa: E402
from test_detector import bus_image_padded  # noqa: E402

video = sys.argv[1] if len(sys.argv) > 1 else os.path.join("samples", "demo_sample.mp4")
FRAME_SKIP, WIDTH = 3, 480
os.makedirs("tests/output", exist_ok=True)

det = SafetyVisionDetector("yolov8n-pose.pt", "ppe_amirt.pt")
det.reset_state()
cap = cv2.VideoCapture(video)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
idx = processed = max_persons = person_frames = 0
tempos = []
while cap.grab():
    idx += 1
    if idx % FRAME_SKIP:
        continue
    ok, frame = cap.retrieve()
    if not ok:
        break
    h0, w0 = frame.shape[:2]
    if w0 > WIDTH:
        frame = cv2.resize(frame, (WIDTH, int(h0 * WIDTH / w0)))
    t = time.perf_counter()
    _, _, stats = det.process_frame(frame, conf=0.42, ppe_conf=0.24, tracker=True, imgsz=384, pose_imgsz=384)
    tempos.append(time.perf_counter() - t)
    processed += 1
    max_persons = max(max_persons, stats["total_persons"])
    person_frames += stats["total_persons"] > 0
media = sum(tempos[1:]) / max(1, len(tempos) - 1)
print(f"Vídeo: {video} | frames: {total} | processados: {processed}")
print(f"Tempo médio por frame: {media * 1000:.0f} ms (~{1 / media:.1f} FPS de processamento)")
print(f"Frames com pessoa: {person_frames} | máx. pessoas num frame: {max_persons}")

# Imagem com pessoas reais, no mesmo tamanho que o app usa (480 px)
img = bus_image_padded()
img = cv2.resize(img, (WIDTH, int(img.shape[0] * WIDTH / img.shape[1])))
d2 = SafetyVisionDetector("yolov8n-pose.pt", "ppe_amirt.pt")
tempos2 = []
for i in range(6):
    t = time.perf_counter()
    out, v, stats = d2.process_frame(img.copy(), conf=0.42, ppe_conf=0.24, tracker=True, imgsz=384, pose_imgsz=384)
    tempos2.append(time.perf_counter() - t)
cv2.imwrite("tests/output/bus_anotado.jpg", out)
m2 = sum(tempos2[1:]) / (len(tempos2) - 1)
print(f"Imagem com pessoas: {stats['total_persons']} pessoas, {stats['non_conforming']} sem EPI; "
      f"{m2 * 1000:.0f} ms por frame com pessoas (~{1 / m2:.1f} FPS)")
print("Flagrantes:", [(x["track_id"], x["type"]) for x in v])
