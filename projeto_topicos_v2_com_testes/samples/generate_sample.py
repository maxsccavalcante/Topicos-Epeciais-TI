"""
Script auxiliar para gerar um vídeo sintético de demonstração (caso o usuário queira testar sem ter um .mp4 em mãos).
Cria pessoas/formas em movimento entrando e saindo de zonas demarcadas.
"""
import cv2
import numpy as np
import os

def create_synthetic_demo_video(output_path="samples/demo_sample.mp4", duration_sec=10, fps=25):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = duration_sec * fps
    
    for f in range(total_frames):
        # Fundo simulando canteiro / piso industrial
        frame = np.full((height, width, 3), 40, dtype=np.uint8)
        
        # Grid no chão
        for y in range(0, height, 40):
            cv2.line(frame, (0, y), (width, y), (55, 55, 55), 1)
        for x in range(0, width, 60):
            cv2.line(frame, (x, 0), (x, height), (55, 55, 55), 1)
            
        # Desenhar uma área de perigo no chão
        cv2.rectangle(frame, (350, 240), (600, 440), (30, 30, 80), -1)
        cv2.putText(frame, "PERIGO - MAQUINARIO", (360, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)
        
        # Simular pessoa 1 caminhando para a direita (fora da zona de perigo)
        p1_x = int(50 + (f * 2.5) % 400)
        p1_y = 200
        # Corpo simples
        cv2.circle(frame, (p1_x, p1_y - 40), 18, (200, 200, 200), -1) # Cabeça
        cv2.rectangle(frame, (p1_x - 18, p1_y - 20), (p1_x + 18, p1_y + 45), (180, 140, 50), -1) # Tronco
        cv2.putText(frame, "Operario 1", (p1_x - 30, p1_y - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Simular pessoa 2 caminhando e entrando na zona de perigo
        p2_x = int(550 - (f * 1.8) % 300)
        p2_y = int(320 + np.sin(f / 15) * 30)
        cv2.circle(frame, (p2_x, p2_y - 40), 18, (220, 220, 220), -1)
        cv2.rectangle(frame, (p2_x - 18, p2_y - 20), (p2_x + 18, p2_y + 45), (50, 100, 180), -1)
        cv2.putText(frame, "Operario 2", (p2_x - 30, p2_y - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        # Timestamp
        cv2.putText(frame, f"Frame: {f}/{total_frames} | SafeVision Simulation", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        out.write(frame)
        
    out.release()
    print(f"Vídeo de demonstração criado em: {output_path}")

if __name__ == "__main__":
    create_synthetic_demo_video()
