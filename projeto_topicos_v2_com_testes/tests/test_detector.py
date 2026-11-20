"""
Testes do SafeVision AI (versão só de EPI, sem zona de risco).

Executar a partir da pasta do projeto:
    python -m unittest discover -s tests -p "test_*.py" -v
"""
import base64
import io
import os
import sys
import unittest
import wave

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.detector import SafetyVisionDetector  # noqa: E402
from src import sound  # noqa: E402


def solid(bgr, h=40, w=40):
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = bgr
    return img


def bare_detector():
    """Instância sem carregar modelos, só para as funções auxiliares."""
    return SafetyVisionDetector.__new__(SafetyVisionDetector)


def bus_image_padded(pad=120):
    """Imagem de exemplo do ultralytics (pessoas sem EPI), com borda para
    que ninguém encoste na beira da tela (regra que bloqueia flagrantes)."""
    import ultralytics
    path = os.path.join(os.path.dirname(ultralytics.__file__), "assets", "bus.jpg")
    img = cv2.imread(path)
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(90, 90, 90))
    return cv2.resize(img, (640, int(img.shape[0] * 640 / img.shape[1])))


# ---------------- 1. Funções auxiliares (sem modelos) ----------------
class TestCorDoCapacete(unittest.TestCase):
    def setUp(self):
        self.d = bare_detector()

    def test_amarelo_industrial(self):
        self.assertTrue(self.d.check_hardhat_color(solid((0, 210, 255))))

    def test_ceu_azul_nao_conta(self):
        self.assertFalse(self.d.check_hardhat_color(solid((235, 206, 135))))

    def test_pele_nao_conta(self):
        self.assertFalse(self.d.check_hardhat_color(solid((140, 170, 220))))

    def test_recorte_vazio(self):
        self.assertFalse(self.d.check_hardhat_color(np.zeros((0, 0, 3), np.uint8)))
        self.assertFalse(self.d.check_hardhat_color(None))


class TestCorDoColete(unittest.TestCase):
    def setUp(self):
        self.d = bare_detector()

    def test_amarelo_limao(self):
        self.assertTrue(self.d.check_vest_color(solid((0, 255, 200))))

    def test_bege_nao_conta(self):
        self.assertFalse(self.d.check_vest_color(solid((160, 200, 220))))

    def test_recorte_vazio(self):
        self.assertFalse(self.d.check_vest_color(None))


class TestFiltroDeCapacete(unittest.TestCase):
    def setUp(self):
        self.d = bare_detector()

    def test_confianca_alta_aceita_sem_olhar_cor(self):
        self.assertTrue(self.d.is_valid_industrial_helmet(solid((128, 128, 128)), conf=0.70))

    def test_cinza_fosco_rejeitado(self):
        self.assertFalse(self.d.is_valid_industrial_helmet(solid((128, 128, 128)), conf=0.30))

    def test_branco_brilhante_aceito(self):
        self.assertTrue(self.d.is_valid_industrial_helmet(solid((255, 255, 255)), conf=0.30))

    def test_cor_vibrante_aceita(self):
        self.assertTrue(self.d.is_valid_industrial_helmet(solid((0, 0, 230)), conf=0.30))


class TestSom(unittest.TestCase):
    def test_bipe_de_alerta(self):
        with wave.open(io.BytesIO(base64.b64decode(sound.ALERT_BEEP_B64))) as w:
            self.assertEqual(w.getnchannels(), 1)
            self.assertEqual(w.getframerate(), 44100)
            self.assertAlmostEqual(w.getnframes() / 44100, 0.22, places=2)


# ---------------- 2. Detector completo (modelos YOLO) ----------------
class TestModelos(unittest.TestCase):
    def test_modelo_ppe_tem_capacete_e_colete(self):
        d = SafetyVisionDetector("yolov8n-pose.pt", "ppe_amirt.pt")
        nomes = [n.lower() for n in d.ppe_model.names.values()]
        self.assertIn("helmet", nomes)
        self.assertIn("vest", nomes)


class ModeloContador:
    """Envolve o modelo de EPI e conta quantas vezes ele é chamado."""
    def __init__(self, real):
        self.real, self.names, self.calls = real, real.names, 0

    def __call__(self, *a, **k):
        self.calls += 1
        return self.real(*a, **k)


class TestProcessFrame(unittest.TestCase):
    def test_frame_vazio(self):
        d = SafetyVisionDetector("yolov8n-pose.pt", "ppe_amirt.pt")
        _, violations, stats = d.process_frame(np.zeros((480, 640, 3), np.uint8))
        self.assertEqual(stats["total_persons"], 0)
        self.assertEqual(stats["compliance_rate"], 100.0)
        self.assertEqual(violations, [])

    def test_modelo_de_epi_nao_roda_sem_pessoas(self):
        d = SafetyVisionDetector("yolov8n-pose.pt", "ppe_amirt.pt")
        d.ppe_model = ModeloContador(d.ppe_model)
        d.process_frame(np.zeros((480, 640, 3), np.uint8))
        self.assertEqual(d.ppe_model.calls, 0)
        # o ByteTrack só confirma uma pessoa nova no 2º frame em que ela aparece
        img = bus_image_padded()
        d.process_frame(img.copy())
        _, _, stats = d.process_frame(img.copy())
        self.assertGreaterEqual(stats["total_persons"], 1)
        self.assertGreaterEqual(d.ppe_model.calls, 1)

    def test_pessoas_sem_epi_viram_flagrante_so_no_5o_frame(self):
        d = SafetyVisionDetector("yolov8n-pose.pt", "ppe_amirt.pt")
        img = bus_image_padded()
        por_frame = []
        for _ in range(6):
            _, v, stats = d.process_frame(img.copy())
            por_frame.append(len(v))
        self.assertGreaterEqual(stats["total_persons"], 1)
        self.assertLess(stats["compliance_rate"], 100.0)
        self.assertEqual(sum(por_frame[:4]), 0, por_frame)
        self.assertGreater(por_frame[4], 0, por_frame)

    def test_reset_state_limpa_a_memoria(self):
        d = SafetyVisionDetector("yolov8n-pose.pt", "ppe_amirt.pt")
        img = bus_image_padded()
        for _ in range(3):
            d.process_frame(img.copy())
        self.assertTrue(d.worker_memory)
        self.assertTrue(d.violation_counter)
        d.reset_state()
        self.assertEqual(d.worker_memory, {})
        self.assertEqual(d.violation_counter, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
