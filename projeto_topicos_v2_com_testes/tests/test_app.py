"""
Teste de ponta a ponta do dashboard (app.py) com o AppTest do Streamlit.

Executar a partir da pasta do projeto:
    python -m unittest tests.test_app -v
"""
import os
import unittest

from streamlit.testing.v1 import AppTest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
APP = os.path.join(ROOT, "app.py")


def iniciar(at):
    return next(b for b in at.button if "Iniciar Auditoria" in b.label)


class TestDashboard(unittest.TestCase):
    def test_abre_sem_erros(self):
        at = AppTest.from_file(APP, default_timeout=120).run()
        self.assertEqual(len(at.exception), 0, [e.message for e in at.exception])
        self.assertEqual(len(at.tabs), 3)
        rotulos = [m.label for m in at.metric]
        self.assertEqual(rotulos, ["Pessoas", "Sem EPI"])

    def test_iniciar_sem_video_mostra_erro(self):
        at = AppTest.from_file(APP, default_timeout=120).run()
        iniciar(at).click().run()
        self.assertTrue(any("selecione" in e.value for e in at.error))

    def test_auditoria_do_video_de_exemplo(self):
        at = AppTest.from_file(APP, default_timeout=600).run()
        at.radio[0].set_value("📹 Vídeo de Exemplo (Demonstração)").run()
        iniciar(at).click().run()
        self.assertEqual(len(at.exception), 0, [e.message for e in at.exception])
        self.assertTrue(any("concluída" in s.value for s in at.success))
        self.assertGreater(len(at.session_state["timeline_data"]), 0)
        rotulos = [m.label for m in at.metric]
        self.assertIn("Duração Analisada", rotulos)


if __name__ == "__main__":
    unittest.main(verbosity=2)
