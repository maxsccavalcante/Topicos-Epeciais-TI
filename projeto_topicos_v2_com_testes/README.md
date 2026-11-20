# 🛡️ SafeVision AI - Monitoramento de Segurança e Auditoria de EPI

Sistema de visão computacional em tempo real para monitoramento de segurança ocupacional, conformidade de EPI (Equipamento de Proteção Individual) e gestão de invasão de áreas restritas em ambientes industriais e canteiros de obras.

---

## 🎯 Objetivo e Escopo do Projeto
Ambientes industriais e canteiros de obra exigem auditoria rigorosa de conformidade com normas de segurança (NR-6 e NR-18). Este projeto automatiza a identificação de pessoas, rastreamento contínuo, verificação de EPI e detecção de invasão de perímetros de risco com geração de alertas e relatórios em tempo real.

---

## 🚀 Tecnologias Utilizadas
* **YOLOv8 (`ultralytics`):** Detecção de objetos em tempo real de alto desempenho.
* **ByteTRACK:** Rastreamento multi-objeto (MOT) para atribuição de IDs únicos persistentes entre frames.
* **OpenCV (`opencv-python`):** Processamento e manipulação de vídeo, teste de inclusão em polígono (`cv2.pointPolygonTest`).
* **Streamlit:** Interface web interativa com controles dinâmicos de vídeo, exibição de feeds e métricas.
* **Plotly:** Visualização interativa de séries temporais de fluxo de pessoas e incidentes de segurança.

---

## 📋 Funcionalidades
1. **Processamento de Vídeo Gravado & Upload:**
   * Suporte a arquivos `.mp4`, `.avi`, `.mov`.
   * Vídeos de demonstração pré-configurados para apresentação imediata.
2. **Classificação Visual com Bounding Boxes:**
   * 🟢 **Verde:** Operário conforme (com EPI).
   * 🟠 **Laranja:** Operário em não-conformidade (falta de capacete/colete).
   * 🔴 **Vermelho:** Invasão de área restrita / zona de risco de maquinário.
3. **Feed de Flagrantes / Evidências:**
   * Recorte e captura automática do infrator no momento da ocorrência.
4. **Relatório Analítico & Exportação:**
   * Gráfico de série temporal mostrando a distribuição de incidentes ao longo do vídeo.
   * Botão de exportação dos dados da auditoria em formato `.CSV`.

---

## 🛠️ Como Executar o Projeto

1. Instale as dependências:
```bash
pip install -r requirements.txt
```

2. Execute o dashboard Streamlit:
```bash
streamlit run app.py
```

3. Abra no navegador o endereço exibido (geralmente `http://localhost:8501`).
