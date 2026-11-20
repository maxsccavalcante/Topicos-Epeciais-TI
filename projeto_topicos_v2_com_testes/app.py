import os
import cv2
import tempfile
import time
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Importar módulos do sistema
import importlib
import src.detector
from src.sound import ALERT_BEEP_B64

# Configuração da página Streamlit
st.set_page_config(
    page_title="SafeVision AI - Monitoramento de Segurança e Tráfego",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização CSS Moderna (Padrão SOC / Central de Operações)
st.markdown("""
<style>
    /* Estilo Geral do Header */
    .brand-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-bottom: 8px;
        border-bottom: 1px solid #E2E8F0;
        margin-bottom: 16px;
    }
    .brand-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.5px;
        margin: 0;
    }
    .brand-sub {
        font-size: 0.98rem;
        color: #64748B;
        margin-top: 2px;
        margin-bottom: 0px;
    }
    
    /* Badges de Status do Sistema */
    .status-badge-ok {
        background-color: #ECFDF5;
        color: #059669;
        border: 1px solid #A7F3D0;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-block;
    }
    .status-badge-alert {
        background-color: #FEF2F2;
        color: #DC2626;
        border: 1px solid #FECACA;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-block;
        animation: pulse 1.5s infinite;
    }
    
    /* Cards de Métricas Estilizados */
    .metric-box {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #3B82F6;
        border-radius: 8px;
        padding: 12px 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 10px;
    }
    .metric-box-danger {
        border-left-color: #EF4444;
    }
    .metric-box-warning {
        border-left-color: #F59E0B;
    }
    .metric-box-success {
        border-left-color: #10B981;
    }
    
    /* Cards do Feed de Incidentes */
    .evidence-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 12px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.65; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR: CONTROLES -----------------
with st.sidebar:
    st.markdown("### ⚙️ Painel de Controle")
    
    input_source = st.radio(
        "Fonte de Entrada:",
        ["📁 Upload de Vídeo (.mp4)", "📹 Vídeo de Exemplo (Demonstração)", "🔴 Webcam ao Vivo"],
        index=0
    )
    
    uploaded_file = None
    if input_source == "📁 Upload de Vídeo (.mp4)":
        uploaded_file = st.file_uploader(
            "Carregar Gravação (.mp4, .avi, .mov):",
            type=["mp4", "avi", "mov", "mkv"],
            help="Carregue o arquivo de vídeo do canteiro ou rodovia para auditoria."
        )
    
    st.divider()
    st.markdown("### 🦺 Auditoria de EPIs (NR-6 / NR-18)")
    check_helmet = st.checkbox("Auditar Uso de Capacete", value=True)
    check_vest = st.checkbox("Auditar Uso de Colete Refletivo", value=True)
    show_ppe_boxes = st.checkbox("Exibir Caixas de EPIs Detectados", value=True)
    
    st.divider()
    st.markdown("### 🔔 Alarmes Sonoros")
    enable_audio_alert = st.checkbox("Ativar Alarme Sonoro de Infração", value=True, help="Emite sinal sonoro imediato se ocorrer infração de EPI.")

    st.divider()
    st.markdown("### ⚡ Desempenho")
    frame_skip = st.slider(
        "Aceleração / Pulo de Frames:",
        min_value=1,
        max_value=5,
        value=3,
        help="Sincroniza o processamento com a taxa de quadros real (25-30 FPS)."
    )
    process_width = st.select_slider(
        "Resolução de Processamento:",
        options=[480, 640, 720],
        value=480,
        help="Garante maior fluidez sem sobrecarregar a CPU."
    )

    st.divider()
    if st.button("🔄 Limpar Cache / Recarregar Sistema", use_container_width=True):
        st.cache_resource.clear()
        st.session_state.violations_log = []
        st.session_state.timeline_data = []
        st.rerun()

# ----------------- INICIALIZAÇÃO DO MODELO -----------------
@st.cache_resource
def load_detector(_version="amirt_v13_head_kpt_crop"):
    importlib.reload(src.detector)
    return src.detector.SafetyVisionDetector("yolov8n-pose.pt", "ppe_amirt.pt")

detector = load_detector()

if "violations_log" not in st.session_state:
    st.session_state.violations_log = []

if "timeline_data" not in st.session_state:
    st.session_state.timeline_data = []

# ----------------- CABEÇALHO PRINCIPAL -----------------
col_h1, col_h2 = st.columns([8, 2])
with col_h1:
    st.markdown('<h1 class="brand-title">🛡️ SafeVision AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="brand-sub">Monitoramento de EPIs em Tempo Real</p>', unsafe_allow_html=True)
with col_h2:
    status_header_placeholder = st.empty()
    status_header_placeholder.markdown('<div style="text-align: right; padding-top: 10px;"><span class="status-badge-ok">🟢 SISTEMA OPERACIONAL</span></div>', unsafe_allow_html=True)

st.write("")

# ----------------- ABAS PRINCIPAIS -----------------
tab_live, tab_analytics, tab_report = st.tabs([
    "🎥 Monitoramento ao Vivo",
    "📊 Gráficos e Tendências",
    "📋 Relatório e Exportação"
])

# Banner de alerta e som
banner_alert_placeholder = st.empty()
sound_placeholder = st.empty()

# ----------------- ABA 1: MONITORAMENTO AO VIVO -----------------
with tab_live:
    col_video, col_alerts = st.columns([7, 3])

    with col_video:
        video_container = st.empty()
        progress_bar = st.progress(0)
        
        # Linha de Métricas Estilizadas
        kpi1, kpi2 = st.columns(2)
        metric_persons = kpi1.empty()
        metric_violations = kpi2.empty()

    with col_alerts:
        st.markdown("#### 🚨 Feed de Flagrantes")
        alerts_container = st.empty()

    # Previsão inicial das métricas
    metric_persons.metric("Pessoas", 0)
    metric_violations.metric("Sem EPI", 0)

    # ----------------- PREPARAÇÃO DO VÍDEO -----------------
    video_path = None
    if input_source == "📁 Upload de Vídeo (.mp4)" and uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        video_path = tfile.name
    elif input_source == "📹 Vídeo de Exemplo (Demonstração)":
        sample_path = os.path.join("samples", "demo_sample.mp4")
        if os.path.exists(sample_path):
            video_path = sample_path
        else:
            st.warning("Vídeo de exemplo não encontrado.")
    elif input_source == "🔴 Webcam ao Vivo":
        video_path = 0

    st.write("")
    start_btn = st.button("▶️ Iniciar Auditoria de Vídeo", type="primary", use_container_width=False)

    if start_btn:
        st.session_state.violations_log = []
        st.session_state.timeline_data = []
        detector.reset_state()
        
        if video_path is None:
            st.error("Por favor, selecione ou faça o upload de um vídeo antes de iniciar.")
        else:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if isinstance(video_path, str) else 0
            video_fps = int(cap.get(cv2.CAP_PROP_FPS)) if isinstance(video_path, str) and cap.get(cv2.CAP_PROP_FPS) > 0 else 25
            
            frame_idx = 0
            recorded_ids = set()
            recent_crops = []
            last_sound_time = 0
            
            while cap.isOpened():
                # Usa grab() nos frames pulados para evitar decodificação desnecessária de vídeo
                if not cap.grab():
                    break
                    
                frame_idx += 1
                
                if frame_skip > 1 and (frame_idx % frame_skip != 0):
                    continue

                ret, frame = cap.retrieve()
                if not ret:
                    break
                    
                # Redimensionamento (só redimensiona se o vídeo for maior que process_width)
                orig_h, orig_w = frame.shape[:2]
                if orig_w > process_width:
                    target_w = process_width
                    target_h = int(orig_h * (target_w / orig_w))
                    frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

                now = time.time()

                # Inferência IA otimizada
                annotated_frame, violations, stats = detector.process_frame(
                    frame,
                    conf=0.42,
                    ppe_conf=0.24,
                    tracker=True,
                    check_helmet=check_helmet,
                    check_vest=check_vest,
                    imgsz=384,
                    pose_imgsz=384,
                    show_ppe_boxes=show_ppe_boxes
                )
                
                # Auto-purga de falsos flagrantes transitórios de trabalhadores comprovados conformes
                conforming_ids = stats.get("conforming_ids", set())
                purged = False
                if conforming_ids:
                    old_len = len(recent_crops)
                    recent_crops = [item for item in recent_crops if not any(f"ID #{cid}" in item[1] for cid in conforming_ids)]
                    if len(recent_crops) != old_len:
                        purged = True
                        st.session_state.violations_log = [v for v in st.session_state.violations_log if v.get("track_id") not in conforming_ids]

                # Registro de novos incidentes
                new_incident = False
                for v in violations:
                    event_key = (v.get("track_id"), v.get("type"))
                    if event_key not in recorded_ids and len(v["crop"]) > 0:
                        recorded_ids.add(event_key)
                        st.session_state.violations_log.append(v)
                        crop_rgb = cv2.cvtColor(v["crop"], cv2.COLOR_BGR2RGB)
                        recent_crops.insert(0, (crop_rgb, f"⚠️ {v['type']} (ID #{v['track_id']}) - {v['time']}"))
                        if len(recent_crops) > 3:
                            recent_crops.pop()
                        new_incident = True
                
                # Atualização do feed
                if new_incident or purged:
                    with alerts_container.container():
                        if len(recent_crops) == 0:
                            st.info("🛡️ Nenhuma infração ativa no momento.")
                        else:
                            for crop_img, caption_txt in recent_crops:
                                st.image(crop_img, caption=caption_txt, width=170)
                
                # Sistema de alarmes
                if new_incident:
                    status_header_placeholder.markdown('<div style="text-align: right; padding-top: 10px;"><span class="status-badge-alert">⚠️ FALTA DE EPI DETECTADA</span></div>', unsafe_allow_html=True)
                    banner_alert_placeholder.warning("⚠️ **ALERTA:** Falta de EPI detectada e registrada no feed!")
                    if enable_audio_alert and (now - last_sound_time > 0.5):
                        sound_placeholder.markdown(f'<audio autoplay src="data:audio/wav;base64,{ALERT_BEEP_B64}"></audio>', unsafe_allow_html=True)
                        last_sound_time = now
                elif not stats.get("has_ppe_infraction"):
                    status_header_placeholder.markdown('<div style="text-align: right; padding-top: 10px;"><span class="status-badge-ok">🟢 SISTEMA OPERACIONAL</span></div>', unsafe_allow_html=True)
                    banner_alert_placeholder.empty()

                # Histórico temporal
                timestamp_sec = round(frame_idx / video_fps, 1)
                st.session_state.timeline_data.append({
                    "Segundo": timestamp_sec,
                    "Pessoas Ativas": stats["total_persons"],
                    "Em Conformidade": stats["conforming"],
                    "Infrações EPI": stats["non_conforming"]
                })
                
                # Exibir frame em JPEG (muito mais rápido que enviar array NumPy/PNG ao Streamlit)
                ok_jpg, jpg_buf = cv2.imencode(".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok_jpg:
                    video_container.image(jpg_buf.tobytes(), use_container_width=True)
                
                # Métricas
                metric_persons.metric("Pessoas", stats["total_persons"])
                metric_violations.metric("Sem EPI", stats["non_conforming"])
                
                if total_frames > 0:
                    progress_bar.progress(min(frame_idx / total_frames, 1.0))
                
            cap.release()
            st.success("✅ Auditoria do vídeo concluída!")

# ----------------- ABA 2: GRÁFICOS E TENDÊNCIAS -----------------
with tab_analytics:
    st.markdown("### 📊 Evolução ao Longo do Tempo")
    
    if len(st.session_state.timeline_data) > 0:
        df_timeline = pd.DataFrame(st.session_state.timeline_data)
        
        col_g1, col_g2 = st.columns([7, 3])
        
        with col_g1:
            fig_timeline = px.line(
                df_timeline,
                x="Segundo",
                y=["Pessoas Ativas", "Em Conformidade", "Infrações EPI"],
                title="Evolução Temporal do Fluxo de Pessoas e Infrações de EPI",
                color_discrete_map={
                    "Pessoas Ativas": "#3B82F6",
                    "Em Conformidade": "#10B981",
                    "Infrações EPI": "#F59E0B"
                }
            )
            fig_timeline.update_layout(
                template="plotly_white",
                height=380,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
            
        with col_g2:
            total_conf = df_timeline["Em Conformidade"].sum()
            total_inf = df_timeline["Infrações EPI"].sum()
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Conforme', 'Falta de EPI'],
                values=[total_conf, total_inf],
                hole=.45,
                marker=dict(colors=['#10B981', '#F59E0B'])
            )])
            fig_pie.update_layout(
                title="Distribuição Acumulada de Status",
                template="plotly_white",
                height=380,
                showlegend=True
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Execute o processamento do vídeo na Aba 1 para visualizar os gráficos analíticos.")

# ----------------- ABA 3: RELATÓRIO E EXPORTAÇÃO -----------------
with tab_report:
    st.markdown("### 📋 Relatório de Ocorrências")
    
    if len(st.session_state.timeline_data) > 0:
        df_timeline = pd.DataFrame(st.session_state.timeline_data)
        
        tot_alerts = int(df_timeline["Infrações EPI"].sum())
        avg_comp = df_timeline["Em Conformidade"].sum() / max(1, df_timeline["Pessoas Ativas"].sum()) * 100
        
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        col_s1.metric("Duração Analisada", f"{df_timeline['Segundo'].max():.1f} s")
        col_s2.metric("Média de Conformidade", f"{avg_comp:.1f}%")
        col_s3.metric("Ocorrências Totais", tot_alerts)
        col_s4.metric("Flagrantes com Evidência", len(st.session_state.violations_log))
        
        st.divider()
        st.markdown("#### 📜 Tabela de Ocorrências com Carimbo de Horário")
        
        if len(st.session_state.violations_log) > 0:
            log_records = []
            for v in st.session_state.violations_log:
                log_records.append({
                    "Horário": v.get("time"),
                    "ID do Trabalhador": f"#{v.get('track_id')}",
                    "Infração": v.get("type"),
                    "Gravidade": "Alta" if "Capacete e Colete" in v.get("type", "") else "Média"
                })
            df_log = pd.DataFrame(log_records)
            st.dataframe(df_log, use_container_width=True)
        else:
            st.success("✅ Nenhuma infração com evidência registrada. Ambiente 100% em conformidade!")
            
        csv_data = df_timeline.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Baixar Dados da Auditoria (.CSV)",
            data=csv_data,
            file_name="relatorio_seguranca_audit.csv",
            mime="text/csv"
        )
    else:
        st.info("Inicie a auditoria para gerar o relatório executivo de conformidade.")
