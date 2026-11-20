import os
from collections import deque
import cv2
import numpy as np
from datetime import datetime

class SafetyVisionDetector:
    def __init__(self, pose_model_path="yolov8n-pose.pt", ppe_model_path="ppe_amirt.pt"):
        """
        Detector Avançado com Arquitetura Híbrida (IA + Visão Cromática Industrial):
        1. YOLOv8 Pose: Localização de articulações e geometria corporal.
        2. YOLOv8 PPE: Classificação de capacete e colete.
        3. Análise Cromática de Segurança: Detecção de capacetes amarelos/azuis industriais imunes a oclusões de costas.
        4. Memória de Conformidade: Impede falsos alarmes transitórios quando o trabalhador vira de costas ou passa por pilares.
        """
        from ultralytics import YOLO
        
        self.pose_model = YOLO(pose_model_path)
        
        if os.path.exists(ppe_model_path):
            self.ppe_model = YOLO(ppe_model_path)
        elif os.path.exists("ppe_model.pt"):
            self.ppe_model = YOLO("ppe_model.pt")
        else:
            self.ppe_model = None
            
        # Memória histórica em janela deslizante por operário (track_id)
        self.worker_memory = {} # {track_id: {"helmet": deque(maxlen=12), "vest": deque(maxlen=12)}}
        self.violation_counter = {}

    def reset_state(self):
        """Limpa a memória de rastreamento e o estado do ByteTrack ao iniciar um novo vídeo."""
        self.worker_memory = {}
        self.violation_counter = {}
        predictor = getattr(self.pose_model, "predictor", None)
        trackers = getattr(predictor, "trackers", None) if predictor is not None else None
        if trackers:
            for t in trackers:
                if hasattr(t, "reset"):
                    t.reset()

    def check_hardhat_color(self, head_crop):
        """
        Inspeciona se a região da cabeça possui coloração vibrante de capacete amarelo industrial
        (alta saturação para evitar confundir com céu azul, pele ou roupas bege).
        """
        if head_crop is None or head_crop.size == 0:
            return False
            
        hsv = cv2.cvtColor(head_crop, cv2.COLOR_BGR2HSV)
        # Amarelo industrial (cobre luz quente interna e externa sem pegar pele/roupa/céu)
        mask_yellow = cv2.inRange(hsv, np.array([18, 90, 120]), np.array([36, 255, 255]))
        pct_yellow = (np.sum(mask_yellow > 0) / (head_crop.shape[0] * head_crop.shape[1])) * 100
        return pct_yellow >= 1.8

    def is_valid_industrial_helmet(self, helmet_crop, conf=0.0):
        """
        Filtra falsos positivos de bonés de tecido/turbantes que a rede neural
        possa confundir com capacetes pelo formato arredondado de perfil.
        """
        if conf >= 0.68:
            return True
        if helmet_crop is None or helmet_crop.size == 0:
            return False
        hsv = cv2.cvtColor(helmet_crop, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        # Requer cor vibrante de EPI (S >= 105) OU branco brilhante puro (S <= 35, V >= 210)
        safety_pixels = np.sum(((s >= 105) & (v >= 95)) | ((s <= 35) & (v >= 210)))
        pct = (safety_pixels / (helmet_crop.shape[0] * helmet_crop.shape[1])) * 100
        return pct >= 10.0

    def check_vest_color(self, torso_crop):
        """
        Detecta colete refletivo amarelo-limão fluorescente no tronco,
        com saturação e matiz estritos para não confundir com roupas bege/marrom ao sol.
        """
        if torso_crop is None or torso_crop.size == 0:
            return False
        hsv = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2HSV)
        mask_lime = cv2.inRange(hsv, np.array([24, 110, 120]), np.array([45, 255, 255]))
        pct_lime = (np.sum(mask_lime > 0) / (torso_crop.shape[0] * torso_crop.shape[1])) * 100
        return pct_lime >= 2.0

    def process_frame(
        self,
        frame,
        conf=0.42,
        ppe_conf=0.24,
        tracker=True,
        check_helmet=True,
        check_vest=True,
        imgsz=384,
        pose_imgsz=384,
        show_ppe_boxes=True,
        **kwargs
    ):
        h, w = frame.shape[:2]
        annotated_frame = frame.copy()

        # 1. Rastreamento e Pose (executado primeiro para evitar inferência de PPE em frames vazios)
        if tracker:
            pose_results = self.pose_model.track(
                frame,
                conf=conf,
                imgsz=pose_imgsz,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False
            )
        else:
            pose_results = self.pose_model(frame, conf=conf, imgsz=pose_imgsz, verbose=False)

        frame_violations = []
        person_count = 0
        conforming_count = 0
        has_ppe_infraction = False

        boxes = pose_results[0].boxes
        keypoints = pose_results[0].keypoints

        # 2. Detecção de Capacetes e Coletes via Modelo PPE (somente se houver pessoas no frame)
        detected_helmets = []
        detected_vests = []
        
        if boxes is not None and len(boxes) > 0 and self.ppe_model is not None and (check_helmet or check_vest):
            ppe_results = self.ppe_model(frame, conf=ppe_conf, imgsz=imgsz, verbose=False)
            if ppe_results[0].boxes is not None:
                for b in ppe_results[0].boxes:
                    cls_id = int(b.cls[0])
                    label = self.ppe_model.names.get(cls_id, "").lower()
                    bx1, by1, bx2, by2 = map(int, b.xyxy[0])
                    b_conf = float(b.conf[0])
                    center = ((bx1 + bx2) / 2, (by1 + by2) / 2)
                    
                    if label == "helmet" and check_helmet:
                        h_crop = frame[max(0, by1):min(h, by2), max(0, bx1):min(w, bx2)]
                        if not self.is_valid_industrial_helmet(h_crop, conf=b_conf):
                            continue
                            
                        detected_helmets.append({
                            "id": len(detected_helmets),
                            "center": center,
                            "box": (bx1, by1, bx2, by2),
                            "conf": b_conf
                        })
                    elif label == "vest" and check_vest:
                        detected_vests.append({
                            "id": len(detected_vests),
                            "center": center,
                            "box": (bx1, by1, bx2, by2),
                            "conf": b_conf
                        })

        persons_data = []
        if boxes is not None and len(boxes) > 0:
            for i, box in enumerate(boxes):
                box_conf = float(box.conf[0]) if box.conf is not None else 0.0
                # Filtro anatômico: descarta falsos positivos em máquinas/objetos (ex.: betoneira)
                valid_kpts = 0
                if keypoints is not None and i < len(keypoints) and keypoints.conf is not None:
                    valid_kpts = int((keypoints.conf[i] > 0.35).sum())
                if valid_kpts < 5 and box_conf < 0.70:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = int(box.id[0]) if box.id is not None else None
                person_h = max(1, y2 - y1)
                
                # Checar se a pessoa está cortada na borda da tela
                is_touching_edge = (x1 <= 8 or y1 <= 8 or x2 >= w - 8 or y2 >= h - 8)
                
                # Cabeça através de pontos faciais ou topo do corpo
                head_x, head_y = None, None
                if keypoints is not None and i < len(keypoints):
                    kpts = keypoints[i].xy[0].cpu().numpy()
                    head_kpts = [pt for pt in kpts[:5] if pt[0] > 0 and pt[1] > 0]
                    if head_kpts:
                        head_x = float(np.mean([pt[0] for pt in head_kpts]))
                        head_y = float(np.mean([pt[1] for pt in head_kpts]))
                        
                if head_x is None:
                    head_x = (x1 + x2) / 2
                    head_y = y1 + person_h * 0.12

                # Recorte focado na cabeça real via keypoints (imune a braços levantados acima da cabeça)
                head_r = max(20, int(person_h * 0.09))
                hy1 = max(0, int(head_y - head_r * 1.3))
                hy2 = min(h, int(head_y + head_r * 0.5))
                hx1 = max(0, int(head_x - head_r))
                hx2 = min(w, int(head_x + head_r))
                head_crop = frame[hy1:hy2, hx1:hx2]

                torso_center = ((x1 + x2) / 2, y1 + person_h * 0.40)
                feet_point = ((x1 + x2) // 2, y2)

                persons_data.append({
                    "idx": i,
                    "box": (x1, y1, x2, y2),
                    "track_id": track_id,
                    "head": (head_x, head_y),
                    "head_crop": head_crop,
                    "torso": torso_center,
                    "feet": feet_point,
                    "height": person_h,
                    "touching_edge": is_touching_edge
                })

        # --- ASSOCIAÇÃO EXCLUSIVA DE CAPACETE ---
        assigned_helmets = set()
        person_has_helmet = {p["idx"]: False for p in persons_data}

        # Checagem preliminar por cor industrial (amarelo/azul de canteiro)
        for p in persons_data:
            if check_helmet and self.check_hardhat_color(p["head_crop"]):
                person_has_helmet[p["idx"]] = True

        dist_pairs = []
        for p in persons_data:
            if person_has_helmet[p["idx"]]:
                continue
            phx, phy = p["head"]
            strict_max_dist = min(38.0, p["height"] * 0.18)
            for h_idx, h_item in enumerate(detected_helmets):
                hcx, hcy = h_item["center"]
                dist = np.hypot(phx - hcx, phy - hcy)
                dist_pairs.append((dist, p["idx"], h_idx, strict_max_dist))

        dist_pairs.sort(key=lambda x: x[0])
        for dist, p_idx, h_idx, max_d in dist_pairs:
            if not person_has_helmet[p_idx] and h_idx not in assigned_helmets:
                if dist <= max_d:
                    person_has_helmet[p_idx] = True
                    assigned_helmets.add(h_idx)
                    if show_ppe_boxes:
                        bx1, by1, bx2, by2 = detected_helmets[h_idx]["box"]
                        b_conf = detected_helmets[h_idx]["conf"]
                        cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), (255, 255, 0), 1)
                        cv2.putText(
                            annotated_frame, f"Capacete {b_conf:.2f}",
                            (bx1, max(0, by1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 0), 1
                        )

        # --- ASSOCIAÇÃO DE COLETE REFLETIVO ---
        person_has_vest = {p["idx"]: False for p in persons_data}
        for p in persons_data:
            px1, py1, px2, py2 = p["box"]
            torso_y1 = py1 + p["height"] * 0.15
            torso_y2 = py1 + p["height"] * 0.78
            
            # 1. Checagem via YOLO
            for v_item in detected_vests:
                vcx, vcy = v_item["center"]
                if (px1 <= vcx <= px2) and (torso_y1 <= vcy <= torso_y2):
                    person_has_vest[p["idx"]] = True
                    if show_ppe_boxes:
                        bx1, by1, bx2, by2 = v_item["box"]
                        b_conf = v_item["conf"]
                        cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), (0, 220, 255), 1)
                        cv2.putText(
                            annotated_frame, f"Colete {b_conf:.2f}",
                            (bx1, max(0, by1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1
                        )
                    break
                    
            # 2. Checagem complementar cromática de colete refletivo fluorescente
            if not person_has_vest[p["idx"]] and check_vest:
                t_crop = frame[max(0, int(torso_y1)):min(h, int(torso_y2)), max(0, px1):min(w, px2)]
                if self.check_vest_color(t_crop):
                    person_has_vest[p["idx"]] = True

        # --- ATUALIZAÇÃO DA MEMÓRIA HISTÓRICA DO OPERÁRIO (JANELA DESLIZANTE) ---
        for p in persons_data:
            tid = p["track_id"]
            idx = p["idx"]
            has_h = person_has_helmet[idx] if check_helmet else True
            has_v = person_has_vest[idx] if check_vest else True
            
            if tid is not None:
                if tid not in self.worker_memory:
                    self.worker_memory[tid] = {"helmet": deque(maxlen=8), "vest": deque(maxlen=8)}
                mem = self.worker_memory[tid]
                mem["helmet"].append(1 if has_h else 0)
                mem["vest"].append(1 if has_v else 0)
                
                # Mantém conformidade apenas se o EPI foi confirmado na maioria dos últimos frames
                if len(mem["helmet"]) >= 3:
                    if check_helmet and (sum(mem["helmet"]) / len(mem["helmet"]) >= 0.50):
                        person_has_helmet[idx] = True
                    if check_vest and (sum(mem["vest"]) / len(mem["vest"]) >= 0.50):
                        person_has_vest[idx] = True

        # --- AVALIAÇÃO FINAL E FLAGRANTES ---
        active_ids = set()
        conforming_ids = set()

        for p in persons_data:
            person_count += 1
            x1, y1, x2, y2 = p["box"]
            track_id = p["track_id"]
            feet_point = p["feet"]
            
            has_h = person_has_helmet[p["idx"]] if check_helmet else True
            has_v = person_has_vest[p["idx"]] if check_vest else True
            is_conforming = (has_h and has_v)
            
            if track_id is not None:
                active_ids.add(track_id)
                if is_conforming:
                    conforming_ids.add(track_id)

            # Persistência temporal para evitar flagrantes em transição
            tid_key = track_id if track_id is not None else f"temp_{p['idx']}"
            if is_conforming:
                self.violation_counter[tid_key] = 0
            else:
                self.violation_counter[tid_key] = self.violation_counter.get(tid_key, 0) + 1

            # Somente confirma flagrante se:
            # 1. Pelo menos 5 frames consecutivos sem EPI
            # 2. O corpo NÃO está cortado pela borda da tela
            is_edge = p["touching_edge"]
            confirmed_infraction = (
                not is_conforming 
                and not is_edge 
                and self.violation_counter[tid_key] >= 5
            )

            # Rótulos
            if not is_conforming:
                has_ppe_infraction = True
                if not has_h and not has_v:
                    color = (0, 0, 255)
                    status_text = "SEM CAPACETE E COLETE"
                    violation_type = "Falta de Capacete e Colete"
                elif not has_h:
                    color = (0, 140, 255)
                    status_text = "SEM CAPACETE"
                    violation_type = "Falta de Capacete"
                else:
                    color = (0, 180, 255)
                    status_text = "SEM COLETE"
                    violation_type = "Falta de Colete Refletivo"
                    
                if confirmed_infraction:
                    crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                    frame_violations.append({
                        "type": violation_type,
                        "track_id": track_id,
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "crop": crop
                    })
            else:
                color = (0, 255, 0)
                if check_helmet and check_vest:
                    status_text = "EPI CONFORME (CAP.+COL.)"
                elif check_helmet:
                    status_text = "CAPACETE CONFORME"
                else:
                    status_text = "COLETE CONFORME"
                conforming_count += 1

            # Bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            id_label = f"ID: #{track_id} | " if track_id is not None else ""
            full_label = f"{id_label}{status_text}"
            
            (tw, th), _ = cv2.getTextSize(full_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated_frame, (x1, max(0, y1 - 22)), (x1 + tw + 6, y1), color, -1)
            cv2.putText(
                annotated_frame, full_label,
                (x1 + 3, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
            )

        # Limpar IDs antigos
        for k in list(self.violation_counter.keys()):
            if isinstance(k, int) and k not in active_ids:
                del self.violation_counter[k]

        stats = {
            "total_persons": person_count,
            "conforming": conforming_count,
            "non_conforming": person_count - conforming_count,
            "compliance_rate": (conforming_count / person_count * 100) if person_count > 0 else 100.0,
            "has_ppe_infraction": has_ppe_infraction,
            "conforming_ids": conforming_ids
        }

        return annotated_frame, frame_violations, stats
