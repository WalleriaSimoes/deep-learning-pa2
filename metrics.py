import numpy as np
import torch

def greedy_match(iou, predicts, trues):
    iou = iou.copy()
    predicts = int(predicts)
    trues = int(trues)
    # Faz o casamento de forma gulosa
    matches_iou = np.full(shape=predicts, fill_value=0.0)
    matches = np.full(shape=predicts, fill_value=None)
    for _ in range(min(predicts, trues)):
        pred_mxs = iou.max(axis=1)
        greedy_pred_isnt = pred_mxs.argmax()
        greedy_true_isnt = iou[greedy_pred_isnt].argmax()
        greedy_iou = iou[greedy_pred_isnt, greedy_true_isnt].max()

        # Apaga virtualmente as instancias prevista e verdadeira
        iou[greedy_pred_isnt, :] = -1
        iou[:, greedy_true_isnt] = -1

        matches_iou[greedy_pred_isnt] = greedy_iou
        matches[greedy_pred_isnt]     = greedy_true_isnt

    return matches_iou, matches


def test_limiares(iou, limiares, predicts, trues):
    matches_iou, _ = greedy_match(iou, predicts, trues)

    true_positives = []
    false_positves = []
    false_negatives = []
    average_precisions = []

    # Calcula TP, FP e FN para cada limiar
    for limiar in limiares:
        tp = (matches_iou >= limiar).sum()
        fp = predicts - tp
        fn = trues - tp
        true_positives.append(tp)
        false_positves.append(fp)
        false_negatives.append(fn)
        average_precisions.append(tp / (tp + fp + fn))

    return true_positives, false_positves, false_negatives, average_precisions


def compute_metrics(iou_matrix, limiares):
    iou = iou_matrix[1:, 1:]
    predicts, trues = iou.shape
    true_positives, false_positves, false_negatives, average_precisions = test_limiares(iou, limiares, predicts, trues)
    mean_average_precision = np.mean(average_precisions)
    abs_count_error = np.abs(predicts - trues)

    return true_positives, false_positves, false_negatives, float(mean_average_precision), float(abs_count_error)


def calculate_iou_matrix(detections_t_minus_1, detections_t):
    """
    Calcula a matriz de IoU entre as deteccoes de dois quadros diferentes.
    Assume o formato MOT padrao para as colunas: [frame, id, x, y, w, h, ...]
    """
    # Extrai coordenadas (x1, y1) e calcula (x2, y2) para T-1
    p_x1, p_y1 = detections_t_minus_1[:, 2], detections_t_minus_1[:, 3]
    p_x2, p_y2 = p_x1 + detections_t_minus_1[:, 4], p_y1 + detections_t_minus_1[:, 5]

    # Extrai coordenadas (x1, y1) e calcula (x2, y2) para T
    t_x1, t_y1 = detections_t[:, 2], detections_t[:, 3]
    t_x2, t_y2 = t_x1 + detections_t[:, 4], t_y1 + detections_t[:, 5]

    # Broadcasting Geometrico
    xA = np.maximum(p_x1[:, None], t_x1[None, :])
    yA = np.maximum(p_y1[:, None], t_y1[None, :])
    xB = np.minimum(p_x2[:, None], t_x2[None, :])
    yB = np.minimum(p_y2[:, None], t_y2[None, :])

    inter_w = np.clip(xB - xA, 0, None)
    inter_h = np.clip(yB - yA, 0, None)
    inter_area = inter_w * inter_h

    area_p = detections_t_minus_1[:, 4] * detections_t_minus_1[:, 5]
    area_t = detections_t[:, 4] * detections_t[:, 5]
    union_area = area_p[:, None] + area_t[None, :] - inter_area

    # Previne divisao por zero
    iou_matrix = np.divide(inter_area, union_area, out=np.zeros_like(inter_area), where=union_area!=0)
    
    return iou_matrix


def compute_intersection(ground_truth, preds, threshold=0.5):
    # Converte o texto em matrizes NumPy 2D
    parts_t = np.array([list(map(float, x.split(','))) for x in ground_truth])
    parts_p = np.array([list(map(float, x.split(','))) for x in preds])

    unique_true_ids = np.unique(parts_t[:, 1])
    unique_preds_ids = np.unique(parts_p[:, 1])

    global_match_matrix = np.zeros((unique_preds_ids.shape[0], unique_true_ids.shape[0]))
    map_t = {id_val: idx for idx, id_val in enumerate(unique_true_ids)}
    map_p = {id_val: idx for idx, id_val in enumerate(unique_preds_ids)}

    previous_match = {}  
    id_switches = 0      

    num_frames = int(parts_t[-1, 0])
    for f in range(num_frames):
        frame_t = parts_t[parts_t[:, 0] == (f + 1)]
        frame_p = parts_p[parts_p[:, 0] == (f + 1)]

        if len(frame_t) == 0 or len(frame_p) == 0:
            continue

        ids_t = frame_t[:, 1]
        ids_p = frame_p[:, 1]

        # A funcao substitui dezenas de linhas por uma so.
        # Passamos frame_p no lugar de T-1, e frame_t no lugar de T
        iou_matrix = calculate_iou_matrix(frame_p, frame_t)

        # Resolve oclusoes neste frame especifico
        matches_iou, matches = greedy_match(iou_matrix, len(ids_p), len(ids_t))
        
        current_match = {}

        for p_idx in range(len(ids_p)):
            t_idx = matches[p_idx]
            
            if t_idx is None or matches_iou[p_idx] < threshold:
                continue
                
            t_idx = int(t_idx)
            real_id_p = ids_p[p_idx]
            real_id_t = ids_t[t_idx]
            
            mat_row = map_p[real_id_p]
            mat_col = map_t[real_id_t]
            global_match_matrix[mat_row, mat_col] += 1
            
            current_match[real_id_t] = real_id_p
            
            if real_id_t in previous_match:
                if previous_match[real_id_t] != real_id_p:
                    id_switches += 1
            
        for t_id, p_id in current_match.items():
            previous_match[t_id] = p_id
            
    return global_match_matrix, unique_preds_ids.shape[0], unique_true_ids.shape[0], id_switches


def compute_idf1(ground_truth, preds, threshold=0.5):
    # Pega os dados brutos da trajetória gerados pela função de cima
    global_match_matrix, num_preds, num_trues, id_switches = compute_intersection(ground_truth, preds, threshold)
    
    # PAREAMENTO 2 (Global): Força a regra 1-para-1 do vídeo inteiro
    matches_counts, _ = greedy_match(global_match_matrix, num_preds, num_trues)
    
    # Matemática da Métrica
    IDTP = np.sum(matches_counts)
    total_true = len(ground_truth)
    total_pred = len(preds)
    
    IDFP = total_pred - IDTP
    IDFN = total_true - IDTP
    
    if (2 * IDTP + IDFP + IDFN) == 0:
        return 0.0, id_switches
        
    idf1_score = (2 * IDTP) / (2 * IDTP + IDFP + IDFN)
    
    # Retorna o score final e a contagem de erros
    return idf1_score, id_switches



def naive_tracker_shift(ground_truth_data, threshold: float=0.5, max_age: int=3):
    """
    Rastreador ingenuo com memoria de K quadros.
    Parametro max_age (K) define quantos quadros uma track sobrevive sem pareamento.
    """
    detections = np.array([list(map(float, x.split(','))) for x in ground_truth_data])
    num_frames = int(detections[-1, 0])
    
    tracked_results = []
    next_new_id = 1 
    
    # Dicionario para rastrear os IDs de tracks ativas: {ID_da_Track: {'bbox': array_da_caixa, 'age': quadros_sem_ver}}
    active_tracks = {}
    
    for f in range(1, num_frames + 1):
        frame_t = detections[detections[:, 0] == f]
        
        if len(frame_t) == 0:
            # Se o quadro estiver vazio, todas as tracks envelhecem
            tracks_to_delete = []
            for tid in active_tracks:
                active_tracks[tid]['age'] += 1
                if active_tracks[tid]['age'] >= max_age:
                    tracks_to_delete.append(tid)
            for tid in tracks_to_delete:
                del active_tracks[tid]
            continue

        if len(active_tracks) == 0:
            # Se nao ha memória, todas as deteccoes nascem como tracks novas
            for i in range(len(frame_t)):
                res = frame_t[i].copy()
                res[1] = next_new_id
                active_tracks[next_new_id] = {'bbox': res, 'age': 0}
                tracked_results.append(res)
                next_new_id += 1
            continue

        # Montar um array NumPy com as caixas das tracks ativas para a matematica vetorial
        active_ids = list(active_tracks.keys())
        active_bboxes = np.array([active_tracks[tid]['bbox'] for tid in active_ids])
        
        # Calcula IoU entre as tracks ativas e as novas deteccoes
        iou_matrix = calculate_iou_matrix(active_bboxes, frame_t)
        
        # Pareamento guloso
        matches_iou, matches = greedy_match(iou_matrix, len(active_bboxes), len(frame_t))
        
        # Mapear as correspondencias aprovadas
        matched_current_to_memory = {}
        matched_memory_indices = set()
        
        for p_idx in range(len(active_bboxes)):
            t_idx = matches[p_idx]
            if t_idx is not None and matches_iou[p_idx] >= threshold:
                matched_current_to_memory[int(t_idx)] = p_idx
                matched_memory_indices.add(p_idx)
                
        # Processar deteccoes do quadro atual
        for t_idx in range(len(frame_t)):
            res = frame_t[t_idx].copy()
            
            if t_idx in matched_current_to_memory:
                # Se casou, recupera o ID, atualiza a posicao zera a idade
                p_idx = matched_current_to_memory[t_idx]
                track_id = active_ids[p_idx]
                
                res[1] = track_id
                active_tracks[track_id]['bbox'] = res
                active_tracks[track_id]['age'] = 0
            else:
                # Se nao casou, nasce uma track nova com idade 0
                track_id = next_new_id
                next_new_id += 1
                
                res[1] = track_id
                active_tracks[track_id] = {'bbox': res, 'age': 0}
                
            tracked_results.append(res)
            
        # Gerir o ciclo de vida das tracks na memoria
        tracks_to_delete = []
        for p_idx, track_id in enumerate(active_ids):
            if p_idx not in matched_memory_indices:
                # Se a track nao encontrou par neste quadro, ela envelhece
                active_tracks[track_id]['age'] += 1
                
                # Se ultrapassou o limite K, a track e marcada para morrer
                if active_tracks[track_id]['age'] >= max_age:
                    tracks_to_delete.append(track_id)
                    
        # Remove definitivamente as tracks mortas da memoria ativa
        for tid in tracks_to_delete:
            del active_tracks[tid]

    return np.array(tracked_results)



def evaluate_trajectories(ground_truth, preds, threshold=0.5):
    """
    Avalia as trajetorias calculando IDF1, ID Switches, Fragmentacoes 
    e Erro de Contagem de identidades unicas.
    """
    parts_t = np.array([list(map(float, x.split(','))) for x in ground_truth])
    parts_p = np.array([list(map(float, x.split(','))) for x in preds])

    unique_true_ids = np.unique(parts_t[:, 1])
    unique_preds_ids = np.unique(parts_p[:, 1]) if len(parts_p) > 0 else np.array([])
    
    num_trues = len(unique_true_ids)
    num_preds = len(unique_preds_ids)

    # Erro de Contagem de Identidades unicas (Analogo ao erro de contagem do PA1)
    id_count_error = abs(num_preds - num_trues)

    global_match_matrix = np.zeros((num_preds, num_trues))
    map_t = {id_val: idx for idx, id_val in enumerate(unique_true_ids)}
    map_p = {id_val: idx for idx, id_val in enumerate(unique_preds_ids)}

    previous_match = {}  
    id_switches = 0
    
    # Historico de rastreamento para contar Fragmentacoes: 
    # Guarda 1 (rastreado) ou 0 (perdido) para cada frame em que o ID real aparece
    track_history = {tid: [] for tid in unique_true_ids}

    # Identifica o numero total de frames no video
    num_frames = int(max(
        np.max(parts_t[:, 0]) if len(parts_t) > 0 else 0, 
        np.max(parts_p[:, 0]) if len(parts_p) > 0 else 0
    ))
    
    for f in range(1, num_frames + 1):
        frame_t = parts_t[parts_t[:, 0] == f]
        frame_p = parts_p[parts_p[:, 0] == f]

        if len(frame_t) == 0:
            continue
            
        ids_t = frame_t[:, 1]
        
        if len(frame_p) == 0:
            for t_id in ids_t:
                track_history[t_id].append(0)
            continue

        ids_p = frame_p[:, 1]

        iou_matrix = calculate_iou_matrix(frame_p, frame_t)
        
        # Pareamento local do frame
        matches_iou, matches = greedy_match(iou_matrix, len(ids_p), len(ids_t))
        
        current_match = {}
        matched_true_ids = set()

        for p_idx in range(len(ids_p)):
            t_idx = matches[p_idx]
            
            if t_idx is None or matches_iou[p_idx] < threshold:
                continue
                
            t_idx = int(t_idx)
            real_id_p = ids_p[p_idx]
            real_id_t = ids_t[t_idx]
            
            mat_row = map_p[real_id_p]
            mat_col = map_t[real_id_t]
            
            # Popula a matriz global para o IDF1
            global_match_matrix[mat_row, mat_col] += 1
            
            current_match[real_id_t] = real_id_p
            matched_true_ids.add(real_id_t)
            
            # Contagem de ID Switches explicita
            if real_id_t in previous_match:
                if previous_match[real_id_t] != real_id_p:
                    id_switches += 1
            
        for t_id, p_id in current_match.items():
            previous_match[t_id] = p_id
            
        # Registo do estado para as Fragmentacoes
        for t_id in ids_t:
            if t_id in matched_true_ids:
                track_history[t_id].append(1) # Foi rastreado neste frame
            else:
                track_history[t_id].append(0) # Foi ocluído/perdido neste frame

    # Contagem de Fragmentações
    fragmentations = 0
    for tid, history in track_history.items():
        # Uma fragmentacao ocorre quando a trajetoria passa de rastreada (1) 
        # para perdida (0) e regressa a rastreada (1) num frame futuro.
        for i in range(1, len(history)):
            if history[i-1] == 1 and history[i] == 0:
                if 1 in history[i:]:
                    fragmentations += 1

    # Calculo do IDF1 (Atribuicao Global 1-para-1)
    if num_preds == 0 or num_trues == 0:
        idf1_score = 0.0
    else:
        matches_counts, _ = greedy_match(global_match_matrix, num_preds, num_trues)
        IDTP = np.sum(matches_counts)
        total_true_boxes = len(parts_t)
        total_pred_boxes = len(parts_p)
        
        IDFP = total_pred_boxes - IDTP
        IDFN = total_true_boxes - IDTP
        
        if (2 * IDTP + IDFP + IDFN) == 0:
            idf1_score = 0.0
        else:
            idf1_score = (2 * IDTP) / (2 * IDTP + IDFP + IDFN)
            
    return {
        "IDF1 Score": idf1_score,
        "ID Switches": id_switches,
        "Fragmentacoes": fragmentations,
        "Erro de Contagem de IDs": id_count_error,
        "Track History": track_history 
    }

def gru_tracker_shift(ground_truth_data, model, device, img_w, img_h, threshold=0.3, max_age=3, min_hits=2, T=8):
    """
    Rastreador Inteligente Final (Trilha A).
    Usa a GRU estritamente para translação (x,y) e congela a escala (w,h).
    """
    detections = np.array([list(map(float, x.split(','))) for x in ground_truth_data])
    if len(detections) == 0: return []
    num_frames = int(detections[-1, 0])
    
    tracked_results = []
    next_new_id = 1 
    active_tracks = {}
    model.eval()
    
    for f in range(1, num_frames + 1):
        frame_t = detections[detections[:, 0] == f]
        active_ids = list(active_tracks.keys())
        predicted_matrix = []
        
        # --- PASSO 1: PREVISÃO GRU (X e Y apenas) ---
        for tid in active_ids:
            track = active_tracks[tid]
            tamanho_necessario = T - 1
            
            # Última posição real observada
            x_last, y_last, w_last, h_last = track['bbox'][2:6]
            
            if len(track['history']) < tamanho_necessario:
                # Inércia Zero
                x, y, w, h = x_last, y_last, w_last, h_last
            else:
                historico = track['history'][-tamanho_necessario:]
                seq_input = torch.tensor(historico, dtype=torch.float32).unsqueeze(0).to(device)
                
                # Removi a inicialização manual do hidden. 
                # Ao passar 'None', o PyTorch cria automaticamente zeros no formato certo (seja GRU, LSTM ou RNN simples).
                with torch.no_grad():
                    pred_norm, _ = model(seq_input, None)
                
                pred_box = pred_norm.view(-1).cpu().numpy()
                
                # REMOVEMOS O TRAVÃO: A GRU assume 100% do controlo da translação
                x = pred_box[0] * img_w
                y = pred_box[1] * img_h
                
                # CONGELAMENTO DE ESCALA: Mantido para impedir caixas palito
                w = w_last
                h = h_last
            
            dummy_box = np.zeros(9)
            dummy_box[0] = f
            dummy_box[1] = tid
            dummy_box[2:6] = [x, y, w, h]
            dummy_box[6:9] = 1.0
            predicted_matrix.append(dummy_box)
             
        predicted_matrix = np.array(predicted_matrix) if len(predicted_matrix) > 0 else np.zeros((0, 9))
        
        # --- PASSO 2: MATCHING ---
        if len(predicted_matrix) > 0 and len(frame_t) > 0:
            iou_matrix = calculate_iou_matrix(predicted_matrix, frame_t)
            matches_iou, matches = greedy_match(iou_matrix, len(predicted_matrix), len(frame_t))
        else:
            matches_iou = [0] * len(predicted_matrix)
            matches = [None] * len(predicted_matrix)
            
        matched_current_to_memory = {}
        matched_memory_indices = set()
        
        for p_idx in range(len(predicted_matrix)):
            t_idx = matches[p_idx]
            if t_idx is not None and matches_iou[p_idx] >= threshold:
                matched_current_to_memory[int(t_idx)] = p_idx
                matched_memory_indices.add(p_idx)
                
        # --- PASSO 3: ATUALIZAÇÃO COM OBSERVAÇÃO REAL ---
        for t_idx in range(len(frame_t)):
            res = frame_t[t_idx].copy()
            norm_obs = [res[2]/img_w, res[3]/img_h, res[4]/img_w, res[5]/img_h]
            
            if t_idx in matched_current_to_memory:
                p_idx = matched_current_to_memory[t_idx]
                track_id = active_ids[p_idx]
                
                res[1] = track_id
                active_tracks[track_id]['bbox'] = res
                active_tracks[track_id]['history'].append(norm_obs)
                active_tracks[track_id]['age'] = 0
                active_tracks[track_id]['hits'] += 1
            else:
                track_id = next_new_id
                next_new_id += 1
                
                res[1] = track_id
                active_tracks[track_id] = {
                    'bbox': res,
                    'age': 0,
                    'history': [norm_obs],
                    'hits': 1 
                }
                
            # Mostra se a track foi confirmada
            if active_tracks[res[1]]['hits'] >= min_hits:
                tracked_results.append(res)
            
        # --- PASSO 4: OCLUSÃO E MORTE ---
        tracks_to_delete = []
        for p_idx, track_id in enumerate(active_ids):
            if p_idx not in matched_memory_indices:
                active_tracks[track_id]['age'] += 1
                
                if active_tracks[track_id]['age'] < max_age:
                    fake_res = predicted_matrix[p_idx].copy()
                    active_tracks[track_id]['bbox'] = fake_res
                    
                    norm_fake = [fake_res[2]/img_w, fake_res[3]/img_h, fake_res[4]/img_w, fake_res[5]/img_h]
                    active_tracks[track_id]['history'].append(norm_fake)
                    
                    if active_tracks[track_id]['hits'] >= min_hits:
                        tracked_results.append(fake_res)
                else:
                    tracks_to_delete.append(track_id)
                    
        for tid in tracks_to_delete:
            del active_tracks[tid]

    return np.array(tracked_results)
