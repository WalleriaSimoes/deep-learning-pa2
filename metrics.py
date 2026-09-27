import numpy as np

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

import numpy as np

def compute_intersection(ground_truth, preds, threshold=0.5):
    # Converte o texto em matrizes NumPy 2D
    parts_t = np.array([list(map(float, x.split(','))) for x in ground_truth])
    parts_p = np.array([list(map(float, x.split(','))) for x in preds])

    unique_true_ids = np.unique(parts_t[:, 1])
    unique_preds_ids = np.unique(parts_p[:, 1])

    # Matriz global do IDF1 e dicionários de mapeamento
    global_match_matrix = np.zeros((unique_preds_ids.shape[0], unique_true_ids.shape[0]))
    map_t = {id_val: idx for idx, id_val in enumerate(unique_true_ids)}
    map_p = {id_val: idx for idx, id_val in enumerate(unique_preds_ids)}

    # Variáveis de memória para os ID Switches
    previous_match = {}  
    id_switches = 0      

    # Loop Temporal
    num_frames = int(parts_t[-1, 0])
    for f in range(num_frames):
        frame_t = parts_t[parts_t[:, 0] == (f + 1)]
        frame_p = parts_p[parts_p[:, 0] == (f + 1)]

        if len(frame_t) == 0 or len(frame_p) == 0:
            continue

        ids_t = frame_t[:, 1]
        t_x1, t_y1 = frame_t[:, 2], frame_t[:, 3]
        t_x2, t_y2 = t_x1 + frame_t[:, 4], t_y1 + frame_t[:, 5]

        ids_p = frame_p[:, 1]
        p_x1, p_y1 = frame_p[:, 2], frame_p[:, 3]
        p_x2, p_y2 = p_x1 + frame_p[:, 4], p_y1 + frame_p[:, 5]

        # Broadcasting Geométrico
        xA = np.maximum(p_x1[:, None], t_x1[None, :])
        yA = np.maximum(p_y1[:, None], t_y1[None, :])
        xB = np.minimum(p_x2[:, None], t_x2[None, :])
        yB = np.minimum(p_y2[:, None], t_y2[None, :])

        inter_w = np.clip(xB - xA, 0, None)
        inter_h = np.clip(yB - yA, 0, None)
        inter_area = inter_w * inter_h

        area_p = frame_p[:, 4] * frame_p[:, 5]
        area_t = frame_t[:, 4] * frame_t[:, 5]
        union_area = area_p[:, None] + area_t[None, :] - inter_area

        iou_matrix = inter_area / union_area

        # Resolve oclusões neste frame específico
        matches_iou, matches = greedy_match(iou_matrix, len(ids_p), len(ids_t))
        
        current_match = {}

        for p_idx in range(len(ids_p)):
            t_idx = matches[p_idx]
            
            # Rejeita se não casou ou se o IoU for menor que o limiar (0.5)
            if t_idx is None or matches_iou[p_idx] < threshold:
                continue
                
            t_idx = int(t_idx)
            real_id_p = ids_p[p_idx]
            real_id_t = ids_t[t_idx]
            
            # Atualiza a Matriz Global para a função de baixo usar depois
            mat_row = map_p[real_id_p]
            mat_col = map_t[real_id_t]
            global_match_matrix[mat_row, mat_col] += 1
            
            current_match[real_id_t] = real_id_p
            
            # Contabilidade dos Switches
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