"""
Parte 2: um único rastreador (associação + nascimento/morte) com modelo de movimento INTERCAMBIÁVEL.
Assim GRU, Kalman e velocidade-zero são comparados com exatamente a mesma associação e os mesmos limiares.
Todas as caixas aqui estão em PIXELS: [x, y, w, h].
Marcação: caixas "fantasma" (previstas sem observação) saem com conf = -1 (usado na galeria de falhas).
"""
import numpy as np
from metrics import calculate_iou_matrix, greedy_match, naive_tracker_shift, evaluate_trajectories, tracks_to_str


class MovimentoZero:
    """Baseline: a caixa prevista é a última caixa conhecida."""
    nome = "Vel. zero"
    def novo(self, box): return {'box': np.array(box, dtype=float)}
    def prever(self, st): return st['box'].copy(), None
    def avancar(self, st, aux, nova_box, observado): return {'box': np.array(nova_box, dtype=float)}


class MovimentoKalman:
    """Filtro de Kalman de velocidade constante (permitido APENAS como baseline de comparação)."""
    nome = "Kalman CV"
    def __init__(self, q_pos=1.0, q_vel=0.1, r_pos=4.0, r_size=9.0):
        self.F = np.eye(8); self.F[:4, 4:] = np.eye(4)
        self.H = np.zeros((4, 8)); self.H[:, :4] = np.eye(4)
        self.Q = np.diag([q_pos] * 4 + [q_vel] * 4).astype(float)
        self.R = np.diag([r_pos, r_pos, r_size, r_size]).astype(float)
    def novo(self, box):
        return {'x': np.r_[np.array(box, dtype=float), np.zeros(4)], 'P': np.diag([10.] * 4 + [100.] * 4)}
    def prever(self, st):
        x = self.F @ st['x']; P = self.F @ st['P'] @ self.F.T + self.Q
        box = x[:4].copy(); box[2:] = np.maximum(box[2:], 1.0)
        return box, (x, P)
    def avancar(self, st, aux, nova_box, observado):
        x, P = aux
        if observado:
            y = np.array(nova_box, dtype=float) - self.H @ x
            S = self.H @ P @ self.H.T + self.R
            K = P @ self.H.T @ np.linalg.inv(S)
            x = x + K @ y; P = (np.eye(8) - K @ self.H) @ P
        return {'x': x, 'P': P}


class MovimentoGRU:
    """
    Trilha A: um estado recorrente POR TRACK. Entrada a cada quadro = [caixa_norm, delta_norm*DELTA_SCALE];
    saída = delta previsto. Sob oclusão a própria previsão vira a próxima entrada (free-running).
    """
    def __init__(self, model, device, img_w, img_h, nome="GRU"):
        import torch
        from GRU import DELTA_SCALE
        self.torch, self.S = torch, DELTA_SCALE
        self.model = model.to(device).eval(); self.device = device
        self.norm = np.array([img_w, img_h, img_w, img_h], dtype=float)
        self.nome = nome
    def novo(self, box): return {'h': None, 'box': np.array(box, dtype=float), 'd': np.zeros(4)}
    def prever(self, st):
        feat = np.r_[st['box'] / self.norm, st['d'] / self.norm * self.S]
        x = self.torch.tensor(feat, dtype=self.torch.float32, device=self.device).view(1, 1, -1)
        with self.torch.no_grad():
            out, h = self.model.forward_seq(x, st['h'])
        delta = out[0, 0].cpu().numpy() / self.S * self.norm
        pred = st['box'] + delta; pred[2:] = np.maximum(pred[2:], 1.0)
        return pred, h
    def avancar(self, st, aux, nova_box, observado):
        nova_box = np.array(nova_box, dtype=float)
        return {'h': aux, 'box': nova_box, 'd': nova_box - st['box']}


def rodar_tracker(det_strings, movimento, threshold=0.3, max_age=10, min_hits=2, emitir_fantasmas=True):
    """
    Regras (idênticas para qualquer modelo de movimento):
      associação : IoU entre caixa PREVISTA da track e detecção observada, matching guloso, limiar fixo
      nascimento : detecção sem par vira track nova (ID sequencial)
      confirmação: a track só é emitida após `min_hits` observações; track tentativa que falha é descartada
      oclusão    : sem par, o estado roda sem observação (usa a própria previsão); fantasma é emitido
      morte      : age >= max_age quadros consecutivos sem observação
    """
    if len(det_strings) == 0:
        return np.zeros((0, 9))
    dets = np.array([list(map(float, s.split(','))) for s in det_strings])
    n_frames = int(dets[:, 0].max())
    tracks, next_id, out = {}, 1, []
    for f in range(1, n_frames + 1):
        frame_t = dets[dets[:, 0] == f]
        ids = list(tracks.keys())
        preds, auxs = [], []
        for tid in ids:
            pb, aux = movimento.prever(tracks[tid]['state'])
            preds.append(pb); auxs.append(aux)
        n_p, n_d = len(ids), len(frame_t)
        matched = {}
        if n_p > 0 and n_d > 0:
            pm = np.zeros((n_p, 9)); pm[:, 2:6] = np.array(preds)
            iou = calculate_iou_matrix(pm, frame_t)
            m_iou, m_idx = greedy_match(iou, n_p, n_d)
            for p in range(n_p):
                if m_idx[p] is not None and m_iou[p] >= threshold:
                    matched[int(m_idx[p])] = p
        matched_tracks = set(matched.values())
        for d in range(n_d):
            row = frame_t[d].copy()
            if d in matched:
                p = matched[d]; tid = ids[p]; tr = tracks[tid]
                tr['state'] = movimento.avancar(tr['state'], auxs[p], row[2:6], True)
                tr['age'] = 0; tr['hits'] += 1
            else:
                tid = next_id; next_id += 1
                tracks[tid] = {'state': movimento.novo(row[2:6]), 'age': 0, 'hits': 1}
            row[1] = tid
            if tracks[tid]['hits'] >= min_hits:
                out.append(row)
        for p, tid in enumerate(ids):
            if p in matched_tracks:
                continue
            tr = tracks[tid]; tr['age'] += 1
            if tr['hits'] < min_hits or tr['age'] >= max_age:
                del tracks[tid]; continue
            tr['state'] = movimento.avancar(tr['state'], auxs[p], preds[p], False)
            if emitir_fantasmas:
                out.append([f, tid, *preds[p], -1.0, 1, 1.0])      # conf=-1 => fantasma
    return np.array(out) if out else np.zeros((0, 9))


def comparar_trackers(dets, gt, movimentos, threshold=0.3, max_age=10, min_hits=2, incluir_naive=True, eval_thr=0.5):
    """Mesmas detecções, mesma associação, mesmos limiares. Devolve (tabela, preds_por_nome)."""
    tabela, preds_por_nome = {}, {}
    def resumo(m):
        return {k: m[k] for k in ("IDF1 Score", "ID Switches", "Fragmentacoes", "Erro de Contagem de IDs")}
    if incluir_naive:
        p = tracks_to_str(naive_tracker_shift(dets, threshold=threshold, max_age=max_age))
        preds_por_nome['Naive (P1)'] = p
        tabela['Naive (P1)'] = resumo(evaluate_trajectories(gt, p, eval_thr))
    for nome, mov in movimentos.items():
        p = tracks_to_str(rodar_tracker(dets, mov, threshold, max_age, min_hits))
        preds_por_nome[nome] = p
        tabela[nome] = resumo(evaluate_trajectories(gt, p, eval_thr))
    return tabela, preds_por_nome


def imprimir_tabela(tabela, titulo=""):
    if titulo: print(titulo)
    print(f"{'modelo':14s} {'IDF1':>7s} {'switches':>9s} {'fragm.':>7s} {'|Δ IDs|':>8s}")
    for nome, m in tabela.items():
        print(f"{nome:14s} {m['IDF1 Score']:7.4f} {m['ID Switches']:9d} {m['Fragmentacoes']:7d} {m['Erro de Contagem de IDs']:8d}")
