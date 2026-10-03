"""
Parte 2 (treino da Trilha A) e correção da Parte 4 (treino com rollout / free-running).
Perda: smooth-L1 sobre a caixa (em unidades de DELTA_SCALE). Treino SÓ nas sequências de treino.
"""
import numpy as np
import torch
import torch.nn.functional as F
from GRU import BBoxTrackerRNN, DELTA_SCALE, features_from_boxes


# ---------- dados ----------
def trajetorias_contiguas(gt_strings, img_w, img_h, min_len=5):
    """Trajetórias por ID (caixas normalizadas, shape (L,4)); quebra onde há lacuna de quadros no GT."""
    por_id = {}
    for s in gt_strings:
        p = s.split(',')
        tid = int(float(p[1]))
        if tid <= 0:
            continue
        por_id.setdefault(tid, []).append((int(float(p[0])), float(p[2]), float(p[3]), float(p[4]), float(p[5])))
    norm = np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
    segs = []
    for linhas in por_id.values():
        linhas.sort()
        atual = [linhas[0]]
        for a in linhas[1:] + [None]:
            if a is not None and a[0] == atual[-1][0] + 1:
                atual.append(a); continue
            if len(atual) >= min_len:
                segs.append(np.array([c[1:] for c in atual], dtype=np.float32) / norm)
            atual = [a] if a is not None else []
    return segs


def montar_janelas(trajs, T, passo=None):
    """Janelas deslizantes de T+1 caixas (T passos de previsão). Retorna array (N, T+1, 4)."""
    passo = passo or max(1, T // 2)
    X = []
    for tr in trajs:
        L = len(tr)
        if L < T + 1:
            continue
        for i in range(0, L - T, passo):
            X.append(tr[i:i + T + 1])
        if (L - T - 1) % passo != 0:
            X.append(tr[L - T - 1:])
    return np.stack(X).astype(np.float32) if X else np.zeros((0, T + 1, 4), dtype=np.float32)


def janelas_das_sequencias(gt_by_seq, tamanhos, seqs, T, min_len=None, passo=None):
    """Concatena as janelas de várias sequências. tamanhos: {seq: (w, h)}."""
    todas = []
    for s in seqs:
        w, h = tamanhos[s]
        trajs = trajetorias_contiguas(gt_by_seq[s], w, h, min_len=min_len or (T + 1))
        todas.append(montar_janelas(trajs, T, passo))
    return np.concatenate(todas, axis=0)


# ---------- perdas ----------
def loss_teacher(model, B):
    """Teacher forcing: toda entrada é caixa REAL. B: (N, T+1, 4)."""
    feats = features_from_boxes(B[:, :-1])
    out, _ = model.forward_seq(feats)
    alvo = (B[:, 1:] - B[:, :-1]) * DELTA_SCALE
    return F.smooth_l1_loss(out, alvo)


def _rollout(model, B, warm):
    """Roda o modelo passo a passo: caixas reais até `warm` passos, depois alimenta com a PRÓPRIA previsão.
    Retorna lista de caixas previstas (N,4) para os quadros 1..T."""
    T = B.size(1) - 1
    box, d, h, preds = B[:, 0], torch.zeros_like(B[:, 0]), None, []
    for t in range(T):
        feat = torch.cat([box, d * DELTA_SCALE], dim=-1).unsqueeze(1)
        out, h = model.forward_seq(feat, h)
        pred = box + out[:, 0] / DELTA_SCALE
        preds.append(pred)
        if t + 1 < warm:                 # ainda "observando"
            d, box = B[:, t + 1] - B[:, t], B[:, t + 1]
        else:                            # "oclusão": usa a própria previsão
            d, box = pred - box, pred
    return preds


def loss_rollout(model, B, warm):
    preds = _rollout(model, B, warm)
    return torch.stack([F.smooth_l1_loss(p * DELTA_SCALE, B[:, t + 1] * DELTA_SCALE) for t, p in enumerate(preds)]).mean()


# ---------- avaliação em janelas ----------
@torch.no_grad()
def erro_por_horizonte(model, B, warm=8, device="cpu"):
    """Erro L1 médio (normalizado) por passo APÓS `warm` observações, em free-running, para:
    GRU, velocidade zero (repete a última caixa observada) e velocidade constante (repete o último delta)."""
    model.to(device).eval()
    B = torch.as_tensor(B, dtype=torch.float32, device=device)
    preds = _rollout(model, B, warm)
    T = B.size(1) - 1
    gru, zero, cv = [], [], []
    ult = B[:, warm - 1]
    vel = B[:, warm - 1] - B[:, warm - 2]
    for k, t in enumerate(range(warm - 1, T)):      # prevê o quadro t+1
        alvo = B[:, t + 1]
        gru.append((preds[t] - alvo).abs().mean().item())
        zero.append((ult - alvo).abs().mean().item())
        cv.append((ult + vel * (k + 1) - alvo).abs().mean().item())
    return np.array(gru), np.array(zero), np.array(cv)


@torch.no_grad()
def loss_val(model, B, modo="teacher", warm=8, device="cpu"):
    model.to(device).eval()
    B = torch.as_tensor(B, dtype=torch.float32, device=device)
    l = loss_teacher(model, B) if modo == "teacher" else loss_rollout(model, B, warm)
    zero = F.smooth_l1_loss(torch.zeros_like(B[:, 1:]), (B[:, 1:] - B[:, :-1]) * DELTA_SCALE)
    return l.item(), zero.item()


# ---------- treino ----------
def treinar_gru(model, janelas_treino, janelas_val=None, epochs=30, lr=2e-3, batch=256, clip=1.0,
                modo="teacher", warm=8, device="cpu", seed=0, verbose=True, cada=5):
    """
    modo='teacher' : teacher forcing (Parte 2).
    modo='rollout' : as primeiras `warm` entradas são reais, as seguintes são as próprias previsões (Parte 4, correção).
    clip=None desliga o gradient clipping.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    model.to(device)
    X = torch.as_tensor(janelas_treino, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = {'treino': [], 'val': [], 'val_zero': []}
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X)); acum, n = 0.0, 0
        for i in range(0, len(X), batch):
            B = X[perm[i:i + batch]].to(device)
            loss = loss_teacher(model, B) if modo == "teacher" else loss_rollout(model, B, warm)
            opt.zero_grad(); loss.backward()
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            acum += loss.item(); n += 1
        hist['treino'].append(acum / n)
        if janelas_val is not None:
            lv, lz = loss_val(model, janelas_val, modo, warm, device)
            hist['val'].append(lv); hist['val_zero'].append(lz)
        if verbose and (ep % cada == 0 or ep == epochs - 1):
            msg = f"[{modo}] época {ep:03d} | treino {hist['treino'][-1]:.5f}"
            if janelas_val is not None:
                msg += f" | val {hist['val'][-1]:.5f} (vel. zero: {hist['val_zero'][-1]:.5f})"
            print(msg)
    model.eval()
    return model, hist
