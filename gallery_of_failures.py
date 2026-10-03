"""
Parte 4: horizonte de memória (analítico e empírico), galeria de falhas, diagnóstico e antes/depois.
Convenção: ground_truth / preds são listas de strings 'frame, id, x, y, w, h, conf, class, vis'.
Caixas fantasma (previstas pela recorrência sem observação) têm conf = -1.
"""
import colorsys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from metrics import calculate_iou_matrix, greedy_match


def _parse(strs):
    return np.array([list(map(float, s.split(','))) for s in strs]) if len(strs) else np.zeros((0, 9))


def cor_por_id(i):
    """Cor RGB (0-255) determinística por identidade."""
    h = (int(i) * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


# ===================== 4.1  HORIZONTE ANALÍTICO: ||dL_t / dh_{t-k}|| =====================
def curva_gradiente(modelo, janelas, T=None, n_amostras=256, device="cpu"):
    """
    Norma de dL_t/dh_{t-k} em função de k (k=0 é o último passo), média sobre `n_amostras` janelas reais.
    janelas: array (N, T'+1, 4) de caixas normalizadas (use janelas do vídeo de validação/teste).
    Funciona para RNN/LSTM/GRU (BBoxTrackerRNN). A perda é a smooth-L1 do delta previsto no ÚLTIMO passo.
    """
    import torch
    import torch.nn.functional as F
    from GRU import features_from_boxes, DELTA_SCALE
    B = torch.as_tensor(janelas[:n_amostras], dtype=torch.float32, device=device)
    T = T or (B.size(1) - 1)
    B = B[:, :T + 1]
    estava_treino = modelo.training
    modelo.to(device).train()                       # cuDNN-RNN só faz backward em modo train
    with torch.backends.cudnn.flags(enabled=False):
        feats = features_from_boxes(B[:, :T])
        hid, hs = None, []
        for t in range(T):
            _, hid = modelo.rnn(feats[:, t:t + 1], hid)
            h = hid[0] if isinstance(hid, tuple) else hid
            h.retain_grad(); hs.append(h)
        pred = modelo.fc(hs[-1][-1])                # (N, 4): delta previsto para o quadro T
        alvo = (B[:, T] - B[:, T - 1]) * DELTA_SCALE
        modelo.zero_grad()
        F.smooth_l1_loss(pred, alvo, reduction="sum").backward()
    normas = np.stack([h.grad[-1].norm(dim=-1).cpu().numpy() for h in hs[::-1]], axis=1)   # (N, T): k = 0..T-1
    modelo.train(estava_treino)
    return {'media': normas.mean(0), 'p25': np.percentile(normas, 25, axis=0), 'p75': np.percentile(normas, 75, axis=0)}


def plot_curvas_gradiente(curvas, titulo=r"Horizonte analítico: $\|\partial L_t/\partial h_{t-k}\|$"):
    """curvas: {nome: saída de curva_gradiente}. Eixo x = k quadros para trás (k=0 é o mais recente)."""
    plt.figure(figsize=(8, 5))
    for nome, c in curvas.items():
        k = np.arange(len(c['media']))
        plt.plot(k, c['media'], marker='o', label=nome)
        plt.fill_between(k, c['p25'], c['p75'], alpha=0.15)
    plt.yscale('log'); plt.xlabel('k = quadros no passado'); plt.ylabel('norma L2 do gradiente (log)')
    plt.title(titulo); plt.grid(True, which='both', linestyle='--', alpha=0.5); plt.legend(); plt.show()


def fator_de_queda(curva, k=8):
    """Quantas vezes a norma cai de k=0 até k (ex.: 'cai 20x em 8 passos')."""
    m = curva['media']; k = min(k, len(m) - 1)
    return float(m[0] / max(m[k], 1e-12))


# ===================== 4.2  HORIZONTE EMPÍRICO =====================
def linha_do_tempo_gt(ground_truth, preds, threshold=0.5):
    """Para cada ID de GT: lista de (quadro, id_previsto_ou_None, visibilidade), usando o mesmo pareamento por quadro da métrica."""
    T, P = _parse(ground_truth), _parse(preds)
    tl = {}
    for f in np.unique(T[:, 0]):
        ft = T[T[:, 0] == f]; fp = P[P[:, 0] == f]
        atrib = {}
        if len(fp) > 0:
            iou = calculate_iou_matrix(fp, ft)
            mi, m = greedy_match(iou, len(fp), len(ft))
            for p in range(len(fp)):
                if m[p] is not None and mi[p] >= threshold:
                    atrib[int(m[p])] = int(fp[p, 1])
        for j, row in enumerate(ft):
            tl.setdefault(int(row[1]), []).append((int(f), atrib.get(j), float(row[8])))
    return tl


def eventos_de_lacuna(timeline):
    """
    Lacuna = quadros consecutivos do GT sem caixa prevista associada.
    resultado: 'sobreviveu' (mesmo ID antes e depois) | 'trocou_id' | 'perdida_ate_o_fim'.
    Trocas diretas (sem lacuna) entram com comprimento 0.
    """
    ev = []
    for tid, seq in timeline.items():
        ultimo, i, n = None, 0, len(seq)
        while i < n:
            f, pid, _ = seq[i]
            if pid is not None:
                if ultimo is not None and pid != ultimo:
                    ev.append(dict(gt_id=tid, f_ini=f, f_fim=f, comprimento=0, resultado='trocou_id', pred_antes=ultimo, pred_depois=pid))
                ultimo = pid; i += 1; continue
            j = i
            while j < n and seq[j][1] is None:
                j += 1
            if ultimo is not None:
                comp = seq[j - 1][0] - seq[i][0] + 1
                if j < n:
                    dep = seq[j][1]; res = 'sobreviveu' if dep == ultimo else 'trocou_id'
                else:
                    dep, res = None, 'perdida_ate_o_fim'
                ev.append(dict(gt_id=tid, f_ini=seq[i][0], f_fim=seq[j - 1][0], comprimento=comp, resultado=res, pred_antes=ultimo, pred_depois=dep))
            i = j
    return ev


def duracoes_oclusao_dataset(ground_truth, vis_thr=0.3):
    """Distribuição de duração de oclusão no GT: corridas de quadros com visibilidade < vis_thr ou ausentes entre duas aparições."""
    T = _parse(ground_truth); out = []
    for tid in np.unique(T[:, 1]):
        if tid <= 0: continue
        rows = T[T[:, 1] == tid]; rows = rows[np.argsort(rows[:, 0])]
        run, prev = 0, None
        for r in rows:
            if prev is not None and r[0] - prev > 1:
                run += int(r[0] - prev - 1)
            if r[8] < vis_thr:
                run += 1
            else:
                if run > 0: out.append(run)
                run = 0
            prev = r[0]
    return np.array(out, dtype=int)


def resumo_empirico(eventos, verbose=True):
    lac = [e for e in eventos if e['comprimento'] > 0]
    sob = np.array([e['comprimento'] for e in lac if e['resultado'] == 'sobreviveu'])
    fal = np.array([e['comprimento'] for e in lac if e['resultado'] != 'sobreviveu'])
    r = dict(n_lacunas=len(lac), n_sobreviveram=len(sob), n_falharam=len(fal),
             sobrevivencia_maxima=int(sob.max()) if len(sob) else 0,
             lacuna_media_sobreviveu=float(sob.mean()) if len(sob) else float('nan'),
             lacuna_media_falha=float(fal.mean()) if len(fal) else float('nan'),
             trocas_diretas=sum(1 for e in eventos if e['comprimento'] == 0))
    if verbose:
        print(f"Lacunas (GT sem predição associada): {r['n_lacunas']} | sobreviveram (mesmo ID): {r['n_sobreviveram']} | falharam: {r['n_falharam']}")
        print(f"Sobrevivência máxima do estado: {r['sobrevivencia_maxima']} quadros")
        print(f"Lacuna média que sobrevive: {r['lacuna_media_sobreviveu']:.1f} | que causa falha: {r['lacuna_media_falha']:.1f} | trocas diretas de ID: {r['trocas_diretas']}")
    return r


def taxa_sobrevivencia_por_duracao(eventos, bins=(1, 3, 6, 11, 21, 10 ** 9)):
    """Fração de lacunas que sobrevivem (mesmo ID) por faixa de duração."""
    lac = [e for e in eventos if e['comprimento'] > 0]
    linhas = []
    for a, b in zip(bins[:-1], bins[1:]):
        grupo = [e for e in lac if a <= e['comprimento'] < b]
        if grupo:
            linhas.append((f"{a}-{b-1}" if b < 10 ** 9 else f">={a}", len(grupo), np.mean([e['resultado'] == 'sobreviveu' for e in grupo])))
    print(f"{'duração':>8s} {'n':>6s} {'%sobrevive':>11s}")
    for nome, n, frac in linhas:
        print(f"{nome:>8s} {n:6d} {100*frac:10.1f}%")
    return linhas


def plot_horizonte_empirico(dur_dataset, eventos, max_age=None, T_treino=None):
    """CDFs: duração de oclusão no dataset vs. lacunas que o estado sobreviveu vs. lacunas que causaram falha."""
    sob = [e['comprimento'] for e in eventos if e['comprimento'] > 0 and e['resultado'] == 'sobreviveu']
    fal = [e['comprimento'] for e in eventos if e['comprimento'] > 0 and e['resultado'] != 'sobreviveu']
    plt.figure(figsize=(8, 5))
    for dados, nome in ((dur_dataset, 'oclusão no dataset (GT)'), (sob, 'lacunas sobrevividas (mesmo ID)'), (fal, 'lacunas com falha (troca/morte)')):
        if len(dados):
            x = np.sort(dados); plt.step(x, np.arange(1, len(x) + 1) / len(x), where='post', label=f"{nome} (n={len(x)})")
    if max_age: plt.axvline(max_age, color='gray', linestyle=':', label=f'max_age={max_age}')
    if T_treino: plt.axvline(T_treino, color='k', linestyle='--', label=f'janela BPTT T={T_treino}')
    plt.xscale('log'); plt.xlabel('duração (quadros)'); plt.ylabel('CDF'); plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.title('Horizonte empírico vs. distribuição de oclusão do dataset'); plt.legend(); plt.show()


def medir_horizonte(ground_truth, preds, threshold=0.5, verbose=True):
    tl = linha_do_tempo_gt(ground_truth, preds, threshold)
    ev = eventos_de_lacuna(tl)
    return tl, ev, resumo_empirico(ev, verbose)


# ===================== 4.3  GALERIA DE FALHAS =====================
def selecionar_falhas(eventos, n=3, dist_min=30):
    """Os n piores eventos de falha (maior lacuna que terminou em troca de ID ou perda), em IDs/instantes distintos."""
    cand = sorted([e for e in eventos if e['resultado'] != 'sobreviveu' and e['comprimento'] > 0], key=lambda e: -e['comprimento'])
    esc = []
    for e in cand:
        if all(e['gt_id'] != o['gt_id'] and abs(e['f_ini'] - o['f_ini']) > dist_min for o in esc):
            esc.append(e)
        if len(esc) == n: break
    return esc


def _ret_tracejado(img, p1, p2, cor, esp=2, seg=6):
    (x1, y1), (x2, y2) = p1, p2
    for (a, b) in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
        L = int(np.hypot(b[0] - a[0], b[1] - a[1])); n = max(1, L // seg)
        for s in range(0, n, 2):
            p = (int(a[0] + (b[0] - a[0]) * s / n), int(a[1] + (b[1] - a[1]) * s / n))
            q = (int(a[0] + (b[0] - a[0]) * (s + 1) / n), int(a[1] + (b[1] - a[1]) * (s + 1) / n))
            cv2.line(img, p, q, cor, esp)


def plot_falha(get_frame, ground_truth, preds, evento, timeline, margem=6, n_quadros=6, pad=80, salvar=None, titulo=None):
    """
    Tira de quadros recortada em torno do alvo: GT (branco; alvo em verde) e predições COLORIDAS POR IDENTIDADE
    (fantasma = caixa prevista pela recorrência, tracejada). Embaixo: ID previsto associado ao alvo ao longo do tempo.
    """
    T, P = _parse(ground_truth), _parse(preds)
    tid = evento['gt_id']
    max_frames = int(T[:, 0].max())
    f_a = max(1, evento['f_ini'] - margem)
    f_b = min(max_frames, evento['f_fim'] + margem)
    alvo = T[(T[:, 1] == tid) & (T[:, 0] >= f_a) & (T[:, 0] <= f_b)]
    H, W = get_frame(int(alvo[0, 0])).shape[:2]
    x0 = int(max(0, alvo[:, 2].min() - pad)); y0 = int(max(0, alvo[:, 3].min() - pad))
    x1 = int(min(W, (alvo[:, 2] + alvo[:, 4]).max() + pad)); y1 = int(min(H, (alvo[:, 3] + alvo[:, 5]).max() + pad))
    sel = np.unique(np.linspace(f_a, f_b, n_quadros).round().astype(int))
    fig = plt.figure(figsize=(3.2 * len(sel), 6.5))
    gs = fig.add_gridspec(2, len(sel), height_ratios=[3, 1.2])
    for j, f in enumerate(sel):
        img = get_frame(f)[y0:y1, x0:x1].copy()
        for r in T[T[:, 0] == f]:
            p1 = (int(r[2] - x0), int(r[3] - y0)); p2 = (int(r[2] + r[4] - x0), int(r[3] + r[5] - y0))
            if int(r[1]) == tid: cv2.rectangle(img, p1, p2, (0, 255, 0), 3)
            else: cv2.rectangle(img, p1, p2, (255, 255, 255), 1)
        for r in P[P[:, 0] == f]:
            cor = cor_por_id(r[1])
            p1 = (int(r[2] - x0), int(r[3] - y0)); p2 = (int(r[2] + r[4] - x0), int(r[3] + r[5] - y0))
            if r[6] < 0: _ret_tracejado(img, p1, p2, cor, 2)
            else: cv2.rectangle(img, p1, p2, cor, 2)
            cv2.putText(img, f"P{int(r[1])}", (p1[0], max(12, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)
        ax = fig.add_subplot(gs[0, j]); ax.imshow(img); ax.axis('off'); ax.set_title(f"t={f}", fontsize=10)
    ax = fig.add_subplot(gs[1, :])
    seq = [s for s in timeline[tid] if f_a <= s[0] <= f_b]
    ax.scatter([s[0] for s in seq], [0] * len(seq), marker='s', s=90,
               c=[(np.array(cor_por_id(s[1])) / 255.0 if s[1] is not None else (0.6, 0.6, 0.6)) for s in seq])
    for s in seq:
        ax.text(s[0], 0.25, "-" if s[1] is None else f"P{s[1]}", ha='center', fontsize=7)
    ax.axvspan(evento['f_ini'] - 0.5, evento['f_fim'] + 0.5, color='red', alpha=0.12)
    ax.set_yticks([]); ax.set_ylim(-0.6, 0.7); ax.set_xlabel(f'quadro  |  ID previsto associado ao GT {tid}  (cinza = sem associação)')
    fig.suptitle(titulo or f"GT {tid}: lacuna de {evento['comprimento']} quadros -> {evento['resultado']}  "
                           f"(P{evento['pred_antes']} → P{evento['pred_depois']})   [verde = alvo GT, tracejado = previsão da recorrência]")
    plt.tight_layout()
    if salvar: plt.savefig(salvar, dpi=110)
    plt.show()


def diagnostico_texto(evento, curva=None, T_treino=None, max_age=None, k_ref=8):
    """Gera a frase de diagnóstico no formato pedido (preencha o 'porquê' com base nos números)."""
    s = f"O objeto GT {evento['gt_id']} fica {evento['comprimento']} quadros sem observação associada ({evento['resultado']})"
    if max_age is not None:
        s += f"; max_age = {max_age}" + (" (< duração da lacuna: a track morre antes de a oclusão acabar)" if evento['comprimento'] >= max_age else "")
    if T_treino is not None:
        s += f"; minha janela de BPTT é {T_treino}" + (" (a lacuna é maior que a janela: o treino nunca viu uma sequência tão longa sem observação)" if evento['comprimento'] > T_treino else "")
    if curva is not None:
        s += f"; a norma do gradiente cai {fator_de_queda(curva, k_ref):.0f}x em {k_ref} passos."
    return s


def galeria_de_falhas(get_frame, ground_truth, preds, timeline, eventos, n=3, **kw):
    sel = selecionar_falhas(eventos, n)
    for e in sel:
        plot_falha(get_frame, ground_truth, preds, e, timeline, **kw)
    return sel


# ===================== 4.4  ANTES / DEPOIS =====================
def tabela_antes_depois(linhas):
    """linhas: {nome: dict(metricas de evaluate_trajectories + resumo_empirico)}"""
    print(f"{'config':26s} {'IDF1':>7s} {'switches':>9s} {'fragm.':>7s} {'sobrev.máx':>11s} {'lac.falha(média)':>17s}")
    for nome, m in linhas.items():
        print(f"{nome:26s} {m['IDF1 Score']:7.4f} {m['ID Switches']:9d} {m['Fragmentacoes']:7d} {m['sobrevivencia_maxima']:11d} {m['lacuna_media_falha']:17.1f}")
