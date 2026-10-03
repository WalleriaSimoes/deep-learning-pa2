"""Parte 0, item 4: baseline no 'piso fácil' e varredura dos botões do gerador."""
import numpy as np
import matplotlib.pyplot as plt
from Ellipses import EllipsesSimulator
from metrics import naive_tracker_shift, evaluate_trajectories, tracks_to_str


def gt_para_deteccoes(ground_truth, vis_min=0.3):
    """Detector 'perfeito, mas que não enxerga o que está (quase) totalmente oculto'. IDs viram -1."""
    out = []
    for linha in ground_truth:
        p = linha.split(',')
        if float(p[8]) < vis_min:
            continue
        p[1] = ' -1'
        out.append(','.join(p))
    return out


def rodar_baseline_sintetico(num_obj=3, speed=0.5, dur_occ=0, num_frames=60, seed=0,
                             threshold=0.5, max_age=3, vis_min=0.3):
    sim = EllipsesSimulator(128, num_obj, dur_occ, speed=speed, seed=seed)
    frames, gt = sim.simulate(num_frames)
    dets = gt_para_deteccoes(gt, vis_min)
    preds = tracks_to_str(naive_tracker_shift(dets, threshold=threshold, max_age=max_age))
    m = evaluate_trajectories(gt, preds)
    return m["IDF1 Score"], m["ID Switches"], m["Fragmentacoes"], m["Erro de Contagem de IDs"]


def baseline_piso_facil(n_seeds=20, **kw):
    """Poucas elipses, lentas, sem oclusão: o IDF1 do baseline ingênuo tem que ficar muito perto de 1."""
    r = np.array([rodar_baseline_sintetico(num_obj=3, speed=0.5, dur_occ=0, seed=s, **kw) for s in range(n_seeds)])
    print(f"Piso fácil (3 elipses, speed=0.5, sem oclusão), {n_seeds} seeds:")
    print(f"  IDF1 = {r[:,0].mean():.4f} ± {r[:,0].std():.4f} | switches médios = {r[:,1].mean():.2f}")
    return r


def varrer(parametro, valores, base=None, n_seeds=10, **kw):
    """Varre UM botão (num_obj | speed | dur_occ) mantendo os outros no piso fácil."""
    base = dict(num_obj=3, speed=0.5, dur_occ=0) | (base or {})
    res = {}
    for v in valores:
        cfg = dict(base); cfg[parametro] = v
        r = np.array([rodar_baseline_sintetico(seed=s, **cfg, **kw) for s in range(n_seeds)])
        res[v] = dict(idf1=(r[:, 0].mean(), r[:, 0].std()), sw=(r[:, 1].mean(), r[:, 1].std()))
    return res


def plot_varreduras(resultados, titulos=None):
    """resultados: dict {parametro: saida de varrer()}. Um painel por botão."""
    n = len(resultados)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (nome, res) in zip(axes, resultados.items()):
        xs = list(res.keys())
        m = np.array([res[x]['idf1'][0] for x in xs]); s = np.array([res[x]['idf1'][1] for x in xs])
        ax.errorbar(range(len(xs)), m, yerr=s, marker='o', capsize=3)
        ax.set_xticks(range(len(xs))); ax.set_xticklabels([str(x) for x in xs])
        ax.set_xlabel(nome); ax.set_ylim(0, 1.05); ax.grid(True, linestyle='--', alpha=0.5)
    axes[0].set_ylabel('IDF1 (baseline ingênuo)')
    fig.suptitle('Onde o baseline por quadro começa a quebrar (sintético)')
    plt.tight_layout(); plt.show()
