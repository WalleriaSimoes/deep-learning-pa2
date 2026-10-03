"""Utilitários para ler GT/detecções do MOT17 direto do MOT17Dataset (sem carregar imagens) e quadros sob demanda."""
import configparser
import numpy as np
from PIL import Image


def gt_strings_por_sequencia(dataset, so_conf_1=True):
    """{seq_name: [ 'frame, id, x, y, w, h, conf, class, vis', ... ]} ordenado por quadro.
    so_conf_1: descarta caixas com flag conf=0 (ignoradas pelo protocolo do MOT17)."""
    out = {}
    for seq, frames in dataset.gts.items():
        linhas = []
        for fid in sorted(frames):
            for r in frames[fid]:
                if so_conf_1 and r[6] == 0:
                    continue
                linhas.append(f"{int(r[0])}, {int(r[1])}, {r[2]:.1f}, {r[3]:.1f}, {r[4]:.1f}, {r[5]:.1f}, {int(r[6])}, {int(r[7])}, {r[8]:.2f}")
        if linhas:
            out[seq] = linhas
    return out


def dets_strings_por_sequencia(dataset, conf_min=0.0):
    """Detecções públicas (det.txt) por sequência, no mesmo formato de 9 colunas (id = -1)."""
    out = {}
    for seq, frames in dataset.dets.items():
        linhas = []
        for fid in sorted(frames):
            for r in frames[fid]:
                if r[6] < conf_min:
                    continue
                linhas.append(f"{int(r[0])}, -1, {r[2]:.1f}, {r[3]:.1f}, {r[4]:.1f}, {r[5]:.1f}, {r[6]:.3f}, 1, 1.0")
        if linhas:
            out[seq] = linhas
    return out


def tamanho_sequencia(dataset, seq_name):
    """(largura, altura) lidos de seqinfo.ini (fallback: primeiro quadro)."""
    d = dataset.root / seq_name
    ini = d / "seqinfo.ini"
    if ini.exists():
        cp = configparser.ConfigParser(); cp.read(ini)
        return int(cp["Sequence"]["imWidth"]), int(cp["Sequence"]["imHeight"])
    return Image.open(d / "img1" / "000001.jpg").size


def tamanhos_sequencias(dataset, seqs):
    return {s: tamanho_sequencia(dataset, s) for s in seqs}


def frame_loader(dataset, seq_name, cache_max=64):
    """Devolve get_frame(n) -> imagem RGB uint8 do quadro n (1-based), com cache simples."""
    cache = {}
    def get_frame(n):
        n = int(n)
        if n not in cache:
            if len(cache) >= cache_max:
                cache.pop(next(iter(cache)))
            cache[n] = np.array(Image.open(dataset.root / seq_name / "img1" / f"{n:06d}.jpg").convert("RGB"))
        return cache[n]
    return get_frame


def frame_loader_lista(frames):
    """Para o sintético: frames = lista de imagens (cinza ou RGB), índice 0 = quadro 1."""
    import cv2
    def get_frame(n):
        img = frames[int(n) - 1]
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim == 2 else img
    return get_frame
