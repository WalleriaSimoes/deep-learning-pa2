import numpy as np
import cv2
import torch
import random
import matplotlib.pyplot as plt


class EllipsesSimulator:
    """
    Gerador de vídeos sintéticos (Parte 0, item 1).

    Parâmetros expostos:
      num_obj       : número de elipses (5 a 15 no enunciado)
      dur_occlusion : nº de quadros em que um alvo fica COMPLETAMENTE escondido atrás de outra elipse
      speed         : velocidade típica (px/quadro); cada objeto usa (rand+0.2)*speed
      noise_std     : amplitude do ruído uniforme somado ao quadro final
      contrast      : em (0, 1]; 1 = elipses bem mais claras que o fundo, valores baixos = pouco contraste
      n_occlusions  : quantos eventos de oclusão forçada ocorrem no vídeo
      seed          : reprodutibilidade

    Oclusão: as elipses são desenhadas do fundo para a frente (z_index). Em cada evento, uma elipse
    "oclusora" (z maior, círculo um pouco maior que o alvo) se aproxima do alvo, passa à frente dele,
    cobre-o por exatamente `dur_occlusion` quadros (visibilidade 0) e depois se afasta. A visibilidade
    do GT é calculada pixel a pixel (fração da elipse que não está coberta por elipses mais à frente).
    """
    def __init__(self, img_size, num_obj, dur_occlusion, speed=1, noise_std=15.0,
                 contrast=1.0, n_occlusions=1, seed=None):
        self.img_size = img_size
        self.num_obj = num_obj
        self.dur_occlusion = int(dur_occlusion)
        self.noise_std = noise_std
        self.contrast = float(np.clip(contrast, 0.05, 1.0))
        self.n_occlusions = n_occlusions
        self.rng = np.random.RandomState(seed) if seed is not None else np.random
        self.speed = (self.rng.rand(num_obj) + 0.2) * speed
        self.ellipses = []
        self.occlusion_events = []

    def _initialize_ellipses(self, bg):
        """Cria o estado inicial (frame 0) de todos os objetos."""
        r = self.rng
        lo = int(bg + 0.4 * self.contrast * (255 - bg))
        hi = int(bg + self.contrast * (255 - bg))
        self.ellipses = []
        for i in range(self.num_obj):
            self.ellipses.append({
                'id': i + 1,
                'x': float(r.randint(25, self.img_size - 25)),
                'y': float(r.randint(25, self.img_size - 25)),
                'vx': r.choice([-1, 1]) * self.speed[i],
                'vy': r.choice([-1, 1]) * self.speed[i],
                'axes': (int(r.randint(5, 20)), int(r.randint(5, 20))),
                'angle': int(r.randint(0, 180)),
                'intensity': int(r.randint(lo, max(lo, hi) + 1)),
                'z_index': i  # ordem de profundidade (maior = mais à frente)
            })

    def _update_physics(self, e):
        """Atualiza a posição e rebate nas bordas."""
        e['x'] += e['vx']
        e['y'] += e['vy']
        if e['x'] < 0 or e['x'] > self.img_size: e['vx'] *= -1
        if e['y'] < 0 or e['y'] > self.img_size: e['vy'] *= -1

    def _schedule_occlusions(self, pos, num_frames, approach=8):
        """Reescreve a trajetória de uma elipse oclusora para que ela cubra o alvo por dur_occlusion quadros."""
        eventos = []
        N = len(self.ellipses)
        dur = min(self.dur_occlusion, num_frames - 2 * approach)
        if self.dur_occlusion <= 0 or N < 2 or self.n_occlusions <= 0 or dur < 1:
            return eventos
        perm = self.rng.permutation(N)
        for k in range(min(self.n_occlusions, N // 2)):
            a, b = int(perm[2 * k]), int(perm[2 * k + 1])      # a = alvo, b = oclusor
            f0 = int(self.rng.randint(approach, num_frames - dur - approach + 1))  # índice 0-based
            orig_b = pos[:, b].copy()
            for f in range(f0 - approach, f0):                 # aproximação
                w = (f - (f0 - approach) + 1) / (approach + 1)
                pos[f, b] = (1 - w) * orig_b[f] + w * pos[f, a]
            for f in range(f0, f0 + dur):                      # cobre o alvo
                pos[f, b] = pos[f, a]
            for f in range(f0 + dur, f0 + dur + approach):     # afastamento
                w = 1 - ((f - (f0 + dur)) + 1) / (approach + 1)
                pos[f, b] = w * pos[f, a] + (1 - w) * orig_b[f]
            A, B = self.ellipses[a], self.ellipses[b]
            r = max(A['axes']) + 4
            B['axes'] = (r, r)
            B['z_index'] = N + 10 + k
            if abs(B['intensity'] - A['intensity']) < 30:      # garante que dê para ver o oclusor
                B['intensity'] = A['intensity'] + 40 if A['intensity'] + 40 <= 255 else A['intensity'] - 40
            eventos.append({'target_id': A['id'], 'occluder_id': B['id'],
                            'start': f0 + 1, 'end': f0 + dur, 'dur': dur})   # 1-based, inclusivo
        return eventos

    def _bbox(self, e):
        center = (int(e['x']), int(e['y']))
        poly = cv2.ellipse2Poly(center, e['axes'], e['angle'], 0, 360, 5)
        x, y, w, h = cv2.boundingRect(poly)
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(self.img_size, x + w), min(self.img_size, y + h)
        return x0, y0, max(1, x1 - x0), max(1, y1 - y0)

    def simulate(self, num_frames=60):
        """Orquestra a simulação, gerando os quadros e o gabarito (com visibilidade real)."""
        bg = int(self.rng.randint(20, 80))          # fundo constante no vídeo (o ruído é que varia)
        self._initialize_ellipses(bg)
        N = len(self.ellipses)

        # 1) pré-simula as trajetórias e agenda as oclusões
        pos = np.zeros((num_frames, N, 2))
        for f in range(num_frames):
            for k, e in enumerate(self.ellipses):
                self._update_physics(e)
                pos[f, k] = (e['x'], e['y'])
        self.occlusion_events = self._schedule_occlusions(pos, num_frames)

        # 2) renderiza do fundo para a frente e mede a visibilidade (da frente para o fundo)
        ordem = sorted(range(N), key=lambda k: self.ellipses[k]['z_index'])
        frames, ground_truth = [], []
        for f in range(num_frames):
            img = np.full((self.img_size, self.img_size), bg, dtype=np.uint8)
            masks, boxes = {}, {}
            for k in ordem:
                e = self.ellipses[k]
                e['x'], e['y'] = pos[f, k]
                m = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
                cv2.ellipse(m, (int(e['x']), int(e['y'])), e['axes'], e['angle'], 0, 360, 1, -1)
                img[m > 0] = e['intensity']
                masks[k] = m > 0
                boxes[k] = self._bbox(e)
            ocupado = np.zeros((self.img_size, self.img_size), dtype=bool)
            vis = {}
            for k in reversed(ordem):
                area = masks[k].sum()
                vis[k] = float(((masks[k]) & ~ocupado).sum() / area) if area > 0 else 0.0
                ocupado |= masks[k]
            for k in range(N):
                x, y, w, h = boxes[k]
                # frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility
                ground_truth.append(f"{f + 1}, {self.ellipses[k]['id']}, {x}, {y}, {w}, {h}, 1, 1, {vis[k]:.2f}")
            frames.append(self._apply_noise(img))
        return frames, ground_truth

    def _apply_noise(self, img):
        noise = self.rng.uniform(0, self.noise_std, (self.img_size, self.img_size))
        return np.clip(img + noise, 0, 255).astype(np.uint8)


def plot_occlusion_event(frames, ground_truth, evento, margem=3, n_max=8, salvar=None):
    """
    Figura exigida na Parte 0: uma trajetória que SOME por N quadros e VOLTA.
    Linha de cima: quadros com a caixa GT do alvo (verde = visível, amarelo = parcial, vermelho = oculto).
    Linha de baixo: visibilidade do alvo ao longo do tempo, com o intervalo de oclusão sombreado.
    """
    tid = evento['target_id']
    alvo = {}
    for linha in ground_truth:
        p = linha.split(',')
        if int(float(p[1])) == tid:
            alvo[int(p[0])] = (int(p[2]), int(p[3]), int(p[4]), int(p[5]), float(p[8]))
    f_ini = max(1, evento['start'] - margem)
    f_fim = min(len(frames), evento['end'] + margem)
    idx = np.unique(np.linspace(f_ini, f_fim, n_max).round().astype(int))

    fig = plt.figure(figsize=(2.6 * len(idx), 6))
    gs = fig.add_gridspec(2, len(idx), height_ratios=[2.2, 1])
    for j, f in enumerate(idx):
        ax = fig.add_subplot(gs[0, j])
        img = cv2.cvtColor(frames[f - 1], cv2.COLOR_GRAY2RGB)
        x, y, w, h, v = alvo[f]
        cor = (0, 220, 0) if v > 0.8 else ((255, 200, 0) if v > 0.05 else (255, 0, 0))
        cv2.rectangle(img, (x, y), (x + w, y + h), cor, 1)
        ax.imshow(img); ax.axis('off')
        ax.set_title(f"t={f}  vis={v:.2f}", fontsize=9)
    ax = fig.add_subplot(gs[1, :])
    fs = sorted(alvo)
    ax.plot(fs, [alvo[f][4] for f in fs], marker='o', ms=3)
    ax.axvspan(evento['start'] - 0.5, evento['end'] + 0.5, color='red', alpha=0.15,
               label=f"oclusão total: {evento['dur']} quadros")
    ax.set_xlabel('quadro'); ax.set_ylabel('visibilidade do alvo'); ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle='--', alpha=0.5); ax.legend(loc='lower left')
    fig.suptitle(f"Elipse {tid} some atrás da elipse {evento['occluder_id']} por {evento['dur']} quadros e volta")
    plt.tight_layout()
    if salvar: plt.savefig(salvar, dpi=120)
    plt.show()


def visualize_ground_truth(frames, ground_truth):
    # Cria cópias coloridas dos frames para que os retângulos fiquem visíveis 
    # e para não alterar os quadros originais que serão usados pela rede neural
    frames_with_bboxes = [cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR) for img in frames]
    
    for annotation in ground_truth:
        # Divide a string da anotação usando a vírgula como separador
        parts = annotation.split(',')
        
        # O MOT17 começa a contar os frames a partir do 1
        # Subtraímos 1 para casar com o índice da lista (que começa em 0)
        f = int(parts[0]) - 1  
        
        # Extrai bb_left, bb_top, bb_width, bb_height
        x = int(parts[2])
        y = int(parts[3])
        w = int(parts[4])
        h = int(parts[5])
        
        # Desenha o retângulo vermelho no frame correspondente
        # cv2.rectangle recebe: imagem, (x_min, y_min), (x_max, y_max), cor (BGR), espessura
        cv2.rectangle(frames_with_bboxes[f], (x, y), (x + w, y + h), (0, 0, 255), 1)
        
    return frames_with_bboxes


def create_noisy_bboxes(frames, ground_truth, percentage, fp_rate=0.05, vis_min=0.0):
    """Simulador de detector: descarta p% das caixas, adiciona ruído e injeta falsos positivos.
    vis_min: um detector real não vê objetos totalmente ocultos; caixas com visibilidade < vis_min também são descartadas."""
    size = len(ground_truth)
    img_size = frames[0].shape[0]

    # Descarta p% 
    new_size = int(size * (1 - percentage)) 
    ground_truth_idx = set(np.random.choice(range(size), new_size, replace=False))

    frames_with_noisy_bboxes = [cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR) for img in frames]
    noisy_annotations = []

    for i in range(size):
        # Pula a iteração se a caixa estiver nos p% descartados (Falsos Negativos)
        if i not in ground_truth_idx:
            continue

        parts = ground_truth[i].split(',')
        if float(parts[8]) < vis_min:
            continue
        f = int(parts[0]) - 1  
        
        # Extrai bb_left, bb_top, bb_width, bb_height
        x = int(parts[2])
        y = int(parts[3])
        w = int(parts[4])
        h = int(parts[5])

        # Adiciona ruído nas coordenadas (-5 a +5 pixels)
        x += np.random.randint(-5, 6)
        y += np.random.randint(-5, 6)
        w += np.random.randint(-5, 6)
        h += np.random.randint(-5, 6)
        
        # Limita (Clip) as coordenadas para não vazarem da imagem de 128x128
        x = max(0, min(x, img_size - 1))
        y = max(0, min(y, img_size - 1))
        w = max(1, min(w, img_size - x))
        h = max(1, min(h, img_size - y))

        # Desenha a caixa ruidosa em vermelho
        cv2.rectangle(frames_with_noisy_bboxes[f], (x, y), (x + w, y + h), (0, 0, 255), 1)
        
        # Reconstrói a anotação corrompida para o seu simulador de métricas
        noisy_annotations.append(f"{f+1}, {parts[1]}, {x}, {y}, {w}, {h}, {parts[6]}, {parts[7]}, {parts[8]}")

    # 4. Injeta falsos positivos (caixas totalmente aleatórias)
    num_false_positives = int(size * fp_rate)
    for _ in range(num_false_positives):
        f = np.random.randint(0, len(frames))
        w = np.random.randint(10, 40)
        h = np.random.randint(10, 40)
        x = np.random.randint(0, img_size - w)
        y = np.random.randint(0, img_size - h)
        
        # Desenha o falso positivo em amarelo (opcional, para visualização no teste)
        cv2.rectangle(frames_with_noisy_bboxes[f], (x, y), (x + w, y + h), (0, 255, 255), 1)
        
        # Adiciona a anotação fantasma com id -1 para indicar falso positivo
        noisy_annotations.append(f"{f+1}, -1, {x}, {y}, {w}, {h}, 1, 1, 1.0")

    return frames_with_noisy_bboxes, noisy_annotations

def gerar_trajetoria_elipse(num_quadros=60, image_size=128):
    """
    Simula o movimento de 1 elipse (caixa limitadora) em linha reta com velocidade constante.
    Adiciona um pequeno ruído para simular imperfeições da detecção (como exigido no PA2).
    """
    # Posição inicial e tamanho aleatórios
    x = random.uniform(10, image_size - 40)
    y = random.uniform(10, image_size - 40)
    w = random.uniform(10, 25)
    h = random.uniform(10, 25)
    
    # Velocidade (pixels por quadro)
    vx = random.uniform(-2, 2)
    vy = random.uniform(-2, 2)
    
    trajetoria = []
    for _ in range(num_quadros):
        # Atualiza a posição
        x += vx
        y += vy
        
        # Injeta ruído sintético (simulador de detector imperfeito)
        ruido_x = random.uniform(-1, 1)
        ruido_y = random.uniform(-1, 1)
        
        caixa = [x + ruido_x, y + ruido_y, w, h]
        trajetoria.append(caixa)
        
    return torch.tensor(trajetoria, dtype=torch.float32)

# --- 2. O Fatiador de Janelas (BPTT) ---
def preparar_janelas_treino(trajetoria, tamanho_janela_T):
    """
    Desliza uma janela sobre a trajetória para criar pares de (Passado, Futuro).
    Retorna os tensores no formato (batch_size, seq_len, features)
    """
    num_quadros = trajetoria.size(0)
    inputs = []
    targets = []
    
    # Desliza a janela pela trajetória
    for i in range(num_quadros - tamanho_janela_T):
        # Pega T quadros
        janela = trajetoria[i : i + tamanho_janela_T]
        
        # O input é do quadro 0 até o T-1 da janela
        seq_in = janela[:-1, :]
        # O target é o último quadro da janela
        alvo = janela[-1, :]
        
        inputs.append(seq_in)
        targets.append(alvo)
        
    # Empilha tudo numa dimensão de batch
    # Formato final inputs: (batch_size, T-1, 4)
    # Formato final targets: (batch_size, 4)
    return torch.stack(inputs), torch.stack(targets)