import numpy as np
import cv2

class EllipsesSimulator:
    def __init__(self, img_size, num_obj, dur_occlusion, speed=1):
        self.img_size = img_size
        self.num_obj = num_obj
        self.dur_occlusion = dur_occlusion
        self.speed = (np.random.rand(num_obj) + 0.2)*speed
        self.ellipses = []
    
    def _initialize_ellipses(self):
        """Cria o estado inicial (frame 0) de todos os objetos."""
        self.ellipses = []
        for i in range(self.num_obj):
            self.ellipses.append({
                'id': i + 1,
                'x': np.random.randint(25, self.img_size - 25),
                'y': np.random.randint(25, self.img_size - 25),
                'vx': np.random.choice([-1, 1]) * self.speed[i],
                'vy': np.random.choice([-1, 1]) * self.speed[i],
                'axes': (np.random.randint(5, 20), np.random.randint(5, 20)),
                'angle': np.random.randint(0, 180),
                'intensity': np.random.randint(100, 255),
                'z_index': i # Mantém a ordem para sobreposição geométrica
            })

    def _update_physics(self, e):
        """Atualiza a posição e calcula colisões com as bordas."""
        e['x'] += e['vx']
        e['y'] += e['vy']
        
        if e['x'] < 0 or e['x'] > self.img_size: e['vx'] *= -1
        if e['y'] < 0 or e['y'] > self.img_size: e['vy'] *= -1

    def _draw_and_get_bbox(self, img, e):
        """Renderiza a elipse na matriz e extrai a caixa delimitadora perfeita."""
        center = (int(e['x']), int(e['y']))
        
        cv2.ellipse(img, center, e['axes'], e['angle'], 0, 360, e['intensity'], -1)
        
        # Extrai os pontos do contorno da elipse desenhada e gera o bounding box
        poly = cv2.ellipse2Poly(center, e['axes'], e['angle'], 0, 360, 5)
        x_min, y_min, w, h = cv2.boundingRect(poly)
        
        return x_min, y_min, w, h

    def _apply_noise(self, img):
        """Injeta ruído no quadro finalizado."""
        noise = np.random.uniform(0, 15.0, (self.img_size, self.img_size))
        return np.clip(img + noise, 0, 255).astype(np.uint8)

    def simulate(self, num_frames=60):
        """Orquestra a simulação, gerando os quadros e o gabarito."""
        self._initialize_ellipses()
        
        frames = []
        ground_truth = []

        for f in range(1, num_frames + 1):
            bg_intensity = np.random.randint(20, 80)
            img = np.full((self.img_size, self.img_size), bg_intensity, dtype=np.uint8)
            
            # Ordena do fundo para a frente para garantir oclusão correta
            self.ellipses.sort(key=lambda e: e['z_index'])
            
            for e in self.ellipses:
                self._update_physics(e)
                x, y, w, h = self._draw_and_get_bbox(img, e)
                
                # frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility
                # Visibilidade inicia em 1.0; a lógica de oclusão de área pode ser inserida depois
                annotation = f"{f}, {e['id']}, {x}, {y}, {w}, {h}, 1, 1, 1.0"
                ground_truth.append(annotation)

            noisy_img = self._apply_noise(img)
            frames.append(noisy_img)
            
        return frames, ground_truth

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


def create_noisy_bboxes(frames, ground_truth, percentage, fp_rate=0.05):
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