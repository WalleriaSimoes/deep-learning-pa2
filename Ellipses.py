import numpy as np
import cv2
import time
from IPython.display import clear_output
import matplotlib.pyplot as plt

class EllipsesSimulator:
    def __init__(self, img_size, num_obj, speed, dur_occlusion):
        self.img_size = img_size
        self.num_obj = num_obj
        self.speed = speed
        self.dur_occlusion = dur_occlusion
        self.ellipses = []
    
    def _initialize_ellipses(self):
        """Cria o estado inicial (frame 0) de todos os objetos."""
        self.ellipses = []
        for i in range(self.num_obj):
            self.ellipses.append({
                'id': i + 1,
                'x': np.random.randint(25, self.img_size - 25),
                'y': np.random.randint(25, self.img_size - 25),
                'vx': np.random.choice([-1, 1]) * self.speed,
                'vy': np.random.choice([-1, 1]) * self.speed,
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
    
def plot_simulation_frames(frames):
    for img in frames:
        clear_output(wait=True) 
        fig, axes = plt.subplots(1, 1)
        axes.imshow(img, cmap='gray')
        axes.set_title("Simulação Contínua")
        axes.axis('off')
        plt.show()
        time.sleep(0.1) # 100ms de pausa 