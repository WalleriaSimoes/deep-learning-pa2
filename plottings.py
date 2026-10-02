import time
from IPython.display import clear_output
import cv2
import matplotlib.pyplot as plt
import numpy as np

def plot_simulation_frames(frames, delay: float=0.1):
    for img in frames:
        clear_output(wait=True) 
        fig, axes = plt.subplots(1, 1)
        axes.imshow(img, cmap='gray', vmin=0, vmax=255)
        axes.set_title("Simulação Contínua")
        axes.axis('off')
        plt.show()
        time.sleep(delay) # 100ms de pausa 

def render_boxes_to_frames(traj_real, traj_pred=None, image_size=128):
    frames = []
    # Garante que estamos lidando com arrays 2D (seq_len, 4)
    if len(traj_real.shape) == 3:
        traj_real = traj_real.squeeze(0)
    if traj_pred is not None and len(traj_pred.shape) == 3:
        traj_pred = traj_pred.squeeze(0)

    for i in range(len(traj_real)):
        # Cria um fundo preto RGB (128, 128, 3)
        img = np.zeros((image_size, image_size, 3), dtype=np.uint8)

        # Extrai coordenadas reais e desenha uma caixa Azul
        xr, yr, wr, hr = map(int, traj_real[i])
        img[max(0, yr):min(image_size, yr+hr), max(0, xr):min(image_size, xr+wr), 2] = 255
        img[max(0, yr+2):min(image_size, yr+hr-2), max(0, xr+2):min(image_size, xr+wr-2), 2] = 0

        # Extrai coordenadas previstas e desenha uma caixa Vermelha
        if traj_pred is not None:
            xp, yp, wp, hp = map(int, traj_pred[i])
            img[max(0, yp):min(image_size, yp+hp), max(0, xp):min(image_size, xp+wp), 0] = 255
            img[max(0, yp+2):min(image_size, yp+hp-2), max(0, xp+2):min(image_size, xp+wp-2), 0] = 0

        frames.append(img)
    return frames


def visualize_ground_truth_RGB(frames, ground_truth, put_id=False, fmt=lambda id: str(id)):
    # Cria copias coloridas dos frames para que os retangulos fiquem visiveis 
    # e para nao alterar os quadros originais que serao usados pela rede neural
    frames_with_bboxes = [cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB) for img in frames]
    
    for annotation in ground_truth:
        # Divide a string da anotacao usando a virgula como separador
        parts = annotation.split(',')
        
        # O MOT17 comeca a contar os frames a partir do 1
        # Subtraimos 1 para casar com o indice da lista (que comeca em 0)
        f = int(parts[0]) - 1  
        
        # Extrai bb_left, bb_top, bb_width, bb_height
        id = int(parts[1])
        x  = int(parts[2])
        y  = int(parts[3])
        w  = int(parts[4])
        h  = int(parts[5])
        
        # Desenha o retangulo vermelho no frame correspondente
        # cv2.rectangle recebe: imagem, (x_min, y_min), (x_max, y_max), cor (BGR), espessura
        rect = cv2.rectangle(frames_with_bboxes[f], (x, y), (x + w, y + h), (0, 0, 255), 1)
        if put_id:
            cv2.putText(rect, fmt(id), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        
    return frames_with_bboxes