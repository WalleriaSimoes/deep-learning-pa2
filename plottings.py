import time
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np

def plot_simulation_frames(frames):
    for img in frames:
        clear_output(wait=True) 
        fig, axes = plt.subplots(1, 1)
        axes.imshow(img, cmap='gray', vmin=0, vmax=255)
        axes.set_title("Simulação Contínua")
        axes.axis('off')
        plt.show()
        time.sleep(0.1) # 100ms de pausa 

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