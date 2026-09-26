import time
from IPython.display import clear_output
import matplotlib.pyplot as plt


def plot_simulation_frames(frames):
    for img in frames:
        clear_output(wait=True) 
        fig, axes = plt.subplots(1, 1)
        axes.imshow(img, cmap='gray', vmin=0, vmax=255)
        axes.set_title("Simulação Contínua")
        axes.axis('off')
        plt.show()
        time.sleep(0.1) # 100ms de pausa 