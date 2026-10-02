import torch
import torch.nn as nn
import numpy as np
import torch
import cv2

# Usamos como base o site https://www.geeksforgeeks.org/machine-learning/gated-recurrent-unit-networks/

class BBoxTrackerGRU(nn.Module):
    """
    Modelo de movimento baseado em GRU para rastreamento de múltiplos objetos.
    Recebe as coordenadas de uma caixa limitadora no tempo t e prevê a caixa no tempo t+1.
    """
    def __init__(self, input_size=4, hidden_size=64, num_layers=1):
        super(BBoxTrackerGRU, self).__init__()
        
        self.input_size = input_size   # (left, top, width, height)
        self.hidden_size = hidden_size # memória da GRU
        self.num_layers = num_layers   # camadas GRU empilhadas
        
        # A Camada GRU: Onde a física e a inércia são aprendidas
        # batch_first=True significa que os tensores terão formato (batch, seq, features)
        self.gru = nn.GRU(
            input_size=self.input_size, 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers, 
            batch_first=True
        )
        
        # Volta para 4 coordenadas
        self.fc = nn.Linear(self.hidden_size, input_size)

    def forward(self, x, hidden):
        """
        Passo forward do modelo.
        
        Parâmetros:
        x: Tensor de entrada com as caixas. Formato: (batch_size, seq_len, input_size)
        hidden: Tensor do estado oculto anterior. Formato: (num_layers, batch_size, hidden_size)
        
        Retorna:
        pred_box: A caixa prevista para o próximo quadro.
        hidden: O estado oculto atualizado.
        """
        # Passa pela GRU
        # out contém as saídas de todos os passos de tempo; hidden é o estado final
        out, hidden = self.gru(x, hidden)
        
        # Pegamos apenas a saída do último passo de tempo da sequência (out[:, -1, :]) e passamos pela camada linear para prever as 4 coordenadas [left, top, w, h]
        pred_box = self.fc(out[:, -1, :])
        
        return pred_box, hidden

    def init_hidden(self, batch_size=1, device="cpu"):
        """
        Inicializa o estado oculto zerado. Deve ser chamado sempre que um novo objeto nasce.
        """
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

def parse_gt_to_tensor_all(ground_truth_list, num_ids):
    """
    Extrai as coordenadas [x, y, w, h] de TODOS os objetos simultaneamente.
    Retorna um Tensor no formato: (num_objetos, seq_len, 4)
    """
    # Cria um dicionário vazio para cada ID
    trajetorias = {i: [] for i in range(1, num_ids + 1)}
    
    for linha in ground_truth_list:
        parts = linha.split(',')
        obj_id = int(float(parts[1]))
        
        if obj_id in trajetorias:
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            trajetorias[obj_id].append([x, y, w, h])
            
    # Empilha as listas num único tensor
    lista_tensores = [torch.tensor(trajetorias[i], dtype=torch.float32) for i in range(1, num_ids + 1)]
    return torch.stack(lista_tensores)

def render_gru_predictions_multi(frames_fundo, traj_real, traj_pred, janela_T):
    """
    Desenha as caixas reais (Azul) e previstas (Vermelho) para múltiplos objetos.
    Espera tensores no formato: (num_objetos, seq_len, 4)
    """
    frames_animados = []
    offset = janela_T - 1
    
    num_objs = traj_pred.shape[0]
    num_frames = traj_pred.shape[1]
    
    for f in range(num_frames):
        frame_idx = f + offset
        if frame_idx >= len(frames_fundo):
            break
            
        img = frames_fundo[frame_idx].copy()
        
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
        # Desenha as caixas de todos os objetos neste quadro
        for obj in range(num_objs):
            # Real (Azul)
            xr, yr, wr, hr = map(int, traj_real[obj, f])
            cv2.rectangle(img, (xr, yr), (xr+wr, yr+hr), (0, 0, 255), 2)
            
            # Prevista (Vermelho)
            xp, yp, wp, hp = map(int, traj_pred[obj, f])
            cv2.rectangle(img, (xp, yp), (xp+wp, yp+hp), (255, 0, 0), 2)
            
        frames_animados.append(img)
        
    return frames_animados