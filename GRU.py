import torch
import torch.nn as nn
import numpy as np
import cv2

# Usamos como base o site https://www.geeksforgeeks.org/machine-learning/gated-recurrent-unit-networks/

# Os deltas de caixa normalizada são minúsculos (~1e-3); multiplicamos por DELTA_SCALE para que
# entrada e alvo tenham magnitude razoável para a rede e para a smooth-L1.
DELTA_SCALE = 100.0


def features_from_boxes(b):
    """b: (N, T, 4) caixas normalizadas -> (N, T, 8) = [caixa, delta_para_caixa_anterior * DELTA_SCALE] (delta_0 = 0)."""
    d = torch.zeros_like(b)
    d[:, 1:] = b[:, 1:] - b[:, :-1]
    return torch.cat([b, d * DELTA_SCALE], dim=-1)


class BBoxTrackerRNN(nn.Module):
    """
    Modelo de movimento recorrente (RNN simples / LSTM / GRU) para a Trilha A.
    Entrada por passo : [caixa_norm(4), delta_norm*DELTA_SCALE(4)]  -> input_size = 8
    Saída por passo   : delta previsto da caixa no quadro seguinte (x DELTA_SCALE)  -> output_size = 4
    """
    def __init__(self, cell_type='GRU', input_size=8, hidden_size=64, num_layers=1, output_size=4):
        super().__init__()
        self.cell_type, self.input_size, self.hidden_size = cell_type, input_size, hidden_size
        self.num_layers, self.output_size = num_layers, output_size
        cls = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}[cell_type]
        self.rnn = cls(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    @property
    def gru(self):          # compatibilidade com código antigo que usa modelo.gru
        return self.rnn

    def forward_seq(self, x, hidden=None):
        """x: (B, T, input_size) -> (saídas de TODOS os passos (B, T, output_size), hidden)."""
        out, hidden = self.rnn(x, hidden)
        return self.fc(out), hidden

    def forward(self, x, hidden=None):
        """Só a saída do último passo (compatível com a versão anterior)."""
        out, hidden = self.rnn(x, hidden)
        return self.fc(out[:, -1, :]), hidden

    def init_hidden(self, batch_size=1, device="cpu"):
        z = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (z, z.clone()) if self.cell_type == 'LSTM' else z


class BBoxTrackerGRU(BBoxTrackerRNN):
    def __init__(self, input_size=8, hidden_size=64, num_layers=1, output_size=4):
        super().__init__('GRU', input_size, hidden_size, num_layers, output_size)


def salvar_checkpoint(model, caminho, extra=None):
    cfg = dict(cell_type=model.cell_type, input_size=model.input_size, hidden_size=model.hidden_size,
               num_layers=model.num_layers, output_size=model.output_size)
    torch.save({'config': cfg, 'state_dict': model.state_dict(), 'extra': extra or {}}, caminho)


def carregar_checkpoint(caminho, device='cpu'):
    ck = torch.load(caminho, map_location=device)
    model = BBoxTrackerRNN(**ck['config']).to(device)
    model.load_state_dict(ck['state_dict']); model.eval()
    return model, ck


def parse_gt_to_tensor_all(ground_truth_list, num_ids):
    """
    Extrai as coordenadas [x, y, w, h] de TODOS os objetos simultaneamente.
    Retorna um Tensor no formato: (num_objetos, seq_len, 4)
    """
    trajetorias = {i: [] for i in range(1, num_ids + 1)}
    for linha in ground_truth_list:
        parts = linha.split(',')
        obj_id = int(float(parts[1]))
        if obj_id in trajetorias:
            trajetorias[obj_id].append([float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])])
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
        for obj in range(num_objs):
            xr, yr, wr, hr = map(int, traj_real[obj, f])
            cv2.rectangle(img, (xr, yr), (xr+wr, yr+hr), (0, 0, 255), 2)
            xp, yp, wp, hp = map(int, traj_pred[obj, f])
            cv2.rectangle(img, (xp, yp), (xp+wp, yp+hp), (255, 0, 0), 2)
        frames_animados.append(img)
    return frames_animados
