import torch
import torch.nn as nn

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