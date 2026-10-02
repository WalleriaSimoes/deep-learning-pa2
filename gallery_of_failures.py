import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

def analisar_gradiente_bptt(modelo, trajetoria_exemplo, T=15):
    """
    Desdobra a GRU passo a passo para extrair a norma do gradiente 
    da Perda (L) em relação aos estados ocultos passados (h_{t-k}).
    """
    modelo.eval() # Usamos eval, mas com gradientes ligados
    modelo.zero_grad()
    
    device = next(modelo.parameters()).device
    trajetoria_exemplo = trajetoria_exemplo.to(device)
    
    # 1. Estado inicial limpo
    hidden = modelo.init_hidden(batch_size=1, device=device)
    historico_hiddens = []
    
    # 2. Forward passo a passo (Manual Unrolling)
    for k in range(T):
        x = trajetoria_exemplo[:, k:k+1, :] # Pega apenas o quadro k
        _, hidden = modelo.gru(x, hidden)
        
        # Diz ao PyTorch para NÃO apagar este gradiente após o backward
        hidden.retain_grad() 
        historico_hiddens.append(hidden)
        
    # 3. Previsão e Perda apenas no último quadro
    ultimo_hidden = historico_hiddens[-1][:, -1, :]
    predicao = modelo.fc(ultimo_hidden)
    alvo_real = trajetoria_exemplo[:, T, :]
    
    loss = torch.nn.functional.smooth_l1_loss(predicao, alvo_real)
    
    # 4. Backward (Onde a magia acontece)
    loss.backward()
    
    # 5. Recolha das normas dos gradientes
    normas = []
    for h in historico_hiddens:
        # Calcula a norma L2 do gradiente daquele estado
        norma = h.grad.norm().item()
        normas.append(norma)
        
    # 6. Plot (O eixo X é a distância no tempo 'k' para trás)
    # Revertemos a lista para que a esquerda (0) seja o passado mais distante
    normas.reverse() 
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(T, 0, -1), normas, marker='o', color='purple', linewidth=2)
    plt.title(r'Horizonte Analítico: Norma do Gradiente $\partial\mathcal{L}_{t}/\partial h_{t-k}$')
    plt.xlabel('Quadros no Passado (k)')
    plt.ylabel('Norma L2 do Gradiente')
    plt.grid(True, linestyle='--')
    plt.show()

def relatorio_oclusoes_empiricas(track_history, max_age_tracker=10):
    """
    Recebe o dicionário 'track_history' (gerado pela sua função evaluate_trajectories)
    e analisa as lacunas de oclusão vs sobrevivência.
    """
    duracoes_oclusoes = []
    sobrevivencias = []
    falhas = []
    
    for tid, history in track_history.items():
        em_oclusao = False
        tamanho_oclusao_atual = 0
        
        for i in range(1, len(history)):
            # Entrou em oclusão (o detetor perdeu o objeto)
            if history[i-1] == 1 and history[i] == 0:
                em_oclusao = True
                tamanho_oclusao_atual = 1
                
            # Mantém-se em oclusão
            elif history[i-1] == 0 and history[i] == 0 and em_oclusao:
                tamanho_oclusao_atual += 1
                
            # Saiu da oclusão (o objeto reapareceu)
            elif history[i-1] == 0 and history[i] == 1 and em_oclusao:
                duracoes_oclusoes.append(tamanho_oclusao_atual)
                # Se ele reapareceu com o MESMO ID (history == 1), o tracker sobreviveu
                sobrevivencias.append(tamanho_oclusao_atual)
                em_oclusao = False
                tamanho_oclusao_atual = 0
                
        # Se a oclusão terminou mas o history nunca voltou a 1, a track morreu/trocou de ID
        if em_oclusao:
            duracoes_oclusoes.append(tamanho_oclusao_atual)
            falhas.append(tamanho_oclusao_atual)
            
    print(f"Total de oclusões no Dataset: {len(duracoes_oclusoes)}")
    print(f"Oclusão Média do Dataset: {np.mean(duracoes_oclusoes):.1f} quadros")
    print(f"Sobrevivência Máxima do Tracker: {np.max(sobrevivencias) if sobrevivencias else 0} quadros")
    print(f"Oclusão Média que causa Falha: {np.mean(falhas) if falhas else 0:.1f} quadros")

def exportar_tira_de_falha(frames_animados, frame_inicio, frame_fim, passo=2):
    """
    Extrai uma tira de imagens espaçadas para mostrar no relatório/apresentação.
    frames_animados: lista de imagens processadas pelo render_gru_predictions.
    """
    tira = []
    # Salta de 'passo' em 'passo' para não inundar o slide com frames repetitivos
    for f in range(frame_inicio, frame_fim + 1, passo):
        img_recortada = frames_animados[f]
        tira.append(img_recortada)
        
    # Concatena horizontalmente usando OpenCV
    tira_horizontal = cv2.hconcat(tira)
    
    plt.figure(figsize=(20, 5))
    plt.imshow(cv2.cvtColor(tira_horizontal, cv2.COLOR_BGR2RGB))
    plt.title(f"Galeria de Falhas: Oclusão Longa (Quadros {frame_inicio} a {frame_fim})")
    plt.axis('off')
    plt.show()

    