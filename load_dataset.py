import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np
from collections import defaultdict

class MOT17Dataset(Dataset):
    def __init__(
        self, 
        root: str, 
        model_base: str, 
        sequences_list: list, # NOVA VARIÁVEL AQUI
        transform=None, 
        train_dataset: bool=True,
        min_confidence: float = 0.0
    ):
        dataset = "train" if train_dataset else "test"
        self.root = Path(root) / dataset
        self.transform = transform
        self.train_dataset = train_dataset
        self.min_confidence = min_confidence
        self.model_base = model_base
        
        # AGORA: só busca imagens dos vídeos que você definiu em 'sequences_list'
        self.images = []
        for seq_number in sequences_list:
            seq_name = f"MOT17-{seq_number:02d}-{self.model_base}"
            seq_path = self.root / seq_name
            
            # Ordenação segura do vídeo
            frames = sorted(seq_path.glob("img1/*.jpg"), key=lambda p: int(p.stem))
            self.images.extend(frames)
            
        assert len(self.images) > 0, f"Erro: Nenhuma imagem encontrada para '{model_base}' em {self.root}."

        # Dicionários em memória: map[seq_name][frame_id] = [bboxes]
        self.gts = defaultdict(lambda: defaultdict(list))
        self.dets = defaultdict(lambda: defaultdict(list))
        
        # Carrega as anotações num formato que não estoura a RAM
        self._load_annotations()

    def _load_annotations(self):
        """Lê det.txt e gt.txt mantendo um formato padrão de 9 colunas para ambos."""
        for seq_path in self.root.glob(f"*{self.model_base}*"):
            if not seq_path.is_dir(): continue
            seq_name = seq_path.name
            
            # 1. Carregar Detecções do Baseline (det.txt)
            det_path = seq_path / "det" / "det.txt"
            if det_path.exists():
                with open(det_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split(',')
                        if len(parts) < 7: continue
                        
                        conf_score = float(parts[6])
                        if conf_score < self.min_confidence:
                            continue
                            
                        frame_id = int(parts[0])
                        
                        # Padronizando para 9 colunas:
                        # frame, id(-1), bb_left, bb_top, bb_width, bb_height, conf, class(1), visibility(1)
                        bbox = [
                            float(parts[0]), # frame
                            float(parts[1]), # id (No det.txt é sempre -1)
                            float(parts[2]), # bb_left
                            float(parts[3]), # bb_top
                            float(parts[4]), # bb_width
                            float(parts[5]), # bb_height
                            float(parts[6]), # conf
                            1.0,             # class (Forçamos 1 = Pedestre)
                            1.0              # visibility (Forçamos 1.0 = 100% visível)
                        ]
                        self.dets[seq_name][frame_id].append(bbox)

            # 2. Carregar Ground Truth (gt.txt)
            if self.train_dataset:
                gt_path = seq_path / "gt" / "gt.txt"
                if gt_path.exists():
                    with open(gt_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split(',')
                            if len(parts) < 9: continue
                            
                            obj_class = int(parts[7])
                            
                            # Opcional: Se quiser carregar TUDO (até carros), comente as duas linhas abaixo
                            if obj_class != 1: 
                                continue 
                            
                            frame_id = int(parts[0])
                            
                            # Padronizando para 9 colunas: leitura direta do formato oficial
                            bbox = [
                                float(parts[0]), # frame
                                float(parts[1]), # id (ID real do pedestre)
                                float(parts[2]), # bb_left
                                float(parts[3]), # bb_top
                                float(parts[4]), # bb_width
                                float(parts[5]), # bb_height
                                float(parts[6]), # conf / flag
                                float(parts[7]), # class
                                float(parts[8])  # visibility
                            ]
                            self.gts[seq_name][frame_id].append(bbox)

    def __len__(self):
        # O tamanho agora é o número total de IMAGENS, lidas uma por vez.
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Descobrir qual imagem estamos processando
        img_path = self.images[idx]
        seq_name = img_path.parent.parent.name
        frame_id = int(img_path.stem)
        
        # 2. Carregar a imagem (como array NumPy para o Albumentations)
        image = np.array(Image.open(img_path).convert("RGB"))
        
        # 3. Puxar as anotações do dicionário
        frame_dets = self.dets[seq_name][frame_id]
        frame_gts = self.gts[seq_name][frame_id] if self.train_dataset else []

        # 4. Converter para Tensores (Mesmo se estiver vazio, retorna tensor de tamanho zero)
        tensor_dets = torch.tensor(frame_dets, dtype=torch.float32) if len(frame_dets) > 0 else torch.zeros((0, 5))
        tensor_gts = torch.tensor(frame_gts, dtype=torch.float32) if len(frame_gts) > 0 else torch.zeros((0, 4))

        # 5. Aplica as transformações (Sintaxe focada em Albumentations)
        if self.transform is not None:
            # Albumentations exige o argumento nomeado 'image=...'
            augmented = self.transform(image=image)
            image = augmented['image']
            
        # 6. Converte NumPy para Tensor do PyTorch, se já não for
        if isinstance(image, np.ndarray):
            # De (H, W, C) para (C, H, W)
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # RETORNO MÁGICO: Imagem, detecções, gabarito e os METADADOS essenciais para agrupar o vídeo
        return image, tensor_dets, tensor_gts, seq_name, frame_id