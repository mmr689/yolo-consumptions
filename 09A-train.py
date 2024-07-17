from ultralytics import YOLO
import torch

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)


def cuda_is_available():
    if torch.cuda.is_available():
        message = "CUDA está disponible. Usando CUDA."
    else:
        message = "CUDA no está disponible. Usando CPU."
    print(message)
    
    return 'cuda' if torch.cuda.is_available() else 'cpu'

results = model.train(data="coco.yaml",
                        epochs=1000,
                        patience=100,
                        dropout= 0.2,
                        single_cls=False,
                        val=False,
                        device=cuda_is_available(),
                        verbose=False,
                        project='results/ultralytics/coco2017',
                        name='run/train',
                        exist_ok=True
                        )