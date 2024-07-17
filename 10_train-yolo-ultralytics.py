"""Aqui entreno unn coco from scratch"""
import os
import shutil
import random
from PIL import Image
import torch
import json
import yaml
from ultralytics import YOLO

def cuda_is_available():
    if torch.cuda.is_available():
        message = "CUDA está disponible. Usando CUDA."
    else:
        message = "CUDA no está disponible. Usando CPU."
    print(message)
    
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def train(results_path, yaml_path):
    model = YOLO("yolov8n.yaml")
    results = model.train(data=yaml_path,
                        epochs=1000,
                        patience=100,
                        dropout= 0.2,
                        single_cls=False,
                        val=False,
                        device=cuda_is_available(),
                        verbose=False,
                        project=results_path,
                        name='run/train',
                        exist_ok=True
                        )

    


if __name__ == "__main__":
    train(results_path= os.path.join(os.getcwd(), 'datasets/COCO2017/'),
          yaml_path=os.path.join(os.getcwd(), 'datasets/COCO2017/config.yaml'))

    print(os.path.join(os.getcwd(), 'datasets/COCO2017/'))
    print(os.path.join(os.getcwd(), 'datasets/COCO2017/config.yaml'))