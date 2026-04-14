import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import re
from torchvision import transforms

class FingerprintDataset(Dataset):
    def __init__(self, root_dir, functions, transform=None):
        self.samples = []
        self.transform = transform
        
        self.label_map = {
            'PSO': 0,
            'RandomSearch': 1,
            'DE': 2,
            'CMAES': 3,
            'SimulatedAnnealing': 4
        }

        for algo in os.listdir(root_dir):
            algo_path = os.path.join(root_dir, algo)
            if not os.path.isdir(algo_path):
                continue

            for file in os.listdir(algo_path):
                match = re.search(r'_f(\d+)_', file)
                if match:
                    func_id = int(match.group(1))
                    
                    if func_id in functions:
                        self.samples.append((
                            os.path.join(algo_path, file),
                            self.label_map[algo]
                        ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('L')  # grayscale
        
        if self.transform:
            image = self.transform(image)

        return image, label
    
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

train_dataset = FingerprintDataset('dataset', functions=[1,2,3], transform=transform)
test_dataset  = FingerprintDataset('dataset', functions=[4,5], transform=transform)