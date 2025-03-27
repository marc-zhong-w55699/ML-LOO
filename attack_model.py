import torch
import torch.nn as nn
import torch.optim as optim
import torchattacks
from torchattacks import CW
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class Attack:
    def __init__(self, model):
        self.model = model
    
    def attack(self, x, *args):
        raise NotImplementedError

class CWAttack(Attack):
    def __init__(self, model, source_samples=2, binary_search_steps=5, cw_learning_rate=5e-3,
                 confidence=0, attack_iterations=1000, attack_initial_const=1e-2):
        super().__init__(model)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        
        self.cw = CW(model, c=attack_initial_const, kappa=confidence, steps=attack_iterations, lr=cw_learning_rate)
        
    def attack(self, x, y=None):
        x = x.to(self.device)
        if y is not None:
            y = y.to(self.device)
        adv = self.cw(x, y)
        return adv

# Example usage
if __name__ == "__main__":
    # Load a pretrained model (for example, a simple CNN trained on MNIST)
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
            self.fc1 = nn.Linear(32 * 28 * 28, 10)
            
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            return x

    model = SimpleModel()
    model.load_state_dict(torch.load("pretrained_model.pth", map_location=torch.device("cpu")))
    
    # Load dataset
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    attack = CWAttack(model)
    
    # Attack first sample
    for images, labels in test_loader:
        adv_images = attack.attack(images, labels)
        break