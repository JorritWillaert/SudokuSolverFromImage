import torch
from torch import nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3)), 
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(36864, 1000),
            nn.Linear(1000, 10)
        )

    def forward(self, x):
        return self.network(x)

if __name__ == "__main__":
    train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=64)
    iterdata = iter(train_dataloader)
    images, labels = next(iterdata)

    net = MNISTNet()
    print(net.forward(images).shape) # Output shape = [N, 10]