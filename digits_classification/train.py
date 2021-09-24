import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import config
from model import MNISTNet
from datasets import create_validation_dataset


def main():
    full_train_dataset = datasets.MNIST(root='dataset/', train=True, transform=config.transform, download=True)
    train_dataloader, val_dataloader = create_validation_dataset(full_train_dataset)

    test_dataset = datasets.MNIST(root='dataset/', train=False, transform=config.transform, download=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS,
                                 pin_memory=config.PIN_MEMORY, shuffle=config.SHUFFLE)

    loss_function = nn.CrossEntropyLoss()

    model = MNISTNet()
    model=model.to(config.DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler()

    print("Training on:", config.DEVICE)
    for epoch in range(config.NUM_EPOCHS):
        print("Epoch:", epoch)
        train_losses = []
        loop = tqdm(train_dataloader)
        model.train()
        train_loss = 0
        for data, targets in loop:
            data = data.to(device=config.DEVICE)
            targets = targets.to(device=config.DEVICE)

            output = model(data)
            train_loss_batch = loss_function(output, targets)

            optimizer.zero_grad()
            train_loss_batch.backward()
            optimizer.step()
            train_loss += train_loss_batch
            train_losses.append(train_loss_batch.item())
        train_loss /= len(train_dataloader)
        print("Train loss:", train_loss.item())

        loop = tqdm(val_dataloader)
        val_losses = []
        model.eval()
        val_loss = 0
        acc = 0
        with torch.no_grad():
            for data, targets in loop:
                data = data.to(device=config.DEVICE)
                targets = targets.to(device=config.DEVICE)

                output = model(data)
                
                val_loss_batch = loss_function(output, targets)
                val_loss += val_loss_batch
                val_losses.append(val_loss_batch.item())
        acc /= len(val_dataloader)
        val_loss /= len(val_dataloader)
        print("Validation loss:", val_loss.item())
        print("Accuracy:", acc)
    plt.figure(0)
    plt.title("Train losses")
    plt.plot(train_losses)
    plt.show()
    plt.figure(1)
    plt.title("Validation losses")
    plt.plot(val_losses)
    plt.show()

if __name__ == "__main__":
    main()