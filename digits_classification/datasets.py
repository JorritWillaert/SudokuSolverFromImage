import numpy as np
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, SubsetRandomSampler
import config

def create_validation_dataset(MNIST_train_dataset):
    number_training_examples = len(MNIST_train_dataset)
    indices = list(range(number_training_examples))
    split = int(config.VALIDATION_SIZE * number_training_examples)

    if config.SHUFFLE:
        np.random.seed(config.RANDOM_SEED)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(MNIST_train_dataset, batch_size=config.BATCH_SIZE,
                              sampler=train_sampler, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(MNIST_train_dataset, batch_size=config.BATCH_SIZE,
                            sampler=val_sampler, num_workers=config.NUM_WORKERS)
    
    return train_loader, val_loader

if __name__ == "__main__":
    train_dataset = datasets.MNIST(root='dataset/', train=True, transform=config.transform, download=True)
    train_loader, val_loader = create_validation_dataset(train_dataset)
    