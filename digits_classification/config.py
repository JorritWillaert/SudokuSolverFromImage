import torch
import torchvision.transforms as transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
NUM_EPOCHS = 100
NUM_WORKERS = 4
SHUFFLE = True
PIN_MEMORY = True
VALIDATION_SIZE = 0.2
RANDOM_SEED = 1 

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307), (0.3081)) 
                                # Obtained with calculate_std_and_mean.py
                               ]) 
