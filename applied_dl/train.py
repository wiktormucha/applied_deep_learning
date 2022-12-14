'''
File to train models
'''
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import pandas as pd
import sys


sys.path.append("../")
from config import *
from utils.utils import count_parameters, IoULoss
from datasets.FreiHAND import FreiHAND
from models.models import CustomHeatmapsModel
from utils.trainer import Trainer

def main():
    '''
    Main function of training pytohn script
    '''
    
    config = {
    "data_dir": DATA_DIR,
    "model_path": CHECKPOINT_DIR,
    "epochs": MAX_EPOCHS,
    "batch_size": BACTH_SIZE,
    "learning_rate": LEARNING_RATE,
    "device": DEVICE,
    "early_stopping": EARLY_STOPPING
    }

    train_dataset = FreiHAND(config=config, set_type="train", img_transform = TRAIN_IMG_TRANSFORM, heatmap_transform = TRAIN_HEATMAP_TRANSFORM)
    
    train_dataloader = DataLoader(
        train_dataset,
        config["batch_size"],
        shuffle=True,
        drop_last=False,
        num_workers=6,
        pin_memory=True
    )

    val_dataset = FreiHAND(config=config, set_type="val", img_transform= VAL_IMG_TRANSFORM, heatmap_transform= None)
    
    val_dataloader = DataLoader(
        val_dataset,
        config["batch_size"],
        shuffle=True,
        drop_last=False,
        num_workers=6,
        pin_memory=True
    )

    model = CustomHeatmapsModel(N_IMG_CHANNELS, N_KEYPOINTS)
    model = model.to(config["device"])

    # If loading weights from checkpoint
    if CONTINUE_FROM_CHECKPOINT:
        model.load_state_dict(torch.load(config["model_path"], map_location=torch.device(config["device"])))
        print("Model's checkpoint loaded")

    print('Number of parameters to learn:', count_parameters(model))

    criterion = IoULoss()
    #  criterion = nn.MSELoss() # Use this loss in case of regression head
    optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"], weight_decay=1e-5, momentum= MOMENTUM)
   
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, factor=0.5, patience=7, verbose=True, threshold=0.00001
    )

    trainer = Trainer(model, criterion, optimizer, config, scheduler)
    print(f'Starting training on device: {config["device"]}')
    model = trainer.train(train_dataloader, val_dataloader)

    plt.plot(trainer.loss["train"], label="train")
    plt.plot(trainer.loss["val"], label="val")
    plt.legend()
    plt.show()

    plt.savefig(f'{EXPERIMENT_NAME}.png')

    df = pd.DataFrame()
    df['train_los'] = trainer.loss["train"]
    df['val_los'] = trainer.loss["val"]

    df.to_csv(f'{EXPERIMENT_NAME}.csv')


if __name__ == '__main__':
    main()