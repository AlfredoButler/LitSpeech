#!/usr/bin/python

# !pip install torchaudio -f https://download.pytorch.org/whl/torch_stable.html
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torch.utils.data
import torch.optim
import warnings

from torchaudio.transforms import FrequencyMasking
warnings.filterwarnings('ignore')
import os
import sys
import time
import copy
import random
import pickle
from typing import List, Tuple, Dict, Any, Callable, Optional

#%%
from hidden_layers import CNNLayerNorm, ResidualCNN, BidirectionalGRU
from text_transform import TextTransform

# TODO: training_step()
# TODO: validation_step()
# TODO: test_step()

BATCH_SIZE = 100

class LitSpeech(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        """[summary]
        """
        super().__init__(*args, **kwargs)

        torch.manual_seed(42)

        # Hard-coded globals    
        self.DATA_DIR = os.path.join('data')
        self.BATCH_SIZE = BATCH_SIZE
        self.loss_fn = nn.CTCLoss(blank=28) # loss function

        # Hyperparameters
        self.lr = 8e-4
        self.epochs = 10    
        n_cnn_layers = 3
        n_rnn_layers = 5
        rnn_dim = 512
        n_class: int = len(TextTransform().char_map) # 29 
        n_feats =  128
        stride = 2
        dropout = 0.1 

        # Transforms
        self.train_audio_transforms = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
            torchaudio.transforms.TimeMasking(time_mask_param=35))
        self.valid_audio_transforms = torchaudio.transforms.MelSpectrogram()
        self.text_transform = TextTransform()

        # ----------------------------------
        # Architecture - LitSpeech
        # ----------------------------------
        n_feats = self.n_feats // 2
        self.hierarchial_cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3//2)
        self.residual_cnn = nn.Sequential(*[
            ResidualCNN(32, 32, kernel = 3, stride = 1, dropout = dropout, 
                          n_feats = n_feats)
            for cnn_layer in range(n_cnn_layers)])
        self.fc = nn.Linear(self.n_feats * 32, rnn_dim)
        self.BiRNN = nn.Sequential(
            *[BidirectionalGRU(
                rnn_dim = rnn_dim if i==0 else rnn_dim * 2, 
                hidden_size = rnn_dim, dropout = dropout, batch_first = (i==0))
            for i in range(n_rnn_layers)])
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim*2, rnn_dim),  # birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class))

    def forward(self, x):
        x = x.to(self.device)
        x = self.residual_cnn(x)
        sizes = x.size()
        batch_dim = sizes[0]
        feature_dim = sizes[1] * sizes[2]
        time_dim = sizes[3]
        x = x.view(batch_dim, feature_dim, time_dim)
        x = x.transpose(1, 2)
        x = self.fc(x)
        x = self.BiRNN(x)
        return self.classifier(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), self.lr)
        return optimizer


    # ----------------------------------
    # Training, validation, and test steps
    # ----------------------------------

    def training_step(self, batch, batch_idx):
        spectrograms, targets, input_lengths, target_lengths = batch
        spectrograms = spectrograms.to(self.device)
        targets = targets.to(self.device)
        # TODO

    def validation_step(self, batch, batch_idx, val: bool = True):
        # TODO
        pass 

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, val = False)

class LitSpeechDataModule(pl.LightningDataModule):
    """Data preparation hooks for LitSpeech

    Attributes:
        batch_size (int): 
    Methods: 
        prepare_data:
        setup(stage: Optional[str] = None):

    """
    def __init__(self):
        self.batch_size = BATCH_SIZE 

    def prepare_data(self) -> None:
        self.train_dataset = torchaudio.datasets.LIBRISPEECH(
            root = self.DATA_DIR, 
            url = "train-clean-100", 
            download = True)
        self.test_dataset = torchaudio.datasets.LIBRISPEECH(
            root = self.DATA_DIR, 
            url = "test-clean", 
            download = True)

    def setup(self, stage: Optional[str] = None):
        if stage in ["fit", None]:
            pass
            # learning rate scheduler
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer = self.configure_optimizers(), max_lr = self.lr, 
                steps_per_epoch = int(len(self.train_dl)), epochs = self.epochs, 
                anneal_strategy = 'linear')

        if stage in ["test", None]:
            pass # TODO

    def get_dataloader(self, stage: str) -> torch.utils.data.DataLoader:
        if stage == "train":
            dataset = self.train_dataset
        elif stage == "val":
             dataset = self.test_dataset
        elif stage == "test":
            dataset = self.test_dataset
        else:
            raise ValueError("`stage` must be in ['train', 'val', 'test'].")
        return torch.utils.data.DataLoader(
            dataset = dataset, batch_size = self.BATCH_SIZE, 
            collate_fn = lambda x: self.data_)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self.get_dataloader("train")
    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return self.get_dataloader("val")
    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return self.get_dataloader("test")


# %%
