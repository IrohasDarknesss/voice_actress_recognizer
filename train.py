import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from preprocessing import VoiceActorDataset, AudioTransform, DataAugmentation
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class VoiceActorClassifier(nn.Module):
    #ぷえぷえ体操いくぷえよ～！  （∩ ๑•﹏•๑｀)ぷえ〜っwwwぷえぷえぷえええ〜っwww (∩ ๑•﹏•๑｀⊃)ぷぇっ～///ぷぇっ～/// (⊂ ๑•﹏•๑｀