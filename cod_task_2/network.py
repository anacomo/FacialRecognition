# from cod_task_1.params import *
import random
import os
import numpy as np
import torch
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm
from util_task2 import *

from params_task2 import *

names = ['bart', 'homer', 'lisa', 'marge']
class Network:
    def __init__(self):
        # * training
        train_paths = []
        for sdir, dirs, files in os.walk(TRAIN_DS):
            for file in files:
                train_paths.append(sdir + '/'+ file)
        # * training data and loader
        train_data = LabeledDataset(train_paths, transform = TRANSFORM_IMG)
        self.train_load = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        
        # * validation
        val_paths = []
        for sdir, dirs, files in os.walk(VAL_DS):
            for file in files:
                val_paths.append(sdir + '/' + file)
        val_data = LabeledDataset(val_paths, transform = TRANSFORM_IMG)
        self.val_load = torch.utils.data.DataLoader(val_data, batch_size = BATCH_SIZE, shuffle = True)

        self.device = 'cuda'

    
    def load_network(self):
        self.model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=6).cuda()
        self.loss_function = torch.nn.CrossEntropyLoss().cuda()
        self.optimmizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)

    # * mode, optimizer, loss_function, train_load, val_load, epochs, device
    def train_network(self, epochs = 20):
        history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': []}
        
        for epoch in range(1, epochs + 1):
            train_acc, train_loss = self.compute_epoch(self.train_load, training=True)
            val_acc, val_loss = self.compute_epoch(self.val_load, training=False)
            
            history['loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            print('Epoch %3d/%3d, train_loss: %5.6f | train_acc: %5.6f | val_loss: %5.6f | val_acc: %5.6f' % \
                (epoch, epochs, train_loss, train_acc, val_loss, val_acc))

    
    def compute_epoch(self, dataload, training = False):
        if training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_correct = 0
        examples = 0

        for x, y in tqdm(dataload):
            if training:
                self.optimmizer.zero_grad()

            x = x.to(self.device)
            y = y.to(self.device)

            pred = self.model.forward(x)
            loss = self.loss_function(pred, y)

            if training:
                loss.backward()
                self.optimmizer.step()

            total_loss += loss.data.item() * x.size(0)
            total_correct  += (torch.max(pred, 1)[1] == y).sum().item()
            examples += x.shape[0]
        
        accuracy = total_correct / examples
        calc_loss = total_loss / len(dataload.dataset)

        return accuracy, calc_loss

    def save_model(self):
        torch.save(self.model, MODEL_PATH)