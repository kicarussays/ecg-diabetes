import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import neurokit2 as nk
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix, r2_score
from src.config import normal_ranges



class Supervised:
    def __init__(self, model, train_loader, val_loader, criterion, args, savepath, logger):            
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.savepath = savepath
        self.criterion = criterion
        self.logger = logger
    
    def train(self):
        os.makedirs(self.savepath, exist_ok=True)
        if self.args.device == 'cpu':
            self.d = 'cpu'
        else:
            self.d = f'cuda:{self.args.device}'
        
        bestauc = 0.0
        patience = 0
        self.model = self.model.to(self.d)
        self.criterion = self.criterion.to(self.d)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        
        
        first_iter = 0
        if os.path.isfile(self.savepath + f'best_supervised_model_{self.args.seed}.tar'):
            print('file exists')
            checkpoint = torch.load(self.savepath + f'best_supervised_model_{self.args.seed}.tar', map_location=self.d)
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])

            first_iter = checkpoint["epoch"] + 1
        
        for epoch in range(first_iter, self.args.max_epoch):
            self.model.train()
            loss_sum = 0
            skipcnt = 0
            
            for lead8, demo, flag in tqdm(self.train_loader):
                if self.args.lead == 1:
                    lead8 = lead8[:, 1, :].view(-1, 1, 5000)
                else:
                    lead8 = lead8[:, :self.args.lead, :]
                # lead1 = lead1.view(-1, 1, 5000)
                    
                lead8 = lead8.type(torch.float32).to(self.d)
                demo = demo.type(torch.float32).to(self.d)
                target = flag.type(torch.LongTensor).to(self.d)
                output = self.model(lead8, demo, self.args.which)

                loss = self.criterion(output, target.view(-1))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_sum += loss.item()

                
            if (epoch + 1) % self.args.valtime == 0:
                val_loss, roc_auc, pr_auc = self.evaluation(self.model, self.val_loader)
                self.logger.info('Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}, AUROC: {:.4f}, AUPRC: {:.4f}'.format(
                            epoch+1, self.args.max_epoch, loss_sum / (len(self.train_loader) - skipcnt), val_loss, roc_auc, pr_auc))
                
                if roc_auc > bestauc + 0.001:
                    self.logger.info(f'Saved Best Model...')
                    saving_dict = {
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "epoch": epoch,
                    }
                    torch.save(saving_dict, self.savepath + f'best_supervised_model_{self.args.seed}.tar')
                    bestauc = roc_auc
                    patience = 0
                else:
                    patience += 1

            if self.args.es and patience > self.args.patience:
                self.logger.info(f'Early Stopping Activated')
                break
        
        load_model = torch.load(self.savepath + f'best_supervised_model_{self.args.seed}.tar', map_location=self.d)
        
        return load_model, bestauc
        
    
    def evaluation(self, model, dl):
        softmax = nn.Softmax()
        with torch.no_grad():
            model.eval()
            vloss, flags = [], []
            
            loss_sum = 0
            skipcnt = 0
            
            for lead8, demo, flag in tqdm(dl):
                if self.args.lead == 1:
                    lead8 = lead8[:, 1, :].view(-1, 1, 5000)
                else:
                    lead8 = lead8[:, :self.args.lead, :]
                    
                lead8 = lead8.type(torch.float32).to(self.d)
                demo = demo.type(torch.float32).to(self.d)
                target = flag.type(torch.LongTensor).to(self.d)
                output = self.model(lead8, demo, self.args.which)

                loss = self.criterion(output, target.view(-1))
                loss_sum += loss.item()
                vloss.append(softmax(output)[:, 1].view(-1).cpu().detach().numpy())
                flags.append(target.view(-1).cpu().detach().numpy().astype(int))
            
            vloss = np.concatenate(vloss)
            flags = np.concatenate(flags)

            fpr, tpr, _ = roc_curve(flags, vloss)
            precision, recall, _ = precision_recall_curve(flags, vloss)
            roc_auc = auc(fpr, tpr)
            pr_auc =  auc(recall, precision)

            return loss_sum / (len(dl) - skipcnt), roc_auc, pr_auc  


