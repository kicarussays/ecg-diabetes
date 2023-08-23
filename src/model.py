from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader


class LeadNet(nn.Module):
    def __init__(self, conv_dim=8, fc_dim=256, lead=12, last_node=2, length=10):
        super().__init__()
        self.lead = lead
        self.length = length
        self.LeadLayers = nn.ModuleList(
            ResNet(conv_dim, fc_dim, lead=1, last_node=32, length=length)
            for _ in range(lead)
        )

        self.fclayer = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, last_node)
        )
    
    def forward(self, ecg, demo, which='both'):
        x = torch.cat([
            self.LeadLayers[n](ecg[:, n, :].contiguous().view(-1, 1, int(500*self.length)), demo, which) for n in range(self.lead)
        ], axis=1)

        x = self.fclayer(x)

        return x


class GradCam:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradient = None
        self.activations = None

        self.model.eval()
        self.register_hooks()

    def register_hooks(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradient = grad_output[0]

        def forward_hook(module, input, output):
            self.activations = output

        target_layer = self.target_layer
        for name, module in self.model.named_modules():
            if name == target_layer:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)

    def forward(self, input):
        return self.model(input)

    def backward(self, class_idx):
        self.model.zero_grad()
        one_hot = torch.zeros(self.gradient.size()).to(input.device)
        one_hot[0][class_idx] = 1
        self.gradient.backward(gradient=one_hot)

    def generate_heatmap(self):
        weights = torch.mean(self.gradient, dim=-1, keepdim=True)
        heatmap = torch.sum(weights * self.activations, dim=1, keepdim=True)
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap)
        return heatmap


class ResNet(nn.Module):
    def __init__(self, conv_dim=8, fc_dim=256, lead=8, last_node=2, length=10):
        super().__init__()
        self.reduce1 = nn.Sequential(
            nn.Conv1d(lead, lead*conv_dim, kernel_size=11, stride=2, padding=11//2),
            nn.BatchNorm1d(lead*conv_dim),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.5)
        )
        self.reduce2 = nn.Sequential(
            nn.Conv1d(lead*conv_dim, lead*conv_dim, kernel_size=7, stride=2, padding=11//2),
            nn.BatchNorm1d(lead*conv_dim),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.5)
        )
        self.reduce3 = nn.Sequential(
            nn.Conv1d(lead*conv_dim, lead*conv_dim, kernel_size=5, stride=2, padding=11//2),
            nn.BatchNorm1d(lead*conv_dim),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.5)
        )
        
        self.resblock1 = nn.Sequential(
            nn.Conv1d(lead*conv_dim, lead*conv_dim, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(lead*conv_dim),
            nn.ReLU(),
            nn.Conv1d(lead*conv_dim, lead*conv_dim, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(lead*conv_dim)
        )
        
        self.activation = nn.ReLU()

        self.reduce4 = nn.Sequential(
            nn.Conv1d(lead*conv_dim, lead*conv_dim*2, kernel_size=5, stride=1, padding=5//2),
            nn.BatchNorm1d(lead*conv_dim*2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.5)
        )
        
        self.resblock2 = nn.Sequential(
            nn.Conv1d(lead*conv_dim*2, lead*conv_dim*2, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(lead*conv_dim*2),
            nn.ReLU(),
            nn.Conv1d(lead*conv_dim*2, lead*conv_dim*2, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(lead*conv_dim*2)
        )
        
        self.reduce5 = nn.Sequential(
            nn.Conv1d(lead*conv_dim*2, lead*conv_dim*4, kernel_size=5, stride=1, padding=5//2),
            nn.BatchNorm1d(lead*conv_dim*4),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.5)
        )
        
        self.resblock3 = nn.Sequential(
            nn.Conv1d(lead*conv_dim*4, lead*conv_dim*4, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(lead*conv_dim*4),
            nn.ReLU(),
            nn.Conv1d(lead*conv_dim*4, lead*conv_dim*4, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(lead*conv_dim*4)
        )

        self.demo_layer = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        self.shared1 = nn.Linear(int(64*length*lead) + 64, fc_dim)
        self.shared1_1 = nn.Linear(int(64*length*lead), fc_dim)
        self.shared2 = nn.BatchNorm1d(fc_dim)
        self.fc2 = nn.Linear(fc_dim, fc_dim)
        self.bn2 = nn.BatchNorm1d(fc_dim)
        self.fc3 = nn.Linear(fc_dim, last_node)
        self.bothlayer = nn.Sequential(
            self.shared1, self.shared2, nn.ReLU(),
            self.fc2, self.bn2, nn.ReLU(),
            self.fc3)
        self.ecgonlylayer = nn.Sequential(
            self.shared1_1, self.shared2, nn.ReLU(),
            self.fc2, self.bn2, nn.ReLU(),
            self.fc3)
        
        self.lastdemo = nn.Linear(64, last_node)
        
        
    def forward(self, ecg, demo, which='both'):
        if which == 'demo':
            demo = self.demo_layer(demo)
            return self.lastdemo(demo)

        else:
            out = self.reduce1(ecg)
            out = self.reduce2(out)
            out = self.reduce3(out)
            
            out = self.activation(self.resblock1(out) + out)
            out = self.activation(self.resblock1(out) + out)
            
            out = self.reduce4(out)
            
            out = self.activation(self.resblock2(out) + out)
            out = self.activation(self.resblock2(out) + out)
            
            out = self.reduce5(out)
            
            out = self.activation(self.resblock3(out) + out)
            out = self.activation(self.resblock3(out) + out)
            
            out = out.view(out.size(0), -1)

            if which == 'ecg': 
                return self.ecgonlylayer(out)
            else:
                demo = self.demo_layer(demo)
                out = torch.cat((out, demo), dim=1)
                return self.bothlayer(out)


class ResNetLead8(nn.Module):
    def __init__(self, conv_dim=8, fc_dim=256, lead=8, last_node=2, demo=False):
        super().__init__()
        if demo:
            self.Lead8Layer = ResNet(conv_dim=conv_dim, fc_dim=fc_dim, lead=lead, last_node=2, length=2.5)
        else:
            self.Lead8Layer = ResNet(conv_dim=conv_dim, fc_dim=fc_dim, lead=lead, last_node=fc_dim, length=2.5)
            self.Lead1Layer = ResNet(conv_dim=conv_dim, fc_dim=fc_dim, lead=1, last_node=fc_dim, length=10)

            self.fclayer = nn.Sequential(
                nn.Linear(fc_dim * 2, fc_dim),
                nn.ReLU(),
                nn.Linear(fc_dim, fc_dim),
                nn.ReLU(),
                nn.Linear(fc_dim, last_node),
            )
    
    def forward(self, ecg8, ecg1, demo, which='both'):
        ecg8 = self.Lead8Layer(ecg8, demo, which)
        if which == 'demo': 
            return ecg8
        else:
            ecg1 = self.Lead1Layer(ecg1, demo, which)
            merge = torch.cat([ecg8, ecg1], axis=1)
            return self.fclayer(merge)


# class prevResNet(nn.Module):
#     def __init__(self, conv_dim=8, fc_dim=256, lead=8, last_node=2):
#         super().__init__()

#         self.res1 = ResNet(conv_dim=conv_dim, fc_dim=fc_dim, lead=8, last_node=256)
#         self.res2 = ResNet(conv_dim=conv_dim, fc_dim=fc_dim, lead=lead, last_node=256)
#         self.fclayer = nn.Sequential(
#             nn.Linear(256 * 2, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, last_node),
#         )
    
#     def forward(self, ecg8, ecg1, demo, prev_ecg8, prev_ecg1, prev_lab, which='both'):
#         prev = self.res1(prev_ecg8, prev_lab, which)
#         curr = self.res2(ecg8, demo, which)

#         merge = torch.cat([prev, curr], axis=1)
#         return self.fclayer(merge)


class prevResNet(nn.Module):
    def __init__(self, conv_dim=8, fc_dim=256, lead=8, last_node=2):
        super().__init__()

        self.res1 = ResNetLead8(conv_dim=conv_dim, fc_dim=fc_dim, lead=8, last_node=256)
        self.res2 = ResNetLead8(conv_dim=conv_dim, fc_dim=fc_dim, lead=lead, last_node=256)
        self.fclayer = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, last_node),
        )
    
    def forward(self, ecg8, ecg1, demo, prev_ecg8, prev_ecg1, prev_lab, which='both'):
        prev = self.res1(prev_ecg8, prev_ecg1, prev_lab, which)
        curr = self.res2(ecg8, ecg1, demo, which)

        merge = torch.cat([prev, curr], axis=1)
        return self.fclayer(merge)


class LabEncoder(nn.Module):
    def __init__(self, 
                 input_dim,
                 hid_dim, 
                 length,
                 n_layers, 
                 n_heads, 
                 pf_dim,
                 dropout, 
                 device,
                 max_length = 10):
        super().__init__()

        self.device = device
        
        self.tl_embedding = nn.Linear(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim,
                                                  dropout, 
                                                  device) 
                                     for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        self.fc1 = nn.Linear(length * hid_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim // 2)
        self.fc3 = nn.Linear(hid_dim // 2, 2)
        self.fclayer = nn.Sequential(self.fc1, self.fc2, self.fc3)
        
        
    def forward(self, ecgtl, labtl, demo, lastecg):
        
        #src = [batch size, src len]
        #src_mask = [batch size, 1, 1, src len]
        src = labtl
        
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        src = self.tl_embedding(src)
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        
        #pos = [batch size, src len]
        
        src = self.dropout((src * self.scale) + self.pos_embedding(pos))
        
        #src = [batch size, src len, hid dim]
        
        for layer in self.layers:
            src, attention = layer(src)
        
        src = self.fclayer(src.view(src.shape[0], -1))        
        #src = [batch size, src len, hid dim]
            
        return src, (attention)
    

class EncoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim,  
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        
        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, 1, 1, src len] 
                
        #self attention
        _src, attention = self.self_attention(src, src, src)
        
        #dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        #positionwise feedforward
        _src = self.positionwise_feedforward(src)
        
        #dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        return src, attention


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value):
        
        batch_size = query.shape[0]
        
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
                
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]
                
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]
                
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        #energy = [batch size, n heads, query len, key len]
        
        attention = torch.softmax(energy, dim = -1)
                
        #attention = [batch size, n heads, query len, key len]
                
        x = torch.matmul(self.dropout(attention), V)
        
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        #x = [batch size, query len, hid dim]
        
        x = self.fc_o(x)
        
        #x = [batch size, query len, hid dim]
        
        return x, attention
        
        
class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        #x = [batch size, seq len, hid dim]
        
        x = self.dropout(torch.relu(self.fc_1(x)))
        
        #x = [batch size, seq len, pf dim]
        
        x = self.fc_2(x)
        
        #x = [batch size, seq len, hid dim]
        
        return x



class LSTM(nn.Module):
    def __init__(self, hidden_size=256):
        super().__init__()
        self.lstm = nn.LSTM(input_size=12, hidden_size=hidden_size, num_layers=2)
        self.demo_layer = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        self.shared1 = nn.Linear(hidden_size + 64, hidden_size)
        self.shared2 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2)
        self.bothlayer = nn.Sequential(
            self.shared1, self.shared2, nn.ReLU(),
            self.fc2, self.bn2, nn.ReLU(),
            self.fc3)
    
    def forward(self, ecg, demo, which='both'):
        ecg = self.lstm(ecg.permute(0, 2, 1))[0][:, -1, :]
        demo = self.demo_layer(demo)
        out = torch.cat((ecg, demo), dim=1)

        return self.bothlayer(out)