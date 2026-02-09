# -*- coding: utf-8 -*-
"""
Main Training Script for FG-MD Framework on VAIS Dataset.
Includes Triplet Dataset definition, Mutual Distillation (MMMMM) loss, 
and the end-to-end training loop.
"""

from __future__ import division, print_function
import os
import sys
import argparse
import numpy as np
import random
import datetime
import time
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

# Project-specific imports
from network_base import AlexNetFc
from testvais import test_MAP, eval_val_dataset, VAISTestDataset
from meters import AverageMeter, loss_store_init, print_loss, remark_loss, reset_loss
from Loss import AFD_semantic, AFD_spatial

# ==============================================================================
# 1. Dataset & Transformation Utilities
# ==============================================================================

def get_transforms(is_train=True):
    """Returns standard ImageNet transforms."""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if is_train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

class VAISTripletDataset(torch.utils.data.Dataset):
    """VAIS Dataset for Triplet-based Cross-Modal Training.
    
    Attributes:
        rgb_list, ir_list: Paths to paired images.
        y_list: One-hot encoded labels.
        topk_*: Matrices storing indices for dynamic triplet mining.
    """
    def __init__(self, txt_path, root_dir, is_train=True, num_classes=6):
        self.root_dir = root_dir
        self.is_train = is_train
        self.transform = get_transforms(is_train)
        
        with open(txt_path, 'r') as f:
            lines = [line.strip().split() for line in f if line.strip()]
        
        self.rgb_list = [os.path.join(root_dir, x[0]) for x in lines]
        self.ir_list = [os.path.join(root_dir, x[1]) for x in lines]
        self.x_list = self.rgb_list # Dummy for length compatibility
        
        labels = [int(x[2]) for x in lines]
        self.y_list = np.eye(num_classes)[labels]
        
        # Initialize Top-K indices for similarity mining
        self.topk = 20
        init_topk = np.zeros((len(lines), self.topk), dtype=int)
        self.update_topk(init_topk, init_topk, init_topk, init_topk, self.topk)

    def update_topk(self, A2B_pos, A2B_neg, B2A_pos, B2A_neg, topk=20):
        """Updates the mining pools based on Hamming distances."""
        self.topk_A2B_pos = A2B_pos
        self.topk_A2B_neg = A2B_neg
        self.topk_B2A_pos = B2A_pos
        self.topk_B2A_neg = B2A_neg
        self.topk = topk

    def __len__(self):
        return len(self.rgb_list)

    def _load_img(self, path):
        img = Image.open(path).convert('RGB')
        return self.transform(img)

    def __getitem__(self, idx):
        k_idx = random.randint(0, self.topk - 1)
        
        # Anchor RGB (A) and paired IR (B)
        img_A = self._load_img(self.rgb_list[idx])
        img_B = self._load_img(self.ir_list[idx])
        
        # Mining Positive/Negative for RGB2IR and IR2RGB
        idx_B2A_pos = self.topk_B2A_pos[idx, k_idx]
        idx_B2A_neg = self.topk_B2A_neg[idx, k_idx]
        idx_A2B_pos = self.topk_A2B_pos[idx, k_idx]
        idx_A2B_neg = self.topk_A2B_neg[idx, k_idx]
        
        img_B2A_pos = self._load_img(self.rgb_list[idx_B2A_pos])
        img_B2A_neg = self._load_img(self.rgb_list[idx_B2A_neg])
        img_A2B_pos = self._load_img(self.ir_list[idx_A2B_pos])
        img_A2B_neg = self._load_img(self.ir_list[idx_A2B_neg])

        return img_A, img_B2A_pos, img_B2A_neg, img_B, img_A2B_pos, img_A2B_neg, self.y_list[idx], torch.tensor(idx)

# ==============================================================================
# 2. Model & Loss Definitions
# ==============================================================================

class FGMD_DistillationLoss(nn.Module):
    """Dual-Attention Mutual Distillation Loss (MMMMM)."""
    def __init__(self, att_f=0.0625, w_sem=(1.0, 1.0, 1.0), w_spa=(1.0, 1.0, 1.0)):
        super().__init__()
        # Channel configurations for MobileNetV2-based backbones
        channels = [64, 192, 384, 256, 256]
        
        # Semantic Distillation Modules (Higher layers)
        self.sem5 = AFD_semantic(channels[4], att_f)
        self.sem4 = AFD_semantic(channels[3], att_f)
        self.sem3 = AFD_semantic(channels[2], att_f)
        
        # Spatial Distillation Modules (Lower layers)
        self.spa3 = AFD_spatial(channels[2])
        self.spa2 = AFD_spatial(channels[1])
        self.spa1 = AFD_spatial(channels[0])
        
        self.register_buffer('w_sem', torch.tensor(w_sem, dtype=torch.float32))
        self.register_buffer('w_spa', torch.tensor(w_spa, dtype=torch.float32))

    def forward(self, out_list_A, out_list_B):
        """Calculates symmetric semantic and spatial attention losses."""
        A1, A2, A3, A4, A5 = out_list_A
        B1, B2, B3, B4, B5 = out_list_B

        # Symmetric Semantic Attention Loss
        loss_sem = self.w_sem[0] * (self.sem5(A5, B5.detach()) + self.sem5(B5, A5.detach())) + \
                   self.w_sem[1] * (self.sem4(A4, B4.detach()) + self.sem4(B4, A4.detach())) + \
                   self.w_sem[2] * (self.sem3(A3, B3.detach()) + self.sem3(B3, A3.detach()))

        # Symmetric Spatial Attention Loss
        loss_spa = self.w_spa[0] * (self.spa3(A3, B3.detach()) + self.spa3(B3, A3.detach())) + \
                   self.w_spa[1] * (self.spa2(A2, B2.detach()) + self.spa2(B2, A2.detach())) + \
                   self.w_spa[2] * (self.spa1(A1, B1.detach()) + self.spa1(B1, A1.detach()))

        return loss_sem + loss_spa

# ==============================================================================
# 3. Core Training Logic
# ==============================================================================

def train_hash(model_A, model_B, mm_supervision, train_dataset, db_dataset_A, db_dataset_B, 
               val_dataset_A, val_dataset_B, srcname_list, args):
    """Main training loop for cross-modal hashing."""
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    
    # Orthogonal initialization for the projection layer
    project_layer = nn.Linear(args.hash_bit, args.class_cnt, bias=False).to(args.device)
    with torch.no_grad():
        nn.init.orthogonal_(project_layer.weight)

    optimizer = torch.optim.Adam([
        {'params': model_A.parameters()}, {'params': model_B.parameters()},
        {'params': project_layer.parameters()}, {'params': mm_supervision.parameters()}
    ], lr=args.lr)
    
    scheduler = StepLR(optimizer, step_size=10, gamma=0.9)
    pdist = nn.PairwiseDistance(2)
    criterion = nn.CrossEntropyLoss().to(args.device)
    
    loss_metrics = ["cls_A", "cls_B", "hash", "tri_A2B", "tri_B2A", "tri_total", "MM", "total"]
    loss_store = loss_store_init(loss_metrics)
    
    total_samples = len(train_dataset)
    all_A_codes = torch.randn(total_samples, args.hash_bit).to(args.device)
    all_B_codes = torch.randn(total_samples, args.hash_bit).to(args.device)
    all_labels = np.array(train_dataset.y_list, dtype=int)
    
    best_MAP = 0.0

    for epoch in range(args.max_epoch):
        # Update Triplet Mining Pools
        with torch.no_grad():
            A2B_pos, A2B_neg, B2A_pos, B2A_neg = calcTopKPostiveNegative(
                torch.sign(all_A_codes), torch.sign(all_B_codes), all_labels, topk=20)
            train_dataset.update_topk(A2B_pos, A2B_neg, B2A_pos, B2A_neg, topk=20)
        
        model_A.train(); model_B.train(); project_layer.train()

        for batch_idx, (A_img, B2A_pos, B2A_neg, B_img, A2B_pos, A2B_neg, labels_onehot, indices) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass
            anc_A, list_A = model_A(A_img.to(args.device))
            anc_B, list_B = model_B(B_img.to(args.device))
            
            # Extracting triplet components
            pos_A2B, _ = model_B(A2B_pos.to(args.device))
            neg_A2B, _ = model_B(A2B_neg.to(args.device))
            pos_B2A, _ = model_A(B2A_pos.to(args.device))
            neg_B2A, _ = model_A(B2A_neg.to(args.device))

            # Losses
            loss_mm = mm_supervision(list_A, list_B)
            y_A, y_B = project_layer(anc_A), project_layer(anc_B)
            
            target = torch.argmax(labels_onehot, dim=1).to(args.device)
            loss_cls = criterion(y_A, target) + criterion(y_B, target)
            loss_hash = torch.mean((torch.abs(anc_A)-1)**2) + torch.mean((torch.abs(anc_B)-1)**2)
            
            # Margin Triplet Loss
            margin = 0.25
            d_A2B_pos = pdist(F.normalize(anc_A), F.normalize(pos_A2B))
            d_A2B_neg = pdist(F.normalize(anc_A), F.normalize(neg_A2B))
            tri_A2B = F.relu(d_A2B_pos - d_A2B_neg + margin).mean()

            d_B2A_pos = pdist(F.normalize(anc_B), F.normalize(pos_B2A))
            d_B2A_neg = pdist(F.normalize(anc_B), F.normalize(neg_B2A))
            tri_B2A = F.relu(d_B2A_pos - d_B2A_neg + margin).mean()
            
            loss_total = 0.1 * loss_cls + loss_hash + (tri_A2B + tri_B2A) + 0.1 * loss_mm
            
            loss_total.backward()
            optimizer.step()

            # Store hash codes for mining
            all_A_codes[indices] = anc_A.detach()
            all_B_codes[indices] = anc_B.detach()

        scheduler.step()
        best_MAP = eval_val_dataset(epoch, model_A, model_B, db_dataset_A, db_dataset_B, 
                                   val_dataset_A, val_dataset_B, srcname_list, args, best_MAP)

# ==============================================================================
# 4. Entry Point
# ==============================================================================

def main():
    # Setup Random Seeds
    seed = 13
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    parser = argparse.ArgumentParser(description='FG-MD Training on VAIS')
    parser.add_argument('--hash_bit', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--data_name', type=str, default='VAIS')
    parser.add_argument('--class_cnt', type=int, default=6)
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data Path Management
    save_dir = f'data/{args.data_name}/model/ours/'
    os.makedirs(save_dir, exist_ok=True)
    args.model_path = os.path.join(save_
