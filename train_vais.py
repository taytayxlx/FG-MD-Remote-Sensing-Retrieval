# -*- coding: utf-8 -*-
"""
FG-MD Training Script for VAIS Dataset.
Optimization Objective: L_total = L_cls + L_hash + L_tri + lambda_dist * L_dist
"""

from __future__ import division, print_function
import os
import argparse
import numpy as np
import random
import time
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

# Custom modules (Ensure these .py files are in the same directory)
from network_base import AlexNetFc
from test_vais import test_MAP, eval_val_dataset, VAISTestDataset
from meters import loss_store_init, print_loss, remark_loss, reset_loss
from Loss import AFD_semantic, AFD_spatial

# ==============================================================================
# 1. Dataset & Transformation
# ==============================================================================

def get_vais_transforms(is_train=True):
    """Standard ImageNet preprocessing for VAIS images."""
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
    """Dataset for Triplet-based Cross-Modal Training with dynamic mining."""
    def __init__(self, txt_path, root_dir, is_train=True):
        self.root_dir = root_dir
        self.is_train = is_train
        self.transform = get_vais_transforms(is_train)
        
        with open(txt_path, 'r') as f:
            lines = [line.strip().split() for line in f if line.strip()]
        
        self.rgb_list = [os.path.join(root_dir, x[0]) for x in lines]
        self.ir_list = [os.path.join(root_dir, x[1]) for x in lines]
        self.y_list = np.eye(6)[[int(x[2]) for x in lines]] # VAIS: 6 classes
        
        self.topk = 20
        init_topk = np.zeros((len(lines), self.topk), dtype=int)
        self.update_topk(init_topk, init_topk, init_topk, init_topk, self.topk)

    def update_topk(self, A2B_pos, A2B_neg, B2A_pos, B2A_neg, topk=20):
        self.topk_A2B_pos, self.topk_A2B_neg = A2B_pos, A2B_neg
        self.topk_B2A_pos, self.topk_B2A_neg = B2A_pos, B2A_neg
        self.topk = topk

    def __len__(self):
        return len(self.rgb_list)

    def _load(self, path):
        return self.transform(Image.open(path).convert('RGB'))

    def __getitem__(self, idx):
        k_idx = random.randint(0, self.topk - 1)
        img_A = self._load(self.rgb_list[idx])
        img_B = self._load(self.ir_list[idx])
        
        # Cross-modal mining components
        img_B2A_pos = self._load(self.rgb_list[self.topk_B2A_pos[idx, k_idx]])
        img_B2A_neg = self._load(self.rgb_list[self.topk_B2A_neg[idx, k_idx]])
        img_A2B_pos = self._load(self.ir_list[self.topk_A2B_pos[idx, k_idx]])
        img_A2B_neg = self._load(self.ir_list[self.topk_A2B_neg[idx, k_idx]])

        return img_A, img_B2A_pos, img_B2A_neg, img_B, img_A2B_pos, img_A2B_neg, self.y_list[idx], torch.tensor(idx)

# ==============================================================================
# 2. Mutual Distillation Loss (L_dist)
# ==============================================================================

class MutualDistillationLoss(nn.Module):
    """Implementation of L_dist: enforces spatial-semantic consistency."""
    def __init__(self, att_f=0.0625, w_sem=(1.0, 1.0, 1.0), w_spa=(1.0, 1.0, 1.0)):
        super().__init__()
        channels = [64, 192, 384, 256, 256]
        self.sem5 = AFD_semantic(channels[4], att_f)
        self.sem4 = AFD_semantic(channels[3], att_f)
        self.sem3 = AFD_semantic(channels[2], att_f)
        self.spa3 = AFD_spatial(channels[2])
        self.spa2 = AFD_spatial(channels[1])
        self.spa1 = AFD_spatial(channels[0])
        self.register_buffer('w_sem', torch.tensor(w_sem))
        self.register_buffer('w_spa', torch.tensor(w_spa))

    def forward(self, list_A, list_B):
        # Semantic Loss (Higher Layers)
        l_sem = self.w_sem[0]*(self.sem5(list_A[4], list_B[4].detach()) + self.sem5(list_B[4], list_A[4].detach())) + \
                self.w_sem[1]*(self.sem4(list_A[3], list_B[3].detach()) + self.sem4(list_B[3], list_A[3].detach())) + \
                self.w_sem[2]*(self.sem3(list_A[2], list_B[2].detach()) + self.sem3(list_B[2], list_A[2].detach()))
        # Spatial Loss (Lower Layers)
        l_spa = self.w_spa[0]*(self.spa3(list_A[2], list_B[2].detach()) + self.spa3(list_B[2], list_A[2].detach())) + \
                self.w_spa[1]*(self.spa2(list_A[1], list_B[1].detach()) + self.spa2(list_B[1], list_A[1].detach())) + \
                self.w_spa[2]*(self.spa1(list_A[0], list_B[0].detach()) + self.spa1(list_B[0], list_A[0].detach()))
        return l_sem + l_spa

# ==============================================================================
# 3. Training Logic
# ==============================================================================

def train_hash(model_A, model_B, distillation_module, train_dataset, db_dataset_A, db_dataset_B, 
               val_dataset_A, val_dataset_B, srcname_list, args):
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    project_layer = nn.Linear(args.hash_bit, args.class_cnt, bias=False).to(args.device)
    nn.init.orthogonal_(project_layer.weight)

    optimizer = torch.optim.Adam([
        {'params': model_A.parameters()}, {'params': model_B.parameters()},
        {'params': project_layer.parameters()}, {'params': distillation_module.parameters()}
    ], lr=args.lr)
    
    scheduler = StepLR(optimizer, step_size=10, gamma=0.9)
    pdist = nn.PairwiseDistance(2)
    criterion_cls = nn.CrossEntropyLoss().to(args.device)
    
    # Hyper-parameter for L_dist (Parameter Sensitivity Analysis)
    lambda_dist = getattr(args, 'lambda_dist', 0.1)

    # Logging labels synchronized with the paper
    loss_metrics = ["cls_A", "cls_B", "hash", "tri_total", "dist", "total"]
    loss_store = loss_store_init(loss_metrics)
    
    all_A_codes = torch.randn(len(train_dataset), args.hash_bit).to(args.device)
    all_B_codes = torch.randn(len(train_dataset), args.hash_bit).to(args.device)
    all_labels = np.array(train_dataset.y_list, dtype=int)
    best_MAP = 0.0

    for epoch in range(args.max_epoch):
        with torch.no_grad():
            from testvais import calcTopKPostiveNegative
            A2B_pos, A2B_neg, B2A_pos, B2A_neg = calcTopKPostiveNegative(
                torch.sign(all_A_codes), torch.sign(all_B_codes), all_labels, topk=20)
            train_dataset.update_topk(A2B_pos, A2B_neg, B2A_pos, B2A_neg, topk=20)
        
        model_A.train(); model_B.train(); project_layer.train()

        for batch_idx, (A_img, B2A_pos, B2A_neg, B_img, A2B_pos, A2B_neg, label_onehot, index) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward
            anc_A, list_A = model_A(A_img.to(args.device))
            anc_B, list_B = model_B(B_img.to(args.device))
            pos_A2B, _ = model_B(A2B_pos.to(args.device))
            neg_A2B, _ = model_B(A2B_neg.to(args.device))
            pos_B2A, _ = model_A(B2A_pos.to(args.device))
            neg_B2A, _ = model_A(B2A_neg.to(args.device))

            # L_dist: Mutual Distillation Loss
            loss_dist = distillation_module(list_A, list_B)
            
            # Classification and Quantization Loss (Coefficient = 1.0)
            target = torch.argmax(label_onehot, dim=1).to(args.device)
            l_cls_A = criterion_cls(project_layer(anc_A), target)
            l_cls_B = criterion_cls(project_layer(anc_B), target)
            loss_hash = torch.mean((torch.abs(anc_A)-1)**2) + torch.mean((torch.abs(anc_B)-1)**2)
            
            # Triplet Loss L_tri (Coefficient = 1.0)
            margin = 0.25
            tri_A2B = F.relu(pdist(F.normalize(anc_A), F.normalize(pos_A2B)) - pdist(F.normalize(anc_A), F.normalize(neg_A2B)) + margin).mean()
            tri_B2A = F.relu(pdist(F.normalize(anc_B), F.normalize(pos_B2A)) - pdist(F.normalize(anc_B), F.normalize(neg_B2A)) + margin).mean()
            tri_total = tri_A2B + tri_B2A

            # Total Objective: L_total = (L_cls_A + L_cls_B) + L_hash + L_tri + lambda_dist * L_dist
            loss_total = (l_cls_A + l_cls_B) + loss_hash + tri_total + lambda_dist * loss_dist
            
            loss_total.backward()
            optimizer.step()

            # Mining cache updates
            all_A_codes[index] = anc_A.detach()
            all_B_codes[index] = anc_B.detach()

            remark_loss(loss_store, l_cls_A, l_cls_B, loss_hash, tri_total, loss_dist, loss_total)

        scheduler.step()
        best_MAP = eval_val_dataset(epoch, model_A, model_B, db_dataset_A, db_dataset_B, 
                                   val_dataset_A, val_dataset_B, srcname_list, args, best_MAP)

# ==============================================================================
# 4. Entry
# ==============================================================================

def main():
    seed = 13
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    
    parser = argparse.ArgumentParser(description='FG-MD Training Routine')
    parser.add_argument('--hash_bit', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--data_name', type=str, default='VAIS')
    parser.add_argument('--lambda_dist', type=float, default=0.1, help='Weight for distillation loss (lambda_dist)')
    parser.add_argument('--class_cnt', type=int, default=6)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Path Setup
    save_path = f'data/{args.data_name}/model/ours/'
    os.makedirs(save_path, exist_ok=True)
    args.model_path = os.path.join(save_path, f'ours_{args.data_name}_{args.hash_bit}')

    # Load Data
    train_ds = VAISTripletDataset('data/VAIS/train_pairs.txt', 'data/VAIS/', is_train=True)
    val_ds_rgb = VAISTestDataset('data/VAIS/test_rgb.txt', 'data/VAIS/', 'RGB', get_vais_transforms(False))
    val_ds_ir = VAISTestDataset('data/VAIS/test_ir.txt', 'data/VAIS/', 'IR', get_vais_transforms(False))

    # Init
    model_rgb = AlexNetFc(None, args).to(args.device)
    model_ir = AlexNetFc(None, args).to(args.device)
    distill_module = MutualDistillationLoss().to(args.device)

    train_hash(model_rgb, model_ir, distill_module, train_ds, val_ds_rgb, val_ds_ir, 
               val_ds_rgb, val_ds_ir, ['RGB', 'IR'], args)

if __name__ == '__main__':
    main()
