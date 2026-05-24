import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, fbeta_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
import pandas as pd
import math
import torch.nn.functional as F

from model import LogTransformer 
from data_loader import get_hdfs_data 


BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 0.0002  
DROPOUT_RATE = 0.2      
WEIGHT_DECAY = 0.01     
GRAD_CLIP = 1.0
OUTPUT_DIR = 'Resultats_BGL' 

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss) 
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss

class EarlyStopping:
    def __init__(self, patience=6, min_delta=0.001): 
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def collate_fn(batch):
    sequences, labels = zip(*batch)
    max_len = max([len(seq) for seq in sequences])
    
    padded_seqs = torch.zeros((len(sequences), max_len), dtype=torch.long)
    
    for i, seq in enumerate(sequences):
        end = len(seq)
        if isinstance(seq, torch.Tensor):
            padded_seqs[i, :end] = seq
        elif isinstance(seq, np.ndarray):
            padded_seqs[i, :end] = torch.from_numpy(seq)
        else:
            padded_seqs[i, :end] = torch.tensor(seq, dtype=torch.long)
            
    return padded_seqs, torch.tensor(labels, dtype=torch.float32)

def evaluate_model(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels in loader:
            sequences, labels = sequences.to(device), labels.to(device)
            
            logits = model(sequences)
            loss = criterion(logits, labels.unsqueeze(1)) 
            total_loss += loss.item()
            
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0
    all_labels = np.array(all_labels).flatten()
    all_probs = np.array(all_probs).flatten()
    
    preds = (all_probs > threshold).astype(int)
    accuracy = 100 * (preds == all_labels).sum() / len(all_labels)
    
    return avg_loss, accuracy, all_labels, preds, all_probs

def save_plots(train_losses, val_losses, val_accuracies):
    try:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss', color='blue')
        plt.plot(val_losses, label='Val Loss', color='orange')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(val_accuracies, label='Val Accuracy', color='green')
        plt.title('Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'training_metrics.png'))
        plt.close()
    except Exception as e:
        print(f"⚠️ Error al guardar gráficas: {e}")

def save_pr_curve(labels, probs):
    precision, recall, _ = precision_recall_curve(labels, probs)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, label=f'AUC = {pr_auc:.3f}')
    plt.title('Test Set Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'test_pr_curve.png'))
    plt.close()

def save_confusion_matrix_plot(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Normal', 'Anomaly'], 
                yticklabels=['Normal', 'Anomaly'],
                annot_kws={"size": 14})
    
    plt.xlabel('Predicción del Modelo', fontsize=12)
    plt.ylabel('Etiqueta Real', fontsize=12)
    plt.title('Matriz de Confusión (Test Set)', fontsize=14)
    
    save_path = os.path.join(OUTPUT_DIR, 'test_confusion_matrix.png')
    plt.savefig(save_path)
    plt.close()

def main():
    print(f"Iniciant Entrenament: BGL Log Anomaly (Train 70% / Val 15% / Test 15%)")
    print(f"Output Dir: {OUTPUT_DIR}/")

    csv_path = 'data/processed/BGL.log_structured.csv'
    label_path = 'data/raw/anomaly_label.csv'
    
    full_dataset = get_hdfs_data(csv_path, label_path)
    targets = np.array(full_dataset.targets)
    
    vocab_size = 0
    for seq in full_dataset.data:
        if len(seq) > 0:
            m = np.max(seq)
            if m > vocab_size: vocab_size = m
    vocab_size = int(vocab_size + 2)
    print(f"Vocab Size: {vocab_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # SPLIT DE DATOS
    indices = np.arange(len(targets))
    
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        indices, targets, test_size=0.30, stratify=targets, random_state=42
    )
    
    val_idx, test_idx, y_val, y_test = train_test_split(
        temp_idx, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )
    
    print(f"Distribució de dades:")
    print(f"   - Train: {len(train_idx)} secuencias")
    print(f"   - Val:   {len(val_idx)} secuencias")
    print(f"   - Test:  {len(test_idx)} secuencias")

    train_sub = Subset(full_dataset, train_idx)
    val_sub = Subset(full_dataset, val_idx)
    test_sub = Subset(full_dataset, test_idx)
    
    count_normal = np.sum(y_train == 0)
    count_anomaly = np.sum(y_train == 1)
    weight_normal = 1.0 / count_normal
    weight_anomaly = 1.0 / count_anomaly
    samples_weight = np.array([weight_anomaly if t == 1 else weight_normal for t in y_train])
    samples_weight = torch.from_numpy(samples_weight).double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    
    train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, sampler=sampler, shuffle=False, collate_fn=collate_fn)
    val_loader = DataLoader(val_sub, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_sub, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = LogTransformer(
        vocab_size=vocab_size, 
        embedding_dim=128,   
        num_heads=4, 
        num_layers=3,       
        dropout=DROPOUT_RATE
    ).to(device)
    
    criterion = FocalLoss(alpha=1.0, gamma=2.0)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    early_stopper = EarlyStopping(patience=6, min_delta=0.001)

    best_val_loss = float('inf')
    best_model_path = os.path.join(OUTPUT_DIR, 'best_model.pth')
    history_train_loss, history_val_loss, history_val_acc = [], [], []

    print("\n" + "="*40)
    print("Començant entrenament (Focal loss + Capacitat augmentada)")
    print("="*40)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for sequences, batch_labels in train_loader:
            sequences, batch_labels = sequences.to(device), batch_labels.to(device)
                        
            optimizer.zero_grad()
            logits = model(sequences)
            
            loss = criterion(logits, batch_labels.unsqueeze(1)) 
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        val_loss, val_acc, _, _, _ = evaluate_model(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        history_train_loss.append(avg_train_loss)
        history_val_loss.append(val_loss)
        history_val_acc.append(val_acc)
        
        print(f"  Epoch [{epoch+1}/{EPOCHS}] | LR: {current_lr:.6f} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%", end="")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print("(Nou millor model guardat)")
        else:
            print("")
            
        early_stopper(val_loss)
        if early_stopper.early_stop:
            print(f"Early Stopping activat en la època {epoch+1}.")
            break

    save_plots(history_train_loss, history_val_loss, history_val_acc)
    
    print("\n" + "*"*60)
    print("Evaluació final (Trobant millor umbral en val i aplicant-lo al test set)")
    print("*"*60)
    
    model.load_state_dict(torch.load(best_model_path))
    
    print("Trobant millor umbral en el Val Set...")
    _, _, val_labels, _, val_probs = evaluate_model(model, val_loader, criterion, device)
    
    best_f2 = 0
    best_thresh = 0.5
    
    for th in np.arange(0.1, 0.99, 0.01):
        preds_temp = (val_probs > th).astype(int)
        current_f2 = fbeta_score(val_labels, preds_temp, beta=2, pos_label=1, zero_division=0)
        if current_f2 > best_f2:
            best_f2 = current_f2
            best_thresh = th
            
    print(f"Millor umbral trobat en Val: {best_thresh:.2f} (F2-Score: {best_f2:.4f})")
    
    print(f"\nAplicant umbral de {best_thresh:.2f} al TEST SET")
    test_loss, test_acc, test_labels, _, test_probs = evaluate_model(model, test_loader, criterion, device, threshold=best_thresh)
    save_pr_curve(test_labels, test_probs)
    
    final_preds = (test_probs > best_thresh).astype(int)
    
    print("\n CLASSIFICACIÓ (TEST SET)")
    print("-" * 50)
    print(classification_report(test_labels, final_preds, target_names=['Normal', 'Anomaly'], zero_division=0))
    print("Matriu de confusió:")
    print(confusion_matrix(test_labels, final_preds))
    save_confusion_matrix_plot(test_labels, final_preds)
    print(f"Process complet. Gràfics i model guardats en {OUTPUT_DIR}/")

if __name__ == '__main__':
    main()