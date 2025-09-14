import os
import gc
import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.utils.class_weight import compute_class_weight

import timm

# =============================================================================
# CONFIGURATION FOR 4-CHANNEL SWIN TRANSFORMER
# =============================================================================
class CFG:
    # --- MODEL & IMAGE SIZE ---
    MODEL_NAME = 'swinv2_tiny_window8_256'
    IMG_SIZE = 256
    BATCH_SIZE = 16

    # --- DATA PATHS ---
    BASE_PATH = "/kaggle/input/aptos2019"
    TRAIN_CSV = os.path.join(BASE_PATH, "train_1.csv")
    VAL_CSV   = os.path.join(BASE_PATH, "valid.csv")
    TRAIN_DIR = os.path.join(BASE_PATH, "train_images", "train_images")
    VAL_DIR   = os.path.join(BASE_PATH, "val_images", "val_images")
    
    # --- PATHS TO YOUR NEW SEGMENTED MASKS ---
    SEG_BASE_PATH = "/kaggle/input/segmentaion-dataset/"
    SEG_TRAIN_DIR = os.path.join(SEG_BASE_PATH, "segmented_outputs_train_1/segmented_outputs_train_1/")
    SEG_VAL_DIR   = os.path.join(SEG_BASE_PATH, "segmented_outputs_val/segmented_outputs_val/")

    # --- TRAINING PIPELINE (Identical to your successful run for fair comparison) ---
    S1_EPOCHS = 15; S1_LR = 5e-5; S1_WEIGHT_DECAY = 0.05; S1_USE_MIXUP = True
    S2_EPOCHS = 15; S2_LR = 1e-5; S2_WEIGHT_DECAY = 0.05; S2_USE_MIXUP = False
    
    # --- GENERAL & SAVING ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 2
    PATIENCE = 5
    SEED = 42
    LABEL_SMOOTHING = 0.05
    # New save paths for this experiment
    SAVE_PATH_S1 = "best_model_swin_seg_stage1.pth"
    SAVE_PATH_FINAL = "best_model_swin_seg_final.pth"

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = True
seed_everything(CFG.SEED)

# =============================================================================
# PREPROCESSING & AUGMENTATIONS
# =============================================================================
def preprocess_ben_graham(image, output_size):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if gray.mean() < 15: image = cv2.resize(image, (output_size, output_size), interpolation=cv2.INTER_AREA)
        else:
            _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea); x, y, w, h = cv2.boundingRect(largest_contour)
                image = image[y:y+h, x:x+w]
            image = cv2.resize(image, (output_size, output_size), interpolation=cv2.INTER_AREA)
    except Exception: image = cv2.resize(image, (output_size, output_size), interpolation=cv2.INTER_AREA)
    b, g, r = cv2.split(image); clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)); g = clahe.apply(g)
    return cv2.merge((b, g, r))

def get_transforms(is_train=True):
    if is_train:
        return A.Compose([
            A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        ])
    else:
        return None

# =============================================================================
# UPGRADED DATASET TO LOAD 4 CHANNELS
# =============================================================================
class Dataset4Channel(Dataset):
    def __init__(self, df, img_dir, seg_dir, transform=None):
        self.df, self.img_dir, self.seg_dir, self.transform = df.reset_index(drop=True), img_dir, seg_dir, transform
        self.post_transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.5]),
            ToTensorV2()
        ])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['id_code'] + '.png')
        seg_path = os.path.join(self.seg_dir, row['id_code'] + '.png')
        
        img = cv2.imread(img_path); img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
        
        img = preprocess_ben_graham(img, CFG.IMG_SIZE)
        mask = cv2.resize(mask, (CFG.IMG_SIZE, CFG.IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']
            
        img_4_channel = np.dstack((img, mask))
        img_4_channel = self.post_transform(image=img_4_channel)['image']
            
        label = torch.tensor(row['diagnosis'], dtype=torch.long)
        return img_4_channel, label

# =============================================================================
# UPGRADED SWIN MODEL TO ACCEPT 4 CHANNELS
# =============================================================================
class Swin4ChannelOrdinal(nn.Module):
    def __init__(self, model_name, num_classes=5, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='avg')
        
        # The first layer in a Swin Transformer is the patch embedding projection
        original_proj = self.backbone.patch_embed.proj
        original_weights = original_proj.weight.clone()

        # Create a new projection layer that accepts 4 input channels
        new_proj = nn.Conv2d(4, original_proj.out_channels, 
                             kernel_size=original_proj.kernel_size, stride=original_proj.stride, 
                             padding=original_proj.padding, bias=(original_proj.bias is not None))
        
        with torch.no_grad():
            new_proj.weight[:, :3] = original_weights
            new_proj.weight[:, 3] = original_weights.mean(dim=1)
        
        # Replace the original 3-channel layer with our new 4-channel layer
        self.backbone.patch_embed.proj = new_proj
        
        feature_dim = self.backbone.num_features
        self.classifier = nn.Sequential(nn.Linear(feature_dim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes - 1))

    def forward(self, x): return self.classifier(self.backbone(x))

# --- Loss functions, training loops, and other utilities are unchanged ---
class WeightedOrdinalFocalLoss(nn.Module):
    def __init__(self, num_classes=5, gamma=2.0, class_weights=None, label_smoothing=0.0):
        super().__init__(); self.num_classes, self.gamma, self.class_weights, self.label_smoothing = num_classes, gamma, class_weights, label_smoothing; self.bce = nn.BCEWithLogitsLoss(reduction='none')
    def forward(self, outputs, targets):
        ordinal_targets = torch.zeros_like(outputs)
        for i, t in enumerate(targets):
            if t > 0: ordinal_targets[i, :t] = 1.0
        if self.label_smoothing > 0.0: ordinal_targets = ordinal_targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        bce = self.bce(outputs, ordinal_targets)
        if self.class_weights is not None:
            weights = self.class_weights[targets].view(-1, 1).expand(-1, outputs.shape[1]); bce = bce * weights
        pt = torch.exp(-bce); focal = (1 - pt) ** self.gamma * bce
        return focal.mean()

class SmoothKappaLoss(nn.Module):
    def __init__(self, num_classes=5, eps=1e-7):
        super().__init__(); self.num_classes, self.eps = num_classes, eps; W = torch.zeros(num_classes, num_classes)
        for i in range(num_classes):
            for j in range(num_classes): W[i,j] = ((i - j)**2) / ((num_classes - 1)**2)
        self.register_buffer("W", W)
    def forward(self, outputs, targets):
        device = outputs.device; B = outputs.size(0); probs = torch.sigmoid(outputs); class_probs = torch.zeros(B, self.num_classes, device=device)
        class_probs[:, 0] = 1 - probs[:, 0]
        for k in range(1, self.num_classes-1): class_probs[:, k] = probs[:, k-1] - probs[:, k]
        class_probs[:, -1] = probs[:, -1]
        class_probs = torch.clamp(class_probs, min=self.eps, max=1.0); one_hot = F.one_hot(targets, num_classes=self.num_classes).float().to(device); conf_mat = torch.matmul(one_hot.T, class_probs)
        hist_true = one_hot.sum(dim=0); hist_pred = class_probs.sum(dim=0); expected = torch.outer(hist_true, hist_pred)
        W = self.W.to(device); obs = torch.sum(W * conf_mat); exp = torch.sum(W * expected); kappa = 1.0 - (B * obs) / (exp + self.eps)
        return 1.0 - kappa

def mixup_data(x, y, alpha=0.4):
    if alpha > 0: lam = np.random.beta(alpha, alpha)
    else: lam = 1
    batch_size = x.size()[0]; index = torch.randperm(batch_size).to(x.device); mixed_x = lam * x + (1 - lam) * x[index, :]; y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def ordinal_to_class(outputs): return torch.sum(torch.sigmoid(outputs) > 0.5, dim=1).long()
def calculate_metrics(outputs, targets):
    preds = ordinal_to_class(outputs).cpu().numpy(); targets_np = targets.cpu().numpy()
    return accuracy_score(targets_np, preds), cohen_kappa_score(targets_np, preds, weights='quadratic')
def clear_memory(): gc.collect(); torch.cuda.empty_cache()

def train_epoch(model, loader, optimizer, criterion, scaler, device, use_mixup):
    model.train(); running_loss = 0.0; all_out, all_t = [], []
    pbar = tqdm(loader, desc="Training", leave=False)
    for images, targets in pbar:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_mixup: images, targets_a, targets_b, lam = mixup_data(images, targets)
        with torch.cuda.amp.autocast():
            outputs = model(images)
            if use_mixup: loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            else: loss = criterion(outputs, targets)
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        running_loss += loss.item(); all_out.append(outputs.detach()); all_t.append(targets.detach()); pbar.set_postfix(loss=loss.item())
    all_out, all_t = torch.cat(all_out), torch.cat(all_t)
    return running_loss / len(loader), *calculate_metrics(all_out, all_t)

def validate_epoch(model, loader, criterion, device):
    model.eval(); running_loss = 0.0; all_out, all_t = [], []
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validating", leave=False)
        for images, targets in pbar:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(): outputs = model(images); loss = criterion(outputs, targets)
            running_loss += loss.item(); all_out.append(outputs); all_t.append(targets)
    all_out, all_t = torch.cat(all_out), torch.cat(all_t)
    return running_loss / len(loader), *calculate_metrics(all_out, all_t)

def main():
    print(f"Device: {CFG.DEVICE}, Model: {CFG.MODEL_NAME} (4-Channel), Image Size: {CFG.IMG_SIZE}")
    train_df = pd.read_csv(CFG.TRAIN_CSV); val_df = pd.read_csv(CFG.VAL_CSV)
    
    train_tf = get_transforms(is_train=True)
    val_tf = get_transforms(is_train=False)

    train_ds = Dataset4Channel(train_df, CFG.TRAIN_DIR, CFG.SEG_TRAIN_DIR, transform=train_tf)
    val_ds   = Dataset4Channel(val_df, CFG.VAL_DIR, CFG.SEG_VAL_DIR, transform=val_tf)

    class_weights_sampler = compute_class_weight('balanced', classes=np.unique(train_df['diagnosis']), y=train_df['diagnosis'])
    sample_weights = np.array([class_weights_sampler[int(l)] for l in train_df['diagnosis']])
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, sampler=sampler, num_workers=CFG.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CFG.BATCH_SIZE*2, shuffle=False, num_workers=CFG.NUM_WORKERS, pin_memory=True)
    
    model = Swin4ChannelOrdinal(CFG.MODEL_NAME).to(CFG.DEVICE)
    class_weights_loss = torch.tensor(class_weights_sampler, dtype=torch.float).to(CFG.DEVICE)
    focal_loss = WeightedOrdinalFocalLoss(num_classes=5, gamma=2.0, class_weights=class_weights_loss, label_smoothing=CFG.LABEL_SMOOTHING)
    kappa_loss = SmoothKappaLoss(num_classes=5)
    
    def hybrid_loss(outputs, targets): return 0.7 * kappa_loss(outputs, targets) + 0.3 * focal_loss(outputs, targets)
    
    scaler = torch.cuda.amp.GradScaler()

    # --- STAGE 1 ---
    print("\n" + "="*50 + "\n     STARTING STAGE 1 (4-Channel)\n" + "="*50)
    opt = optim.AdamW(model.parameters(), lr=CFG.S1_LR, weight_decay=CFG.S1_WEIGHT_DECAY)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG.S1_EPOCHS)
    best_val_qwk, patience_counter = -1, 0

    for epoch in range(CFG.S1_EPOCHS):
        clear_memory(); print(f"\nEpoch {epoch+1}/{CFG.S1_EPOCHS}")
        train_loss, train_acc, train_qwk = train_epoch(model, train_loader, opt, focal_loss, scaler, CFG.DEVICE, CFG.S1_USE_MIXUP)
        val_loss, val_acc, val_qwk = validate_epoch(model, val_loader, focal_loss, CFG.DEVICE)
        sched.step()
        print(f"Train -> Loss:{train_loss:.4f} Acc:{train_acc:.4f} QWK:{train_qwk:.4f}")
        print(f"Valid -> Loss:{val_loss:.4f} Acc:{val_acc:.4f} QWK:{val_qwk:.4f}")
        if val_qwk > best_val_qwk:
            print(f"Val QWK improved from {best_val_qwk:.4f} to {val_qwk:.4f}. Saving model...")
            best_val_qwk, patience_counter = val_qwk, 0
            torch.save(model.state_dict(), CFG.SAVE_PATH_S1)
        else:
            patience_counter += 1
            if patience_counter >= CFG.PATIENCE: print("Early stopping in Stage 1."); break
    
    # --- STAGE 2 ---
    print("\n" + "="*50 + "\n     STARTING STAGE 2 (4-Channel)\n" + "="*50)
    if os.path.exists(CFG.SAVE_PATH_S1):
        model.load_state_dict(torch.load(CFG.SAVE_PATH_S1))
    else:
        print("No Stage 1 model was saved. Continuing with the current model.")

    opt = optim.AdamW(model.parameters(), lr=CFG.S2_LR, weight_decay=CFG.S2_WEIGHT_DECAY)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG.S2_EPOCHS)
    best_val_qwk_stage2, patience_counter = best_val_qwk, 0

    for epoch in range(CFG.S2_EPOCHS):
        clear_memory(); print(f"\nEpoch {epoch+1}/{CFG.S2_EPOCHS}")
        train_loss, train_acc, train_qwk = train_epoch(model, train_loader, opt, hybrid_loss, scaler, CFG.DEVICE, CFG.S2_USE_MIXUP)
        val_loss, val_acc, val_qwk = validate_epoch(model, val_loader, hybrid_loss, CFG.DEVICE)
        sched.step()
        print(f"Train -> Loss:{train_loss:.4f} Acc:{train_acc:.4f} QWK:{train_qwk:.4f}")
        print(f"Valid -> Loss:{val_loss:.4f} Acc:{val_acc:.4f} QWK:{val_qwk:.4f}")
        if val_qwk > best_val_qwk_stage2:
            print(f"Val QWK improved from {best_val_qwk_stage2:.4f} to {val_qwk:.4f}. Saving final model...")
            best_val_qwk_stage2, patience_counter = val_qwk, 0
            torch.save(model.state_dict(), CFG.SAVE_PATH_FINAL)
        else:
            patience_counter += 1
            if patience_counter >= CFG.PATIENCE: print("Early stopping in Stage 2."); break

    print(f"\nTraining Finished!\nFinal Best QWK: {best_val_qwk_stage2:.4f}")

if __name__ == "__main__":
    main()
