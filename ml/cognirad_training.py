"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  CogniRad — Task 1.8 & 1.9 : Production Training Pipeline (Google Colab)  ║
║  Dataset : RadioML 3-Class Remapped (radioml_remapped.hdf5)                ║
║  Model   : SpectrumClassifier (ResNet1D + Attention)                       ║
║  Target  : >93% validation accuracy in 30 epochs                          ║
║                                                                            ║
║  MEMORY STRATEGY v3: Preprocess HDF5 → memory-mapped .npy on local SSD.   ║
║  Enables num_workers=2 for parallel data loading.                          ║
║  GPU utilization: ~5 GB VRAM. Total time: ~6-8 hours for 30 epochs.        ║
╚══════════════════════════════════════════════════════════════════════════════╝

HOW TO USE ON GOOGLE COLAB
──────────────────────────
1. Upload this file + the layer_1/ folder to Google Drive under:
       MyDrive/CogniRad/
   So the structure looks like:
       MyDrive/CogniRad/cognirad_training.py
       MyDrive/CogniRad/layer_1/residual_block.py
       MyDrive/CogniRad/layer_1/resnet1d.py
       MyDrive/CogniRad/layer_1/spectrum_classifier.py
       MyDrive/dataset/radioml_remapped.hdf5

2. Open a new Colab notebook with GPU runtime
3. Run these cells:

   Cell 1 (keep-alive):
       %%javascript
       function KeepAlive() {
           console.log("Keeping alive " + new Date());
           document.querySelector("colab-connect-button").click();
       }
       setInterval(KeepAlive, 60000);

   Cell 2 (runs everything — Drive mount is automatic):
       %run /content/drive/MyDrive/CogniRad/cognirad_training.py

   NOTE: If Cell 2 fails because Drive isn't mounted yet, add this cell BEFORE it:
       from google.colab import drive
       drive.mount('/content/drive')
"""

import os
import sys
import time
import gc
import json
import shutil
from datetime import datetime

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — Single source of truth
# ═══════════════════════════════════════════════════════════════════════════════
CFG = {
    # ── Paths ────────────────────────────────────────────────────────────
    'dataset_path'    : '/content/drive/MyDrive/dataset/radioml_remapped.hdf5',
    'local_X_path'    : '/content/radioml_X.npy',     # Preprocessed on local SSD
    'local_y_path'    : '/content/radioml_y.npy',     # Preprocessed on local SSD
    'project_dir'     : '/content/drive/MyDrive/CogniRad',
    'output_dir'      : '/content/drive/MyDrive/CogniRad/training_output',
    'model_save_path' : '/content/drive/MyDrive/CogniRad/training_output/spectrum_classifier.pt',
    'best_model_path' : '/content/drive/MyDrive/CogniRad/training_output/spectrum_classifier_best.pt',
    'resume_path'     : '/content/drive/MyDrive/CogniRad/training_output/checkpoint_latest.pt',
    'local_ckpt_path' : '/content/checkpoint_latest.pt',   # local SSD backup (network-safe)

    # ── Model Architecture (must match Task 1.7) ────────────────────────
    'base_filters'    : 48,
    'num_heads'       : 8,
    'attn_dropout'    : 0.1,
    'fc_dropout'      : 0.4,
    'num_classes'     : 3,
    'in_channels'     : 2,
    'seq_len'         : 1024,

    # ── Data Split ──────────────────────────────────────────────────────
    'train_ratio'     : 0.70,
    'val_ratio'       : 0.15,
    'test_ratio'      : 0.15,

    # ── Training Hyperparameters ────────────────────────────────────────
    'batch_size'      : 512,        # better gradient noise for generalization (~2.1 GB VRAM)
    'learning_rate'   : 0.001,
    'weight_decay'    : 1e-4,
    'epochs'          : 30,         # full production run
    'scheduler_factor': 0.5,        # LR reduction factor
    'scheduler_patience': 3,        # epochs without improvement before LR drop
    'num_workers'     : 2,          # matches Colab's 2 CPU cores (avoids worker contention)

    # ── Early Stopping ──────────────────────────────────────────────────
    'early_stop_patience' : 7,       # stop if val_loss doesn't improve for 7 epochs

    # ── Normalization probe ─────────────────────────────────────────────
    'norm_probe_size' : 50000,      # more probe samples for better stats

    # ── Reproducibility ─────────────────────────────────────────────────
    'seed'            : 42,

    # ── Class names ─────────────────────────────────────────────────────
    'class_names'     : ['BUSY', 'FREE', 'JAMMED'],
}


# ═══════════════════════════════════════════════════════════════════════════════
#  GOOGLE DRIVE HEALTH CHECK — Detect & fix zombie mounts after session restart
# ═══════════════════════════════════════════════════════════════════════════════
def ensure_drive_mounted():
    """
    Detect stale/zombie Google Drive mounts and re-mount automatically.

    PROBLEM: When a Colab session dies and you reconnect, the old Drive mount
    at /content/drive/ becomes a zombie — os.path.exists() returns True but
    actual file reads fail with 'errno 107: Transport endpoint is not connected'.

    SOLUTION: Actually try to list a directory on Drive. If it fails, force
    unmount and re-mount.
    """
    import subprocess

    drive_root = '/content/drive/MyDrive'

    # Step 1: Check if Drive is mounted at all
    if not os.path.exists('/content/drive'):
        print("  ℹ Drive not mounted. Mounting now...")
        from google.colab import drive
        drive.mount('/content/drive')
        print("  ✓ Google Drive mounted successfully")
        return

    # Step 2: Check if the mount is a ZOMBIE (path exists but can't read)
    try:
        os.listdir(drive_root)
        # If we get here, Drive is alive
        print("  ✓ Google Drive connection is healthy")
        return
    except OSError as e:
        error_num = getattr(e, 'errno', None)
        print(f"\n  ⚠ STALE DRIVE MOUNT DETECTED (errno={error_num})")
        print(f"    Error: {e}")
        print(f"    This happens when Colab session restarts but old mount lingers.")
        print(f"    Fixing automatically...\n")

    # Step 3: Force unmount the zombie
    try:
        print("    Step 1/2: Force unmounting stale drive...")
        subprocess.run(['fusermount', '-uz', '/content/drive'], check=False,
                       capture_output=True, timeout=10)
        # Give OS a moment to clean up
        time.sleep(2)
    except Exception as e:
        print(f"    ⚠ fusermount failed ({e}), trying direct unmount...")
        try:
            subprocess.run(['umount', '-l', '/content/drive'], check=False,
                           capture_output=True, timeout=10)
            time.sleep(2)
        except Exception:
            pass

    # Step 4: Clean up mount point if needed
    if os.path.exists('/content/drive'):
        try:
            os.listdir('/content/drive')
        except OSError:
            # Still broken — remove the stale mount point
            try:
                os.rmdir('/content/drive')
            except Exception:
                pass

    # Step 5: Re-mount fresh
    print("    Step 2/2: Re-mounting Google Drive...")
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)

    # Step 6: Verify the fresh mount
    try:
        os.listdir(drive_root)
        print("  ✓ Google Drive re-mounted and verified successfully!")
    except OSError as e:
        print(f"  ❌ CRITICAL: Drive still not readable after remount: {e}")
        print(f"     Please manually run: drive.mount('/content/drive', force_remount=True)")
        raise RuntimeError("Cannot access Google Drive. Please remount manually.")


# ═══════════════════════════════════════════════════════════════════════════════
#  SETUP
# ═══════════════════════════════════════════════════════════════════════════════
def setup_environment():
    """Seed everything, detect device, create output dirs."""
    torch.manual_seed(CFG['seed'])
    np.random.seed(CFG['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CFG['seed'])
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True   # fixed input size → big speedup

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(CFG['output_dir'], exist_ok=True)

    return device


def banner(text, width=72):
    print("\n" + "═" * width)
    print(f"  {text}")
    print("═" * width)


def section(text):
    print(f"\n── {text} {'─' * max(4, 62 - len(text))}")


# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET PREPROCESSING — HDF5 → Memory-Mapped .npy (ONE-TIME)
# ═══════════════════════════════════════════════════════════════════════════════
def preprocess_dataset():
    """
    Convert HDF5 dataset to memory-mapped .npy files on local SSD.

    This is a ONE-TIME cost (~5-8 minutes) that pays off massively:
    - HDF5 random reads: ~2 seconds per batch (GPU-starving)
    - Memory-mapped .npy: ~0.01 seconds per batch (GPU-saturating)

    The .npy files live on Colab's local SSD (/content/).
    If they already exist (from a previous run in the same session),
    this step is skipped entirely.

    CRITICAL ADVANTAGE: Unlike HDF5, memory-mapped files are FORK-SAFE.
    This means we can use num_workers=2 in DataLoader for parallel
    data loading — the GPU never waits for data.
    """
    if os.path.exists(CFG['local_X_path']) and os.path.exists(CFG['local_y_path']):
        # Verify the files are valid
        try:
            X_check = np.load(CFG['local_X_path'], mmap_mode='r')
            y_check = np.load(CFG['local_y_path'], mmap_mode='r')
            n = X_check.shape[0]
            print(f"  ✓ Preprocessed files found on local SSD")
            print(f"    X: {X_check.shape} | y: {y_check.shape}")
            print(f"    Samples: {n:,}")
            del X_check, y_check
            return n
        except Exception:
            print("  ⚠ Cached files corrupted, re-preprocessing...")

    section("Preprocessing HDF5 → Memory-Mapped .npy (one-time cost)")
    print(f"  Source: {CFG['dataset_path']}")
    print(f"  This takes ~5-8 minutes. After this, every epoch runs 100× faster.\n")
    t0 = time.time()

    with h5py.File(CFG['dataset_path'], 'r') as f:
        n = f['X'].shape[0]
        print(f"  Dataset size: {n:,} samples")
        print(f"  HDF5 X shape: {f['X'].shape}")

        # ── Step 1: Compute normalization stats from probe ──────────────
        probe_size = min(CFG['norm_probe_size'], n)
        probe_idx = np.sort(np.random.choice(n, probe_size, replace=False))
        probe_data = np.array(f['X'][list(probe_idx)], dtype=np.float32)  # (probe, 1024, 2)

        ch_mean = np.zeros(2, dtype=np.float32)
        ch_std  = np.ones(2, dtype=np.float32)
        for ch in range(2):
            ch_mean[ch] = probe_data[:, :, ch].mean()
            ch_std[ch]  = probe_data[:, :, ch].std()
            if ch_std[ch] < 1e-8:
                ch_std[ch] = 1.0

        del probe_data
        gc.collect()

        print(f"  ✓ Normalization stats (from {probe_size:,} probe samples):")
        print(f"    Channel I: mean={ch_mean[0]:.6f}, std={ch_std[0]:.6f}")
        print(f"    Channel Q: mean={ch_mean[1]:.6f}, std={ch_std[1]:.6f}")

        # ── Step 2: Save norm stats for inference ───────────────────────
        norm_path = os.path.join(CFG['output_dir'], 'norm_stats.json')
        with open(norm_path, 'w') as nf:
            json.dump({'ch_mean': ch_mean.tolist(), 'ch_std': ch_std.tolist()}, nf, indent=2)
        print(f"  ✓ Saved norm_stats.json → {norm_path}")

        # ── Step 3: Create output .npy files ────────────────────────────
        # Pre-allocate memory-mapped files on local SSD
        X_out = np.lib.format.open_memmap(
            CFG['local_X_path'], mode='w+', dtype=np.float32,
            shape=(n, 2, 1024)  # Already transposed for Conv1D!
        )
        y_out = np.lib.format.open_memmap(
            CFG['local_y_path'], mode='w+', dtype=np.int64,
            shape=(n,)
        )

        # ── Step 4: Read HDF5 in chunks, normalize, transpose, write ───
        chunk_size = 50000
        num_chunks = (n + chunk_size - 1) // chunk_size

        for i in range(num_chunks):
            start = i * chunk_size
            end   = min(start + chunk_size, n)
            pct   = end / n * 100

            # Read chunk from HDF5
            chunk = np.array(f['X'][start:end], dtype=np.float32)  # (chunk, 1024, 2)

            # Normalize per channel
            for ch in range(2):
                chunk[:, :, ch] = (chunk[:, :, ch] - ch_mean[ch]) / ch_std[ch]

            # Transpose: (chunk, 1024, 2) → (chunk, 2, 1024) for Conv1D
            X_out[start:end] = chunk.transpose(0, 2, 1)

            print(f"\r    Processing: {end:>10,} / {n:,}  ({pct:5.1f}%)", end='')
            del chunk

        print()  # newline

        # Write labels
        y_out[:] = np.array(f['y'][:], dtype=np.int64)

    # Flush to disk
    del X_out, y_out
    gc.collect()

    elapsed = time.time() - t0
    print(f"\n  ✓ Preprocessing complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"    X saved: {CFG['local_X_path']} (normalized, transposed)")
    print(f"    y saved: {CFG['local_y_path']}")

    # Verify file sizes
    x_size = os.path.getsize(CFG['local_X_path']) / 1e9
    y_size = os.path.getsize(CFG['local_y_path']) / 1e6
    print(f"    X size: {x_size:.1f} GB | y size: {y_size:.1f} MB")

    return n


# ═══════════════════════════════════════════════════════════════════════════════
#  TASK 1.8 — RadioMLDataset (Memory-Mapped — Maximum GPU Throughput)
# ═══════════════════════════════════════════════════════════════════════════════
class RadioMLDataset(Dataset):
    """
    Ultra-fast PyTorch Dataset using memory-mapped numpy files.

    WHY THIS IS FAST:
    ─────────────────
    1. Data lives as .npy on fast local SSD (not Google Drive)
    2. Memory-mapped = OS handles caching automatically
    3. No HDF5 overhead per read
    4. FORK-SAFE → enables num_workers=2 for parallel loading
    5. Data is PRE-normalized and PRE-transposed (zero per-sample cost)

    RAM footprint: ~100 MB (OS manages the page cache transparently)
    """

    def __init__(self, x_path, y_path):
        super().__init__()
        self.X = np.load(x_path, mmap_mode='r')  # (N, 2, 1024) — lazy, no RAM
        self.y = np.load(y_path, mmap_mode='r')   # (N,) — lazy, no RAM
        self._n = self.X.shape[0]

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        x = torch.from_numpy(np.array(self.X[idx]))  # (2, 1024) copy from mmap → tensor
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


def normalize_batch_energy(x):
    """
    Per-sample RMS normalization on the whole batch at once.
    This keeps the structural-learning benefit while avoiding expensive
    per-sample CPU work inside Dataset.__getitem__.
    """
    energy = x.pow(2).mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-8)
    return x / energy


# ═══════════════════════════════════════════════════════════════════════════════
#  TASK 1.8 — Data Splitting & DataLoaders
# ═══════════════════════════════════════════════════════════════════════════════
def create_dataloaders(dataset):
    """
    Split dataset into 70% train / 15% val / 15% test.
    Returns three DataLoaders with parallel workers.
    """
    section("Creating Train/Val/Test Split (70/15/15)")

    n = len(dataset)
    indices = np.arange(n)
    np.random.shuffle(indices)

    n_train = int(n * CFG['train_ratio'])
    n_val   = int(n * CFG['val_ratio'])

    train_idx = indices[:n_train].tolist()
    val_idx   = indices[n_train:n_train + n_val].tolist()
    test_idx  = indices[n_train + n_val:].tolist()

    print(f"\n  Total samples  : {n:,}")
    print(f"  Training       : {len(train_idx):>10,}  ({len(train_idx)/n*100:.1f}%)")
    print(f"  Validation     : {len(val_idx):>10,}  ({len(val_idx)/n*100:.1f}%)")
    print(f"  Test           : {len(test_idx):>10,}  ({len(test_idx)/n*100:.1f}%)")

    # Verify no overlap
    s_train, s_val, s_test = set(train_idx), set(val_idx), set(test_idx)
    assert len(s_train & s_val) == 0, "Train/Val overlap!"
    assert len(s_train & s_test) == 0, "Train/Test overlap!"
    assert len(s_val & s_test) == 0, "Val/Test overlap!"
    print("  ✓ No data leakage — splits are mutually exclusive")

    # Save exact indices — mandatory for reproducible evaluation
    idx_dir = CFG['output_dir']
    np.save(os.path.join(idx_dir, 'train_idx.npy'), np.array(train_idx))
    np.save(os.path.join(idx_dir, 'val_idx.npy'),   np.array(val_idx))
    np.save(os.path.join(idx_dir, 'test_idx.npy'),  np.array(test_idx))
    print(f"  ✓ Split indices saved to {idx_dir}")

    # Class balance
    y_all = np.load(CFG['local_y_path'], mmap_mode='r')
    print(f"\n  Class balance per split:")
    print(f"  {'Split':<12} {'BUSY':>8} {'FREE':>8} {'JAMMED':>8}")
    print("  " + "─" * 40)
    for name, idx_list in [('Train', train_idx), ('Val', val_idx), ('Test', test_idx)]:
        labels = y_all[idx_list]
        counts = [int((labels == c).sum()) for c in range(3)]
        pcts   = [c / len(idx_list) * 100 for c in counts]
        print(f"  {name:<12} {pcts[0]:>7.1f}% {pcts[1]:>7.1f}% {pcts[2]:>7.1f}%")

    # Create Subset datasets
    train_set = Subset(dataset, train_idx)
    val_set   = Subset(dataset, val_idx)
    test_set  = Subset(dataset, test_idx)

    nw = CFG['num_workers']

    train_loader = DataLoader(
        train_set,
        batch_size=CFG['batch_size'],
        shuffle=True,
        num_workers=nw,          # PARALLEL loading — mmap is fork-safe!
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if nw > 0 else False,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=CFG['batch_size'],
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=True if nw > 0 else False,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=CFG['batch_size'],
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=True if nw > 0 else False,
    )

    print(f"\n  DataLoaders created (num_workers={nw}, memory-mapped mode):")
    print(f"    Train : {len(train_loader):>6} batches × {CFG['batch_size']} = "
          f"{len(train_loader) * CFG['batch_size']:,} samples/epoch")
    print(f"    Val   : {len(val_loader):>6} batches")
    print(f"    Test  : {len(test_loader):>6} batches")

    return train_loader, val_loader, test_loader, train_idx, val_idx, test_idx


# ═══════════════════════════════════════════════════════════════════════════════
#  TASK 1.9 — Training Engine
# ═══════════════════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    """Train for one epoch. Returns average loss and accuracy."""
    model.train()
    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        x = normalize_batch_energy(x)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        # Gradient clipping — prevents explosive gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        total_loss    += loss.item() * x.size(0)
        preds          = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += x.size(0)

        # Progress every 100 batches
        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(loader):
            running_loss = total_loss / total_samples
            running_acc  = total_correct / total_samples * 100
            pct = (batch_idx + 1) / len(loader) * 100
            print(f"\r    Epoch {epoch:>2}/{total_epochs} │ "
                  f"Batch {batch_idx+1:>5}/{len(loader)} ({pct:5.1f}%) │ "
                  f"Loss: {running_loss:.4f} │ Acc: {running_acc:6.2f}%", end='')

    avg_loss = total_loss / total_samples
    avg_acc  = total_correct / total_samples * 100
    print()  # newline after progress
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate on val/test set. Returns loss, accuracy, per-class accuracy."""
    model.eval()
    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    class_correct = [0] * CFG['num_classes']
    class_total   = [0] * CFG['num_classes']

    all_preds  = []
    all_labels = []

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        x = normalize_batch_energy(x)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss    += loss.item() * x.size(0)
        preds          = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += x.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

        for c in range(CFG['num_classes']):
            mask = y == c
            class_correct[c] += (preds[mask] == y[mask]).sum().item()
            class_total[c]   += mask.sum().item()

    avg_loss = total_loss / total_samples
    avg_acc  = total_correct / total_samples * 100

    class_acc = []
    for c in range(CFG['num_classes']):
        if class_total[c] > 0:
            class_acc.append(class_correct[c] / class_total[c] * 100)
        else:
            class_acc.append(0.0)

    return avg_loss, avg_acc, class_acc, np.array(all_preds), np.array(all_labels)


def compute_confusion_matrix(preds, labels, num_classes):
    """Compute confusion matrix."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true, pred in zip(labels, preds):
        cm[true][pred] += 1
    return cm


# ═══════════════════════════════════════════════════════════════════════════════
#  VISUALIZATION — Training Curves
# ═══════════════════════════════════════════════════════════════════════════════
def plot_training_curves(history, output_dir):
    """Plot loss and accuracy curves. Saves to drive."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle('CogniRad — Training Progress',
                 color='white', fontsize=14, fontweight='bold')

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    ax1.set_facecolor('#0d1117')
    ax1.plot(epochs, history['train_loss'], 'o-', color='#2196F3',
             linewidth=2, markersize=4, label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 's-', color='#E91E63',
             linewidth=2, markersize=4, label='Val Loss')
    ax1.set_xlabel('Epoch', color='white')
    ax1.set_ylabel('Cross-Entropy Loss', color='white')
    ax1.set_title('Loss Curves', color='white', fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.tick_params(colors='white')
    ax1.grid(True, alpha=0.15, color='white')
    for spine in ax1.spines.values():
        spine.set_edgecolor('#444')

    # Accuracy
    ax2.set_facecolor('#0d1117')
    ax2.plot(epochs, history['train_acc'], 'o-', color='#2196F3',
             linewidth=2, markersize=4, label='Train Acc')
    ax2.plot(epochs, history['val_acc'], 's-', color='#E91E63',
             linewidth=2, markersize=4, label='Val Acc')
    ax2.axhline(93, color='#4CAF50', linestyle='--', alpha=0.7,
                linewidth=1.5, label='Target (93%)')
    ax2.set_xlabel('Epoch', color='white')
    ax2.set_ylabel('Accuracy (%)', color='white')
    ax2.set_title('Accuracy Curves', color='white', fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.tick_params(colors='white')
    ax2.grid(True, alpha=0.15, color='white')
    for spine in ax2.spines.values():
        spine.set_edgecolor('#444')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"  ✓ Saved → {save_path}")


def plot_confusion_matrix(cm, output_dir, title='Confusion Matrix'):
    """Plot confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    im = ax.imshow(cm, cmap='plasma', interpolation='nearest')
    plt.colorbar(im, ax=ax)

    classes = CFG['class_names']
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, color='white', fontsize=11)
    ax.set_yticklabels(classes, color='white', fontsize=11)
    ax.set_xlabel('Predicted', color='white', fontsize=12)
    ax.set_ylabel('True', color='white', fontsize=12)
    ax.set_title(title, color='white', fontsize=14, fontweight='bold')

    for i in range(len(classes)):
        for j in range(len(classes)):
            total_row = cm[i].sum()
            pct = cm[i][j] / total_row * 100 if total_row > 0 else 0
            text_color = 'white' if cm[i][j] < cm.max() / 2 else 'black'
            ax.text(j, i, f'{cm[i][j]:,}\n({pct:.1f}%)',
                    ha='center', va='center', color=text_color,
                    fontsize=10, fontweight='bold')

    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"  ✓ Saved → {save_path}")


def plot_lr_schedule(history, output_dir):
    """Plot learning rate over epochs."""
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    epochs = range(1, len(history['lr']) + 1)
    ax.plot(epochs, history['lr'], 'o-', color='#FFA726',
            linewidth=2, markersize=5)
    ax.set_xlabel('Epoch', color='white')
    ax.set_ylabel('Learning Rate', color='white')
    ax.set_title('Learning Rate Schedule', color='white',
                 fontsize=13, fontweight='bold')
    ax.set_yscale('log')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.15, color='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'lr_schedule.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"  ✓ Saved → {save_path}")


def plot_per_class_accuracy(history, output_dir):
    """Plot per-class accuracy over epochs."""
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    epochs = range(1, len(history['val_class_acc']) + 1)
    colors = ['#00BCD4', '#4CAF50', '#E91E63']

    for cls_id, (cls_name, color) in enumerate(zip(CFG['class_names'], colors)):
        accs = [epoch_accs[cls_id] for epoch_accs in history['val_class_acc']]
        ax.plot(epochs, accs, 'o-', color=color, linewidth=2,
                markersize=4, label=cls_name)

    ax.axhline(93, color='white', linestyle='--', alpha=0.3,
               linewidth=1, label='Target (93%)')
    ax.set_xlabel('Epoch', color='white')
    ax.set_ylabel('Accuracy (%)', color='white')
    ax.set_title('Per-Class Validation Accuracy',
                 color='white', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.15, color='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'per_class_accuracy.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"  ✓ Saved → {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN — Production Training Loop
# ═══════════════════════════════════════════════════════════════════════════════
def safe_save_checkpoint(state, drive_path, local_path):
    """
    Network-safe checkpoint save.
    1. Save to LOCAL SSD first (instant, never fails)
    2. Copy to Google Drive (may fail if network drops)
    3. If Drive copy fails, local copy survives for manual recovery
    """
    # Step 1: Save locally (instant, no network)
    torch.save(state, local_path)

    # Step 2: Copy to Drive (network-dependent)
    try:
        shutil.copyfile(local_path, drive_path)
    except Exception as e:
        print(f"  ⚠ Drive save failed ({e}), but local checkpoint is safe at {local_path}")


def main():
    total_start = time.time()

    banner("CogniRad — Production Training Pipeline v3")
    print(f"  Started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── Google Drive Health Check (prevents errno 107 on session restart) ─
    section("Google Drive Health Check")
    ensure_drive_mounted()

    # ── Environment Setup ────────────────────────────────────────────────
    device = setup_environment()
    print(f"\n  Device   : {device}")
    if torch.cuda.is_available():
        print(f"  GPU      : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  PyTorch  : {torch.__version__}")
    print(f"  Seed     : {CFG['seed']}")

    # ── Import Model ─────────────────────────────────────────────────────
    section("Importing SpectrumClassifier")
    sys.path.insert(0, os.path.join(CFG['project_dir'], 'layer_1'))
    from spectrum_classifier import SpectrumClassifier

    model = SpectrumClassifier(
        num_classes  = CFG['num_classes'],
        in_channels  = CFG['in_channels'],
        base_filters = CFG['base_filters'],
        num_heads    = CFG['num_heads'],
        attn_dropout = CFG['attn_dropout'],
        fc_dropout   = CFG['fc_dropout'],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ Model loaded — {total_params:,} params ({trainable:,} trainable)")

    # ── Task 1.8 — Preprocess & Load Dataset ─────────────────────────────
    section("Task 1.8 — Dataset Preprocessing")

    if not os.path.exists(CFG['dataset_path']):
        print(f"\n  ❌ Dataset not found at: {CFG['dataset_path']}")
        print("  Please upload radioml_remapped.hdf5 to Google Drive.")
        return

    n_samples = preprocess_dataset()

    # Load memory-mapped dataset
    section("Loading Memory-Mapped Dataset")
    dataset = RadioMLDataset(CFG['local_X_path'], CFG['local_y_path'])
    print(f"  ✓ Dataset ready — {len(dataset):,} samples")
    print(f"    X shape: {dataset.X.shape}")
    print(f"    Data is PRE-normalized and PRE-transposed (zero overhead per batch)")

    # Print class distribution
    y_all = np.load(CFG['local_y_path'], mmap_mode='r')
    for cls_id, cls_name in enumerate(CFG['class_names']):
        count = int((y_all == cls_id).sum())
        pct = count / len(dataset) * 100
        print(f"    {cls_name:<8}: {count:>10,}  ({pct:5.1f}%)")

    # Smoke test
    print(f"\n  Smoke test — passing one batch through model...")
    x_test, y_test = dataset[0]
    x_test = x_test.unsqueeze(0).to(device)
    with torch.no_grad():
        logits_test = model(x_test)
    print(f"  ✓ Input: {tuple(x_test.shape)} → Output: {tuple(logits_test.shape)}")

    # ── Task 1.8 — Create DataLoaders ────────────────────────────────────
    train_loader, val_loader, test_loader, train_idx, val_idx, test_idx = \
        create_dataloaders(dataset)

    # ── Task 1.9 — Training Setup ────────────────────────────────────────
    section("Task 1.9 — Training Configuration")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CFG['learning_rate'],
        weight_decay=CFG['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=CFG['scheduler_factor'],
        patience=CFG['scheduler_patience'],
    )

    print(f"  Optimizer   : Adam (lr={CFG['learning_rate']}, "
          f"weight_decay={CFG['weight_decay']})")
    print(f"  Loss        : CrossEntropyLoss (label_smoothing=0.05)")
    print(f"  Scheduler   : ReduceLROnPlateau "
          f"(factor={CFG['scheduler_factor']}, patience={CFG['scheduler_patience']})")
    print(f"  Grad clip   : max_norm=5.0")
    print(f"  Epochs      : {CFG['epochs']}")
    print(f"  Batch size  : {CFG['batch_size']}")
    print(f"  num_workers : {CFG['num_workers']} (parallel, memory-mapped)")

    # ── Training History ─────────────────────────────────────────────────
    history = {
        'train_loss':    [],
        'train_acc':     [],
        'val_loss':      [],
        'val_acc':       [],
        'val_class_acc': [],
        'lr':            [],
    }

    best_val_acc  = 0.0
    best_val_loss = float('inf')
    best_epoch    = 0
    start_epoch   = 1

    # ── Resume from checkpoint if available ──────────────────────────────
    # Check Drive checkpoint first, then local fallback
    resume_from = None
    if os.path.exists(CFG['resume_path']):
        resume_from = CFG['resume_path']
    elif os.path.exists(CFG['local_ckpt_path']):
        resume_from = CFG['local_ckpt_path']
        print(f"  ⚠ Drive checkpoint missing, using local fallback")

    if resume_from is not None:
        section("Resuming from checkpoint")
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch   = ckpt['epoch'] + 1
        best_val_acc  = ckpt.get('best_val_acc', 0.0)
        best_epoch    = ckpt.get('best_epoch', 0)
        history       = ckpt.get('history', history)
        print(f"  ✓ Resumed from epoch {ckpt['epoch']}")
        print(f"    Best val acc so far : {best_val_acc:.2f}%")
        print(f"    Continuing from epoch {start_epoch}/{CFG['epochs']}")
        if start_epoch > CFG['epochs']:
            print(f"  ✓ Training already complete! Skipping to evaluation.")
    else:
        print(f"\n  No checkpoint found — starting fresh from epoch 1")

    # ══════════════════════════════════════════════════════════════════════
    #  TRAINING LOOP
    # ══════════════════════════════════════════════════════════════════════
    banner("Training Begins")

    print(f"\n  {'Epoch':>5} │ {'Train Loss':>10} │ {'Train Acc':>9} │ "
          f"{'Val Loss':>10} │ {'Val Acc':>9} │ {'LR':>10} │ {'Status'}")
    print("  " + "─" * 85)

    for epoch in range(start_epoch, CFG['epochs'] + 1):
        epoch_start = time.time()

        # ── Train ────────────────────────────────────────────────────────
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, CFG['epochs']
        )

        # ── Validate ─────────────────────────────────────────────────────
        val_loss, val_acc, val_class_acc, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        # ── Scheduler Step ───────────────────────────────────────────────
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']

        # ── Record History ───────────────────────────────────────────────
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_class_acc'].append(val_class_acc)
        history['lr'].append(current_lr)

        # ── Best Model Checkpoint ────────────────────────────────────────
        status = ""
        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_val_loss = val_loss
            best_epoch    = epoch
            status = "★ BEST"

            # Save best model
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'optimizer':   optimizer.state_dict(),
                'scheduler':   scheduler.state_dict(),
                'val_acc':     val_acc,
                'val_loss':    val_loss,
                'config':      CFG,
            }, CFG['best_model_path'])
            # Also save best model locally as backup
            shutil.copyfile(CFG['best_model_path'], '/content/best_model_backup.pt')

        if current_lr != new_lr:
            status += " ↓LR"

        epoch_time = time.time() - epoch_start

        # ── Epoch Summary ────────────────────────────────────────────────
        print(f"  {epoch:>5} │ {train_loss:>10.4f} │ {train_acc:>8.2f}% │ "
              f"{val_loss:>10.4f} │ {val_acc:>8.2f}% │ {current_lr:>10.6f} │ "
              f"{status}  ({epoch_time:.0f}s)")

        # Per-class breakdown every 5 epochs
        if epoch % 5 == 0 or epoch == 1 or epoch == CFG['epochs']:
            for cls_id, cls_name in enumerate(CFG['class_names']):
                print(f"          {cls_name:<8}: {val_class_acc[cls_id]:6.2f}%")

        # ── Save resume checkpoint EVERY epoch (network-safe) ───────────
        safe_save_checkpoint({
            'epoch':        epoch,
            'model_state':  model.state_dict(),
            'optimizer':    optimizer.state_dict(),
            'scheduler':    scheduler.state_dict(),
            'best_val_acc': best_val_acc,
            'best_epoch':   best_epoch,
            'history':      history,
            'config':       CFG,
        }, CFG['resume_path'], CFG['local_ckpt_path'])

        # Save training curves periodically
        if epoch % 5 == 0 or epoch == CFG['epochs']:
            plot_training_curves(history, CFG['output_dir'])
            plot_per_class_accuracy(history, CFG['output_dir'])
            plot_lr_schedule(history, CFG['output_dir'])

    # ══════════════════════════════════════════════════════════════════════
    #  FINAL EVALUATION ON TEST SET
    # ══════════════════════════════════════════════════════════════════════
    banner("Final Evaluation — Test Set")

    # Load best model
    checkpoint = torch.load(CFG['best_model_path'], map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    print(f"  Loaded best model from epoch {checkpoint['epoch']} "
          f"(val_acc={checkpoint['val_acc']:.2f}%)")

    test_loss, test_acc, test_class_acc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )

    print(f"\n  ══════════════════════════════════════════")
    print(f"  Test Loss     : {test_loss:.4f}")
    print(f"  Test Accuracy : {test_acc:.2f}%")
    print(f"  ══════════════════════════════════════════")
    for cls_id, cls_name in enumerate(CFG['class_names']):
        print(f"    {cls_name:<8}: {test_class_acc[cls_id]:6.2f}%")

    target_met = test_acc >= 93.0
    print(f"\n  Target (>93%): {'✅ ACHIEVED' if target_met else '❌ NOT MET'}")

    # Confusion Matrix
    cm = compute_confusion_matrix(test_preds, test_labels, CFG['num_classes'])
    plot_confusion_matrix(cm, CFG['output_dir'], title='Test Set — Confusion Matrix')

    print(f"\n  Confusion Matrix:")
    print(f"  {'':>10} {'BUSY':>8} {'FREE':>8} {'JAMMED':>8}")
    print("  " + "─" * 38)
    for i, cls_name in enumerate(CFG['class_names']):
        row_str = "  ".join(f"{cm[i][j]:>6,}" for j in range(3))
        print(f"  {cls_name:<10} {row_str}")

    # ── Save Final Model ─────────────────────────────────────────────────
    section("Saving Final Artifacts")

    torch.save({
        'epoch':       CFG['epochs'],
        'model_state': model.state_dict(),
        'config':      CFG,
        'test_acc':    test_acc,
        'test_loss':   test_loss,
        'class_acc':   test_class_acc,
    }, CFG['model_save_path'])
    print(f"  ✓ Final model   → {CFG['model_save_path']}")

    # Save training history
    history_path = os.path.join(CFG['output_dir'], 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  ✓ History       → {history_path}")

    # Save config
    config_path = os.path.join(CFG['output_dir'], 'config.json')
    with open(config_path, 'w') as f:
        json.dump(CFG, f, indent=2)
    print(f"  ✓ Config        → {config_path}")

    # ══════════════════════════════════════════════════════════════════════
    #  FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    total_time = time.time() - total_start

    banner("TRAINING COMPLETE — FINAL REPORT")

    print(f"""
  Model         : SpectrumClassifier (ResNet1D + 8-Head Attention)
  Parameters    : {total_params:,}
  Dataset       : {len(dataset):,} samples (BUSY/FREE/JAMMED)
  Split         : {CFG['train_ratio']*100:.0f}/{CFG['val_ratio']*100:.0f}/{CFG['test_ratio']*100:.0f} (train/val/test)
  Data Pipeline : Memory-mapped .npy (parallel workers, zero copy)

  Best Epoch    : {best_epoch}/{CFG['epochs']}
  Best Val Acc  : {best_val_acc:.2f}%
  Test Accuracy : {test_acc:.2f}%
  Test Loss     : {test_loss:.4f}

  Per-Class Test Accuracy:
    BUSY         : {test_class_acc[0]:.2f}%
    FREE         : {test_class_acc[1]:.2f}%
    JAMMED       : {test_class_acc[2]:.2f}%

  Target (>93%) : {'✅ ACHIEVED' if target_met else '❌ NOT MET'}
  Total Time    : {total_time/60:.1f} minutes

  Saved Artifacts:
    → {CFG['model_save_path']}
    → {CFG['best_model_path']}
    → {CFG['output_dir']}/training_curves.png
    → {CFG['output_dir']}/confusion_matrix.png
    → {CFG['output_dir']}/per_class_accuracy.png
    → {CFG['output_dir']}/lr_schedule.png
    → {CFG['output_dir']}/training_history.json
    → {CFG['output_dir']}/config.json
    → {CFG['output_dir']}/norm_stats.json
""")

    banner(f"CogniRad Training Complete — {test_acc:.2f}% Test Accuracy "
           f"{'✅' if target_met else '❌'}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    main()
