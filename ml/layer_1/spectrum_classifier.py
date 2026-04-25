"""
╔══════════════════════════════════════════════════════════════════════════╗
║  CogniRad — Task 1.7 : SpectrumClassifier with Attention               ║
║  Class  : SpectrumClassifier (PyTorch nn.Module)                        ║
║  Purpose: ResNet1D → MultiheadAttention → GAP → Linear(384, 3)         ║
║  Output : Shape trace, attention heatmaps, entropy, gradient flow      ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys
import math

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — single source of truth for all hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
CFG = {
    'base_filters'  : 48,           # must match Task 1.6 exactly
    'num_heads'     : 8,            # attention heads
    'attn_dropout'  : 0.1,          # dropout inside attention layers
    'fc_dropout'    : 0.4,          # dropout before classifier
    'num_classes'   : 3,            # BUSY=0, FREE=1, JAMMED=2
    'seq_len'       : 1024,         # IQ sample length
    'in_channels'   : 2,            # I and Q
    'embed_dim'     : 48 * 8,       # = 384, derived — never hardcode
    'batch_size'    : 8,            # for verification passes only
    'plot_dir'      : 'plots',
    'seed'          : 42,
}

# ── Seed everything for reproducibility ──────────────────────────────────
torch.manual_seed(CFG['seed'])
np.random.seed(CFG['seed'])

# ── Import building blocks from previous tasks ──────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from residual_block import ResidualBlock1D
from resnet1d import ResNet1D

# ── Paths ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR  = os.path.join(SCRIPT_DIR, CFG['plot_dir'])
os.makedirs(PLOTS_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def banner(text, width=66):
    print("\n" + "═" * width)
    print(f"  {text}")
    print("═" * width)


def section(text):
    print(f"\n── {text} {'─' * max(4, 58 - len(text))}")


def ok(msg):
    print(f"  ✅  {msg}")


def fail(msg):
    print(f"  ❌  {msg}")


def info(msg):
    print(f"  ℹ   {msg}")


def generate_synthetic_signal(cls_name, seq_len=1024):
    """Generate a synthetic IQ signal for one of the three classes."""
    t = np.linspace(0, 1, seq_len)
    if cls_name == 'BUSY':
        phases = np.random.choice([0, np.pi/2, np.pi, 3*np.pi/2], seq_len)
        I = np.cos(2 * np.pi * 8 * t + phases) + 0.1 * np.random.randn(seq_len)
        Q = np.sin(2 * np.pi * 8 * t + phases) + 0.1 * np.random.randn(seq_len)
    elif cls_name == 'FREE':
        I = np.random.normal(0, 0.3, seq_len)
        Q = np.random.normal(0, 0.3, seq_len)
    else:  # JAMMED
        I = np.random.normal(0, 2.5, seq_len)
        Q = np.random.normal(0, 2.5, seq_len)
    return np.stack([I, Q], axis=0).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#  THE CLASS — SpectrumClassifier
# ═══════════════════════════════════════════════════════════════════════════════
class SpectrumClassifier(nn.Module):
    """
    Full CogniRad classifier: ResNet1D backbone → Self-Attention → GAP → Linear.

    Architecture
    ────────────
        Input: (batch, 2, 1024)  — raw I/Q samples
          │
          ResNet1D backbone (no GAP)     → (batch, 384, 128)
          │
          Transpose                      → (batch, 128, 384)
          │
          MultiheadAttention (8 heads)   → (batch, 128, 384)
          + Residual Connection + LayerNorm
          │
          Transpose back                 → (batch, 384, 128)
          │
          Global Average Pooling         → (batch, 384)
          │
          Dropout (0.4)                  → (batch, 384)
          │
          Linear(384, 3)                 → (batch, 3) logits
          │
        Output: (batch, 3)  — class logits [BUSY, FREE, JAMMED]

    Parameters
    ──────────
        num_classes   : number of output classes (default 3)
        in_channels   : input channels, 2 for I and Q (default 2)
        base_filters  : backbone filter base (default 48)
        num_heads     : number of attention heads (default 8)
        attn_dropout  : dropout inside attention (default 0.1)
        fc_dropout    : dropout before classifier (default 0.4)
    """

    def __init__(self,
                 num_classes  = CFG['num_classes'],
                 in_channels  = CFG['in_channels'],
                 base_filters = CFG['base_filters'],
                 num_heads    = CFG['num_heads'],
                 attn_dropout = CFG['attn_dropout'],
                 fc_dropout   = CFG['fc_dropout']):
        super().__init__()

        embed_dim = base_filters * 8  # 384

        # ── Validation ───────────────────────────────────────────────────
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by "
            f"num_heads ({num_heads}). "
            f"Per-head dimension = {embed_dim}/{num_heads} "
            f"= {embed_dim/num_heads} — must be integer."
        )

        # ── Backbone — ResNet1D with GAP removed ────────────────────────
        self.backbone = ResNet1D(
            num_classes=num_classes,
            in_channels=in_channels,
            base_filters=base_filters
        )

        # ── Multi-Head Self-Attention ───────────────────────────────────
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True          # (batch, seq, features)
        )

        # ── LayerNorm after residual connection ─────────────────────────
        self.norm = nn.LayerNorm(embed_dim)

        # ── Global Average Pooling (moved from ResNet1D) ────────────────
        self.gap = nn.AdaptiveAvgPool1d(1)

        # ── Dropout before classifier ───────────────────────────────────
        self.dropout = nn.Dropout(p=fc_dropout)

        # ── Classifier head ─────────────────────────────────────────────
        self.classifier = nn.Linear(embed_dim, num_classes)

        # Store config
        self._embed_dim    = embed_dim
        self._num_heads    = num_heads
        self._num_classes  = num_classes
        self._base_filters = base_filters

    @property
    def embed_dim(self):
        return self._embed_dim

    def forward(self, x, return_attention=False):
        """
        Forward pass.

        Args:
            x: (batch, 2, 1024) — raw IQ samples
            return_attention: if True, also return per-head attention weights.
                Keep this False for training to avoid the large attention-weight tensor.

        Returns:
            logits: (batch, 3) — class logits
            attn_weights (optional): (batch, 8, 128, 128) — per-head weights
        """
        # Backbone feature extraction
        x = self.backbone(x)
        # x: (batch, 384, 128)

        # Transpose for attention: channels → embedding dim
        x = x.transpose(1, 2)
        # x: (batch, 128, 384)

        # Self-attention with residual connection
        attn_out, attn_weights = self.attention(
            x, x, x,
            need_weights=return_attention,
            average_attn_weights=False
        )
        # attn_out:     (batch, 128, 384)
        # attn_weights: (batch, 8, 128, 128) — per head when requested

        x = self.norm(x + attn_out)
        # x: (batch, 128, 384) — residual + normalized

        # Transpose back for GAP: embedding → channels
        x = x.transpose(1, 2)
        # x: (batch, 384, 128)

        # Global average pooling
        x = self.gap(x).squeeze(-1)
        # x: (batch, 384)

        # Regularization
        x = self.dropout(x)
        # x: (batch, 384)

        # Classification
        logits = self.classifier(x)
        # logits: (batch, 3)

        if return_attention:
            return logits, attn_weights
        return logits


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — Shape Verification
# ═══════════════════════════════════════════════════════════════════════════════
def verify_all_shapes(model):
    """Run every shape transition explicitly."""
    model = model.to(DEVICE).eval()
    x = torch.randn(CFG['batch_size'], CFG['in_channels'],
                     CFG['seq_len']).to(DEVICE)

    print(f"\n{'='*65}")
    print(f"  SpectrumClassifier — Shape Propagation Trace")
    print(f"{'='*65}")

    steps = [
        ('Input', tuple(x.shape)),
    ]

    with torch.no_grad():
        # Backbone
        feat = model.backbone(x)
        steps.append(('After backbone', tuple(feat.shape)))

        # Transpose
        feat_t = feat.transpose(1, 2)
        steps.append(('After transpose (→attn)', tuple(feat_t.shape)))

        # Attention
        attn_out, weights = model.attention(
            feat_t, feat_t, feat_t,
            need_weights=True, average_attn_weights=False)
        steps.append(('After attention', tuple(attn_out.shape)))
        steps.append(('Attention weights', tuple(weights.shape)))

        # Residual + norm
        normed = model.norm(feat_t + attn_out)
        steps.append(('After residual+norm', tuple(normed.shape)))

        # Transpose back
        normed_t = normed.transpose(1, 2)
        steps.append(('After transpose (→GAP)', tuple(normed_t.shape)))

        # GAP
        pooled = model.gap(normed_t).squeeze(-1)
        steps.append(('After GAP+squeeze', tuple(pooled.shape)))

        # Classifier
        logits = model.classifier(model.dropout(pooled))
        steps.append(('Logits (output)', tuple(logits.shape)))

    ed = CFG['embed_dim']
    bs = CFG['batch_size']
    nh = CFG['num_heads']
    seq_out = 128  # 1024 / 8 from strides

    expected = {
        'Input'                   : (bs, 2, 1024),
        'After backbone'          : (bs, ed, seq_out),
        'After transpose (→attn)' : (bs, seq_out, ed),
        'After attention'         : (bs, seq_out, ed),
        'Attention weights'       : (bs, nh, seq_out, seq_out),
        'After residual+norm'     : (bs, seq_out, ed),
        'After transpose (→GAP)'  : (bs, ed, seq_out),
        'After GAP+squeeze'       : (bs, ed),
        'Logits (output)'         : (bs, CFG['num_classes']),
    }

    all_pass = True
    for name, shape in steps:
        exp = expected.get(name)
        if exp:
            match = tuple(shape) == exp
            status = '✅' if match else f'❌ EXPECTED {exp}'
            if not match:
                all_pass = False
        else:
            status = 'ℹ'
        print(f"  {name:<35} {str(tuple(shape)):<28} {status}")

    print(f"\n  Overall: {'ALL PASS ✅' if all_pass else 'FAILURES DETECTED ❌'}")
    print(f"{'='*65}")
    return all_pass


# ═══════════════════════════════════════════════════════════════════════════════
#  VISUALIZATION 1 — Architecture Flow Diagram
# ═══════════════════════════════════════════════════════════════════════════════
def plot_architecture_flow(model):
    """Draw the complete forward pass as a flowchart."""
    fig, ax = plt.subplots(figsize=(12, 16))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)

    # Component definitions: (y, label, shape_str, color)
    components = [
        (18.5, 'Input (Raw IQ)',        '(B, 2, 1024)',       '#64B5F6'),
        (16.5, 'ResNet1D Backbone',     '(B, 384, 128)',      '#2196F3'),
        (14.5, 'Transpose',             '(B, 128, 384)',      '#78909C'),
        (12.5, 'Multi-Head Attention',  '(B, 128, 384)',      '#AB47BC'),
        (11.0, '+ Residual + LayerNorm','(B, 128, 384)',      '#CE93D8'),
        (9.0,  'Transpose Back',        '(B, 384, 128)',      '#78909C'),
        (7.0,  'Global Avg Pool',       '(B, 384)',           '#66BB6A'),
        (5.0,  'Dropout (0.4)',         '(B, 384)',           '#FFA726'),
        (3.0,  'Linear(384, 3)',        '(B, 3)',             '#EF5350'),
        (1.0,  'Output (Logits)',       'BUSY / FREE / JAMMED','#E53935'),
    ]

    for y, label, shape_str, color in components:
        # Box
        rect = plt.Rectangle((2, y - 0.5), 6, 1.0,
                              facecolor=color + '33', edgecolor=color,
                              linewidth=2, zorder=2)
        ax.add_patch(rect)
        ax.text(5, y + 0.05, label, ha='center', va='center',
                color='white', fontsize=11, fontweight='bold', zorder=3)
        ax.text(8.5, y, shape_str, ha='left', va='center',
                color='#aaa', fontsize=9, fontstyle='italic', zorder=3)

    # Arrows between components
    for i in range(len(components) - 1):
        y1 = components[i][0] - 0.5
        y2 = components[i + 1][0] + 0.5
        ax.annotate('', xy=(5, y2), xytext=(5, y1),
                    arrowprops=dict(arrowstyle='->', color='white',
                                   lw=1.5, connectionstyle='arc3,rad=0'))

    # Residual bypass arrow
    ax.annotate('', xy=(1.5, 11.0), xytext=(1.5, 14.5),
                arrowprops=dict(arrowstyle='->', color='#CE93D8',
                                lw=2.0, linestyle='--',
                                connectionstyle='arc3,rad=-0.3'))
    ax.text(0.5, 12.8, 'Residual\nBypass', ha='center', va='center',
            color='#CE93D8', fontsize=8, fontstyle='italic')

    # Title
    ax.set_title('SpectrumClassifier — Architecture Flow\n'
                 f'Total Parameters: {sum(p.numel() for p in model.parameters()):,}',
                 color='white', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, 'architecture_flow.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    info(f"Plot saved → plots/architecture_flow.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  VISUALIZATION 2 — Attention Head Heatmaps (per class)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_attention_heatmaps(model):
    """8 attention heads × 3 classes = 24 heatmaps."""
    model = model.to(DEVICE).eval()

    classes = ['BUSY', 'FREE', 'JAMMED']
    fig, axes = plt.subplots(3, 8, figsize=(28, 12))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle('Attention Head Heatmaps — Per Class (rows) × Per Head (cols)\n'
                 'Before training: near-uniform (expected). After training: structured.',
                 color='white', fontsize=13, fontweight='bold')

    for row, cls_name in enumerate(classes):
        sig = generate_synthetic_signal(cls_name)
        x = torch.tensor(sig, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            _, attn_weights = model(x, return_attention=True)
        # attn_weights: (1, 8, 128, 128)
        weights_np = attn_weights[0].cpu().numpy()

        for head in range(8):
            ax = axes[row, head]
            ax.set_facecolor('#0d1117')
            im = ax.imshow(weights_np[head], aspect='auto',
                           cmap='plasma', interpolation='nearest',
                           vmin=0, vmax=weights_np[head].max())
            if row == 0:
                ax.set_title(f'Head {head}', color='white', fontsize=9,
                             fontweight='bold')
            if head == 0:
                ax.set_ylabel(cls_name, color='white', fontsize=10,
                              fontweight='bold')
            ax.tick_params(colors='white', labelsize=5)
            for spine in ax.spines.values():
                spine.set_edgecolor('#333')

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, 'attention_heatmaps.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    info(f"Plot saved → plots/attention_heatmaps.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  VISUALIZATION 3 — Attention Entropy Per Head
# ═══════════════════════════════════════════════════════════════════════════════
def compute_attention_entropy(attn_weights):
    """
    Compute mean entropy per attention head.

    Args:
        attn_weights: (batch, num_heads, seq, seq)
    Returns:
        (num_heads,) numpy array of mean entropy values
    """
    eps = 1e-9
    # Entropy per row per head: -(p * log(p)).sum(dim=-1)
    entropy = -(attn_weights * torch.log(attn_weights + eps)).sum(dim=-1)
    # entropy: (batch, num_heads, seq)
    mean_entropy = entropy.mean(dim=(0, 2))  # (num_heads,)
    return mean_entropy.cpu().numpy()


def plot_attention_entropy(model):
    """Bar chart of entropy per head for each class."""
    model = model.to(DEVICE).eval()

    classes = ['BUSY', 'FREE', 'JAMMED']
    class_colors = {'BUSY': '#00BCD4', 'FREE': '#4CAF50', 'JAMMED': '#E91E63'}

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    max_entropy = math.log(128)  # theoretical maximum
    bar_width = 0.25
    x_pos = np.arange(8)

    for i, cls_name in enumerate(classes):
        sig = generate_synthetic_signal(cls_name)
        x = torch.tensor(sig, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            _, attn_weights = model(x, return_attention=True)
        entropy = compute_attention_entropy(attn_weights)
        ax.bar(x_pos + i * bar_width, entropy,
               bar_width, color=class_colors[cls_name],
               alpha=0.8, label=cls_name, edgecolor='white', linewidth=0.3)

    ax.axhline(max_entropy, color='#ff4444', linestyle='--',
               linewidth=1.5, alpha=0.7, label=f'Max entropy (log 128 ≈ {max_entropy:.2f})')
    ax.set_xlabel('Attention Head', color='white', fontsize=10)
    ax.set_ylabel('Mean Entropy (nats)', color='white', fontsize=10)
    ax.set_title('Attention Entropy Per Head — Lower = More Focused',
                 color='white', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos + bar_width)
    ax.set_xticklabels([f'Head {i}' for i in range(8)])
    ax.legend(fontsize=9)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, 'attention_entropy.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    info(f"Plot saved → plots/attention_entropy.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  VISUALIZATION 4 — Mean Attention Per Position Per Class
# ═══════════════════════════════════════════════════════════════════════════════
def plot_attention_per_position(model):
    """Which time positions does the model find important?"""
    model = model.to(DEVICE).eval()

    classes = ['BUSY', 'FREE', 'JAMMED']
    class_colors = {'BUSY': '#00BCD4', 'FREE': '#4CAF50', 'JAMMED': '#E91E63'}

    fig, ax = plt.subplots(figsize=(16, 5))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    for cls_name in classes:
        sig = generate_synthetic_signal(cls_name)
        x = torch.tensor(sig, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            _, attn_weights = model(x, return_attention=True)
        # attn_weights: (1, 8, 128, 128)
        # Average over heads, then average over query positions → key importance
        key_importance = attn_weights[0].mean(dim=0).mean(dim=0).cpu().numpy()
        # key_importance: (128,) — how attended-to each position is

        ax.fill_between(range(128), key_importance, alpha=0.3,
                        color=class_colors[cls_name])
        ax.plot(key_importance, linewidth=1.2,
                color=class_colors[cls_name], label=cls_name)

    ax.set_xlabel('Temporal Position (128 time steps)', color='white', fontsize=10)
    ax.set_ylabel('Mean Attention Received', color='white', fontsize=10)
    ax.set_title('Key Importance Per Position — Which time steps matter most?\n'
                 '(Before training: flat and overlapping — expected)',
                 color='white', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, 'attention_position_importance.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    info(f"Plot saved → plots/attention_position_importance.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  VISUALIZATION 5 — Classifier Weight Distribution
# ═══════════════════════════════════════════════════════════════════════════════
def plot_classifier_weights(model):
    """Visualize the (3, 384) weight matrix as three line plots."""
    model = model.to(DEVICE).eval()

    weights = model.classifier.weight.detach().cpu().numpy()
    # weights: (3, 384)

    class_names  = ['BUSY', 'FREE', 'JAMMED']
    class_colors = ['#00BCD4', '#4CAF50', '#E91E63']

    fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True)
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle('Classifier Weight Templates — Linear(384, 3)\n'
                 'Before training: random. After training: class-discriminative patterns.',
                 color='white', fontsize=13, fontweight='bold')

    for ax, cls_name, color, w_row in zip(axes, class_names, class_colors, weights):
        ax.set_facecolor('#0d1117')
        ax.bar(range(len(w_row)), w_row, color=color, alpha=0.7, width=1.0)
        ax.axhline(0, color='white', lw=0.5, alpha=0.4)
        ax.set_ylabel(f'{cls_name}', color='white', fontsize=10, fontweight='bold')
        ax.tick_params(colors='white', labelsize=7)
        ax.text(0.98, 0.85, f'mean={w_row.mean():.4f}  std={w_row.std():.4f}',
                transform=ax.transAxes, color='white', fontsize=8, ha='right',
                bbox=dict(boxstyle='round', facecolor='#333', alpha=0.7))
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

    axes[-1].set_xlabel('Feature dimension (0–383)', color='white', fontsize=10)
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, 'classifier_weights.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    info(f"Plot saved → plots/classifier_weights.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  VISUALIZATION 6 — Parameter Budget Pie Chart
# ═══════════════════════════════════════════════════════════════════════════════
def plot_parameter_budget(model):
    """Show parameter distribution across components."""
    components = {
        'Backbone'   : sum(p.numel() for p in model.backbone.parameters()),
        'Attention'  : sum(p.numel() for p in model.attention.parameters()),
        'LayerNorm'  : sum(p.numel() for p in model.norm.parameters()),
        'Classifier' : sum(p.numel() for p in model.classifier.parameters()),
    }
    colors = ['#2196F3', '#9C27B0', '#FF9800', '#4CAF50']
    total  = sum(components.values())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('#0d1117')

    # Pie chart
    ax1.set_facecolor('#0d1117')
    wedges, texts, autotexts = ax1.pie(
        components.values(),
        labels=[f"{k}\n{v:,}" for k, v in components.items()],
        colors=colors, autopct='%1.1f%%',
        textprops={'color': 'white', 'fontsize': 9},
        wedgeprops={'edgecolor': '#0d1117', 'linewidth': 2},
        pctdistance=0.75)
    for at in autotexts:
        at.set_color('white')
        at.set_fontsize(10)
        at.set_fontweight('bold')
    ax1.set_title(f'Parameter Distribution — Total: {total:,}',
                  color='white', fontsize=12, fontweight='bold')

    # Horizontal bar chart
    ax2.set_facecolor('#0d1117')
    bars = ax2.barh(list(components.keys()), list(components.values()),
                    color=colors, edgecolor='white', linewidth=0.3)
    ax2.set_xlabel('Parameter Count', color='white', fontsize=10)
    ax2.tick_params(colors='white')
    for spine in ax2.spines.values():
        spine.set_edgecolor('#444')
    for bar, (name, count) in zip(bars, components.items()):
        pct = count / total * 100
        ax2.text(count + total * 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{count:,} ({pct:.1f}%)', va='center',
                 color='white', fontsize=9)
    ax2.set_title('Absolute Parameter Counts',
                  color='white', fontsize=12, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, 'parameter_budget.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    info(f"Plot saved → plots/parameter_budget.png")

    return components, total


# ═══════════════════════════════════════════════════════════════════════════════
#  VISUALIZATION 7 — Gradient Flow Through Full Model
# ═══════════════════════════════════════════════════════════════════════════════
def check_gradient_flow(model):
    """Verify all parameters receive gradients, including attention."""
    model = model.to(DEVICE).train()
    x = torch.randn(CFG['batch_size'], CFG['in_channels'],
                     CFG['seq_len']).to(DEVICE)
    y = torch.randint(0, CFG['num_classes'],
                      (CFG['batch_size'],)).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()

    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()

    names, means, maxes = [], [], []
    for name, param in model.named_parameters():
        if param.grad is not None:
            names.append(name)
            means.append(param.grad.abs().mean().item())
            maxes.append(param.grad.abs().max().item())

    fig, axes = plt.subplots(2, 1, figsize=(20, 10))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle('Gradient Flow Through SpectrumClassifier\n'
                 '(All bars must be non-zero — zeros = dead layers)',
                 color='white', fontsize=13, fontweight='bold')

    for ax, vals, title, color in zip(
            axes, [means, maxes],
            ['Mean |gradient|', 'Max |gradient|'],
            ['#2196F3', '#E91E63']):
        ax.set_facecolor('#0d1117')

        # Color-code by component
        bar_colors = []
        for n in names:
            if 'backbone' in n:
                bar_colors.append('#2196F3')
            elif 'attention' in n:
                bar_colors.append('#9C27B0')
            elif 'norm' in n:
                bar_colors.append('#FF9800')
            elif 'classifier' in n:
                bar_colors.append('#4CAF50')
            else:
                bar_colors.append(color)

        ax.bar(range(len(vals)), vals, color=bar_colors, alpha=0.75)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=90, fontsize=5, color='white')
        ax.set_ylabel(title, color='white', fontsize=9)
        ax.tick_params(colors='white')
        ax.set_yscale('log')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

        zero_count = sum(1 for v in vals if v < 1e-8)
        status = "✅ All layers active" if zero_count == 0 \
                 else f"❌ {zero_count} dead layers"
        ax.text(0.98, 0.95, status, transform=ax.transAxes,
                color='white', fontsize=10, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.8))

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, 'classifier_gradient_flow.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    info(f"Plot saved → plots/classifier_gradient_flow.png")

    dead_count = sum(1 for v in means if v < 1e-8)

    # Print critical parameter gradients
    print(f"\n  {'Parameter':<45} {'|grad| mean':>14}")
    print("  " + "─" * 63)
    critical_params = [
        'attention.in_proj_weight',
        'attention.out_proj.weight',
        'norm.weight',
        'norm.bias',
        'classifier.weight',
        'classifier.bias',
    ]
    for crit in critical_params:
        for n, m in zip(names, means):
            if crit in n:
                status = "✅" if m > 1e-8 else "❌ DEAD"
                print(f"  {n:<45} {m:>14.2e}  {status}")

    return dead_count


# ═══════════════════════════════════════════════════════════════════════════════
#  STRESS TESTS
# ═══════════════════════════════════════════════════════════════════════════════
def run_stress_tests(model):
    """Run all 4 stress tests. Returns dict of results."""
    results = {}

    # ── Test 1: train/eval mode toggle ───────────────────────────────────
    section("Stress Test 1 — Train/Eval Mode Toggle")
    model = model.to(DEVICE)
    bs = CFG['batch_size']
    x = torch.randn(bs, CFG['in_channels'], CFG['seq_len']).to(DEVICE)
    test1_ok = True

    try:
        model.train()
        out = model(x)
        assert out.shape == (bs, CFG['num_classes']), \
            f"Train mode output shape wrong: {out.shape}"
        assert isinstance(out, torch.Tensor)
        ok(f"Train mode: output shape {tuple(out.shape)} correct")

        model.eval()
        with torch.no_grad():
            out, weights = model(x, return_attention=True)
        assert out.shape == (bs, CFG['num_classes']), \
            f"Eval mode output shape wrong: {out.shape}"
        assert weights.shape == (bs, CFG['num_heads'], 128, 128), \
            f"Attention weights shape wrong: {weights.shape}"

        # Verify attention weights sum to 1 across key dimension
        row_sums = weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4), \
            f"Attention weights don't sum to 1! Max deviation: {(row_sums - 1.0).abs().max():.6f}"
        ok(f"Eval mode: output shape {tuple(out.shape)}, "
           f"attn weights {tuple(weights.shape)}, rows sum to 1.0")
    except AssertionError as e:
        fail(str(e))
        test1_ok = False
    except Exception as e:
        fail(f"Unexpected error: {e}")
        test1_ok = False

    results['train_eval_toggle'] = test1_ok

    # ── Test 2: Untrained prediction bias check ─────────────────────────
    section("Stress Test 2 — Untrained Prediction Bias Check")
    model.eval()
    test2_ok = True

    logits_list = []
    for _ in range(100):
        x_rand = torch.randn(32, CFG['in_channels'], CFG['seq_len']).to(DEVICE)
        with torch.no_grad():
            logits_out = model(x_rand)
            probs = torch.softmax(logits_out, dim=1)
            logits_list.append(probs.mean(dim=0))

    mean_probs = torch.stack(logits_list).mean(dim=0).cpu().numpy()
    class_names = ['BUSY', 'FREE', 'JAMMED']
    print(f"\n  Mean class probabilities over 100 batches × 32 samples:")
    for cls_name, p in zip(class_names, mean_probs):
        bar = "█" * int(p * 60)
        deviation = abs(p - 1/3)
        status = "✅" if deviation < 0.10 else "⚠️  biased"
        print(f"    {cls_name:<8} {p:.4f}  ({deviation:+.4f} from 0.333)  {bar}  {status}")
        if deviation >= 0.10:
            test2_ok = False

    if test2_ok:
        ok("No class bias — all near 33.3%")
    else:
        fail("Class bias detected — may affect training convergence")

    results['prediction_bias'] = test2_ok

    # ── Test 3: Batch size independence ──────────────────────────────────
    section("Stress Test 3 — Batch Size Independence")
    model.eval()
    test3_ok = True
    print()

    for bs_test in [1, 4, 8, 16, 32, 64]:
        try:
            x_bs = torch.randn(bs_test, CFG['in_channels'],
                                CFG['seq_len']).to(DEVICE)
            with torch.no_grad():
                out_bs = model(x_bs)
            expected = (bs_test, CFG['num_classes'])
            actual = tuple(out_bs.shape)
            assert actual == expected, f"Expected {expected}, got {actual}"
            if torch.isnan(out_bs).any():
                fail(f"  Batch {bs_test:3d} → NaN detected!")
                test3_ok = False
            else:
                print(f"  Batch {bs_test:3d} → {actual}  ✅")
        except Exception as e:
            fail(f"  Batch {bs_test:3d} → {e}")
            test3_ok = False

    results['batch_independence'] = test3_ok

    # ── Test 4: Determinism check ───────────────────────────────────────
    section("Stress Test 4 — Determinism Check (eval mode)")
    torch.manual_seed(42)
    model.eval()
    test4_ok = True

    x_det = torch.randn(4, CFG['in_channels'], CFG['seq_len']).to(DEVICE)
    with torch.no_grad():
        out1 = model(x_det)
        out2 = model(x_det)

    if torch.allclose(out1, out2, atol=1e-6):
        ok("Deterministic — identical outputs for identical inputs")
    else:
        max_diff = (out1 - out2).abs().max().item()
        fail(f"Non-deterministic! Max diff: {max_diff:.2e}")
        test4_ok = False

    results['determinism'] = test4_ok

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN — RUN ALL CHECKS AND VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':

    banner("CogniRad  Task 1.7 — SpectrumClassifier with Attention")

    print(f"\n  Device       : {DEVICE}")
    print(f"  PyTorch      : {torch.__version__}")
    print(f"  Plots        : {PLOTS_DIR}")
    print(f"  embed_dim    : {CFG['embed_dim']}")
    print(f"  num_heads    : {CFG['num_heads']}")
    print(f"  per-head dim : {CFG['embed_dim'] // CFG['num_heads']}")

    # ── Instantiate ──────────────────────────────────────────────────────
    model = SpectrumClassifier()
    info(f"SpectrumClassifier instantiated — embed_dim={model.embed_dim}")

    # ── STEP 1: Shape Propagation Trace ──────────────────────────────────
    section("Step 1 — Shape Propagation Trace")
    shapes_ok = verify_all_shapes(model)

    # ── STEP 2: Visualizations ──────────────────────────────────────────
    section("Step 2 — Architecture Flow Diagram")
    plot_architecture_flow(model)

    section("Step 3 — Attention Head Heatmaps")
    print("  Generating 8 heads × 3 classes = 24 heatmaps...\n")
    plot_attention_heatmaps(model)

    section("Step 4 — Attention Entropy Analysis")
    plot_attention_entropy(model)

    section("Step 5 — Attention Position Importance")
    plot_attention_per_position(model)

    section("Step 6 — Classifier Weight Distribution")
    plot_classifier_weights(model)

    section("Step 7 — Parameter Budget Analysis")
    param_components, total_params = plot_parameter_budget(model)
    print(f"\n  Parameter Budget:")
    for comp, count in param_components.items():
        pct = count / total_params * 100
        print(f"    {comp:<12} : {count:>10,}  ({pct:5.1f}%)")
    print(f"    {'TOTAL':<12} : {total_params:>10,}")

    # ── STEP 3: Gradient Flow ───────────────────────────────────────────
    section("Step 8 — Gradient Flow Through Full Model")
    print("  Running forward+backward to verify all layers receive gradients...\n")
    dead_layers = check_gradient_flow(model)

    # ── STEP 4: Stress Tests ────────────────────────────────────────────
    banner("Stress Tests")
    test_results = run_stress_tests(model)

    # ═════════════════════════════════════════════════════════════════════
    #  FINAL SUMMARY
    # ═════════════════════════════════════════════════════════════════════
    banner("TASK 1.7 — VERIFICATION SUMMARY")

    ed = CFG['embed_dim']
    nh = CFG['num_heads']
    bs = CFG['batch_size']

    attn_sum_ok = True
    try:
        model.eval()
        x_check = torch.randn(bs, 2, 1024).to(DEVICE)
        with torch.no_grad():
            _, w_check = model(x_check, return_attention=True)
        row_sums = w_check.sum(dim=-1)
        attn_sum_ok = torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)
    except Exception:
        attn_sum_ok = False

    param_total = sum(p.numel() for p in model.parameters())

    checklist = [
        (f"embed_dim % num_heads == 0",                True),
        (f"  ({ed} ÷ {nh} = {ed // nh})",              True),
        (f"Shape: backbone out ({bs}, {ed}, 128)",      shapes_ok),
        (f"Shape: post-transpose ({bs}, 128, {ed})",    shapes_ok),
        (f"Shape: attention out ({bs}, 128, {ed})",     shapes_ok),
        (f"Shape: attn weights ({bs}, {nh}, 128, 128)", shapes_ok),
        (f"Shape: after norm ({bs}, 128, {ed})",        shapes_ok),
        (f"Shape: after GAP ({bs}, {ed})",              shapes_ok),
        (f"Shape: logits ({bs}, 3)",                    shapes_ok),
        ("Attention weights sum to 1.0",               attn_sum_ok),
        (f"Gradient flow — {dead_layers} dead layers",  dead_layers == 0),
        ("Untrained probs near 33.3% each",            test_results['prediction_bias']),
        ("Determinism in eval mode",                   test_results['determinism']),
        ("Batch size stress test",                     test_results['batch_independence']),
        ("Train/eval mode toggle",                     test_results['train_eval_toggle']),
    ]

    print()
    print(f"  {'Check':<50} {'Status'}")
    print(f"  {'─' * 60}")
    for label, passed in checklist:
        icon = "✅" if passed else "❌"
        print(f"  {label:<50} {icon}")

    print(f"\n  Total parameters: {param_total:,}")
    for comp, count in param_components.items():
        pct = count / total_params * 100
        print(f"    {comp:<12} : {count:>10,}  ({pct:5.1f}%)")

    print(f"""
  Saved plots:
    → plots/architecture_flow.png
    → plots/attention_heatmaps.png
    → plots/attention_entropy.png
    → plots/attention_position_importance.png
    → plots/classifier_weights.png
    → plots/parameter_budget.png
    → plots/classifier_gradient_flow.png

  Output feeds into:
    → Task 1.8 — Training (RadioML dataset + training loop)
""")

    all_passed = all(p for _, p in checklist)
    if all_passed:
        banner("Task 1.7 Complete — SpectrumClassifier ready for Task 1.8 ✅")
    else:
        banner("Task 1.7 — SOME CHECKS FAILED — review above ❌")

    print()
