# Re-IQA: Unsupervised Learning for Image Quality Assessment - Complete Codebase Documentation

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture Overview](#architecture-overview)
3. [Directory Structure](#directory-structure)
4. [Detailed File Explanations](#detailed-file-explanations)
5. [Key Concepts](#key-concepts)
6. [Data Flow](#data-flow)
7. [Training Pipeline](#training-pipeline)

---

## Project Overview

**Re-IQA** is an unsupervised learning framework for **No-Reference Image Quality Assessment (NR-IQA)** in the wild. It was published at IEEE/CVF CVPR 2023 and achieves state-of-the-art performance across multiple IQA databases.

### Key Features:
- **Unsupervised Learning**: Does not require labeled quality scores
- **Dual Module Approach**: 
  - Quality-Aware Module (trained on distorted images)
  - Content-Aware Module (trained on clean ImageNet images)
- **Momentum Contrast (MoCo)**: Uses self-supervised contrastive learning
- **Multi-Scale Processing**: Analyzes images at multiple scales
- **Distributed Training**: Supports multi-node, multi-GPU training via DDP

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Re-IQA Architecture                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input Image                                                    │
│      │                                                          │
│      ├──────────────────┬──────────────────┐                    │
│      │                  │                  │                    │
│  Quality-Aware      Content-Aware     Linear Regressor         │
│  Module             Module                                      │
│  (KADIS-700k,       (ImageNet)                                  │
│   UGC data)                                                     │
│      │                  │                  │                    │
│      └──────────────────┼──────────────────┘                    │
│                         │                                       │
│                   Feature Fusion (Concatenation)                │
│                         │                                       │
│                    IQA Score (0-1)                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Two Training Stages:

**Stage 1: Quality-Aware Module**
- Trains on artificially distorted images (KADIS-700k) and natural images (COCO, AVA, etc.)
- Learns to recognize and extract features of image quality
- Uses MoCo-v2 with custom augmentations and distortions
- Output: Quality-aware representations

**Stage 2: Content-Aware Module**
- Trains on ImageNet using vanilla MoCo-v2
- Learns generic image content features
- Supports two different modality streams when needed

**Stage 3: Linear Regression**
- Concatenates quality-aware + content-aware features
- Trains a simple ridge/elastic net regressor
- Maps combined features to IQA scores

---

## Directory Structure

```
ReIQA/
├── csv_files/
│   └── moco_train.csv                 # Paths to training images
│
├── datasets/                           # Data loading and processing
│   ├── __pycache__/
│   ├── RandAugment.py                # Data augmentation library
│   ├── dataset.py                    # Dataset classes
│   ├── iqa_distortions.py            # Image distortion functions
│   └── util.py                       # Dataset utility functions
│
├── learning/                           # Training logic
│   ├── __pycache__/
│   ├── base_trainer.py               # Base class for trainers
│   ├── contrast_trainer.py           # MoCo/Contrastive learning trainer
│   ├── linear_trainer.py             # Linear regression trainer
│   └── util.py                       # Training utilities
│
├── memory/                             # Memory bank implementation
│   ├── __pycache__/
│   ├── alias_multinomial.py          # Alias multinomial sampling
│   ├── build_memory.py               # Factory for memory types
│   ├── mem_bank.py                   # Memory bank implementation
│   └── mem_moco.py                   # MoCo-style memory cache
│
├── moco/                               # MoCo framework
│   ├── __pycache__/
│   ├── __init__.py
│   ├── builder.py                    # MoCo model builder
│   ├── loader.py                     # Data loading utilities
│   ├── optimizer.py                  # LARS optimizer
│   └── __init__.py
│
├── networks/                           # Neural network architectures
│   ├── __pycache__/
│   ├── build_backbone.py             # Factory for model creation
│   ├── build_linear.py               # Linear layers builder
│   ├── resnet.py                     # ResNet implementation
│   ├── resnet_cmc.py                 # CMC variant (Cross-Modal)
│   ├── resnest.py                    # ResNest variant
│   └── util.py                       # Network utilities
│
├── options/                            # Configuration/argument parsing
│   ├── __pycache__/
│   ├── base_options.py               # Base option parser
│   ├── test_options.py               # Test configuration
│   └── train_options.py              # Training configuration
│
├── sample_images/                      # Example images for inference
│
├── main_contrast.py                    # Main training script
├── demo_quality_aware_feats.py        # Extract quality-aware features
├── demo_content_aware_feats.py        # Extract content-aware features
├── requirements.txt                    # Python dependencies
├── LICENSE
└── README.MD                           # Project readme

```

---

## Detailed File Explanations

### 1. **options/** - Configuration Management

#### `options/base_options.py`
```python
"""
Base configuration class that defines all command-line arguments
Inherited by TrainOptions and TestOptions
"""

class BaseOptions(object):
    def __init__(self):
        # Predefined configs for different methods (InsDis, CMC, MoCo, etc.)
        self.override_dict = {
            'InsDis':  ['RGB', False, 'bank', 'A', 'linear', 0.07],
            'MoCov2':  ['RGB', False, 'moco', 'B', 'mlp',    0.2],
            # ... other methods
        }
        
    def initialize(self, parser):
        # Add all training parameters
        parser.add_argument('--csv_path', type=str, default='./contrast_train.csv',
                           help='Path to CSV file with image paths')
        parser.add_argument('--epochs', type=int, default=30,
                           help='Number of training epochs')
        parser.add_argument('--batch_size', type=int, default=60,
                           help='Batch size for training')
        parser.add_argument('--n_aug', type=int, default=7,
                           help='Number of augmentations per image')
        # ... many more parameters
```

**Key Parameters:**
- `--method`: Choose predefined training method (MoCov2 for Re-IQA)
- `--batch_size`: Batch size (630 for Re-IQA training)
- `--epochs`: Number of epochs to train (40 for Re-IQA)
- `--learning_rate`: Learning rate (12 for Re-IQA with batch scaling)
- `--optimizer`: SGD, AdamW, or LARS
- `--n_aug`: Number of augmentations per image
- `--n_distortions`: Number of image distortions to apply

#### `options/train_options.py`
Inherits from BaseOptions and adds training-specific arguments like:
- Resume checkpoint
- Learning rate decay settings
- Warmup settings
- Mixed precision (AMP) settings

#### `options/test_options.py`
Inherits from BaseOptions and adds testing-specific arguments

---

### 2. **networks/** - Neural Network Architectures

#### `networks/build_backbone.py` - Model Factory

```python
class RGBSingleHead(nn.Module):
    """
    Single-head RGB model with projection head
    
    Architecture:
    Input -> ResNet50 (encoder) -> Projection Head -> Features
    
    Parameters:
    - name: ResNet variant (resnet50, resnet50x2, resnet50x4)
    - head: Type of projection head ('linear' or 'mlp')
    - feat_dim: Output feature dimension (usually 128)
    """
    
    def __init__(self, name='resnet50', head='linear', feat_dim=128):
        # name parsing for width multipliers (x2, x4)
        name, width = self._parse_width(name)
        dim_in = int(2048 * width)  # ResNet50 base has 2048 channels
        
        # Build encoder (ResNet backbone)
        self.encoder = model_dict[name](width=width)
        
        # Build projection head
        if head == 'linear':
            # Simple linear projection + L2 normalization
            self.head = nn.Sequential(
                nn.Linear(dim_in, feat_dim),
                Normalize(2)  # L2 normalization
            )
        elif head == 'mlp':
            # MLP with hidden layer (MoCov2 style)
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),      # hidden layer
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim),    # output layer
                Normalize(2)                     # L2 normalization
            )

class RGBMultiHeads(RGBSingleHead):
    """
    Multi-head RGB model with Jigsaw puzzle branch
    Adds auxiliary jigsaw prediction head for PIRL-style learning
    """
```

**Model Modes:**
- `mode=0`: Normal encoder (training) - applies projection head
- `mode=1`: Momentum encoder (target computation) - applies projection head
- `mode=2`: Testing mode - only returns encoder features (no head)

#### `networks/resnet.py` - ResNet Implementation

```python
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution for dimension adjustment"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    """
    ResNet basic building block
    
    Structure:
    Input -> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> Add(skip) -> ReLU -> Output
    
    expansion = 1 (output channels = input channels)
    """

class Bottleneck(nn.Module):
    """
    ResNet bottleneck block (for ResNet50+)
    
    Structure:
    Input -> Conv1x1 (reduce) -> Conv3x3 -> Conv1x1 (expand) -> BN -> Add(skip) -> ReLU -> Output
    
    expansion = 4 (output channels = 4 * input channels)
    Uses fewer parameters than using 3x3 convolutions
    """
```

**ResNet Backbone: 4 Stages**
1. **Initial Conv**: Conv2d(3, 64, 7×7, stride=2) + BN + ReLU
2. **Layer1** (64 channels): Multiple residual blocks
3. **Layer2** (128 channels): Downsample, multiple residual blocks
4. **Layer3** (256 channels): Downsample, multiple residual blocks
5. **Layer4** (512 channels): Downsample, multiple residual blocks
6. **Global Average Pooling**: Compress spatial dimensions
7. **2048-dim Features**: Output from ResNet50

#### `networks/build_linear.py` - Linear Classifier Builder

```python
"""
Builds simple linear classifiers for:
1. Supervised fine-tuning
2. Image classification (CIFAR, STL10)
3. Custom linear regression heads
"""
```

---

### 3. **datasets/** - Data Processing

#### `datasets/dataset.py` - Dataset Classes

```python
class ImageFolderInstance(datasets.ImageFolder):
    """
    PyTorch ImageFolder variant that returns both image and its index
    
    Returns:
    - img: Tensor of shape (C, H, W)
    - index: Integer index of the image in dataset
    - jigsaw_image (optional): Jigsaw-transformed image for PIRL
    
    Attributes:
    - two_crop: Apply transformation twice to get two augmented views
    - jigsaw_transform: Apply jigsaw puzzle transformation
    """

class IQAImageClass(data.Dataset):
    """
    Custom dataset for IQA training with quality distortions
    
    Key Features:
    - Loads images from CSV file with paths
    - Applies multiple augmentations per image (n_aug)
    - Applies multiple scales (n_scale)
    - Applies multiple distortions (n_distortions)
    - Generates crops with controlled overlap (Overlap Link Area - OLA)
    
    Parameters:
    - csv_path: Path to CSV with image paths
    - n_aug: Number of augmentations per image (default: 7)
    - n_scale: Number of scales (1=original, 2=original+half-scale)
    - n_distortions: Number of distortion types
    - patch_size: Crop size (default: 224)
    - swap_crops: Swap crop pairs for diversity
    - min_OLA: Minimum overlap ratio (10%)
    - max_OLA: Maximum overlap ratio (30%)
    """
    
    def iqa_transformations(self, choice, im, level):
        """
        Apply different distortion types
        
        Distortion Types:
        1. Gaussian Blur: Blurs image with Gaussian kernel
        2. Lens Blur: Simulates optical lens blur
        3. Color Diffuse: Adds color diffusion
        4. Color Shift: Shifts color channels
        5. Color Saturate: Changes saturation
        6. Saturate: Adjusts overall saturation
        """
```

#### `datasets/iqa_distortions.py` - Image Distortion Functions

```python
def imblurgauss(im, level):
    """
    Gaussian blur distortion
    levels = [0.1, 0.5, 1, 2, 5] - sigma values
    """

def imblurlens(im, level):
    """
    Lens blur approximation using circular morphology
    levels = [1, 2, 4, 6, 8] - radius values
    Implements lens point spread function
    """

def imcolordiffuse(im, level):
    """Apply color diffusion effect"""

def imcolorshift(im, level):
    """Shift image color channels"""

def imcolorsaturate(im, level):
    """Adjust color saturation"""

def imsaturate(im, level):
    """Adjust overall saturation"""
```

#### `datasets/util.py` - Dataset Utility Functions

```python
def build_contrast_loader(args, ngpus_per_node):
    """
    Build data loader for contrastive learning
    
    Returns:
    - train_dataset: IQAImageClass or ImageFolderInstance
    - train_loader: DataLoader with distributed sampler
    - train_sampler: DistributedSampler for synchronization
    
    Features:
    - Distributed sampling for DDP
    - Two-crop augmentation for contrastive learning
    - Optional jigsaw transformation
    """
```

#### `datasets/RandAugment.py`
Implements RandAugment data augmentation:
- Randomly selects augmentation operations
- Applies with random magnitude strength
- Improves robustness of learned features

---

### 4. **memory/** - Memory Bank for Contrastive Learning

#### `memory/mem_moco.py` - MoCo Memory Implementations

```python
class BaseMoCo(nn.Module):
    """
    Base class for MoCo-style memory cache
    
    Key Concepts:
    - K: Size of memory bank (typically 65536)
    - T: Temperature parameter for softmax (typically 0.07-0.2)
    - index: Circular pointer to update memory
    
    Memory Update Strategy (FIFO):
    When new samples arrive, oldest samples are replaced
    """
    
    def _update_pointer(self, bsz):
        """Circular pointer update: index = (index + batch_size) % K"""
        self.index = (self.index + bsz) % self.K
    
    def _update_memory(self, k, queue):
        """
        Update memory buffer with new keys
        
        Process:
        1. Calculate output indices: (index + i) % K for each sample
        2. Use index_copy_ to place new samples in memory
        """
    
    def _compute_logit(self, q, k, queue):
        """
        Compute contrastive logits
        
        Formula:
        logits = [pos_score | neg_scores...] / temperature
        
        Where:
        - pos_score = dot(query, positive_key) [1 score]
        - neg_scores = dot(query, memory_bank) [K scores]
        """

class RGBMoCo(BaseMoCo):
    """
    Single-modal MoCo memory
    
    Memory: [K, n_dim] - stores n_dim dimensional embeddings
    
    forward(q, k, q_jig=None, all_k=None):
        q: Query embeddings [batch_size, n_dim]
        k: Key embeddings [batch_size, n_dim]
        q_jig: Jigsaw query embeddings (optional)
        all_k: All gathered keys from multi-GPU (for synchronization)
        
        Returns:
        - logits: Contrastive predictions [batch_size, K+1]
        - labels: All zeros [batch_size] (positive is index 0)
        - logits_jig: Jigsaw predictions (if q_jig provided)
    """
    
    def forward(self, q, k, q_jig=None, all_k=None):
        bsz = q.size(0)
        k = k.detach()  # Keys don't need gradients
        
        # Clone memory to prevent in-place modification issues
        queue = self.memory.clone().detach()
        logits = self._compute_logit(q, k, queue)
        
        # Labels: positive key is always at index 0
        # Others (65536) are negatives
        labels = torch.zeros(bsz, dtype=torch.long).cuda()
        
        # Update memory with new keys
        all_k = all_k if all_k is not None else k
        self._update_memory(all_k, self.memory)
        self._update_pointer(all_k.size(0))
        
        return logits, labels

class CMCMoCo(BaseMoCo):
    """
    Cross-Modal Contrastive (CMC) memory for two modalities
    
    Maintains separate memory banks for two modalities
    Example: RGB vs Grayscale comparison
    """
```

#### `memory/mem_bank.py` - Memory Bank (alternative to MoCo queue)

```python
"""
Memory bank implementation for instance discrimination
Similar to MoCo but with different update strategy
"""
```

#### `memory/build_memory.py` - Memory Factory

```python
def build_mem(args, n_sample):
    """
    Factory function to create appropriate memory module
    
    Returns:
    - RGBMoCo: For single modal training
    - CMCMoCo: For cross-modal training
    
    Parameters:
    - args.nce_k: Size of memory bank (K)
    - args.nce_t: Temperature (T)
    - args.modal: RGB or CMC
    """
```

#### `memory/alias_multinomial.py`
Implements efficient sampling from multinomial distribution using alias method

---

### 5. **learning/** - Training Components

#### `learning/base_trainer.py` - Base Trainer Class

```python
class BaseTrainer(object):
    """
    Base class for all trainers
    
    Core Responsibilities:
    1. Distributed training setup (DDP)
    2. Learning rate scheduling
    3. Checkpoint management
    4. Logging to TensorBoard
    """
    
    def init_ddp_environment(self, gpu, ngpus_per_node):
        """
        Initialize Distributed Data Parallel (DDP) environment
        
        Process:
        1. Set current GPU device
        2. Initialize process group
        3. Create local groups for ShuffleBN
        """
    
    def adjust_learning_rate(self, optimizer, epoch):
        """
        Adjust learning rate based on epoch
        
        Strategies:
        - Cosine annealing: LR = eta_min + (LR - eta_min) * (1 + cos(pi*epoch/epochs))/2
        - Step decay: LR = LR * decay_rate^steps
        """
    
    def warmup_learning_rate(self, epoch, batch_id, total_batches, optimizer):
        """
        Learning rate warmup in first few epochs
        Gradually increase from warmup_from to warmup_to
        """
```

#### `learning/contrast_trainer.py` - Contrastive Learning Trainer

```python
class ContrastTrainer(BaseTrainer):
    """
    Trainer for MoCo-style contrastive learning
    
    Key Methods:
    1. train(): Main training loop
    2. wrap_up(): Prepare models for distributed training
    3. resume_model(): Load checkpoint
    4. broadcast_memory(): Synchronize memory across GPUs
    """
    
    def wrap_up(self, model, model_ema, optimizer):
        """
        Wrap models for distributed training
        
        Steps:
        1. Move models to current GPU
        2. Apply mixed precision (optional)
        3. Wrap with DistributedDataParallel (DDP)
        4. Initialize momentum encoder
        """
    
    def broadcast_memory(self, contrast):
        """
        Synchronize memory buffer across all processes
        Important for consistent memory across GPUs
        """
    
    def train(self, epoch, train_loader, model, model_ema, contrast, criterion, optimizer):
        """
        Single epoch training loop
        
        For each batch:
        1. Get augmented image pair (x1, x2)
        2. Forward pass through model
        3. Compute logits via contrast memory
        4. Compute loss (CrossEntropyLoss)
        5. Backward pass and optimizer step
        6. Update momentum encoder
        
        Returns: [loss, accuracy, jig_loss, jig_acc]
        """
```

#### `learning/linear_trainer.py`
Trainer for linear regression on frozen features

#### `learning/util.py` - Training Utilities

```python
def accuracy(output, target, topk=(1,)):
    """Compute top-k accuracy"""

class AverageMeter(object):
    """Maintains running average and current value"""
```

---

### 6. **moco/** - Momentum Contrast Framework

#### `moco/builder.py` - MoCo Model

```python
class MoCo(nn.Module):
    """
    MoCo (Momentum Contrast) Model
    
    Architecture:
    - base_encoder: Query encoder (updated by gradient)
    - momentum_encoder: Key encoder (updated by momentum)
    - Projection MLPs: Convert features to embedding space
    
    Key Property:
    momentum_encoder.params = m * momentum_encoder.params + (1-m) * base_encoder.params
    """
    
    def __init__(self, base_encoder, dim=256, mlp_dim=4096, T=1.0):
        # Create two encoders with same architecture
        self.base_encoder = base_encoder(num_classes=mlp_dim)
        self.momentum_encoder = base_encoder(num_classes=mlp_dim)
        
        # Initialize momentum encoder with base encoder weights
        for param_b, param_m in zip(self.base_encoder.parameters(), 
                                     self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)
            param_m.requires_grad = False  # Don't update by gradient
    
    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """
        Momentum update of momentum encoder
        
        Formula: encoder_m ← m * encoder_m + (1-m) * encoder
        This creates exponential moving average of base encoder
        """
    
    def forward(self, x1, x2, m):
        """
        Forward pass for training
        
        Process:
        1. Compute query features from x1 and x2 (with gradients)
        2. Update momentum encoder
        3. Compute key features from x1 and x2 (no gradients)
        4. Compute contrastive loss
        """

class MoCo_ResNet(MoCo):
    """
    MoCo implementation using ResNet backbone
    Builds projection and predictor MLPs
    """
```

#### `moco/optimizer.py` - LARS Optimizer

```python
"""
LARS (Layer-wise Adaptive Rate Scaling) Optimizer
Used for large-batch training (batch size > 256)

Key Feature:
- Adapts learning rate per layer based on weight norm
- Prevents layer from becoming too large during updates
- formula: lr_i = base_lr * ||w_i|| / (||∇L(w_i)|| + λ*||w_i||)
"""
```

#### `moco/loader.py` - Data Loading Utilities

```python
"""
Utilities for:
1. Creating train/val loaders
2. Handling distributed sampling
3. Collating batch samples
"""
```

---

### 7. **Main Entry Points**

#### `main_contrast.py` - Training Script

```python
"""
Main training script using MoCo framework
Implements distributed training with DDP
"""

def main():
    """
    Main function:
    1. Parse arguments
    2. Setup distributed environment
    3. Spawn worker processes
    """

def main_worker(gpu, ngpus_per_node, args):
    """
    Worker function for each GPU process
    
    Steps:
    1. Initialize DDP environment
    2. Build model and model_ema
    3. Build dataset and loader
    4. Build memory bank
    5. Create optimizer (SGD/AdamW/LARS)
    6. Training loop:
        For each epoch:
        - Adjust learning rate
        - Train for one epoch
        - Log metrics
        - Save checkpoint
    """
    
    # Training loop pseudo-code
    for epoch in range(start_epoch, args.epochs + 1):
        # Set epoch for sampler (ensures different random samples per epoch)
        train_sampler.set_epoch(epoch)
        
        # Adjust learning rate based on epoch
        trainer.adjust_learning_rate(optimizer, epoch)
        
        # Run one training epoch
        outs = trainer.train(epoch, train_loader, model, model_ema,
                            contrast, criterion, optimizer)
        
        # Log to tensorboard
        trainer.logging(epoch, outs, optimizer.param_groups[0]['lr'])
        
        # Save checkpoint
        trainer.save(model, model_ema, contrast, optimizer, epoch)
```

**Key Command:**
```bash
python main_contrast.py \
    --method MoCov2 \
    --head mlp \
    --batch_size 630 \
    --epochs 40 \
    --learning_rate 12 \
    --optimizer LARS \
    --csv_path ./csv_files/moco_train.csv \
    --model_path ./expt0 \
    --multiprocessing-distributed \
    --world-size 6
```

#### `demo_quality_aware_feats.py` - Extract Quality-Aware Features

```python
"""
Extract quality-aware features from trained model
"""

def run_inference():
    """
    Process:
    1. Load trained quality-aware model checkpoint
    2. Load image and half-scale version
    3. Pass through encoder (no projection head)
    4. Concatenate full-scale and half-scale features
    5. Save to numpy file
    
    Output: [2048 * 2,] dimensional feature vector
    """
    
    # Load model
    model, _ = build_model(args)
    checkpoint = torch.load('./reiqa_ckpts/quality_aware_r50.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    
    # Load image
    image = Image.open(img_path).convert('RGB')
    image2 = image.resize((image.size[0]//2, image.size[1]//2))  # Half-scale
    
    # Convert to tensors
    img1 = transforms.ToTensor()(image).unsqueeze(0)
    img2 = transforms.ToTensor()(image2).unsqueeze(0)
    
    # Extract features (mode=2 means testing mode - only encoder)
    with torch.no_grad():
        feat1 = model.module.encoder(img1.to(args.device))
        feat2 = model.module.encoder(img2.to(args.device))
        # Concatenate: [batch_size, 4096]
        feat = torch.cat((feat1, feat2), dim=1).detach().cpu().numpy()
    
    # Save as numpy
    np.save(save_path + filename + '_quality_aware_features.npy', feat)
```

#### `demo_content_aware_feats.py` - Extract Content-Aware Features

Similar to quality-aware but loads content-aware model from ImageNet training.

---

## Key Concepts

### 1. **Contrastive Learning**
- **Positive Pair**: Same image with two different augmentations
- **Negative Pairs**: Different images in the batch + memory bank
- **Objective**: Maximize similarity between positive pairs, minimize with negatives

### 2. **Momentum Contrast (MoCo)**
```
Regular Gradient: ∇L w.r.t query and key (both updated by gradient descent)
Momentum Contrast: Query encoder updated by gradient, 
                   Key encoder updated by exponential moving average
Benefit: More stable features in memory bank
```

### 3. **Memory Bank**
- **Size**: K = 65536 (fixed size queue)
- **Update**: Circular FIFO queue (oldest replaced by newest)
- **Temperature**: T = 0.07-0.2 (controls softmax sharpness)
- **Purpose**: Store large population of negative keys

### 4. **Multi-Scale Processing**
- **Full Scale**: Original image resolution
- **Half Scale**: 50% of original resolution
- **Feature Concatenation**: [full_features, half_features]
- **Rationale**: Multi-scale analysis captures both fine and coarse details

### 5. **Image Quality Distortions**
Used during training to teach quality awareness:
1. **Blur**: Gaussian blur, lens blur
2. **Color**: Saturation, color shift, diffusion
3. **Levels**: [0, 1, 2, 3, 4] severity levels

### 6. **Distributed Data Parallel (DDP)**
- **Process 0**: Master process (saves checkpoints, logs)
- **Processes 1-N**: Worker processes
- **Synchronization**: AllReduce for loss averaging, memory broadcast
- **Batch Size**: Effective batch = local_batch * num_processes * num_nodes

### 7. **L2 Normalization**
```
normalized_feature = feature / ||feature||_2
Ensures all features lie on unit hypersphere
Enables cosine similarity: sim(a,b) = dot(normalize(a), normalize(b))
```

---

## Data Flow

### Training Data Pipeline

```
CSV File with image paths
     ↓
IQAImageClass Dataset
     ├─ Load image from disk
     ├─ Apply distortion (blur, color, etc.)
     ├─ Extract multiple crops with overlap control
     ├─ Apply augmentation (RandAugment, rotation, flip, etc.)
     └─ Returns: (augmented_image_1, augmented_image_2, index)
        [Note: May return jigsaw variant too]
     ↓
DistributedSampler (ensures no overlap across processes)
     ↓
DataLoader (batch_size=630/6=105 per GPU)
     ↓
Training Loop
```

### Forward Pass (Training)

```
Batch of Images [batch_size=105, 3, 224, 224]
     ↓
Split into two augmented views:
  x1 [batch, 3, 224, 224]
  x2 [batch, 3, 224, 224]
     ↓
Process x1 (query):
  encoder(x1) → [batch, 2048]
  projection_head(features) → q1 [batch, 128]
     ↓
Process x2 (with momentum):
  momentum_encoder(x2) → [batch, 2048] (no gradient)
  projection_head(features) → k2 [batch, 128] (no gradient)
     ↓
Compute similarity logits:
  logits = [dot(q1, k2) | dot(q1, memory)] / T
         [batch, 65537]  (1 positive + 65536 negatives)
     ↓
Contrastive Loss:
  CrossEntropyLoss(logits, labels=0)
  Labels all 0 because positive is at index 0
     ↓
Backward + Optimizer Update
     ↓
Momentum Update:
  momentum_encoder.params = 0.999 * momentum_encoder.params + 0.001 * encoder.params
```

### Testing/Inference

```
Single Image
     ↓
encoder(image) → [batch, 2048]
     ↓
[NO Projection Head in testing mode]
     ↓
Output: Raw 2048-d features
     ↓
Quality-aware features: 4096-d (full + half scale)
Content-aware features: 4096-d (full + half scale)
     ↓
Concatenate: [4096 + 4096] = 8192-d
     ↓
Linear Regressor: 8192-d → 1-d (IQA score)
```

---

## Training Pipeline

### Phase 1: Quality-Aware Module Training

**Configuration:**
```python
batch_size = 630 (across 6 nodes × 6 GPUs = 36 GPUs)
epochs = 40
learning_rate = 12 (scaled by batch size)
optimizer = LARS
n_aug = 11 (number of augmentations)
n_scale = 2 (full scale + half scale)
n_distortions = 1
patch_size = 160
```

**Data:**
- KADIS-700k: 700K distorted images
- COCO: 330K images
- AVA: ~250K natural images
- Blur Dataset: 631 images
- VOC2012: ~50K images

**Augmentations:**
- RandAugment (random operations)
- Geometric (rotation, flip, crop)
- Distortions (blur, color shift, saturation)
- Multi-scale (full scale + 0.5x scale)

**Learning Rate Schedule:**
```
Warmup: 0 → 12 over first epoch
Decay: Cosine annealing to 0.12 over 40 epochs
```

**Loss Function:**
```
L = CrossEntropyLoss(logits, labels)
Where logits include 1 positive + 65536 negatives
```

### Phase 2: Content-Aware Module Training

Uses vanilla MoCo-v2 on ImageNet with default settings

### Phase 3: Linear Regression

```python
features = [quality_aware_features, content_aware_features]
# Both ~4096-d, total ~8192-d

# Train on IQA dataset with labels
regressor = Ridge(alpha=0.001)  # or ElasticNet
regressor.fit(features, quality_scores)

# Output: IQA scores [0, 1]
```

---

## Common Issues and Debugging

### Memory Broadcast Not Synced
```python
# Solution: Call broadcast_memory after wrapping models
trainer.broadcast_memory(contrast)
```

### Distributed Training Hangs
```python
# Often caused by uneven batch distribution
# Ensure: batch_size % (world_size) == 0
```

### Poor Final IQA Scores
```python
# Try:
1. Train for 40 epochs (not 25)
2. Use LARS optimizer (better than SGD)
3. Tune linear regressor (Ridge/ElasticNet)
4. Adjust learning rate scaling
```

---

## References

- **Paper**: Saha et al., "Re-IQA: Unsupervised Learning for Image Quality Assessment in the Wild", CVPR 2023
- **MoCo**: He et al., "Momentum Contrast for Unsupervised Visual Representation Learning"
- **LARS**: You et al., "Large Batch Optimization for Deep Learning: Training BERT in 76 Minutes"

---

## Conclusion

**Re-IQA** combines:
1. **MoCo-v2** for self-supervised feature learning
2. **Quality-aware** module trained on distorted images
3. **Content-aware** module trained on natural images
4. **Linear regression** to map combined features to IQA scores

The architecture is modular, extensible, and achieves state-of-the-art results on multiple IQA benchmarks!

