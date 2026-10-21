# -*- coding: utf-8 -*-
"""Train BudgetUniRepLKNet and save its best validation checkpoint."""

# Model architecture (shapes omit the batch dimension):
# Input: RGB resized with bicubic interpolation and antialiasing so the shorter
# side is 144, then center-cropped to 128 x 128. During training, apply random
# horizontal flip, affine transforms, and color jitter before tensor conversion
# and normalization (mean 0.5, std 1.0 per channel), then random erasing.
# Stem: overlapping Conv 3 x 3, 3 -> 14, stride 2, padding 1 -> BN:
# 14 x 64 x 64 (406 params). No activation in the stem.

# Each custom UniRepLKNet-inspired block has two parallel spatial branches:
#   (1) DW Conv 3 x 3, dilation 1, padding 1 -> BN.
#   (2) DW Conv 5 x 5, dilation 2, padding 4 -> BN (effective 9 x 9 span).
# Sum branches -> SE channel attention -> PW Conv C -> 3C -> GELU
# -> GRN(3C) -> PW Conv 3C -> C -> add the original input.
# Each depthwise branch operates at C channels; output shape is unchanged.
# No activation or BatchNorm follows the projection or residual addition.

# SE (explicit channel attention): spatial global average pool per channel,
# -> PW Conv C -> max(4, C // 4) -> ReLU -> PW Conv back to C -> sigmoid.
# Multiply every channel by its learned, input-dependent channel weight.
# GRN (channel-response normalization): spatial L2 norms normalized by their
# channel mean per image; gamma and beta begin at zero, with an identity path.
# GRN is channel-wise modulation, not a conventional attention module.

# Spatial attention (CBAM-style): compute channel-wise mean and max maps of
# C x H x W features, concatenate them to 2 x H x W, then Conv k x k (2 -> 1)
# with bias and padding k // 2 -> sigmoid -> shared H x W spatial weights.
# Multiply the input by this map; shape stays unchanged, no residual addition.

# Stage 1: 2 blocks at C=14, each with 1,922 params: 14 x 64 x 64
# (3,844 params). Stage 1 spatial attention: k=3 (19 params).
# Downsampling: DW Conv 3 x 3, stride 2, padding 1 -> BN -> PW Conv
# 14 -> 24 -> BN: 24 x 32 x 32 (538 params); no activation.
# Stage 2: 1 block at C=24, 3C=72: 24 x 32 x 32 (4,830 params).
# Head: BN(24) -> spatial attention k=5, padding 2 (51 params)
# -> adaptive global average pool -> 24 x 1 x 1 -> flatten
# -> Linear 24 -> 10 logits (250 params). Head total: 349 params.
# All tensors use channels-first layout. SE and spatial attention use biases;
# depthwise and pointwise feature convolutions do not. No LayerNorm,
# DropPath, train-time reparameterization, or output softmax.
# Total: 406 + 3,844 + 19 + 538 + 4,830 + 349 = 9,986 parameters.

import logging
import sys
from copy import deepcopy
from time import perf_counter

import torch
import torchvision
import torchvision.transforms.v2 as v2
from torch import nn
from torchvision.transforms import InterpolationMode

from challenge_test import Test


# Adapted from the official Meta ConvNeXt V2 GRN implementation:
# https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/utils.py
class GRN(nn.Module):
    """Normalize global channel responses for input shaped [B, C, H, W]."""

    def __init__(self, channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = 1e-6

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + self.eps)
        return self.gamma * (x * Nx) + self.beta + x


# UniRepLKNet reference: https://github.com/AILab-CVC/UniRepLKNet
class SEAttention(nn.Module):
    """SE channel attention with a small bottleneck for narrow feature maps."""

    def __init__(self, channels):
        super().__init__()
        hidden_channels = max(4, channels // 4)
        self.weights = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.weights(x)


class UniRepLKNetBlock(nn.Module):
    """Parallel local/dilated depthwise branches, SE, and 3C GRN expansion."""

    def __init__(self, channels):
        super().__init__()
        expanded_channels = 3 * channels

        self.local_branch = nn.Sequential(
            nn.Conv2d(
                channels, channels, kernel_size=3, padding=1,
                groups=channels, bias=False,
            ),
            nn.BatchNorm2d(channels),
        )
        self.dilated_branch = nn.Sequential(
            nn.Conv2d(
                channels, channels, kernel_size=5, dilation=2, padding=4,
                groups=channels, bias=False,
            ),
            nn.BatchNorm2d(channels),
        )
        self.channel_attention = SEAttention(channels)
        self.channel_mixing = nn.Sequential(
            nn.Conv2d(channels, expanded_channels, kernel_size=1, bias=False),
            nn.GELU(),
            GRN(expanded_channels),
            nn.Conv2d(expanded_channels, channels, kernel_size=1, bias=False),
        )

    def forward(self, x):
        spatial_features = self.local_branch(x) + self.dilated_branch(x)
        spatial_features = self.channel_attention(spatial_features)
        return x + self.channel_mixing(spatial_features)


# Spatial attention reference: https://github.com/elbuco1/CBAM
class SpatialAttention(nn.Module):
    """Scale [B, C, H, W] features with one [B, 1, H, W] attention map.

    Channel mean and max maps feed a convolution and sigmoid; the resulting
    spatial weights are shared across channels.
    """

    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size % 2 == 1

        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=True,
        )

    def forward(self, x):
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map = torch.amax(x, dim=1, keepdim=True)

        combined = torch.cat([avg_map, max_map], dim=1)
        attention = torch.sigmoid(self.conv(combined))

        return x * attention


class BudgetUniRepLKNet(nn.Module):
    """Three 3C blocks over two stages with SE, GRN, and spatial attention."""

    def __init__(self):
        super().__init__()

        # Overlapping convolutional stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 14, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(14),
        )

        # Stage 1: 2 dual-branch blocks, 14 channels
        self.stage1 = nn.Sequential(
            UniRepLKNetBlock(14),
            UniRepLKNetBlock(14),
        )

        # Attend to Stage 1 spatial features
        self.stage1_attention = SpatialAttention(kernel_size=3)

        # Downsample features and widen from 14 to 24 channels
        self.downsampling = nn.Sequential(
            nn.Conv2d(14, 14, kernel_size=3, stride=2, padding=1, groups=14,
                      bias=False),
            nn.BatchNorm2d(14),
            nn.Conv2d(14, 24, kernel_size=1, bias=False),
            nn.BatchNorm2d(24),
        )

        # Stage 2: 1 dual-branch block, 24 channels
        self.stage2 = nn.Sequential(
            UniRepLKNetBlock(24),
        )

        # Classifier with a final spatial attention map
        self.classifier = nn.Sequential(
            nn.BatchNorm2d(24),
            SpatialAttention(kernel_size=5),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(24, 10),
        )

    def forward(self, images):
        features = self.stem(images)

        features = self.stage1(features)
        features = self.stage1_attention(features)

        features = self.downsampling(features)
        features = self.stage2(features)

        return self.classifier(features)


class Trainer(Test):
    """Provide the challenge interface and manage the training workflow."""

    def __init__(self, data_root="./dataset", epochs=200, batch_size=32,
                 learning_rate=1e-3, patience=25, seed=42, warmup_epochs=10,
                 log_path="log.txt", min_learning_rate=1e-6,
                 weight_decay=0.02, ema_decay=0.999, label_smoothing=0.05,
                 random_erasing_probability=0.35, gradient_clip_norm=1.0,
                 warmup_start_factor=0.1, augmentation_probability=0.5):
        super().__init__()
        self.data_root = data_root
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.seed = seed
        self.warmup_epochs = warmup_epochs
        self.min_learning_rate = min_learning_rate
        self.weight_decay = weight_decay
        self.ema_decay = ema_decay
        self.label_smoothing = label_smoothing
        self.random_erasing_probability = random_erasing_probability
        self.gradient_clip_norm = gradient_clip_norm
        self.warmup_start_factor = warmup_start_factor
        self.augmentation_probability = augmentation_probability
        self.checkpoint_path = "model.torch"
        self.log_path = log_path

    def create_model(self):
        return BudgetUniRepLKNet()

    def create_transform(self, inference=True):
        """Resize and center-crop RGB; sample augmentations only for training."""
        transforms = [v2.Resize((128, 128), antialias=True)]

        if not inference:
            # Sample each augmentation group independently for every image.
            transforms.extend([
                v2.RandomHorizontalFlip(p=self.augmentation_probability),
                v2.RandomApply([
                    v2.RandomAffine(
                        degrees=10,
                        translate=(0.05, 0.05),
                        scale=(0.9, 1.1),
                        interpolation=InterpolationMode.BILINEAR,
                        fill=128,
                    ),
                ], p=self.augmentation_probability),
                v2.RandomApply([
                    v2.ColorJitter(brightness=0.1, contrast=0.1),
                ], p=self.augmentation_probability),
            ])

        transforms.extend([
            torchvision.transforms.ToTensor(),
            v2.Normalize(mean=(0.5, 0.5, 0.5), std=(1.0, 1.0, 1.0)),
        ])

        if not inference:
            transforms.append(v2.RandomErasing(
                p=self.random_erasing_probability,
                scale=(0.02, 0.06),
                ratio=(0.5, 2.0),
                value=0.0,  # Mid-gray in normalized image space.
            ))

        return torchvision.transforms.Compose(transforms)

    def save_model(self, model):
        """Save the state dictionary expected by the challenge evaluator."""
        torch.save(model.state_dict(), self.checkpoint_path)

    def train(self):
        """Train on all training images and validate after each epoch."""
        run_started = perf_counter()

        # Configure Logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(self.log_path, mode="w", encoding="utf-8"),
            ],
            force=True,
        )
        logger = logging.getLogger(__name__)

        torch.set_num_threads(10) # Set CPU threads
        torch.manual_seed(self.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pin_memory = self.device.type == "cuda"
        self.use_amp = (
            self.device.type == "cuda"
            and torch.amp.autocast_mode.is_autocast_available("cuda")
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Create Dataset
        train_data = torchvision.datasets.ImageFolder(
            root=f"{self.data_root}/train",
            transform=self.create_transform(inference=False),
        )
        val_data = torchvision.datasets.ImageFolder(
            root=f"{self.data_root}/val",
            transform=self.create_transform(inference=True),
        )
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size, shuffle=True, num_workers=0,
            pin_memory=self.pin_memory,
        )
        val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=self.batch_size, shuffle=False, num_workers=0,
            pin_memory=self.pin_memory,
        )

        # Create Model
        self.model = self.create_model().to(self.device)
        parameter_count = sum(p.numel() for p in self.model.parameters())
        if parameter_count > 10_000:
            raise ValueError(f"Model has too many parameters: {parameter_count}.")

        # Create EMA Model
        self.ema_model = deepcopy(self.model).eval().requires_grad_(False)

        # Create Optimizer
        decay_parameters = []
        no_decay_parameters = []
        # Only convolutional and linear weights receive decay; BN and GRN do not.
        # https://github.com/facebookresearch/ConvNeXt-V2/blob/main/optim_factory.py
        for module in self.model.modules():
            for name, parameter in module.named_parameters(recurse=False):
                if not parameter.requires_grad:
                    continue
                if isinstance(module, (nn.Conv2d, nn.Linear)) and name == "weight":
                    decay_parameters.append(parameter)
                else:
                    no_decay_parameters.append(parameter)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay_parameters, "weight_decay": self.weight_decay},
                {"params": no_decay_parameters, "weight_decay": 0.0},
            ],
            lr=self.learning_rate,
            weight_decay=0.0,
        )

        # Schedule Learning Rate
        steps_per_epoch = len(train_loader)
        total_steps = self.epochs * steps_per_epoch
        warmup_steps = self.warmup_epochs * steps_per_epoch

        if 0 < warmup_steps < total_steps:
            warmup = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=self.warmup_start_factor,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=self.min_learning_rate,
            )
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_steps],
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=self.min_learning_rate,
            )

        # Log Run Configuration
        logger.info(
            f"Model: {self.model.__class__.__name__} | "
            f"Parameters: {parameter_count} | Device: {self.device} | "
            f"PyTorch: {torch.__version__} | torchvision: {torchvision.__version__}"
        )
        logger.info(
            f"Data root: {self.data_root} | "
            f"Train images: {len(train_data)} | Validation images: {len(val_data)} | "
            f"Batch size: {self.batch_size} | Seed: {self.seed} | "
            f"CPU threads: {torch.get_num_threads()} | "
            f"Data loader workers: {train_loader.num_workers}"
        )
        logger.info(f"Classes: {train_data.classes}")
        logger.info(f"Training transform:\n{train_data.transform}")
        logger.info(f"Validation/inference transform:\n{val_data.transform}")
        logger.info(
            f"Optimizer: {self.optimizer.__class__.__name__} | "
            f"LR: {self.learning_rate:.2e} | "
            f"Warmup: {self.warmup_epochs} epochs | "
            f"Warmup start factor: {self.warmup_start_factor} | "
            f"Total: {self.epochs} epochs | "
            f"Steps per epoch: {steps_per_epoch} | "
            f"Total steps: {total_steps} | "
            f"Warmup steps: {warmup_steps} | "
            f"Weight decay: {self.weight_decay} | "
            f"Min LR: {self.min_learning_rate:.2e} | "
            f"Scheduler: {self.scheduler.__class__.__name__} (per optimizer step)"
        )
        logger.info("Weight decay applies to convolutional and linear weights only.")
        logger.info(
            f"Label smoothing: {self.label_smoothing} | "
            f"Augmentation probability: {self.augmentation_probability} | "
            f"Random erasing: {self.random_erasing_probability} | "
            f"Gradient clip norm: {self.gradient_clip_norm} | "
            f"EMA decay: {self.ema_decay} | CUDA AMP: {self.use_amp}"
        )
        logger.info(
            f"Early stopping patience: {self.patience} epochs | "
            f"Checkpoint: {self.checkpoint_path} | Log: {self.log_path}"
        )
        logger.info(
            "Checkpoint selection: highest validation accuracy, then lowest "
            "validation loss, across normal and EMA models."
        )

        best_accuracy = -1.0
        best_val_loss = float("inf")
        best_source = None
        best_epoch = 0
        epochs_without_improvement = 0
        completed_epochs = 0
        stopped_early = False

        # Train Model
        logger.info(f"Training on {self.device}; {parameter_count} model parameters.")
        for epoch in range(1, self.epochs + 1):
            # CUDA work is asynchronous; synchronize at timing boundaries.
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            epoch_started = perf_counter()

            train_loss = self._train_epoch(train_loader)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            train_seconds = perf_counter() - epoch_started

            # Validate Model
            val_started = perf_counter()
            val_loss, val_accuracy = self._evaluate(self.model, val_loader)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            val_seconds = perf_counter() - val_started

            # Validate EMA Model
            ema_val_started = perf_counter()
            ema_val_loss, ema_val_accuracy = self._evaluate(self.ema_model, val_loader)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            ema_val_seconds = perf_counter() - ema_val_started

            # Save Best Model
            candidate_model = self.model
            candidate_source = "normal"
            candidate_accuracy = val_accuracy
            candidate_loss = val_loss
            if (
                ema_val_accuracy > candidate_accuracy
                or (ema_val_accuracy == candidate_accuracy and ema_val_loss < candidate_loss)
            ):
                candidate_model = self.ema_model
                candidate_source = "ema"
                candidate_accuracy = ema_val_accuracy
                candidate_loss = ema_val_loss

            improved = (
                candidate_accuracy > best_accuracy
                or (candidate_accuracy == best_accuracy and candidate_loss < best_val_loss)
            )
            if improved:
                best_accuracy = candidate_accuracy
                best_val_loss = candidate_loss
                best_source = candidate_source
                best_epoch = epoch
                self.save_model(candidate_model)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            epoch_seconds = perf_counter() - epoch_started
            logger.info(
                f"Epoch {epoch:02d}/{self.epochs}: "
                f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
                f"val_acc={val_accuracy:.6f}, "
                f"ema_val_loss={ema_val_loss:.6f}, ema_val_acc={ema_val_accuracy:.6f}, "
                f"lr={self.optimizer.param_groups[0]['lr']:.2e}, "
                f"epochs_without_improvement={epochs_without_improvement}/{self.patience}, "
                f"train_time={train_seconds:.2f}s, "
                f"val_time={val_seconds:.2f}s, "
                f"ema_val_time={ema_val_seconds:.2f}s, "
                f"epoch_time={epoch_seconds:.2f}s",
            )
            if improved:
                logger.info(
                    f"  Saved new best {best_source} checkpoint to {self.checkpoint_path}: "
                    f"val_acc={best_accuracy:.6f}, val_loss={best_val_loss:.6f}",
                )
            completed_epochs = epoch
            if epochs_without_improvement >= self.patience:
                stopped_early = True
                logger.info(f"Early stopping after {epoch} epochs.")
                break

        logger.info(
            f"Completed epochs: {completed_epochs}/{self.epochs} | "
            f"Stopped early: {stopped_early}"
        )
        logger.info(f"Best validation accuracy: {best_accuracy:.6f}")
        logger.info(f"Best validation loss: {best_val_loss:.6f}")
        logger.info(f"Best checkpoint source: {best_source}; epoch: {best_epoch}")
        logger.info(f"Best checkpoint: {self.checkpoint_path}")
        logger.info(f"Total training run time: {perf_counter() - run_started:.2f}s")

    def _train_epoch(self, loader):
        """Update model weights using one complete pass through the loader."""
        self.model.train()
        loss_sum = 0.0
        sample_count = 0

        for images, labels in loader:
            images = images.to(self.device, non_blocking=self.pin_memory)
            labels = labels.to(self.device, non_blocking=self.pin_memory)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.gradient_clip_norm,
            )
            previous_scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # An overflow skips the optimizer step and reduces the scaler value.
            if self.scaler.get_scale() >= previous_scale:
                self.scheduler.step()
                self._update_ema()

            loss_sum += loss.item() * labels.size(0)
            sample_count += labels.size(0)

        return loss_sum / sample_count

    @torch.no_grad()
    def _update_ema(self):
        """Average learned parameters and copy current BatchNorm statistics.
        # https://github.com/facebookresearch/ConvNeXt-V2/blob/main/engine_finetune.py"""
        for ema_parameter, parameter in zip(
            self.ema_model.parameters(), self.model.parameters()
        ):
            ema_parameter.lerp_(parameter, 1.0 - self.ema_decay)

        # Copy buffers, including floating running statistics and integer counters.
        for ema_buffer, buffer in zip(self.ema_model.buffers(), self.model.buffers()):
            ema_buffer.copy_(buffer)

    def _evaluate(self, model, loader):
        """Measure float32 validation loss and accuracy for one model."""
        model.eval()
        loss_sum = 0.0
        correct = 0
        sample_count = 0

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device, non_blocking=self.pin_memory)
                labels = labels.to(self.device, non_blocking=self.pin_memory)
                logits = model(images)

                loss_sum += self.criterion(logits, labels).item() * labels.size(0)
                correct += (logits.argmax(dim=1) == labels).sum().item()
                sample_count += labels.size(0)

        return loss_sum / sample_count, correct / sample_count


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
