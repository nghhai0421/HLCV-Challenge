# -*- coding: utf-8 -*-
"""Train BudgetResNet_Wide and save its best validation checkpoint."""

# Model architecture (shapes omit the batch dimension):
# Input: RGB resized to 3 x 128 x 128 with antialiasing, converted to a tensor,
# then normalized with mean 0.5 and standard deviation 1.0 per channel.
# Stem: standard Conv 5 x 5, 3 -> 12, stride 1, padding 2 -> BatchNorm -> ReLU
# produces 12 x 128 x 128 (924 params), then MaxPool 2 x 2, stride 2 -> 12 x 64 x 64.
# Each bottleneck uses Conv 1 x 1 -> BN -> ReLU -> depthwise Conv 3 x 3 -> BN -> ReLU
# -> Conv 1 x 1 -> BN, followed by shortcut addition and a final ReLU.
# Only the middle convolution is depthwise, with one group per hidden channel.
# Normal bottlenecks use an identity shortcut and stride 1 throughout.
# Downsampling bottlenecks use stride 2 in the depthwise 3 x 3 convolution
# and a learned Conv 1 x 1, stride 2 -> BN projection shortcut.
# All convolution padding uses zeros. All convolutions have no bias.
# The 3 x 3 convolutions use padding 1; 1 x 1 convolutions have no padding.

# Stage 1:
#   Block 1: 12 -> 18 -> 18 -> 12, identity: 12 x 64 x 64 (690 params).
#   Block 2: 12 -> 18 -> 18 -> 12, identity: 12 x 64 x 64 (690 params).
#   Block 3: 12 -> 18 -> 18 -> 20, downsampling: 20 x 32 x 32 (1,130 params).
#   Stage total: 2,510 params.
# Stage 2:
#   Block 1: 20 -> 28 -> 28 -> 20, identity: 20 x 32 x 32 (1,524 params).
#   Block 2: 20 -> 28 -> 28 -> 20, identity: 20 x 32 x 32 (1,524 params).
#   Block 3: 20 -> 28 -> 28 -> 40, downsampling: 40 x 16 x 16 (3,004 params).
#   Stage total: 6,052 params.
# Each downsampling block halves the spatial resolution in both branches.
# Adaptive global average pooling averages each 16 x 16 feature map: 40 x 1 x 1.
# Flatten: 40 features. Linear with bias: 40 -> 10 class logits (410 params).
# Total: 9,896 trainable parameters, including both projection shortcuts.

import logging
import sys
from time import perf_counter

import torch
import torchvision
import torchvision.transforms.v2 as v2
from torch import nn
from torchvision.transforms import InterpolationMode

from challenge_test import Test


class Bottleneck(nn.Module):
    """Refine features with a depthwise middle convolution and identity shortcut."""

    def __init__(self, channels, bottleneck_channels):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(channels, bottleneck_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(),
            nn.Conv2d(
                bottleneck_channels, bottleneck_channels,
                kernel_size=3, padding=1, groups=bottleneck_channels, bias=False,
            ),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(),
            nn.Conv2d(bottleneck_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, images):
        residual = self.residual(images)
        return self.relu(residual + images)


class DownsamplingBottleneck(nn.Module):
    """Downsample both branches with a learned projection shortcut."""

    def __init__(self, in_channels, bottleneck_channels, out_channels):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(),
            nn.Conv2d(
                bottleneck_channels, bottleneck_channels,
                kernel_size=3, stride=2, padding=1,
                groups=bottleneck_channels, bias=False,
            ),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(),
            nn.Conv2d(
                bottleneck_channels, out_channels, kernel_size=1, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=2, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU()

    def forward(self, images):
        residual = self.residual(images)
        shortcut = self.shortcut(images)
        return self.relu(residual + shortcut)


class BudgetResNet_Wide(nn.Module):
    """Classify 128 x 128 RGB images with two stages and 9,896 parameters."""

    def __init__(self):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Stage 1
        self.stage1 = nn.Sequential(
            Bottleneck(12, 18),
            Bottleneck(12, 18),
            DownsamplingBottleneck(12, 18, 20),
        )

        # Stage 2
        self.stage2 = nn.Sequential(
            Bottleneck(20, 28),
            Bottleneck(20, 28),
            DownsamplingBottleneck(20, 28, 40),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(40, 10),
        )

    def forward(self, images):
        features = self.stem(images)
        features = self.stage1(features)
        features = self.stage2(features)
        return self.classifier(features)


class Trainer(Test):
    """Provide the challenge interface and manage the training workflow."""

    def __init__(self, data_root="./dataset", epochs=100, batch_size=32,
                 learning_rate=1e-3, patience=20, seed=42, warmup_epochs=5,
                 log_path="log.txt"):
        super().__init__()
        self.data_root = data_root
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.seed = seed
        self.warmup_epochs = warmup_epochs
        self.checkpoint_path = "model.torch"
        self.log_path = log_path

    def create_model(self):
        return BudgetResNet_Wide()

    def create_transform(self, inference=True):
        """Resize and normalize RGB; sample augmentations only for training."""
        augmentation_probability = 0.5
        transforms = [v2.Resize((128, 128), antialias=True)]

        if not inference:
            # Sample each augmentation group independently for every image.
            transforms.extend([
                v2.RandomHorizontalFlip(p=augmentation_probability),
                v2.RandomApply([
                    v2.RandomAffine(
                        degrees=10,
                        scale=(0.9, 1.1),
                        interpolation=InterpolationMode.BILINEAR,
                        fill=128,
                    ),
                ], p=augmentation_probability),
                v2.RandomApply([
                    v2.ColorJitter(brightness=0.1, contrast=0.1),
                ], p=augmentation_probability),
            ])

        transforms.extend([
            torchvision.transforms.ToTensor(),
            v2.Normalize(mean=(0.5, 0.5, 0.5), std=(1.0, 1.0, 1.0)),
        ])

        if not inference:
            transforms.append(v2.RandomErasing(
                p=augmentation_probability,
                scale=(0.02, 0.08),
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

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate,
            weight_decay=1e-5,
        )

        # Warm Up Learning Rate
        self.warmup_scheduler = None
        if self.warmup_epochs > 0:
            self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.warmup_epochs * len(train_loader),
            )

        best_accuracy = -1.0
        best_val_loss = float("inf")
        epochs_without_improvement = 0

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
            val_loss, val_accuracy = self._evaluate(val_loader)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            val_seconds = perf_counter() - val_started

            # Save Best Model
            improved = (
                val_accuracy > best_accuracy
                or (val_accuracy == best_accuracy and val_loss < best_val_loss)
            )
            if improved:
                best_accuracy = val_accuracy
                best_val_loss = val_loss
                self.save_model(self.model)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            epoch_seconds = perf_counter() - epoch_started
            logger.info(
                f"Epoch {epoch:02d}/{self.epochs}: "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                f"val_acc={val_accuracy:.3f}, "
                f"lr={self.optimizer.param_groups[0]['lr']:.2e}, "
                f"train_time={train_seconds:.2f}s, "
                f"val_time={val_seconds:.2f}s, "
                f"epoch_time={epoch_seconds:.2f}s",
            )
            if improved:
                logger.info(
                    f"  Saved new best checkpoint to {self.checkpoint_path}",
                )
            if epochs_without_improvement >= self.patience:
                logger.info(f"Early stopping after {epoch} epochs.")
                break

        logger.info(f"Best validation accuracy: {best_accuracy:.3f}")
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
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0
            )
            self.optimizer.step()
            if self.warmup_scheduler is not None:
                self.warmup_scheduler.step()

            loss_sum += loss.item() * labels.size(0)
            sample_count += labels.size(0)

        return loss_sum / sample_count

    def _evaluate(self, loader):
        """Measure mean loss and image accuracy without updating weights."""
        self.model.eval()
        loss_sum = 0.0
        correct = 0
        sample_count = 0

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device, non_blocking=self.pin_memory)
                labels = labels.to(self.device, non_blocking=self.pin_memory)
                logits = self.model(images)

                loss_sum += self.criterion(logits, labels).item() * labels.size(0)
                correct += (logits.argmax(dim=1) == labels).sum().item()
                sample_count += labels.size(0)

        return loss_sum / sample_count, correct / sample_count


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
