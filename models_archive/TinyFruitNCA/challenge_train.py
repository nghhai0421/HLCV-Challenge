# -*- coding: utf-8 -*-
"""Train the Tiny Fruit NCA and save its best validation checkpoint."""
# Reference repo: https://github.com/mspitzna/NCAtorch

# Model architecture (shapes omit the batch dimension):
# Input: RGB resized to 3 x 48 x 48 with antialiasing, converted to a tensor,
# then normalized with mean 0.5 and standard deviation 1.0 per channel.
# Initial state: 20 x 48 x 48, containing 3 fixed RGB channels, 7 hidden
# channels, and 10 class channels. Hidden and class channels start at zero.
# Repeat the following update 16 times, sharing the same weights at every step:
# Residual perception branch (3,732 params):
#   Conv 3 x 3, 20 -> 12 -> ReLU -> Conv 3 x 3, 12 -> 12.
#   Add a Conv 1 x 1, 20 -> 12 skip projection, then apply ReLU.
#   Both spatial convolutions use circular padding 1: 12 x 48 x 48.
# Dilated perception branch (4,008 params):
#   Conv 5 x 5, 20 -> 8, dilation 2, circular padding 4 -> LeakyReLU(0.2).
#   The effective kernel spans 9 x 9 pixels; output: 8 x 48 x 48.
# Concatenate both branches: 20 x 48 x 48.
# Update network (1,988 params): Conv 1 x 1, 20 -> 48 -> LeakyReLU(0.2)
#   -> Conv 1 x 1, 48 -> 20. Discard the first 3 output channels so RGB stays fixed.
#   Initialize the final convolution's weights with normal std 0.001 and bias 0.
# Sample an independent update mask per image, pixel, and step with probability
# 0.5, shared across the 17 evolving channels, during both training and inference.
# Add the masked update to the hidden and class channels and reattach fixed RGB.
# All convolutions use stride 1 and have biases. Spatial resolution stays 48 x 48;
# there is no BatchNorm or intermediate pooling.
# Readout: take the final 10 class maps and globally average over height and width
# to obtain 10 class logits. Optionally return the 10 x 48 x 48 pixel logits.
# No linear classifier or output softmax; argmax of image logits selects the class.
# Total: 9,728 trainable parameters, independent of the number of update steps.

from time import perf_counter

import torch
import torchvision
import torchvision.transforms.v2 as v2
from torch import nn
from torchvision.transforms import InterpolationMode

from challenge_test import Test


class ResidualPerception(nn.Module):
    """Extract local features with a residual convolution branch."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, padding=1, padding_mode="circular"
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, padding=1, padding_mode="circular"
        )
        self.skip = nn.Conv2d(in_channels, out_channels, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = self.skip(x)
        y = self.relu(self.conv1(x))
        y = self.conv2(y)
        return self.relu(y + residual)


class TinyFruitNCA(nn.Module):
    """Evolve 7 hidden and 10 class channels while holding RGB input fixed."""

    def __init__(self, steps=16, fire_rate=0.5):
        super().__init__()
        self.steps = steps
        self.fire_rate = fire_rate
        self.input_channels = 3
        self.hidden_channels = 7
        self.num_classes = 10

        state_channels = self.input_channels + self.hidden_channels + self.num_classes
        residual_channels = 12
        dilated_channels = 8
        perception_channels = residual_channels + dilated_channels
        update_channels = 48

        self.residual_perception = ResidualPerception(
            state_channels, residual_channels
        )
        self.dilated_perception = nn.Sequential(
            nn.Conv2d(
                state_channels, dilated_channels, kernel_size=5,
                dilation=2, padding=4, padding_mode="circular",
            ),
            nn.LeakyReLU(0.2),
        )
        self.update_model = nn.Sequential(
            nn.Conv2d(perception_channels, update_channels, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(update_channels, state_channels, kernel_size=1),
        )
        nn.init.normal_(self.update_model[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.update_model[-1].bias)

    def forward(self, images, return_pixel_logits=False):
        """Map RGB [B, 3, H, W] to image logits or per-pixel class logits."""
        batch_size, _, height, width = images.shape
        evolving = images.new_zeros(
            (batch_size, self.hidden_channels + self.num_classes, height, width)
        )
        state = torch.cat([images, evolving], dim=1)

        for _ in range(self.steps):
            residual_features = self.residual_perception(state)
            dilated_features = self.dilated_perception(state)
            perception = torch.cat([residual_features, dilated_features], dim=1)
            delta = self.update_model(perception)[:, self.input_channels:]

            if self.fire_rate < 1.0:
                mask = (
                    torch.rand(
                        batch_size, 1, height, width, device=images.device
                    ) < self.fire_rate
                ).to(delta.dtype)
                delta = delta * mask

            evolving = evolving + delta
            state = torch.cat([images, evolving], dim=1)

        pixel_logits = evolving[:, -self.num_classes:]
        if return_pixel_logits:
            return pixel_logits
        return pixel_logits.mean(dim=(2, 3))


class Trainer(Test):
    """Provide the challenge interface and manage the training workflow."""

    def __init__(self, data_root="./dataset", epochs=50, batch_size=32,
                 learning_rate=1e-3, patience=20, seed=42, warmup_epochs=5):
        super().__init__()
        self.data_root = data_root
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.seed = seed
        self.warmup_epochs = warmup_epochs
        self.checkpoint_path = "model.torch"

    def create_model(self):
        return TinyFruitNCA()

    def create_transform(self, inference=True):
        """Resize and normalize RGB; sample augmentations only for training."""
        augmentation_probability = 0.5
        transforms = [v2.Resize((48, 48), antialias=True)]

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
        print(f"Training on {self.device}; {parameter_count} model parameters.")
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
            print(
                f"Epoch {epoch:02d}/{self.epochs}: "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                f"val_acc={val_accuracy:.3f}, "
                f"lr={self.optimizer.param_groups[0]['lr']:.2e}, "
                f"train_time={train_seconds:.2f}s, "
                f"val_time={val_seconds:.2f}s, "
                f"epoch_time={epoch_seconds:.2f}s",
                flush=True,
            )
            if improved:
                print(
                    f"  Saved new best checkpoint to {self.checkpoint_path}",
                    flush=True,
                )
            if epochs_without_improvement >= self.patience:
                print(f"Early stopping after {epoch} epochs.", flush=True)
                break

        print(f"Best validation accuracy: {best_accuracy:.3f}")
        print(f"Best checkpoint: {self.checkpoint_path}")
        print(f"Total training run time: {perf_counter() - run_started:.2f}s")

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
