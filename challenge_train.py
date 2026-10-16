# -*- coding: utf-8 -*-
"""Train the Tiny Fruit NCA and save its best validation checkpoint."""

import random

import torch
from torch import nn
import torchvision
import torchvision.transforms.v2 as v2

from challenge_test import Test


class ResidualPerception(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(20, 12, 3, padding=1, padding_mode="circular")
        self.conv2 = nn.Conv2d(12, 12, 3, padding=1, padding_mode="circular")
        self.skip = nn.Conv2d(20, 12, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = self.skip(x)
        y = self.relu(self.conv1(x))
        y = self.conv2(y)
        return self.relu(y + residual)


class TinyFruitNCA(nn.Module):
    def __init__(self, steps=32, fire_rate=1.0):
        super().__init__()
        self.steps = steps
        self.fire_rate = fire_rate
        self.residual_perception = ResidualPerception()
        self.dilated_perception = nn.Sequential(
            nn.Conv2d(
                20, 8, kernel_size=5, dilation=2, padding=4,
                padding_mode="circular"
            ),
            nn.LeakyReLU(0.2),
        )
        self.update_model = nn.Sequential(
            nn.Conv2d(20, 48, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(48, 20, kernel_size=1),
        )
        nn.init.normal_(self.update_model[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.update_model[-1].bias)

    def forward(self, images, return_pixel_logits=False):
        batch_size, _, height, width = images.shape
        evolving = images.new_zeros((batch_size, 17, height, width))
        state = torch.cat([images, evolving], dim=1)

        for _ in range(self.steps):
            p1 = self.residual_perception(state)
            p2 = self.dilated_perception(state)
            perception = torch.cat([p1, p2], dim=1)
            delta = self.update_model(perception)[:, 3:]

            if self.fire_rate < 1.0:
                mask = (
                    torch.rand(
                        batch_size, 1, height, width, device=images.device
                    ) < self.fire_rate
                ).to(delta.dtype)
                delta = delta * mask

            evolving = evolving + delta
            state = torch.cat([images, evolving], dim=1)

        pixel_logits = evolving[:, -10:]
        if return_pixel_logits:
            return pixel_logits
        return pixel_logits.mean(dim=(2, 3))


class Trainer(Test):
    def __init__(self):
        super().__init__()

    def create_model(self):
        return TinyFruitNCA(steps=32, fire_rate=1.0)

    def create_transform(self, inference=True):
        transforms = []
        if not inference:
            transforms.append(v2.RandomHorizontalFlip(p=0.5))
        transforms.extend([
            v2.Resize((64, 64), antialias=True),
            torchvision.transforms.ToTensor(),
            v2.Normalize(mean=(0.5, 0.5, 0.5), std=(1.0, 1.0, 1.0)),
        ])
        return torchvision.transforms.Compose(transforms)

    def save_model(self, model):
        torch.save(model.state_dict(), "model.torch")

    def train(self, data_root="./dataset", epochs=30, batch_size=32,
              learning_rate=1e-3, patience=7):
        random.seed(42)
        torch.manual_seed(42)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_data = torchvision.datasets.ImageFolder(
            root=f"{data_root}/train",
            transform=self.create_transform(inference=False),
        )
        val_data = torchvision.datasets.ImageFolder(
            root=f"{data_root}/val",
            transform=self.create_transform(inference=True),
        )

        pin_memory = device.type == "cuda"
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True, num_workers=0,
            pin_memory=pin_memory,
        )
        val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=batch_size, shuffle=False, num_workers=0,
            pin_memory=pin_memory,
        )

        model = self.create_model().to(device)
        parameter_count = sum(p.numel() for p in model.parameters())
        if parameter_count > 10_000:
            raise ValueError(f"Model has too many parameters: {parameter_count}.")

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=1e-5
        )
        best_accuracy = -1.0
        best_val_loss = float("inf")
        epochs_without_improvement = 0

        print(f"Training on {device}; {parameter_count} model parameters.")
        for epoch in range(1, epochs + 1):
            model.train()
            train_loss_sum = 0.0
            train_count = 0
            for images, labels in train_loader:
                images = images.to(device, non_blocking=pin_memory)
                labels = labels.to(device, non_blocking=pin_memory)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss_sum += loss.item() * labels.size(0)
                train_count += labels.size(0)

            model.eval()
            val_loss_sum = 0.0
            correct = 0
            val_count = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device, non_blocking=pin_memory)
                    labels = labels.to(device, non_blocking=pin_memory)
                    logits = model(images)
                    val_loss_sum += criterion(logits, labels).item() * labels.size(0)
                    correct += (logits.argmax(dim=1) == labels).sum().item()
                    val_count += labels.size(0)

            train_loss = train_loss_sum / train_count
            val_loss = val_loss_sum / val_count
            val_accuracy = correct / val_count
            print(
                f"Epoch {epoch:02d}/{epochs}: "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                f"val_acc={val_accuracy:.3f}"
            )

            improved = (
                val_accuracy > best_accuracy
                or (val_accuracy == best_accuracy and val_loss < best_val_loss)
            )
            if improved:
                best_accuracy = val_accuracy
                best_val_loss = val_loss
                self.save_model(model)
                epochs_without_improvement = 0
                print("  Saved new best checkpoint to model.torch")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping after {epoch} epochs.")
                    break

        print(f"Best validation accuracy: {best_accuracy:.3f}")
        print("Best checkpoint: model.torch")


if __name__ == "__main__":
    Trainer().train()
