import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
import wandb
from dataset import PolygonColorDataset
from model import UNet

# Config
config = {
    "epochs": 30,
    "batch_size": 8,
    "lr": 1e-4,
    "img_size": 128
}

wandb.init(project="polygon-coloring-public", config=config)

# Transforms
transform = T.Compose([
    T.Resize((config["img_size"], config["img_size"])),
    T.ToTensor()
])

# Datasets
train_dataset = PolygonColorDataset("dataset/dataset", split="training", transform=transform)
val_dataset = PolygonColorDataset("dataset/dataset", split="validation", transform=transform)

# Force val dataset to share train's color mapping
val_dataset.colors = train_dataset.colors
val_dataset.color_to_idx = train_dataset.color_to_idx

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
n_colors = len(train_dataset.colors)
model = UNet(in_channels=1, n_classes=3, n_colors=n_colors).to(device)

# Loss & Optimizer
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=config["lr"])

# Training loop
for epoch in range(config["epochs"]):
    model.train()
    train_loss = 0.0
    for polygon, color, target in train_loader:
        polygon, color, target = polygon.to(device), color.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(polygon, color)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for polygon, color, target in val_loader:
            polygon, color, target = polygon.to(device), color.to(device), target.to(device)
            output = model(polygon, color)
            loss = criterion(output, target)
            val_loss += loss.item()

    wandb.log({
        "train_loss": train_loss / len(train_loader),
        "val_loss": val_loss / len(val_loader)
    })

    print(f"Epoch [{epoch+1}/{config['epochs']}], "
          f"Train Loss: {train_loss/len(train_loader):.4f}, "
          f"Val Loss: {val_loss/len(val_loader):.4f}")

# Save model
torch.save(model.state_dict(), "unet_polygon.pth")
