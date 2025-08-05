import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class PolygonColorDataset(Dataset):
    def __init__(self, root_dir, split="training", transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        json_path = os.path.join(root_dir, split, "data.json")
        with open(json_path, "r") as f:
            self.data = json.load(f)

        # Get unique colors (note: key is 'colour')
        self.colors = sorted({item["colour"] for item in self.data})
        self.color_to_idx = {c: i for i, c in enumerate(self.colors)}

        self.inputs_dir = os.path.join(root_dir, split, "inputs")
        self.outputs_dir = os.path.join(root_dir, split, "outputs")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        polygon_img = Image.open(
            os.path.join(self.inputs_dir, sample["input_polygon"])
        ).convert("L")

        target_img = Image.open(
            os.path.join(self.outputs_dir, sample["output_image"])
        ).convert("RGB")

        # One-hot encode color
        color_idx = self.color_to_idx[sample["colour"]]
        color_onehot = torch.zeros(len(self.colors))
        color_onehot[color_idx] = 1.0

        if self.transform:
            polygon_img = self.transform(polygon_img)
            target_img = self.transform(target_img)

        return polygon_img, color_onehot, target_img
