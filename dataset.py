import os
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class PaliGemmaDataset(Dataset):
    def __init__(self, data_dir, json_file, processor, max_length=512):
        self.data_dir = data_dir
        self.processor = processor
        self.max_length = max_length

        with open(os.path.join(data_dir, json_file), "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.data_dir, item["image_file"])
        text = item["text"]

        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze()

        encoding = self.processor.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding.input_ids.squeeze()
        attention_mask = encoding.attention_mask.squeeze()

        labels = input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = (
            -100
        )  # CrossEntropyLoss ignores index -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": labels,
        }


def create_dataloaders(
    data_dir, train_json, val_json, processor, batch_size, num_workers=-1
):
    train_dataset = PaliGemmaDataset(data_dir, train_json, processor)
    val_dataset = PaliGemmaDataset(data_dir, val_json, processor)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, val_dataloader
