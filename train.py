import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from pali_gemma import PaliGemmaForConditionalGen


def train(
    model: PaliGemmaForConditionalGen,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    device: str,
    num_epochs: int,
    learning_rate: float,
):
    model = torch.compile(model, mode="max-autotune", fullgraph=True)
    model.to(device)

    
    torch.set_float32_matmul_precision("high")
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler()  # gradient scaler

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Training loop
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_dataloader, desc="Training"):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                logits = outputs["logits"]
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), labels.view(-1)
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss:.4f}")

        # Validation loop
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                loss = outputs.loss
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"Average validation loss: {avg_val_loss:.4f}")

    print("Training completed.")
