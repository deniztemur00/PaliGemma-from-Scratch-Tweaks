from utils import PaliGemmaProcessor
from train import train
from dataset import create_dataloaders
from pali_gemma import PaliGemmaForConditionalGen
from configs import PaliGemmaConfig
import torch
from transformers import LlamaTokenizerFast


def main():
    model = PaliGemmaForConditionalGen(config=PaliGemmaConfig)
    tokenizer = LlamaTokenizerFast.from_pretrained(
        "hf-internal-testing/llama-tokenizer"
    )
    processor = PaliGemmaProcessor(
        tokenizer=tokenizer, num_image_tokens=196, image_size=224  # (224/16)^2 = 196
    )

    train_dataloader, val_dataloader = create_dataloaders(
        data_dir="./data",
        train_json="sample_data.json",
        val_json="sample_data.json",
        processor=processor,
        batch_size=5,
    )

    train(
        model,
        train_dataloader,
        val_dataloader,
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_epochs=5,
        learning_rate=1e-5,
    )
