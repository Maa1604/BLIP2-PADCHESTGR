# train/train_blip2.py

import os
import random
from typing import Optional

import torch
from torch.utils.data import random_split
from transformers import Trainer, TrainingArguments

from mydatasets.padchestgr_dataset import PadChestDataset
from mymodels.blip2_padchestgr import build_model_and_processor


def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    # --------------------------
    # Config (edit as you like)
    # --------------------------
    csv_path = os.environ.get("PADCHEST_CSV", "padchest_train.csv")  # path to your CSV
    lang = os.environ.get("LANG", "es")  # "es" or "en"
    output_dir = os.environ.get("OUTPUT_DIR", "./blip2-padchest-lora")
    num_train_epochs = int(os.environ.get("EPOCHS", "5"))
    per_device_train_batch_size = int(os.environ.get("BATCH_SIZE", "4"))
    learning_rate = float(os.environ.get("LR", "2e-4"))
    save_steps = int(os.environ.get("SAVE_STEPS", "500"))
    eval_split = float(os.environ.get("EVAL_SPLIT", "0.1"))  # 10% eval by default
    report_to = os.environ.get("REPORT_TO", "none")  # set to "wandb" if you use it

    seed_everything(42)

    # --------------------------
    # Data
    # --------------------------
    full_ds = PadChestDataset(csv_path=csv_path, lang=lang)

    if 0.0 < eval_split < 1.0:
        eval_len = max(1, int(len(full_ds) * eval_split))
        train_len = len(full_ds) - eval_len
        train_ds, eval_ds = random_split(full_ds, [train_len, eval_len])
    else:
        train_ds, eval_ds = full_ds, None

    # --------------------------
    # Model + Processor
    # --------------------------
    model, processor = build_model_and_processor()

    # Get collate_fn from dataset (NEW)
    data_collator = full_ds.build_collate_fn(processor)

    # --------------------------
    # Training
    # --------------------------
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        save_steps=save_steps,
        save_total_limit=2,
        fp16=True,
        report_to=report_to,
        logging_steps=50,
        evaluation_strategy="steps" if eval_ds is not None else "no",
        eval_steps=save_steps,
        remove_unused_columns=False,  # important for vision-text models
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=processor.tokenizer,
        data_collator=data_collator,  # now provided by the dataset
    )

    trainer.train()


if __name__ == "__main__":
    main()
