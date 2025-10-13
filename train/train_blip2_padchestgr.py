import os
import sys
import random
import importlib

sys.path.append(os.path.abspath(os.path.join(os.path.abspath(os.getcwd()), os.pardir)))

import torch
from transformers import Trainer, TrainingArguments

from mydatasets.padchestgr_dataset import PadChestDataset
from mymodels.blip2_padchestgr import build_model_and_processor, build_grounding_model_and_processor
from paths import DICT_CSV_PADCHESTGR_PATH


def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    # --------------------------
    # Config (edit as you like)
    # --------------------------
    output_dir = os.environ.get("OUTPUT_DIR", "../EXPERIMENTS")
    num_train_epochs = int(os.environ.get("EPOCHS", "5"))
    per_device_train_batch_size = int(os.environ.get("BATCH_SIZE", "4"))
    learning_rate = float(os.environ.get("LR", "2e-4"))
    save_steps = int(os.environ.get("SAVE_STEPS", "500"))

    # Simple switch:
    # GROUNDED=0 → normal BLIP-2
    # GROUNDED=1 → region-aware BLIP-2
    grounded = 0

    seed_everything(42)

    # --------------------------
    # Data
    # --------------------------
    train_ds = PadChestDataset(
        csv_path=DICT_CSV_PADCHESTGR_PATH["train"],
        grounded=grounded
    )
    eval_ds = PadChestDataset(
        csv_path=DICT_CSV_PADCHESTGR_PATH["validation"],
        grounded=grounded
    )
    test_ds = PadChestDataset(
        csv_path=DICT_CSV_PADCHESTGR_PATH["test"],
        grounded=grounded
    )

    # --------------------------
    # Model + Processor
    # --------------------------
    if grounded:
        # Use your grounded builder
        model, processor = build_grounding_model_and_processor()
    else:
        model, processor = build_model_and_processor()

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
        logging_steps=50,
        evaluation_strategy="steps" if eval_ds is not None else "no",
        eval_steps=save_steps,
        remove_unused_columns=False,  # important for vision-text models
    )

    # Use the dataset-provided collate
    data_collator = train_ds.build_collate_fn(processor)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=processor.tokenizer,
        data_collator=data_collator,
    )

    trainer.train()


if __name__ == "__main__":
    main()
