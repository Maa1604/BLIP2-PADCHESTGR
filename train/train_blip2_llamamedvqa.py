import os
import sys
import argparse
import multiprocessing
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Project root (adjust if needed)
sys.path.append(os.path.abspath(os.path.join(os.path.abspath(os.getcwd()), os.pardir)))

# ---- your modules / paths ----
from myscorers.bleu.bleu import Bleu
from myscorers.rouge.rouge import Rouge
from myscorers.bertscore.bertscore import BertScorer
# from myscorers.chexbert.chexbert import myF1ChexBert
# from myscorers.myradgraph.myradgraph import myRadGraph

from paths import DICT_CSV_LLAMAMEDVQA_PATH

from mydatasets.llamamedvqa_dataset import LlamaMedVQADataset
from mymodels.blip2_padchestgr import build_model_and_processor

torch.set_float32_matmul_precision('medium')


# ----------------------------
# Args
# ----------------------------
parser = argparse.ArgumentParser(description="Train BLIP-2 on LLaMA-MedVQA")

parser.add_argument('--exp_name', type=str, required=True, help='Experiment name.')
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--accumulate_grad_batches', type=int, default=32)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--hnm', action='store_true', help='(placeholder) Hard Negative Mining not used here.')
parser.add_argument('--save_every', type=int, default=0, help='If >0, additionally save checkpoint every N epochs.')
parser.add_argument('--max_new_tokens', type=int, default=128, help='Max tokens for generation.')
parser.add_argument('--num_beams', type=int, default=2, help='Beams for generation.')

train_csv = DICT_CSV_LLAMAMEDVQA_PATH["train"]
val_csv   = DICT_CSV_LLAMAMEDVQA_PATH["validation"]


args = parser.parse_args()

print("*" * 20)
print("exp_name:", args.exp_name)
print("epochs:", args.epochs)
print("batch_size:", args.batch_size)
print("accumulate_grad_batches:", args.accumulate_grad_batches)
print("lr:", args.lr)
print("num_beams:", args.num_beams)
print("max_new_tokens:", args.max_new_tokens)
print("*" * 30)

EXP_DIR_PATH = os.path.join("../EXPERIMENTS", args.exp_name)
os.makedirs(EXP_DIR_PATH, exist_ok=True)



# ----------------------------
# Scorers
# ----------------------------
bleu_scorer = Bleu(n=4)
rougel_scorer = Rouge(rouges=['rougeL'])
# f1cxb_scorer = myF1ChexBert()
bert_scorer = BertScorer()
# radgraph_scorer = myRadGraph(reward_level='partial')



# ----------------------------
# Model & Processor
# ----------------------------
model, processor = build_model_and_processor()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ----------------------------
# Dataset & DataLoaders
# ----------------------------
train_ds = LlamaMedVQADataset(csv_path=train_csv)
val_ds   = LlamaMedVQADataset(csv_path=val_csv)

train_collate = train_ds.build_collate_fn(processor)
val_collate   = val_ds.build_collate_fn(processor)

num_workers = max(1, multiprocessing.cpu_count() - 1)
print("Num workers:", num_workers)

train_loader = DataLoader(
    train_ds,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=num_workers,
    collate_fn=train_collate,
    pin_memory=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=1,          # eval one-by-one for simpler decoding
    shuffle=False,
    num_workers=num_workers,
    collate_fn=val_collate,
    pin_memory=True
)


# ----------------------------
# Optimizer & Scheduler
# ----------------------------
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.8)

# ----------------------------
# Saver
# ----------------------------

import pandas as pd
import os

def collect_and_save_predictions(hyps, refs, l_hyps, l_refs, batch, exp_dir, epoch):
    """
    Clean, collect, and save generated predictions for an epoch.

    Args:
        hyps (List[str]): raw generated hypotheses from model.generate().
        refs (List[str]): raw reference texts from the batch.
        l_hyps (List[str]): global list of hypotheses to extend.
        l_refs (List[str]): global list of references to extend.
        batch (Dict): current batch, used to fetch image_ids if available.
        exp_dir (str): experiment directory path.
        epoch (int): current epoch number.
    """

    # --- Clean up and collect ---
    clean_hyps = [h.strip() for h in hyps]
    clean_refs = [r.strip() for r in refs]
    l_hyps.extend(clean_hyps)
    l_refs.extend(clean_refs)

    # Get image IDs if available
    image_ids = batch.get("image_ids", [f"sample_{i}" for i in range(len(clean_hyps))])

    # --- Save to CSV ---
    os.makedirs(exp_dir, exist_ok=True)
    csv_path = os.path.join(exp_dir, f"epoch_{epoch}_predictions.csv")

    df = pd.DataFrame({
        "image_id": image_ids,
        "reference": clean_refs,
        "hypothesis": clean_hyps
    })

    # Append if exists, else create
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, index=False, mode='a', header=False)

# ----------------------------
# Training
# ----------------------------
model.train()  # we'll switch to eval() during validation
best_f1cxb = float("-inf")
best_bertscore = float("-inf")
best_rg = float("-inf")
epoch_best_rg = 0
epoch_best_f1cxb = 0
epoch_best_bertscore = 0



print("\n---- Start Training ----")
for epoch in range(args.epochs):

    # ========== Train ==========
    model.train()
    optimizer.zero_grad()
    train_loss = 0.0

    with tqdm(iter(train_loader), desc=f"Train Epoch {epoch}", unit="batch") as tepoch:
        for step, batch in enumerate(tepoch):
            pixel_values   = batch['pixel_values'].to(device, dtype=torch.bfloat16 if next(model.parameters()).dtype==torch.bfloat16 else torch.float32)
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)
            # print("prixel_values:\n\n")
            # print(pixel_values)
            # print("input_ids:\n\n")
            # print(input_ids)
            # print("attention_mask:\n\n")
            # print(attention_mask)
            # print("labels:\n\n")
            # print(labels)
            # exit()

            forward_kwargs = dict(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            outputs = model(**forward_kwargs)
            loss = outputs.loss
            # print(loss)
            
            loss.backward()

            if (step + 1) % args.accumulate_grad_batches == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item()
            tepoch.set_postfix(loss=loss.item())

        # flush remaining grads
        optimizer.step()
        optimizer.zero_grad()

    # Normalize by number of optimizer steps (roughly)
    steps_per_epoch = max(1, len(train_loader) // max(1, args.accumulate_grad_batches))
    train_loss /= steps_per_epoch

    # ========== Validation ==========
    model.eval()
    val_loss = 0.0
    l_refs, l_hyps = [], []

    with torch.no_grad():
        with tqdm(iter(val_loader), desc=f"Val Epoch {epoch}", unit="batch") as tepoch:
            for batch in tepoch:
                pixel_values   = batch['pixel_values'].to(device, dtype=torch.bfloat16 if next(model.parameters()).dtype==torch.bfloat16 else torch.float32)
                input_ids      = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels         = batch['labels'].to(device)

                forward_kwargs = dict(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                out = model(**forward_kwargs)
                loss = out.loss
                val_loss += loss.item()

                # ---- generation ----
                gen_ids = model.generate(
                    pixel_values=pixel_values,
                    input_ids=input_ids,               # <— ADD
                    attention_mask=attention_mask,     # <— ADD
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False
                )
                hyps = processor.batch_decode(gen_ids, skip_special_tokens=True)
                refs = batch['references']  # list[str] (bs=1 here)

                collect_and_save_predictions(hyps, refs, l_hyps, l_refs, batch, EXP_DIR_PATH, epoch)

                tepoch.set_postfix(loss=loss.item())

    val_loss /= len(val_loader)

    # ========== Metrics ==========
    calculated_bleu = bleu_scorer(l_refs, l_hyps)[0]
    calculated_rougel = rougel_scorer(refs=l_refs, hyps=l_hyps)[0]
    # calculated_f1cxb = f1cxb_scorer.calculate(l_refs, l_hyps)
    calculated_bertscore = bert_scorer(l_hyps, l_refs)
    # calculated_rg = radgraph_scorer(l_refs, l_hyps)

    # ========== Checkpointing ==========
    def _save_model(tag: str):
        path = os.path.join(EXP_DIR_PATH, f"{tag}_epoch{epoch}.pt")
        torch.save(model.state_dict(), path)
        return path

    # if calculated_f1cxb > best_f1cxb:
    #     # remove previous best if exists
    #     if epoch_best_f1cxb != 0:
    #         try:
    #             os.remove(os.path.join(EXP_DIR_PATH, f"best_f1cxb_epoch{epoch_best_f1cxb}.pt"))
    #         except OSError:
    #             pass
    #     best_f1cxb = calculated_f1cxb
    #     epoch_best_f1cxb = epoch
    #     _save_model("best_f1cxb")

    # if calculated_rg > best_rg:
    #     if epoch_best_rg != 0:
    #         try:
    #             os.remove(os.path.join(EXP_DIR_PATH, f"best_rg_epoch{epoch_best_rg}.pt"))
    #         except OSError:
    #             pass
    #     best_rg = calculated_rg
    #     epoch_best_rg = epoch
    #     _save_model("best_rg")

    if calculated_bertscore > best_bertscore:
        if epoch_best_bertscore != 0:
            try:
                os.remove(os.path.join(EXP_DIR_PATH, f"best_bertscore_epoch{epoch_best_bertscore}.pt"))
            except OSError:
                pass
        best_bertscore = calculated_bertscore
        epoch_best_bertscore = epoch
        _save_model("best_bertscore")

    if args.save_every > 0 and ((epoch + 1) % args.save_every == 0):
        _save_model("checkpoint")

    # ========== Logging ==========
    print(f"\nEpoch {epoch}")
    print(f"\tTrain Loss: {train_loss:.6f}")
    print(f"\tVal   Loss: {val_loss:.6f}")
    print(f"\tBLEU4:      {calculated_bleu}")
    print(f"\tRougeL:     {calculated_rougel}")
    # print(f"\tF1cXb:      {calculated_f1cxb}")
    print(f"\tBERTscore:  {calculated_bertscore}")
    # print(f"\tRadGraph:   {calculated_rg}")

    with open(os.path.join(EXP_DIR_PATH, "log.txt"), 'a') as f:
        f.write(f"EPOCH: {epoch}\n")
        f.write(f"\tBLEU4:\t\t{calculated_bleu}\n")
        f.write(f"\tRougeL:\t\t{calculated_rougel}\n")
        # f.write(f"\tF1cXb:\t\t{calculated_f1cxb}\n")
        f.write(f"\tBERTscore:\t{calculated_bertscore}\n")
        # f.write(f"\tRG:\t\t{calculated_rg}\n")
        f.write(f"\tTrLoss:\t\t{train_loss}\n")
        f.write(f"\tValLoss:\t{val_loss}\n")
        f.write("------------------------------\n")

    lr_scheduler.step(val_loss)

# Save final weights
torch.save(model.state_dict(), os.path.join(EXP_DIR_PATH, "last_model.pt"))
print("Training finished. Saved last_model.pt")
