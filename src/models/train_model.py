import random
import copy

import torch
from tqdm.auto import tqdm
import numpy as np

from settings import RANDOM_STATE

random.seed(RANDOM_STATE),
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
torch.cuda.manual_seed_all(RANDOM_STATE)


def train_epoch(model, data_loader, loss_function, optimizer, scheduler, device):
    model.to(device)
    model.train()
    total_loss = 0
    dl_size = len(data_loader)
    preds = []
    targets = []

    for batch in tqdm(data_loader):
        for key in batch:
            batch[key] = batch[key].to(device)

        optimizer.zero_grad()
        logits = model(batch)

        preds.append(logits.argmax(dim=1))
        targets.append(batch["labels"])

        loss = loss_function(logits, batch["labels"])
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    #For every epoch, not batch
    scheduler.step()
        
    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)
    acc = (targets == preds).sum() / preds.shape[0]
    metrics = {"Train Loss": total_loss / dl_size, "Train Accuracy": acc.item()}
    return metrics


def eval_epoch(model, data_loader, loss_function, device):
    model.to(device)
    model.eval()
    total_loss = 0
    preds = []
    targets = []
    dl_size = len(data_loader)

    for batch in tqdm(data_loader):
        for key in batch:
            batch[key] = batch[key].to(device)

        with torch.no_grad():
            logits = model(batch)
            preds.append(logits.argmax(dim=1))
            targets.append(batch["labels"])

        loss = loss_function(logits, batch["labels"])
        total_loss += loss.item()

    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)
    acc = (targets == preds).sum() / preds.shape[0]
    metrics = {"Eval Loss": total_loss / dl_size, "Eval Accuracy": acc.item()}
    return metrics


def single_model(
    model,
    train_loader,
    val_loader,
    loss_function,
    optimizer,
    scheduler,
    device=torch.device("cpu"),
    epochs: int = 8,
    start_epoch=0,
):
    loss_function.to(device)
    model.to(device)
    for epoch_i in range(0, epochs):
        if epoch_i >= start_epoch:
            train_metrics = train_epoch(
                model, train_loader, loss_function, optimizer, scheduler, device
            )
            eval_metrics = eval_epoch(model, val_loader, loss_function, device)
            print("EPOCH", epoch_i)
            print(train_metrics)
            print(eval_metrics)


def train_model_early_stopping(
    model,
    train_loader,
    val_loader,
    loss_function,
    optimizer,
    scheduler,
    device=torch.device("cpu"),
    early_stopping: int = 2,
    eps: int = 1e-3,
):
    loss_function.to(device)
    model.to(device)
    es = early_stopping
    max_acc = 0
    epoch = 0
    while es > 0:
        train_metrics = train_epoch(
            model, train_loader, loss_function, optimizer, scheduler, device
        )
        eval_metrics = eval_epoch(model, val_loader, loss_function, device)
        epoch += 1
        print("EPOCH", epoch)
        print(train_metrics)
        print(eval_metrics)
        acc = eval_metrics["Eval Accuracy"]
        if acc > max_acc + eps:
            best_model = copy.deepcopy(model)
            es = early_stopping
            max_acc = acc
        else:
            es -= 1

    return best_model
