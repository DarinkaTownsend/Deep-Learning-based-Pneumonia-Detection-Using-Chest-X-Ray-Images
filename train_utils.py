import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        running_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs=10,
    scheduler=None,
    save_path="best_model.pt",
    print_every=50
):

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(enumerate(train_loader, start=1), total=len(train_loader))
        pbar.set_description(f"Epoch {epoch}/{epochs}")

        for step, (x, y) in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            # stats
            running_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            if step % print_every == 0 or step == 1:
                avg_loss_so_far = running_loss / max(total, 1)
                acc_so_far = correct / max(total, 1)
                pbar.set_postfix({"train_loss": f"{avg_loss_so_far:.4f}",
                                  "train_acc": f"{acc_so_far:.4f}"})

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch}/{epochs} | "
            f"LR {current_lr:.2e} | "
            f"train_loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val_loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path} (val_loss={best_val_loss:.4f})")

    return history

def plot_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    # ---- Loss curves ----
    plt.figure(figsize=(6,4))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.show()

    # ---- Accuracy curves ----
    plt.figure(figsize=(6,4))
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.show()

@torch.no_grad()
def evaluate_test(model, loader, device, class_names=None):
    model.eval()

    all_preds = []
    all_labels = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        preds = torch.argmax(logits, dim=1)

        all_preds.append(preds.cpu())
        all_labels.append(y.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    total_acc = (all_preds == all_labels).float().mean().item()

    num_classes = int(all_labels.max().item()) + 1
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for t, p in zip(all_labels, all_preds):
        cm[t, p] += 1

    per_class_acc = {}
    for c in range(num_classes):
        correct_c = cm[c, c].item()
        total_c = cm[c, :].sum().item()
        acc_c = correct_c / total_c if total_c > 0 else 0.0
        name = class_names[c] if class_names else str(c)
        per_class_acc[name] = acc_c

    return total_acc, per_class_acc, cm, all_preds, all_labels


def plot_confusion_matrix(cm, class_names):
    cm_np = cm.numpy()
    plt.figure(figsize=(4,4))
    plt.imshow(cm_np)
    plt.title("Confusion Matrix (Test)")
    plt.xticks(range(len(class_names)), class_names, rotation=45)
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(cm_np.shape[0]):
        for j in range(cm_np.shape[1]):
            plt.text(j, i, str(cm_np[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.show()


def compute_classification_metrics(cm, class_names):
    metrics = {}
    num_classes = cm.shape[0]

    for i in range(num_classes):
        TP = cm[i, i].item()
        FP = cm[:, i].sum().item() - TP
        FN = cm[i, :].sum().item() - TP

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[class_names[i]] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    macro_precision = sum(m["precision"] for m in metrics.values()) / num_classes
    macro_recall = sum(m["recall"] for m in metrics.values()) / num_classes
    macro_f1 = sum(m["f1_score"] for m in metrics.values()) / num_classes

    print("\nClass-wise Metrics:")
    for cls, m in metrics.items():
        print(f"{cls}:")
        print(f"  Precision: {m['precision']:.4f}")
        print(f"  Recall:    {m['recall']:.4f}")
        print(f"  F1-score:  {m['f1_score']:.4f}")

    print("\nMacro Average:")
    print(f"  Precision: {macro_precision:.4f}")
    print(f"  Recall:    {macro_recall:.4f}")
    print(f"  F1-score:  {macro_f1:.4f}")

    return metrics


def _ema(series, beta=0.6):
    if series is None or len(series) == 0:
        return []
    smoothed = []
    s = series[0]
    for x in series:
        s = beta * s + (1 - beta) * x
        smoothed.append(s)
    return smoothed


def plot_three_histories(hist1, hist2, hist3,
                         names=("Task 1.1 (Scratch)", "Task 1.2 (Pretrained)", "Task 1.2.1 (Balanced)"),
                         ema_beta=0.6):
    histories = [hist1, hist2, hist3]

    for hist, name in zip(histories, names):
        epochs = list(range(1, len(hist["train_loss"]) + 1))

        train_loss = hist["train_loss"]
        val_loss = hist["val_loss"]
        val_loss_s = _ema(val_loss, beta=ema_beta)

        plt.figure(figsize=(6,4))
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.scatter(epochs, val_loss, label="Val Loss (raw)", marker="o")
        plt.plot(epochs, val_loss_s, label=f"Val Loss (EMA β={ema_beta})")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss Curves — {name}")
        plt.legend()
        plt.show()

    for hist, name in zip(histories, names):
        epochs = list(range(1, len(hist["train_acc"]) + 1))

        train_acc = hist["train_acc"]
        val_acc = hist["val_acc"]
        val_acc_s = _ema(val_acc, beta=ema_beta)

        plt.figure(figsize=(6,4))
        plt.plot(epochs, train_acc, label="Train Acc")
        plt.scatter(epochs, val_acc, label="Val Acc (raw)", marker="o")
        plt.plot(epochs, val_acc_s, label=f"Val Acc (EMA β={ema_beta})")

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy Curves — {name}")
        plt.legend()
        plt.show()

def compare_experiments(results_dict):
    names = list(results_dict.keys())

    accuracies = []
    macro_f1 = []
    macro_precision = []
    macro_recall = []

    for name in names:
        acc = results_dict[name]["accuracy"]
        metrics = results_dict[name]["metrics"]

        precision_vals = [m["precision"] for m in metrics.values()]
        recall_vals = [m["recall"] for m in metrics.values()]
        f1_vals = [m["f1_score"] for m in metrics.values()]

        accuracies.append(acc)
        macro_precision.append(np.mean(precision_vals))
        macro_recall.append(np.mean(recall_vals))
        macro_f1.append(np.mean(f1_vals))

    print("\n===== EXPERIMENT COMPARISON =====\n")
    print(f"{'Model':<15} {'Acc':<10} {'MacroPrec':<12} {'MacroRec':<12} {'MacroF1':<10}")
    print("-"*60)
    for i, name in enumerate(names):
        print(f"{name:<15} "
              f"{accuracies[i]:<10.4f} "
              f"{macro_precision[i]:<12.4f} "
              f"{macro_recall[i]:<12.4f} "
              f"{macro_f1[i]:<10.4f}")

    x = np.arange(len(names))
    width = 0.2

    plt.figure(figsize=(10,5))
    plt.bar(x - 1.5*width, accuracies, width, label="Accuracy")
    plt.bar(x - 0.5*width, macro_precision, width, label="Macro Precision")
    plt.bar(x + 0.5*width, macro_recall, width, label="Macro Recall")
    plt.bar(x + 1.5*width, macro_f1, width, label="Macro F1")

    plt.xticks(x, names)
    plt.ylabel("Score")
    plt.ylim(0,1)
    plt.title("Model Comparison")
    plt.legend()
    plt.show()
