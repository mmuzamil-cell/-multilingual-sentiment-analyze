import os
import argparse
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    TrainerCallback
)
from torch.utils.data import Dataset

from config import (
    MODEL_NAME, 
    FINE_TUNED_MODEL_DIR, 
    DATA_DIR,
    MAX_LEN, 
    BATCH_SIZE, 
    EPOCHS, 
    LEARNING_RATE, 
    RANDOM_SEED, 
    EVAL_DIR,
    get_logger
)
from preprocessing.cleaner import preprocess_review
from preprocessing.lang_detector import detect_language

logger = get_logger("train")

# Set random seed
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class ReviewDataset(Dataset):
    """Custom Dataset class for Hugging Face Trainer."""
    def __init__(self, reviews, labels, tokenizer, max_len):
        self.reviews = reviews
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        label = self.labels[item]

        encoding = self.tokenizer(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(eval_pred):
    """Computes training/validation metrics."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

class LossTrackingCallback(TrainerCallback):
    """Custom callback to record training and validation loss history."""
    def __init__(self):
        super().__init__()
        self.history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # We record logs containing loss, learning rate, eval_loss, etc.
            self.history.append(logs.copy())

def save_training_plots(history: list, output_dir: Path):
    """Plots training and validation loss/accuracy curves and saves them."""
    epochs = []
    train_loss = []
    val_loss = []
    val_acc = []
    
    # Extract metrics from history
    # The Trainer logs training loss and eval metrics at separate intervals or at epoch end
    for entry in history:
        epoch = entry.get("epoch")
        if epoch is None:
            continue
        
        # Check if train loss
        if "loss" in entry:
            # Match or append train loss
            train_loss.append((epoch, entry["loss"]))
        # Check if validation loss/accuracy
        if "eval_loss" in entry:
            val_loss.append((epoch, entry["eval_loss"]))
        if "eval_accuracy" in entry:
            val_acc.append((epoch, entry["eval_accuracy"]))
            
    # Convert to dataframes or sorted arrays
    train_loss = sorted(train_loss, key=lambda x: x[0])
    val_loss = sorted(val_loss, key=lambda x: x[0])
    val_acc = sorted(val_acc, key=lambda x: x[0])
    
    # Plot Loss Curve
    plt.figure(figsize=(10, 5))
    if train_loss:
        t_epochs, t_losses = zip(*train_loss)
        plt.plot(t_epochs, t_losses, label="Training Loss", color="royalblue", marker="o")
    if val_loss:
        v_epochs, v_losses = zip(*val_loss)
        plt.plot(v_epochs, v_losses, label="Validation Loss", color="crimson", marker="x")
        
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=150)
    plt.close()
    
    # Plot Accuracy Curve
    if val_acc:
        plt.figure(figsize=(10, 5))
        va_epochs, va_values = zip(*val_acc)
        plt.plot(va_epochs, va_values, label="Validation Accuracy", color="forestgreen", marker="s")
        plt.title("Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(output_dir / "accuracy_curve.png", dpi=150)
        plt.close()
        
    logger.info(f"Training plots saved to {output_dir}")

def train_model(quick_train: bool = False):
    """Main training routine."""
    logger.info("Initializing XLM-RoBERTa fine-tuning...")
    
    # Load dataset splits
    train_path = DATA_DIR / "train.csv"
    val_path = DATA_DIR / "val.csv"
    
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError("Dataset files not found. Run dataset_builder.py first.")
        
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    # In quick train mode, take a very small subset for fast verification
    if quick_train:
        logger.info("QUICK TRAIN MODE: Using 16 samples for training and validation.")
        train_df = train_df.head(16)
        val_df = val_df.head(8)
        epochs = 1
        logging_steps = 1
        eval_steps = 1
        save_steps = 2
    else:
        epochs = EPOCHS
        logging_steps = 20
        eval_steps = 50
        save_steps = 100
        
    # Preprocess text and detect language (just in case)
    logger.info("Preprocessing training data...")
    train_df["cleaned_review"] = train_df.apply(lambda r: preprocess_review(r["review"], r["language"]), axis=1)
    val_df["cleaned_review"] = val_df.apply(lambda r: preprocess_review(r["review"], r["language"]), axis=1)
    
    # Load Tokenizer
    logger.info(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Create datasets
    train_dataset = ReviewDataset(
        reviews=train_df["cleaned_review"].values,
        labels=train_df["label"].values,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    val_dataset = ReviewDataset(
        reviews=val_df["cleaned_review"].values,
        labels=val_df["label"].values,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device for training: {device}")
    
    # Load Pretrained XLM-RoBERTa
    logger.info(f"Loading pretrained model: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=3 # Negative (0), Neutral (1), Positive (2)
    )
    
    # Setup Training Arguments
    training_args = TrainingArguments(
        output_dir=str(FINE_TUNED_MODEL_DIR / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=BATCH_SIZE if not quick_train else 4,
        per_device_eval_batch_size=BATCH_SIZE if not quick_train else 4,
        warmup_ratio=0.1 if not quick_train else 0.0,
        weight_decay=0.01,
        learning_rate=LEARNING_RATE,
        logging_dir=str(EVAL_DIR / "logs"),
        logging_steps=logging_steps,
        evaluation_strategy="epoch" if not quick_train else "steps",
        eval_steps=eval_steps if quick_train else None,
        save_strategy="epoch" if not quick_train else "steps",
        save_steps=save_steps if quick_train else None,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="loss" if quick_train else "f1",
        greater_is_better=False if quick_train else True,
        report_to="none" # Disable integrations like wandb
    )
    
    loss_tracker = LossTrackingCallback()
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[loss_tracker]
    )
    
    # Train model
    logger.info("Starting training run...")
    trainer.train()
    
    # Load best model checkpoints
    logger.info("Training complete. Saving final model and tokenizer...")
    trainer.save_model(str(FINE_TUNED_MODEL_DIR))
    tokenizer.save_pretrained(str(FINE_TUNED_MODEL_DIR))
    logger.info(f"Model successfully saved to {FINE_TUNED_MODEL_DIR}")
    
    # Run evaluation on validation set
    logger.info("Evaluating model on validation set...")
    eval_results = trainer.evaluate()
    logger.info(f"Validation metrics: {eval_results}")
    
    # Save training plots
    save_training_plots(loss_tracker.history, EVAL_DIR)
    
    # Get predictions on validation set for detailed reports
    predictions = trainer.predict(val_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = val_df["label"].values
    
    # Generate Confusion Matrix
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=["Negative", "Neutral", "Positive"],
        yticklabels=["Negative", "Neutral", "Positive"]
    )
    plt.title("Confusion Matrix - Validation Set")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(EVAL_DIR / "confusion_matrix.png", dpi=150)
    plt.close()
    
    # Generate and save Classification Report
    report = classification_report(
        true_labels, 
        pred_labels, 
        target_names=["Negative", "Neutral", "Positive"],
        output_dict=True
    )
    
    report_path = EVAL_DIR / "classification_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
        
    logger.info(f"Saved classification report to {report_path}")
    
    # Save a human-readable text report
    report_txt = classification_report(
        true_labels, 
        pred_labels, 
        target_names=["Negative", "Neutral", "Positive"]
    )
    with open(EVAL_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_txt)
        
    # Write metadata info
    model_metadata = {
        "base_model": MODEL_NAME,
        "epochs": epochs,
        "batch_size": BATCH_SIZE if not quick_train else 4,
        "learning_rate": LEARNING_RATE,
        "val_accuracy": eval_results.get("eval_accuracy", 0.0),
        "val_f1": eval_results.get("eval_f1", 0.0),
        "val_loss": eval_results.get("eval_loss", 0.0),
        "status": "Fully trained" if not quick_train else "Verification trained"
    }
    with open(FINE_TUNED_MODEL_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(model_metadata, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Perform a quick train run with few samples to verify code works.")
    args = parser.parse_args()
    
    train_model(quick_train=args.quick)
