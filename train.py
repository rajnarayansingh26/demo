import pandas as pd
from datasets import Dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments
)

# =========================
# LOAD DATA
# =========================

expanded = pd.read_csv("expanded_dataset.csv")
master = pd.read_csv("master_dataset.csv")

df = pd.concat([expanded, master], ignore_index=True)

df = df[["incorrect", "correct"]]

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

print("Total samples:", len(df))

# =========================
# PREPARE DATA
# =========================

df["input_text"] = "fix grammar: " + df["incorrect"]
df["target_text"] = df["correct"]

dataset = Dataset.from_pandas(
    df[["input_text", "target_text"]]
)

# =========================
# TOKENIZER
# =========================

model_name = "t5-small"

tokenizer = T5Tokenizer.from_pretrained(model_name)

def preprocess(example):

    model_inputs = tokenizer(
        example["input_text"],
        max_length=128,
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        example["target_text"],
        max_length=128,
        truncation=True,
        padding="max_length"
    )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

tokenized_dataset = dataset.map(preprocess)

# =========================
# MODEL
# =========================

model = T5ForConditionalGeneration.from_pretrained(
    model_name
)

# =========================
# TRAINING ARGUMENTS
# =========================

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    save_strategy="epoch",
    logging_steps=50,
    learning_rate=5e-5,
)

# =========================
# TRAINER
# =========================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

trainer.train()

# =========================
# SAVE MODEL
# =========================

model.save_pretrained("trained_model")
tokenizer.save_pretrained("trained_model")

print("Training Completed")