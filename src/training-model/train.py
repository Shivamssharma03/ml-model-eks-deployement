import os
from pathlib import Path

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

# ----------------------------------------
# Load pretrained model
# ----------------------------------------

model_name = "microsoft/phi-2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)
model.config.pad_token_id = tokenizer.pad_token_id

# ----------------------------------------
# Dataset Path
# ----------------------------------------

# When running on SageMaker
train_channel = os.environ.get("SM_CHANNEL_TRAINING")

if train_channel:
    DATASET_PATH = os.path.join(
        train_channel,
        "dataset.jsonl"
    )
    print(f"Using SageMaker dataset: {DATASET_PATH}")

# Local execution
else:
    ROOT_DIR = Path(__file__).resolve().parents[2]

    DATASET_PATH = ROOT_DIR / "Data-set" / "data.jsonl"

    print(f"Using Local dataset: {DATASET_PATH}")

# ----------------------------------------
# Load Dataset
# ----------------------------------------

dataset = load_dataset(
    "json",
    data_files=str(DATASET_PATH)
)

# ----------------------------------------
# Tokenization
# ----------------------------------------

def preprocess(example):

    text = (
        f"Instruction: {example['instruction']}\n"
        f"Response: {example['response']}"
    )

    result = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
    )

    result["labels"] = result["input_ids"].copy()

    return result


tokenized_dataset = dataset.map(preprocess)

# ----------------------------------------
# Data Collator
# ----------------------------------------

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# ----------------------------------------
# Output Directory
# ----------------------------------------

OUTPUT_DIR = os.environ.get(
    "SM_OUTPUT_DATA_DIR",
    "./output"
)

MODEL_DIR = os.environ.get(
    "SM_MODEL_DIR",
    "./Artifacts/trained_model"
)

os.makedirs(MODEL_DIR, exist_ok=True)

# ----------------------------------------
# Training Arguments
# ----------------------------------------

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    num_train_epochs=10,

    per_device_train_batch_size=1,

    learning_rate=5e-5,

    weight_decay=0.01,

    save_strategy="epoch",

    logging_steps=1,

    report_to="none",
)

# ----------------------------------------
# Trainer
# ----------------------------------------

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=tokenized_dataset["train"],

    data_collator=data_collator,
)

# ----------------------------------------
# Train
# ----------------------------------------

trainer.train()

# ----------------------------------------
# Save Model
# ----------------------------------------

trainer.save_model(MODEL_DIR)

tokenizer.save_pretrained(MODEL_DIR)

print(f"Model saved to {MODEL_DIR}")

print("Training completed successfully.")
