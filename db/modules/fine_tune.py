# db/fine_tune_peft.py
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# -----------------------------
# Configuration
# -----------------------------
BASE_MODEL = "tiiuae/falcon-rw-1b"  # small, fast model for demo
TOKENIZER_MODEL = BASE_MODEL
FINE_TUNE_OUTPUT_DIR = "data/rag_llm"

# Training hyperparameters
BATCH_SIZE = 2
EPOCHS = 3
LEARNING_RATE = 3e-4
MAX_SEQ_LEN = 512

# Hardware/performance
GRADIENT_ACCUMULATION_STEPS = 4
FP16 = True

# Logging / checkpoints
SAVE_STEPS = 100
LOGGING_STEPS = 50

# LoRA / QLoRA config
LORA_R = 4
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
TARGET_MODULES = ["query_key_value"] # ["q_proj", "v_proj"]  # attention projection layers
LORA_TASK_TYPE = "CAUSAL_LM"

# -----------------------------
# Fine-tuning function
# -----------------------------
def fine_tune_insurance_llm(data_path: str = "db/insurance_data.json"):
    print(f"🚀 Loading dataset from: {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Flatten 'context' field
    for item in raw_data:
        context = item.pop("context", {})
        if isinstance(context, dict):
            context_str = ", ".join(
                f"{k}:{'|'.join(v) if isinstance(v, list) else v}" 
                for k, v in context.items()
            )
        elif isinstance(context, list):
            context_str = ", ".join(str(c) for c in context)
        else:
            context_str = str(context)
        item["context"] = context_str

    dataset = Dataset.from_list(raw_data)

    # Format instruction-response text
    def format_prompt(example):
        return {
            "text": f"### Instruction:\n{example['instruction']}\n\n"
                    f"### Context:\n{example['context']}\n\n"
                    f"### Response:\n{example['response']}"
        }

    dataset = dataset.map(format_prompt)

    # -----------------------------
    # Load model & tokenizer
    # -----------------------------
    print(f"📦 Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    # -----------------------------
    # Prepare model for k-bit training (QLoRA)
    # -----------------------------
    model = prepare_model_for_kbit_training(model)

    # -----------------------------
    # Configure LoRA with PEFT
    # -----------------------------
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=LORA_TASK_TYPE
    )

    model = get_peft_model(model, lora_config)
    print(f"✅ LoRA layers added via PEFT to {TARGET_MODULES}")

    # -----------------------------
    # Tokenization
    # -----------------------------
    def tokenize_function(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding="max_length"
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # -----------------------------
    # Training setup
    # -----------------------------
    training_args = TrainingArguments(
        output_dir=FINE_TUNE_OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=FP16,
        logging_steps=LOGGING_STEPS,
        save_total_limit=2,
        save_steps=SAVE_STEPS,
        report_to="none",
        remove_unused_columns=False
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )

    # -----------------------------
    # Train & Save
    # -----------------------------
    print(f"🔧 Starting QLoRA fine-tuning on {len(dataset)} samples...")
    trainer.train()

    print(f"💾 Saving fine-tuned model to {FINE_TUNE_OUTPUT_DIR}...")
    model.save_pretrained(FINE_TUNE_OUTPUT_DIR)
    tokenizer.save_pretrained(FINE_TUNE_OUTPUT_DIR)
    print("✅ Fine-tuning complete!")

    return FINE_TUNE_OUTPUT_DIR

# Optional CLI
if __name__ == "__main__":
    fine_tune_insurance_llm()
