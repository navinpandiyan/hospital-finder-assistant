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

from settings.config import FINE_TUNE_OUTPUT_DIR

# -----------------------------
# Training Hyperparameters
# -----------------------------
BASE_MODEL = "meta-llama/Llama-2-7b-hf"   # Hugging Face Mistral instruct model
TOKENIZER_MODEL = BASE_MODEL                    # Usually same as base model

BATCH_SIZE = 1                                 # Per-device batch size (lower for 7B models)
EPOCHS = 3                                     # Fine-tuning epochs
LEARNING_RATE = 2e-4                           # Learning rate
MAX_SEQ_LEN = 512                               # Max token length for inputs

# Gradient accumulation for effective batch size
GRADIENT_ACCUMULATION_STEPS = 8               # Increase for memory-constrained GPUs

# Mixed precision
FP16 = True

# -----------------------------
# Logging / Checkpoints
# -----------------------------
SAVE_STEPS = 50                                # Save checkpoint every N steps
LOGGING_STEPS = 20                              # Log every N steps

# -----------------------------
# QLoRA / LoRA Configuration
# -----------------------------
LOAD_IN_4BIT = True                            # 4-bit quantization
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj"]         # Mistral uses 'q_proj' and 'v_proj' in transformer layers
LORA_TASK_TYPE = "CAUSAL_LM"

# -----------------------------
# Optional / Advanced
# -----------------------------
USE_SAFETENSORS = True                         # Save model in safe serialization
OPTIMIZER = "adamw_torch"
GRADIENT_CHECKPOINTING = True                  # Save memory during training

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
