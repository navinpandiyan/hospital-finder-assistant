# db/fine_tune_peft_windows.py
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
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
TARGET_MODULES = ["q_proj", "v_proj"]
LORA_TASK_TYPE = "CAUSAL_LM"

# -----------------------------
# Fine-tuning function (QLoRA)
# -----------------------------
def fine_tune_insurance_llm(data_path: str = "db/insurance_data.json"):
    print(f"🚀 Loading dataset from: {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Flatten 'context' into string
    for item in raw_data:
        context = item.pop("context", {})
        if isinstance(context, dict):
            context_str = ", ".join(
                f"{k}:{'|'.join(v) if isinstance(v, list) else v}" for k, v in context.items()
            )
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
    # Load model & tokenizer (QLoRA)
    # -----------------------------
    print(f"📦 Loading base model in 4-bit mode (QLoRA): {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=LOAD_IN_4BIT,                  # ✅ Quantized 4-bit weights
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",          # NormalFloat4 (best for QLoRA)
        bnb_4bit_compute_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="cuda:0",                  # Automatically spread across GPU/CPU
        low_cpu_mem_usage=True
    )

    model.gradient_checkpointing_enable()

    # -----------------------------
    # Prepare model for QLoRA
    # -----------------------------
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=LORA_TASK_TYPE
    )

    model = get_peft_model(model, lora_config)
    print(f"✅ QLoRA layers added via PEFT to {TARGET_MODULES}")

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
    print("✅ QLoRA fine-tuning complete!")

    return FINE_TUNE_OUTPUT_DIR


# Optional CLI
if __name__ == "__main__":
    fine_tune_insurance_llm()
