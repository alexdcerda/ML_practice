import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset

# Choose device: use MPS if available (for Apple Silicon), else CPU.
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Load a small text dataset (wikitext for demonstration)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
# This provides 'train', 'validation', and 'test' splits.

# 2. Initialize tokenizer and model
checkpoint = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
model = GPT2LMHeadModel.from_pretrained(checkpoint)

# GPT-2 small doesn't have pad_token, so use the eos_token as pad.
tokenizer.pad_token = tokenizer.eos_token

# Move model to the chosen device.
model.to(device)

# 3. Preprocess / tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 4. Prepare data collator (for language modeling tasks)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # GPT-2 is a causal language model.
)

# 5. Create training arguments and Trainer
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned-wikitext",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    num_train_epochs=1,   # For demonstration, run only 1 epoch.
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator
)

# 6. Fine-tune the model
trainer.train()

# 7. Generate text with the fine-tuned model
prompt_text = "how would you build a pytorch model for efficient tuning and training?"
input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)

model.eval()
with torch.no_grad():
    output_ids = model.generate(input_ids, max_length=50, do_sample=True, top_k=50)
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Generated:", generated_text)