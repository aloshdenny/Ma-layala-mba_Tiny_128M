# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments

tokenizer = AutoTokenizer.from_pretrained("ai21labs/Jamba-tiny-random")
model = AutoModelForCausalLM.from_pretrained("ai21labs/Jamba-tiny-random",  trust_remote_code=True)

from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig

dataset=load_dataset("rajeshradhakrishnan/malayalam_wiki")

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=3,
    logging_dir='./logs',
    logging_steps=10,
    learning_rate=2e-3
)

lora_config = LoraConfig(
    r=8,
    target_modules=["embed_tokens", "x_proj", "in_proj", "out_proj"],
    task_type="CAUSAL_LM",
    bias="none"
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    peft_config=lora_config,
    train_dataset=dataset["train"],
    dataset_text_field="text",
)

trainer.train()