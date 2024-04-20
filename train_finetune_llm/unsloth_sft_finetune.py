from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# Get LAION dataset
url = "https://huggingface.co/datasets/laion/OIG/resolve/main/unified_chip2.jsonl"
dataset = load_dataset("json", cache_dir="/data2/cmdir/home/villm/data/unified_chip2", data_files = {"train" : url}, split = "train")

import wandb, os
wandb.login()

wandb_project = "FoxBrain_LLM"
if len(wandb_project) > 0:
    os.environ["WANDB_PROJECT"] = wandb_project

# 4bit pre quantized models we support - 4x faster downloading!
fourbit_models = [
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/llama-2-7b-bnb-4bit",
    "unsloth/llama-2-13b-bnb-4bit",
    "unsloth/codellama-34b-bnb-4bit",
    "unsloth/tinyllama-bnb-4bit",
] # Go to https://huggingface.co/unsloth for more 4-bit models!

# Load Llama model
cach_dir_="/data2/cmdir/home/villm/model_weights/mistral"
model_path="/data2/cmdir/home/villm/model_weights/mistral/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/41b61a33a2483885c981aa79e0df6b32407ed873"
max_seq_length = 2048 # Supports RoPE Scaling interally, so choose any!

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name =model_path , # Supports Llama, Mistral - replace this!
    cache_dir=cach_dir_,
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,)

# Do model patching and add fast LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], #"gate_proj", "up_proj", "down_proj",
    lora_alpha = 32,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = True,
    random_state = 3407,
    max_seq_length = max_seq_length,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    tokenizer = tokenizer,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        max_steps = 10,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        output_dir = "/data2/cmdir/home/villm/training_output/",
        optim = "adamw_8bit",
        seed = 3407,
        report_to="wandb",           # Comment this out if you don't want to use weights & baises
        # run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"    
    ),

)
trainer.train()

output_dir="/data2/cmdir/home/villm/training_output"
trainer.train()
trainer.save_model(output_dir)

# 7. save
output_dir = os.path.join(output_dir, "final_checkpoint")
trainer.model.save_pretrained(output_dir)
# Go to https://github.com/unslothai/unsloth/wiki for advanced tips like
# (1) Saving to GGUF / merging to 16bit for vLLM
# (2) Continued training from a saved LoRA adapter
# (3) Adding an evaluation loop / OOMs
# (4) Cutomized chat templates 