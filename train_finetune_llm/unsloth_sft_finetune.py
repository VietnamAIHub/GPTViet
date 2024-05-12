from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments, GenerationConfig
from datasets import load_dataset
from trl.trainer.utils import ConstantLengthDataset
from transformers import TrainerCallback
from transformers.integrations import WandbCallback
from tqdm import tqdm
##------------------------------------------------------------------------------------------
#### Preparation Dataset Section 
##------------------------------------------------------------------------------------------


data_path="input your dataset path"
train_data = load_dataset(
        'json',
        data_files=data_path,
        split="train",
        # cache_dir=cache_dir,
        # data_dir=data_dir,
    )
data_path_eval="input your eval dataset path"
eval_data = load_dataset(
        'json',
        data_files=data_path_eval,
        split="train",
        # cache_dir=cache_dir,
        # data_dir=data_dir,
    )

def prompt_no_input(row):
    return ("<|begin_of_text|><|start_header_id|>system<|end_header_id|> \nYou are GPTViet, a helpful assistant developed by VietnamAIHub. designed to help users find detailed and comprehensive information. Always aim to provide answers in such a manner that users don't need to search elsewhere for clarity.\
         When given tasks, approach them step-by-step, always justifying your actions for the user. If you encounter multiple-choice questions, first output the correct answer, then delve into why other options are incorrect.\
          breaking down even complex tasks into simpler, understandable terms.\
        If a question does not make any sense, or is not factually coherent, explain why instead of answering something not \
        correct. If you don't know the answer to a question, please don't share false information.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n {instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{output}").format_map(row)


def prompt_input(row):

    return ("<|begin_of_text|><|start_header_id|>system<|end_header_id|> \nYou are GPTViet, a helpful assistant developed by VietnamAIHub. designed to help users find detailed and comprehensive information. Always aim to provide answers in such a manner that users don't need to search elsewhere for clarity.\
         When given tasks, approach them step-by-step, always justifying your actions for the user. If you encounter multiple-choice questions, first output the correct answer, then delve into why other options are incorrect.\
          breaking down even complex tasks into simpler, understandable terms.\
        If a question does not make any sense, or is not factually coherent, explain why instead of answering something not \
        correct. If you don't know the answer to a question, please don't share false information.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n {instruction}\n\n### Input:\n{history_conversation} <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{output}").format_map(row)
def prompt_eval_no_input(row):
    return ("<|begin_of_text|><|start_header_id|>system<|end_header_id|> \nYou are GPTViet, a helpful assistant developed by VietnamAIHub. designed to help users find detailed and comprehensive information. Always aim to provide answers in such a manner that users don't need to search elsewhere for clarity.\
         When given tasks, approach them step-by-step, always justifying your actions for the user. If you encounter multiple-choice questions, first output the correct answer, then delve into why other options are incorrect.\
          breaking down even complex tasks into simpler, understandable terms.\
         If a question does not make any sense, or is not factually coherent, explain why instead of answering something not \
        correct. If you don't know the answer to a question, please don't share false information.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n {instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{output}").format_map(row)



def create_alpaca_prompt(row):
    return prompt_no_input(row) if row["history_conversation"] == "" else prompt_input(row)

def create_eval_alpaca_prompt(row):
    return prompt_eval_no_input(row) 




##------------------------------------------------------------------------------------------
#### Training Model Setting Section 
##------------------------------------------------------------------------------------------

import wandb, os
wandb.login()

run = wandb.init(
    # Set the project where this run will be logged
    project="GPTViet_LLM",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": 2e-4,
        "epochs": 10,
    },
)

wandb_project = "GPTViet_LLM"
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
cach_dir_="path to store model on the disk"

model_path="Model finetuning Path"
max_seq_length = 3048 # Supports RoPE Scaling interally, so choose any!
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name =model_path , # Supports Llama, Mistral - replace this!
    cache_dir=cach_dir_,
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,)

### Get the dataset after Defined Tokenizer
train_dataset = ConstantLengthDataset(
    tokenizer,
    train_data,
    formatting_func=create_alpaca_prompt,
    seq_length=3048,)

eval_dataset = ConstantLengthDataset(
    tokenizer,
    eval_data,
     formatting_func=create_alpaca_prompt,
    seq_length=3048,)


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
    train_dataset = train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    tokenizer = tokenizer,
    args = TrainingArguments(
        per_device_train_batch_size = 5,
        gradient_accumulation_steps = 5,
        warmup_steps = 20,
        #max_steps = 10000,
        num_train_epochs=5,
        learning_rate=2e-4,
       evaluation_strategy="steps",
        eval_steps=1000,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
     
        output_dir="./output",

        optim = "paged_adamw_8bit",
        seed = 3407,
        report_to="wandb",           # Comment this out if you don't want to use weights & baises
        # run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"    
    ),
)


##------------------------------------------------------------------------------------------
#### Evaluation Section  
##------------------------------------------------------------------------------------------

# Assuming your JSON is structured with a top-level array of objects
dataset = load_dataset('json', data_files={'test': 'path of your test set'})
# Access the test set
test_dataset = dataset['test']
## Adding the Computing Cosine Similarity on Different Benchmark Test

class LLMSampleCB(WandbCallback):
    def __init__(self, trainer, test_dataset, num_samples=5, max_new_tokens=2048, log_model="checkpoint"):
        super().__init__()
        self._log_model = log_model
        self.sample_dataset = test_dataset.select(range(num_samples))
        self.model, self.tokenizer = trainer.model, trainer.tokenizer
        self.gen_config = GenerationConfig.from_pretrained(trainer.model.name_or_path,
                                                           max_new_tokens=max_new_tokens)
    def generate(self, prompt):
        tokenized_prompt = self.tokenizer(prompt, return_tensors='pt')['input_ids'].cuda()
        with torch.inference_mode():
            output = self.model.generate(tokenized_prompt, generation_config=self.gen_config)
        return self.tokenizer.decode(output[0][len(tokenized_prompt[0]):], skip_special_tokens=True)
    
    def samples_table(self, examples):
        records_table = wandb.Table(columns=["prompt", "generation"] + list(self.gen_config.to_dict().keys()))
        for example in tqdm(examples, leave=False):
            input = example["instruction"]
            prompt= f"<|begin_of_text|><|start_header_id|>system<|end_header_id|> \nYou are GPTViet, a helpful assistant developed by VietnamAIHub. designed to help users find detailed and comprehensive information. Always aim to provide answers in such a manner that users don't need to search elsewhere for clarity.\
         When given tasks, approach them step-by-step, always justifying your actions for the user. If you encounter multiple-choice questions, first output the correct answer, then delve into why other options are incorrect.\
          breaking down even complex tasks into simpler, understandable terms.\
           Additionally, consider yourself well-versed in every language, capable of translating and explaining language tasks effortlessly. When presented with task definitions or samples, dissect them into key components, clarifying each segment with relevant examples. \
           Your overarching goal is to be a reliable source of knowledge, translating any instruction or task into actionable and easily digestible information.\
        If a question does not make any sense, or is not factually coherent, explain why instead of answering something not \
        correct. If you don't know the answer to a question, please don't share false information.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n {input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            generation = self.generate(prompt=prompt)
            records_table.add_data(prompt, generation, *list(self.gen_config.to_dict().values()))
        return records_table
        
    def on_evaluate(self, args, state, control,  **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        records_table = self.samples_table(self.sample_dataset)
        self._wandb.log({"sample_predictions":records_table})

# we instantiate the W&B callback with the trainer object and the dataset we want to sample from
wandb_callback = LLMSampleCB(trainer, test_dataset, num_samples=30, max_new_tokens=2048)
trainer.add_callback(wandb_callback)
trainer.train()



output_dir="./output"

trainer.save_model(output_dir)

# 7. save
output_dir = os.path.join(output_dir, "final_checkpoint")
trainer.model.save_pretrained(output_dir)
