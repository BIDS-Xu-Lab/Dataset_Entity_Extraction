from unsloth import FastLanguageModel, PatchDPOTrainer
from unsloth import is_bfloat16_supported
PatchDPOTrainer()
import torch
from trl import DPOTrainer
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

savepath = "/home/gy237/project/llama3/unsloth/fine-tuned model/unsloth_model_3.1"
max_seq_length = 8100 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 15 trillion tokens model 2x faster!
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # We also uploaded 4bit for 405b!
    "unsloth/Mistral-Nemo-Base-2407-bnb-4bit", # New Mistral 12b 2x faster!
    "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    "unsloth/mistral-7b-v0.3-bnb-4bit",        # Mistral v3 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!
]  # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)


## 加载数据集
train_file = "/home/gy237/project/llama3/unsloth/filtered_data/traindata.json"
train_data = load_dataset("json", data_files = {"train" : train_file}, split = "train")

eval_file = "/home/gy237/project/llama3/unsloth/filtered_data/testdata.json"
eval_data = load_dataset("json", data_files = {"eval" : eval_file}, split = "eval")

# # 模型扩展
# 对目标模型进行参数扩展技术的配置增强原有 FastLanguageModel模型的功能优化内存使用, 并提升处理长文本序列的能力
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 64,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 42,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    max_seq_length = max_seq_length
)


# # 设置训练环境
# 利用SFTTrainer和一系列的TeainingArguments来配置和执行自然语言模型的微调, 指定优化器，学习率调度器
# 使用了指定的数据集,优化器，学习率调度器
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_data,
    eval_dataset = eval_data,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(    # one day 58250 steps
        do_eval = True,
        eval_strategy = "steps",
        eval_steps = 500,
        save_strategy = "steps",
        save_steps = 2500,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 2,
        warmup_steps = 2500,
        max_steps = 60000,
        learning_rate = 1e-5,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 50,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 42,
        output_dir = "/home/gy237/project/llama3/unsloth/output",
    ),
)

trainer_stats = trainer.train()

# 保存模型
model.save_pretrained(savepath)

print("All over")

# 加载模型
if False:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = savepath, # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference

    # alpaca_prompt = You MUST copy from above!

    inputs = tokenizer(
    [
        alpaca_prompt.format(
            "what in famous tall tower in paris", # instruction
            "", # input
            "", # output - leave this blank for generation!
        )
    ], return_tensors = "pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
    tokenizer.batch_decode(outputs)
    print(outputs)