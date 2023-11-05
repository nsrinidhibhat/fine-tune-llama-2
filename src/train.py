import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training 
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer
import pandas as pd
import os
from dotenv import load_dotenv  
from huggingface_hub import login

class Train:

  def __init__(self):
    self.data = "tatsu-lab/alpaca"
    self.train_size = "train[:10]"  #Picking first 10 rows for faster training
    self.model_id = "meta-llama/Llama-2-7b-hf"

  def train_llama(self):

    # Load environment variables from .env file
    load_dotenv()
    # Access the API key using the variable name defined in the .env file
    hf_key = os.getenv("HUGGING_FACE_API_KEY")
    login(token = hf_key)

    train_dataset = load_dataset(self.data, split=self.train_size)

    base_model_id = self.model_id
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config, device_map='auto', trust_remote_code=True)
    model.resize_token_embeddings(len(tokenizer))
    model = prepare_model_for_int8_training(model)
    peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias='none', task_type="CAUSAL_LM")
    model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
      output_dir="llama-7b-tuned-alpaca",
      per_device_train_batch_size=4,
      optim="adamw_torch", 
      logging_steps=20, 
      learning_rate=2e-4, 
      fp16=True,
      warmup_ratio=0.1, 
      lr_scheduler_type="linear", 
      num_train_epochs=1,
      save_strategy="epoch", 
    )

    trainer = SFTTrainer(
      model=model, 
      train_dataset=train_dataset,
      dataset_text_field="text",
      max_seq_length=1024, 
      tokenizer=tokenizer, 
      args=training_args, 
      packing=True,
      peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model('model_ft/fine_tuned_llama-7B')
