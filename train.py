import os
import json
import torch

# ==========================================
# 🚀 魔法咒語：強行繞過 Hugging Face 的 flash_attn 檢查
# ==========================================
from unittest.mock import patch
import transformers.dynamic_module_utils as dmu

_original_get_imports = dmu.get_imports

def _custom_get_imports(filename):
    imports = _original_get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports

dmu.get_imports = _custom_get_imports
# ==========================================

from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# ==========================================
# 1. 簡化版 Dataset (只負責讀資料，不負責轉換)
# ==========================================
class Florence2Dataset(Dataset):
    def __init__(self, jsonl_path):
        self.data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['image']).convert("RGB")
        return item['prefix'], item['text'], image

# ==========================================
# 2. 定義 Collate Function (負責打包 Batch 並動態對齊長度)
# ==========================================
def get_collate_fn(processor):
    def collate_fn(batch):
        # 將 batch 拆解成提示詞、答案、圖片三個列表
        prompts, target_texts, images = zip(*batch)
        
        # 處理輸入 (圖片 + Prompt)
        # padding=True 代表自動對齊這個 batch 裡最長的資料
        inputs = processor(
            text=list(prompts), 
            images=list(images), 
            return_tensors="pt", 
            padding=True 
        )
        
        # 處理標準答案 (Labels)
        labels = processor.tokenizer(
            text=list(target_texts),
            return_tensors="pt",
            padding=True
        ).input_ids
        
        inputs['labels'] = labels
        return inputs
    return collate_fn

# ==========================================
# 3. 模型與微調主程式
# ==========================================
def main():
    model_id = "microsoft/Florence-2-base-ft"
    train_file = "/workspace/grounding_train.jsonl" 
    val_file = "/workspace/grounding_val.jsonl"     
    output_dir = "./florence2-custom-model"

    print("⏳ 正在載入 Processor 與 基礎模型...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        trust_remote_code=True,
        device_map="cuda" 
    )

    print("🛠️ 正在套用 LoRA 配置...")
    config = LoraConfig(
        r=8, 
        lora_alpha=16, 
        target_modules=["q_proj", "v_proj"], 
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
        bias="none"
    )
    
    model = get_peft_model(model, config)
    model.print_trainable_parameters() 

    print("📚 正在準備資料集...")
    # 注意：這裡不再傳入 processor，因為處理邏輯移到 collate_fn 了
    train_dataset = Florence2Dataset(train_file)
    val_dataset = Florence2Dataset(val_file)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,              
        per_device_train_batch_size=2,   # 若記憶體不足，請改為 1
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,   
        learning_rate=5e-4,              
        evaluation_strategy="steps",
        eval_steps=50,                   
        save_strategy="steps",
        save_steps=50,
        logging_steps=10,                
        fp16=True,                       
        optim="adamw_torch",
        remove_unused_columns=False      
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=get_collate_fn(processor) # 掛載我們寫好的動態打包器
    )

    print("🚀 開始煉丹！(開始訓練...)")
    trainer.train()

    print(f"✅ 訓練完成！模型已儲存至：{output_dir}")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

if __name__ == "__main__":
    main()