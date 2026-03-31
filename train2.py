import os
import json
import torch
import argparse
import logging
from PIL import Image
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
from transformers import EarlyStoppingCallback
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        # 【工業級升級】加入防護網，防止單一壞圖摧毀整個訓練
        try:
            image = Image.open(item['image']).convert("RGB")
        except Exception as e:
            logger.warning(f"圖片讀取失敗: {item['image']}, 錯誤: {e}. 建立全黑替代圖片。")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            
        return item['prefix'], item['text'], image

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


def parse_args():
    # 【工業級升級】使用 argparse 動態接收外部指令
    parser = argparse.ArgumentParser(description="Florence-2 Industrial Training Script")
    parser.add_argument("--model_id", type=str, default="microsoft/Florence-2-base-ft")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--val_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=200)
    return parser.parse_args()

def main():
    args = parse_args()

    # 只有主 GPU 才會印出準備訊息，避免畫面被洗版
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        logger.info(f"⏳ 正在載入 Processor 與 基礎模型: {args.model_id}...")

    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, trust_remote_code=True, device_map="cuda")

    config = LoraConfig(
        r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], 
        task_type="CAUSAL_LM", lora_dropout=0.05, bias="none"
    )
    model = get_peft_model(model, config)

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        model.print_trainable_parameters() 
        logger.info("📚 正在準備資料集...")

    train_dataset = Florence2Dataset(args.train_file)
    val_dataset = Florence2Dataset(args.val_file)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,             
        evaluation_strategy="steps",     
        eval_steps=100,                   
        save_strategy="steps",           
        save_steps=100,                   
        load_best_model_at_end=True,     
        metric_for_best_model="eval_loss", 
        greater_is_better=False,          
        per_device_train_batch_size=args.batch_size,   
        gradient_accumulation_steps=4,   
        learning_rate=args.lr,                              
        logging_steps=10,                
        fp16=True,                       
        optim="adamw_torch",
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to="tensorboard" # 【工業級升級】開啟儀表板追蹤
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=get_collate_fn(processor),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        logger.info("🚀 開始煉丹！(開始訓練...)")
        
    # 【工業級升級】自動尋找 Checkpoint 斷點續傳
    trainer.train(resume_from_checkpoint=True) 

    # 【工業級升級】只有 Rank 0 (主 GPU) 有資格進行最終存檔！
    if trainer.is_world_process_zero():
        logger.info(f"✅ 訓練完成！模型已儲存至：{args.output_dir}")
        model.save_pretrained(args.output_dir)
        processor.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()