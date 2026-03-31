import os
import re
import torch
from glob import glob
from PIL import Image
from tqdm import tqdm

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

from transformers import AutoProcessor, AutoModelForCausalLM
from peft import PeftModel

def florence_to_yolo(x1, y1, x2, y2):
    """
    將 Florence-2 的座標 (0-1000) 轉換為 YOLO 格式 (0-1 之間的標準化數值)
    """
    x1, y1, x2, y2 = x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0
    
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))
    
    return x_center, y_center, w, h

def main():
    # ==========================================
    # ⚙️ 基本設定
    # ==========================================
    base_model_id = "microsoft/Florence-2-base-ft"
    lora_path = r"C:\Users\5-005-072\Desktop\Florence\model" 
    
    input_img_dir = r"C:\Users\5-005-072\Documents\AIQC_Data\projects\scratch\dataset\original\images"
    output_txt_dir = r"C:\Users\5-005-072\Documents\AIQC_Data\projects\scratch\dataset\original\labels" # 你要存 YOLO txt 的資料夾 (如果不存在會自動建立)
    
    # 🌟 【關鍵修改】建立你要找的目標與 YOLO ID 的對應表 (字典)
    # 格式為: {"Florence 要找的文字": YOLO的類別ID}
    targets_dict = {
        "White scratch on the metal surface": 0
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ 目前使用的運算設備: {device}")

    # ==========================================
    # 1. 載入模型與權重 (省略部分印出訊息保持版面簡潔)
    # ==========================================
    processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id, trust_remote_code=True).to(device)
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()

    os.makedirs(output_txt_dir, exist_ok=True)
    image_paths = glob(os.path.join(input_img_dir, "*.[jp][pn]*[g]"))
    print(f"📁 總共找到 {len(image_paths)} 張圖片，開始進行多類別自動標註...")

    # ==========================================
    # 2. 開始批次處理與自動標註
    # ==========================================
    for img_path in tqdm(image_paths):
        image = Image.open(img_path).convert("RGB")
        
        # 準備一個清單，用來收集這張圖片「所有類別」的 YOLO 標註結果
        all_yolo_lines = []
        
        # 🌟 【關鍵修改】對同一張圖片，輪流詢問不同的目標
        for target_text, class_id in targets_dict.items():
            task_prompt = f"<CAPTION_TO_PHRASE_GROUNDING>{target_text}"
            
            inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(device)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    early_stopping=False,
                    #do_sample=False,+num_beams=3 標精準度，減少誤報，但可能漏掉一些框
                    #do_sample=True, num_beams=1 標召回率，可能會多報一些框，但不漏掉真正的框
                    do_sample=False,
                    num_beams=4
                )
            
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            matches = re.findall(r"<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>", generated_text)
            
            # 將這個類別找到的框，轉換後加入清單中
            for match in matches:
                x1, y1, x2, y2 = map(int, match)
                x_center, y_center, w, h = florence_to_yolo(x1, y1, x2, y2)
                # 這裡會自動套用迴圈當下的 class_id (0, 1, 或 2)
                all_yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

        # ------------------------------------------
        # 3. 將這張圖片所有類別的結果，一次性寫入 txt 檔
        # ------------------------------------------
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(output_txt_dir, f"{base_name}.txt")
        
        with open(txt_path, "w", encoding="utf-8") as f:
            f.writelines(all_yolo_lines)

    print(f"✅ 自動標註完成！所有 YOLO 標註檔已儲存至：{output_txt_dir}")

if __name__ == "__main__":
    main()