import os
import re
import torch
from glob import glob
from PIL import Image
from tqdm import tqdm # 用來顯示進度條

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
    # 轉為 0.0 ~ 1.0 的浮點數
    x1, y1, x2, y2 = x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0
    
    # 計算中心點與長寬
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    
    # 限制在 0-1 之間，避免稍微超出邊界
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))
    
    return x_center, y_center, w, h

def main():
    # ==========================================
    # ⚙️ 基本設定 (請依照你的需求修改)
    # ==========================================
    base_model_id = "microsoft/Florence-2-base-ft"
    # 指向工作桌上的模型資料夾
    lora_path = "/workspace/florence2-custom-model/checkpoint-1200" 
    
    # 【情境 A：你要標註原本桌面的 Cardboard 測試圖】
    input_img_dir = "/dataset/S25036-大塚製藥視覺排除疊包模組/100ml saline yolov8/valid/images" 
    
    # 【情境 B：如果你現在要大顯身手，直接標註 NAS 裡的 P600 照片！】
    # input_img_dir = "/dataset/某個裝照片的子資料夾名稱" 
    
    # 輸出的標註檔存回你的桌面，方便你用 Windows 查看
    output_txt_dir = "/workspace/yolo_labels"
    
    target_text = "Single 100ml saline bag"              # 你要找的物件名稱
    task_prompt = f"<CAPTION_TO_PHRASE_GROUNDING>{target_text}"
    yolo_class_id = 0                # 對應到 YOLO 的 class id (例如 0 代表 box)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ 目前使用的運算設備: {device}")

    # ==========================================
    # 1. 載入模型與權重
    # ==========================================
    print("⏳ 正在載入基礎模型與 Processor...")
    processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id, trust_remote_code=True).to(device)

    print("🛠️ 正在掛載 LoRA 權重...")
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval() # 設定為評估模式

    # 確保輸出資料夾存在
    os.makedirs(output_txt_dir, exist_ok=True)

    # 抓取所有圖片路徑 (支援 jpg, png 等)
    image_paths = glob(os.path.join(input_img_dir, "*.[jp][pn]*[g]"))
    print(f"📁 總共找到 {len(image_paths)} 張圖片，開始進行自動標註...")

    # ==========================================
    # 2. 開始批次處理與自動標註
    # ==========================================
    for img_path in tqdm(image_paths):
        # 讀取圖片
        image = Image.open(img_path).convert("RGB")
        
        # 將輸入轉換為模型可以吃進去的 Tensor
        inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(device)
        
        # 進行推理 (產生預測文字)
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024, # 設大一點，避免圖中框太多被截斷
                early_stopping=False,
                do_sample=False,
                num_beams=3
            )
        
        # 將 Token 解碼回人類看得懂的文字 (例如: box<loc_144><loc_25><loc_911><loc_955>)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        
        # ------------------------------------------
        # 3. 解析預測文字並轉換為 YOLO 格式
        # ------------------------------------------
        # 使用正規表達式 (Regex) 抓出所有 <loc_X><loc_Y><loc_X><loc_Y> 的組合
        # Florence-2 座標順序為: [x1, y1, x2, y2]
        matches = re.findall(r"<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>", generated_text)
        
        # 建立輸出的 txt 檔案名稱 (把附檔名 .jpg/.png 換成 .txt)
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(output_txt_dir, f"{base_name}.txt")
        
        with open(txt_path, "w", encoding="utf-8") as f:
            for match in matches:
                # 取得原本 0~1000 的座標
                x1, y1, x2, y2 = map(int, match)
                
                # 轉換成 YOLO 格式
                x_center, y_center, w, h = florence_to_yolo(x1, y1, x2, y2)
                
                # 寫入 txt 檔
                f.write(f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

    print(f"✅ 自動標註完成！所有 YOLO 標註檔已儲存至：{output_txt_dir}")

if __name__ == "__main__":
    main()