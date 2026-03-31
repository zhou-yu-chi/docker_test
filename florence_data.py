import os
import json
import random
from collections import defaultdict

def extract_data_from_project(yolo_images_dir, yolo_labels_dir, descriptive_classes):
    """
    修改為：不直接寫入檔案，而是回傳處理好的字典清單 (List of Dictionaries)
    """
    project_entries = []
    
    for label_filename in os.listdir(yolo_labels_dir):
        if not label_filename.endswith('.txt'):
            continue

        # 支援 jpg, png 等常見格式
        image_filename = label_filename.replace('.txt', '.jpg')
        image_path = os.path.join(yolo_images_dir, image_filename)

        if not os.path.exists(image_path):
            # 如果 .jpg 找不到，試試看 .png
            image_filename_png = label_filename.replace('.txt', '.png')
            image_path_png = os.path.join(yolo_images_dir, image_filename_png)
            if os.path.exists(image_path_png):
                image_path = image_path_png
            else:
                print(f"⚠️ 找不到圖片 {image_path}，已跳過。")
                continue

        label_path = os.path.join(yolo_labels_dir, label_filename)
        objects_by_phrase = defaultdict(list)

        with open(label_path, 'r', encoding='utf-8') as labelfile:
            for line in labelfile:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                class_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])

                # 取得該專案對應的文字
                if class_id >= len(descriptive_classes):
                    continue
                phrase = descriptive_classes[class_id]

                # 轉換座標 (0~999)
                xmin = max(0.0, cx - (w / 2))
                ymin = max(0.0, cy - (h / 2))
                xmax = min(1.0, cx + (w / 2))
                ymax = min(1.0, cy + (h / 2))

                loc_xmin = min(999, max(0, int(round(xmin * 999))))
                loc_ymin = min(999, max(0, int(round(ymin * 999))))
                loc_xmax = min(999, max(0, int(round(xmax * 999))))
                loc_ymax = min(999, max(0, int(round(ymax * 999))))

                loc_string = f"<loc_{loc_xmin}><loc_{loc_ymin}><loc_{loc_xmax}><loc_{loc_ymax}>"
                objects_by_phrase[phrase].append(loc_string)

        # 將處理好的資料放入清單
        for phrase, loc_list in objects_by_phrase.items():
            combined_target_text = "".join([f"{phrase}{loc}" for loc in loc_list])
            data_entry = {
                "image": image_path,
                "prefix": f"<CAPTION_TO_PHRASE_GROUNDING>{phrase}",
                "text": combined_target_text
            }
            project_entries.append(data_entry)
            
    return project_entries

# ==========================================
# 🚀 參數設定區 (支援多專案自動合併與切割)
# ==========================================
if __name__ == "__main__":
    
    # 輸出的訓練集與驗證集路徑
    train_file = "/mnt/nfs/data/master_grounding_train.jsonl"
    val_file = "/mnt/nfs/data/master_grounding_val.jsonl"
    
    # 設定驗證集的比例 (0.2 代表 20% 驗證集，80% 訓練集)
    val_ratio = 0.2 
    
    projects = [
        {
            "name": "大塚100製藥_點滴",
            "images_dir": "/prefactor/data/data_1/100ml_saline/images",
            "labels_dir": "/prefactor/data/data_1/100ml_saline/labels",
            "classes": ["Overlapping 100ml saline bags", "Single 100ml saline bag"] 
        },
        {
            "name": "大塚250製藥_點滴",
            "images_dir": "/prefactor/data/250 ml saline yolov8/train/images",
            "labels_dir": "/prefactor/data/250 ml saline yolov8/train/labels",
            "classes": ["Overlapping 250ml saline bags", "Single 250ml saline bag"] 
        },
        {
            "name": "大塚500製藥_點滴",
            "images_dir": "/prefactor/data/500ml_saline_yolov8/500ml saline yolov8/train/images",
            "labels_dir": "/prefactor/data/500ml_saline_yolov8/500ml saline yolov8/train/labels",
            "classes": ["Overlapping 500ml saline bags", "Single 500ml saline bag"] 
        },
        {
             "name": "筆",
             "images_dir": "/prefactor/data/data_1/2025-Computex展-PCB檢測 & 餅乾檢測 & 筆角度 & 咖啡豆正反面/2025-Computex展-PCB檢測 & 餅乾檢測 & 筆角度 & 咖啡豆正反面/aetina_pen/images/train",
             "labels_dir": "/prefactor/data/data_1/2025-Computex展-PCB檢測 & 餅乾檢測 & 筆角度 & 咖啡豆正反面/2025-Computex展-PCB檢測 & 餅乾檢測 & 筆角度 & 咖啡豆正反面/aetina_pen/labels/train",
             "classes": ["gray pen"] 
        },
        {
             "name": "咖啡豆",
             "images_dir": "/prefactor/data/data_1/2025-Computex展-PCB檢測 & 餅乾檢測 & 筆角度 & 咖啡豆正反面/2025-Computex展-PCB檢測 & 餅乾檢測 & 筆角度 & 咖啡豆正反面/coffeebeans/images/train",
             "labels_dir": "/prefactor/data/data_1/2025-Computex展-PCB檢測 & 餅乾檢測 & 筆角度 & 咖啡豆正反面/2025-Computex展-PCB檢測 & 餅乾檢測 & 筆角度 & 咖啡豆正反面/coffeebeans/labels/train",
             "classes": ["good coffee beans","Broken coffee beans"] 
        },
        {
             "name": "北捐",
             "images_dir": "/prefactor/data/data_1/tp_blood/tp_blood/images",
             "labels_dir": "/prefactor/data/data_1/tp_blood/tp_blood/labels",
             "classes": ["SST採血管"] 
        },
        {
             "name": "toolbox",
             "images_dir": "/prefactor/data/toolbox/toolbox/images",
             "labels_dir": "/prefactor/data/toolbox/toolbox/labels",
             "classes": ["screw","plier","hammer","wrench","screwdriver"] 
        },
        {
             "name": "虎門_牙孔",
             "images_dir": "/prefactor/data/虎門_牙孔/虎門_牙孔/images",
             "labels_dir": "/prefactor/data/虎門_牙孔/虎門_牙孔/labels",
             "classes": ["Debris inside the threaded hole"] 
        },
        {
             "name": "藥丸",
             "images_dir": "/prefactor/data/藥丸/藥丸/images",
             "labels_dir": "/prefactor/data/藥丸/藥丸/labels",
             "classes": ["Paran_Front","Paran_Back","Acetal_Front","Acetal_Back","Piant_Front","Piant_Back","Panadol_Front","Panadol_Back","Splotin_Front","Splotin_Back","Somin_Front","Somin_Back","Mopride_Front","Mopride_Back","Mysozyme_Front","Mysozyme_Back","Weisufu_Front","Weisufu_Back","Biofermin_Front","Biofermin_Back","Prednisolone_Front","Prednisolone_Back_085516","Perofen","Troches_Front","Troches_Back","Amoxicillin_Cap","Amoxicillin_Synmosa","Unfradine","Sulpiho","Yonice_Collagen","Undiarrhea","Anpin","Lopedin"]
        },
        {
             "name": "束聯_金屬刮傷",
             "images_dir": "/prefactor/data/束聯/束聯/images",
             "labels_dir": "/prefactor/data/束聯/束聯/labels",
             "classes": ["Metal_Scratch"]
        }
    ]

    # 用來裝所有專案合併後的大清單
    all_data = []

    print("🚀 開始收集專案資料...")
    for p in projects:
        print(f"🔄 正在處理專案: 【{p['name']}】...")
        entries = extract_data_from_project(
            yolo_images_dir=p["images_dir"], 
            yolo_labels_dir=p["labels_dir"], 
            descriptive_classes=p["classes"]
        )
        all_data.extend(entries)
        print(f"✅ 【{p['name']}】 讀取完成！共抓到 {len(entries)} 筆標註資料。\n")

    # 1. 將所有資料徹底洗牌 (Random Shuffle)
    print("🔀 正在將所有專案的資料進行隨機洗牌...")
    random.shuffle(all_data)

    # 2. 計算切分點 (80% 訓練，20% 驗證)
    total_len = len(all_data)
    val_len = int(total_len * val_ratio)
    train_len = total_len - val_len

    train_data = all_data[:train_len]
    val_data = all_data[train_len:]

    # 3. 分別寫入兩個獨立的 JSONL 檔案
    print(f"💾 正在寫入 訓練集 ({train_len} 筆) 至 {train_file}...")
    with open(train_file, 'w', encoding='utf-8') as f:
        for entry in train_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"💾 正在寫入 驗證集 ({val_len} 筆) 至 {val_file}...")
    with open(val_file, 'w', encoding='utf-8') as f:
        for entry in val_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"🎉 恭喜！多專案合併與 {100-val_ratio*100}% / {val_ratio*100}% 切割已完美完成！")