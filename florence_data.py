import os
import json
from collections import defaultdict

def yolo_to_phrase_grounding(yolo_images_dir, yolo_labels_dir, descriptive_classes, output_jsonl_path):
    """
    將 YOLO 格式轉為 Florence-2 <CAPTION_TO_PHRASE_GROUNDING> 任務格式
    會自動處理同一張圖中出現「多個相同物件」的情況。
    """
    with open(output_jsonl_path, 'w', encoding='utf-8') as outfile:
        
        for label_filename in os.listdir(yolo_labels_dir):
            if not label_filename.endswith('.txt'):
                continue

            image_filename = label_filename.replace('.txt', '.jpg')
            image_path = os.path.join(yolo_images_dir, image_filename)

            if not os.path.exists(image_path):
                print(f"⚠️ 找不到圖片 {image_path}，已跳過。")
                continue

            label_path = os.path.join(yolo_labels_dir, label_filename)
            
            # 使用字典來「群組化」相同類別的座標
            # 格式: { "brown box": ["<loc_1><loc_2><loc_3><loc_4>", "<loc_5>..."], "rusty screw": [...] }
            objects_by_phrase = defaultdict(list)

            with open(label_path, 'r', encoding='utf-8') as labelfile:
                for line in labelfile:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue

                    class_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])

                    # 取得你「改編」後的形容詞片語
                    if class_id >= len(descriptive_classes):
                        continue
                    phrase = descriptive_classes[class_id]

                    # 轉換為左上與右下座標，並映射到 [0, 999]
                    xmin = max(0.0, cx - (w / 2))
                    ymin = max(0.0, cy - (h / 2))
                    xmax = min(1.0, cx + (w / 2))
                    ymax = min(1.0, cy + (h / 2))

                    loc_xmin = min(999, max(0, int(round(xmin * 999))))
                    loc_ymin = min(999, max(0, int(round(ymin * 999))))
                    loc_xmax = min(999, max(0, int(round(xmax * 999))))
                    loc_ymax = min(999, max(0, int(round(ymax * 999))))

                    # 記錄這組座標
                    loc_string = f"<loc_{loc_xmin}><loc_{loc_ymin}><loc_{loc_xmax}><loc_{loc_ymax}>"
                    objects_by_phrase[phrase].append(loc_string)

            # 針對這張圖片中出現的「每一種」物件，獨立生成一行訓練資料
            for phrase, loc_list in objects_by_phrase.items():
                
                # 將「片語」與「它所有的座標」組合起來
                # 組合結果會是: "brown box<loc...><loc...>brown box<loc...><loc...>"
                combined_target_text = "".join([f"{phrase}{loc}" for loc in loc_list])
                
                # Florence-2 Phrase Grounding 格式
                data_entry = {
                    "image": image_path,
                    "prefix": f"<CAPTION_TO_PHRASE_GROUNDING>{phrase}", # 把你的形容詞放在 Prefix
                    "text": combined_target_text                       # 輸出所有的座標
                }
                
                outfile.write(json.dumps(data_entry, ensure_ascii=False) + '\n')

    print(f"✅ 完美轉換！JSONL 檔案已儲存至：{output_jsonl_path}")

# ==========================================
# 參數設定區
# ==========================================
if __name__ == "__main__":
    # 這裡就是你「改編」的地方！把原本 YOLO 冷冰冰的類別，換成自然的英文片語
    # 假設 YOLO class 0 是箱子，class 1 是螺絲
    my_descriptive_phrases = [
        "Overlapping 100ml saline bags",
        "Single 100ml saline bag"         # 對應 class_id 0
    ] 
    
    # 把 C:\...\Florence 替換成 /workspace
    images_dir = "C:\\Users\\5-005-072\\Desktop\\Florence\\data\\Cardboard.v1i.yolov8\\train\\images" 
    labels_dir = "C:\\Users\\5-005-072\\Desktop\\Florence\\data\\Cardboard.v1i.yolov8\\train\\labels"
    
    # 建議加上 /workspace/ 確保檔案存在外面的資料夾，不會隨著 Docker 關閉而消失
    output_file = "grounding_train03262.jsonl"

    yolo_to_phrase_grounding(images_dir, labels_dir, my_descriptive_phrases, output_file)