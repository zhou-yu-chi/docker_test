import os
import numpy as np
from glob import glob

def yolo_to_corners(x_center, y_center, w, h):
    """將 YOLO 格式轉為左上角(x1, y1)與右下角(x2, y2)座標，方便計算面積"""
    x1 = x_center - w / 2
    y1 = y_center - h / 2
    x2 = x_center + w / 2
    y2 = y_center + h / 2
    return [x1, y1, x2, y2]

def calculate_iou(box1, box2):
    """計算兩個框的 IoU (交併比)"""
    # 取得重疊區域的座標
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # 如果沒有重疊
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # 計算交集面積
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # 計算各自的面積
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 計算聯集面積
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area

def read_yolo_labels(file_path):
    """讀取 YOLO txt 檔，回傳 box 列表"""
    boxes = []
    if not os.path.exists(file_path):
        return boxes
    with open(file_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) >= 5: # class, x, y, w, h
                # 取出座標並轉換格式
                x, y, w, h = map(float, parts[1:5])
                boxes.append(yolo_to_corners(x, y, w, h))
    return boxes

def main():
    # ==========================================
    # ⚙️ 請在這裡填入你的資料夾路徑 (記得加 r)
    # ==========================================
    gt_dir = r"C:\Users\5-005-072\Desktop\Florence\my_labels"    # 你 100% 準確的手標 txt 資料夾
    pred_dir = r"C:\Users\5-005-072\Desktop\Florence\yolo_labels" # 剛才模型產生的 txt 資料夾
    
    iou_threshold = 0.5 # 認定為「圈中」的及格線 (通常設 0.5)
    
    # 統計用變數
    total_gt_boxes = 0   # 真實存在的總框數
    total_pred_boxes = 0 # 模型圈出的總框數
    true_positives = 0   # 模型圈對的數量
    iou_list = []        # 記錄所有配對成功的 IoU 用來算平均

    # 抓取你手標資料夾內所有的 txt 檔
    gt_files = glob(os.path.join(gt_dir, "*.txt"))
    
    print("⏳ 開始比對計算評分...\n")

    for gt_file in gt_files:
        filename = os.path.basename(gt_file)
        pred_file = os.path.join(pred_dir, filename)

        gt_boxes = read_yolo_labels(gt_file)
        pred_boxes = read_yolo_labels(pred_file)

        total_gt_boxes += len(gt_boxes)
        total_pred_boxes += len(pred_boxes)

        # 為了避免同一個預測框重複配對，做個標記
        matched_pred_indices = set()

        # 對每一個真實框，去模型預測的框裡面找「最像的」(IoU 最高)
        for gt_box in gt_boxes:
            best_iou = 0
            best_pred_idx = -1

            for i, pred_box in enumerate(pred_boxes):
                if i in matched_pred_indices:
                    continue # 這個預測框已經配對過了
                
                iou = calculate_iou(gt_box, pred_box)
                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = i

            # 如果最高的 IoU 大於及格線，算作「答對了 (True Positive)」
            if best_iou >= iou_threshold:
                true_positives += 1
                matched_pred_indices.add(best_pred_idx)
                iou_list.append(best_iou)

    # ==========================================
    # 📊 計算最終成績
    # ==========================================
    precision = true_positives / total_pred_boxes if total_pred_boxes > 0 else 0
    recall = true_positives / total_gt_boxes if total_gt_boxes > 0 else 0
    
    # F1-Score 是綜合精確率與召回率的終極分數
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    avg_iou = np.mean(iou_list) if iou_list else 0

    print("================ 🏆 模型成績單 ================")
    print(f"🔸 你的標準答案總共有: {total_gt_boxes} 個框")
    print(f"🔸 模型總共圈出了: {total_pred_boxes} 個框")
    print(f"🔸 模型成功圈對了: {true_positives} 個框 (IoU >= {iou_threshold})")
    print("-----------------------------------------------")
    print(f"🎯 平均 IoU (圈得準不準): {avg_iou:.2%} (只計算有圈中的部分)")
    print(f"✅ 精確率 Precision (沒亂圈的比例): {precision:.2%}")
    print(f"🔍 召回率 Recall (沒漏圈的比例): {recall:.2%}")
    print(f"🌟 綜合表現 F1-Score: {f1_score:.2%}")
    print("===============================================")

if __name__ == "__main__":
    main()