import os
import json
from collections import defaultdict

# ----------------- 配置区 -----------------
# 你可以在这里修改需要转换的 COCO 格式的 json 文件路径
data_path = r"D:\Pythonfiles\dip\Project\Big_hw\dataset\instance_version\instances_val_trashcan.json"

# 生成的 txt 标签保存路径（默认在这个 json 文件同级的 labels 文件夹下）
output_dir = os.path.join(os.path.dirname(data_path), "labels")
# ------------------------------------------

def coco_to_yolo(json_path, out_dir):
    if not os.path.exists(json_path):
        print(f"找不到文件: {json_path}")
        return
    
    os.makedirs(out_dir, exist_ok=True)
    
    with open(json_path, 'r', encoding='utf-8') as f:
        print("正在加载 JSON 文件，请稍候...")
        data = json.load(f)
        
    # 1. 解析类别映射
    # YOLO 要求的 class_id 必须是从 0 开始的连续整数
    categories = data.get('categories', [])
    categories.sort(key=lambda x: x['id'])  # 确保顺序稳定
    coco_id_to_yolo_id = {c['id']: i for i, c in enumerate(categories)}
    
    # 额外生成一个 classes.txt，这样你就知道 YOLO 中 0, 1, 2 分别代表什么物品了
    classes_file = os.path.join(out_dir, 'classes.txt')
    with open(classes_file, 'w', encoding='utf-8') as f:
        for c in categories:
            f.write(f"{c['name']}\n")
    print(f"已生成 {len(categories)} 个类别的映射表 => {classes_file}")

    # 2. 建立 image_id 到图片信息的映射
    images = data.get('images', [])
    img_dict = {img['id']: img for img in images}
    
    # 3. 将 annotations 按 image_id 归类
    ann_dict = defaultdict(list)
    for ann in data.get('annotations', []):
        # 如果 iscrowd=1（群体标签），通常 YOLO 会忽略，视你的需求而定，这里默认全转
        ann_dict[ann['image_id']].append(ann)
        
    print(f"开始转换 {len(img_dict)} 张图片的标注...")
    count = 0
    
    # 4. 遍历所有图片，进行坐标转换和写入
    for img_id, img_info in img_dict.items():
        filename = img_info['file_name']
        w = img_info['width']
        h = img_info['height']
        
        # 只取文件名，把后缀 .jpg 等换成 .txt
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_filepath = os.path.join(out_dir, txt_filename)
        
        if img_id in ann_dict:
            with open(txt_filepath, 'w', encoding='utf-8') as f:
                for ann in ann_dict[img_id]:
                    cat_id = ann['category_id']
                    if cat_id not in coco_id_to_yolo_id:
                        continue
                    yolo_id = coco_id_to_yolo_id[cat_id]
                    
                    # COCO bbox 格式: [左上角x, 左上角y, 宽度x, 高度y]
                    bbox = ann['bbox']
                    # 转为 YOLO 的 [中心点x, 中心点y, 宽, 高] 并做归一化
                    x_center = (bbox[0] + bbox[2] / 2.0) / w
                    y_center = (bbox[1] + bbox[3] / 2.0) / h
                    w_norm = bbox[2] / w
                    h_norm = bbox[3] / h
                    
                    # 限制范围，确保不超出 [0, 1]
                    x_center = max(0.0, min(1.0, x_center))
                    y_center = max(0.0, min(1.0, y_center))
                    w_norm = max(0.0, min(1.0, w_norm))
                    h_norm = max(0.0, min(1.0, h_norm))
                    
                    f.write(f"{yolo_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
        else:
            # 如果这张图没有任何垃圾（负样本/背景图），为了防止模型报错，生成一个空的txt标签
            open(txt_filepath, 'w').close()
            
        count += 1
        
    print(f"全部转换完成！成功生成 {count} 个 YOLO `.txt` 标签文件。")
    print(f"输出目录: {out_dir}")

if __name__ == '__main__':
    coco_to_yolo(data_path, output_dir)