import os
import json
from collections import defaultdict

data_path = r"D:\Pythonfiles\dip\Project\Big_hw\dataset\instance_version\instances_val_trashcan.json"

output_dir = os.path.join(os.path.dirname(data_path), "labels")

def coco_to_yolo(json_path, out_dir):
    if not os.path.exists(json_path):
        print(f"Failed: file not found: {json_path}")
        return
    
    os.makedirs(out_dir, exist_ok=True)
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    categories = data.get('categories', [])
    categories.sort(key=lambda x: x['id'])
    coco_id_to_yolo_id = {c['id']: i for i, c in enumerate(categories)}
    
    classes_file = os.path.join(out_dir, 'classes.txt')
    with open(classes_file, 'w', encoding='utf-8') as f:
        for c in categories:
            f.write(f"{c['name']}\n")
    images = data.get('images', [])
    img_dict = {img['id']: img for img in images}
    
    ann_dict = defaultdict(list)
    for ann in data.get('annotations', []):
        ann_dict[ann['image_id']].append(ann)
        
    count = 0
    
    for img_id, img_info in img_dict.items():
        filename = img_info['file_name']
        w = img_info['width']
        h = img_info['height']
        
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_filepath = os.path.join(out_dir, txt_filename)
        
        if img_id in ann_dict:
            with open(txt_filepath, 'w', encoding='utf-8') as f:
                for ann in ann_dict[img_id]:
                    cat_id = ann['category_id']
                    if cat_id not in coco_id_to_yolo_id:
                        continue
                    yolo_id = coco_id_to_yolo_id[cat_id]
                    
                    bbox = ann['bbox']
                    x_center = (bbox[0] + bbox[2] / 2.0) / w
                    y_center = (bbox[1] + bbox[3] / 2.0) / h
                    w_norm = bbox[2] / w
                    h_norm = bbox[3] / h
                    
                    x_center = max(0.0, min(1.0, x_center))
                    y_center = max(0.0, min(1.0, y_center))
                    w_norm = max(0.0, min(1.0, w_norm))
                    h_norm = max(0.0, min(1.0, h_norm))
                    
                    f.write(f"{yolo_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
        else:
            open(txt_filepath, 'w').close()
            
        count += 1
        
    print(f"Done: {out_dir}")

if __name__ == '__main__':
    coco_to_yolo(data_path, output_dir)
