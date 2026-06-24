from __future__ import annotations

import argparse
import shutil
from pathlib import Path


CLASSES = {
    0: "trash_bottle",
    1: "trash_bag",
    2: "trash_can",
    3: "trash_cup",
}


def read_yolo(path: Path, has_conf: bool, conf_thres: float) -> list[dict]:
    rows = []
    if not path.exists() or path.name == "classes.txt":
        return rows

    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 5:
            continue

        cls = int(float(parts[0]))
        box = tuple(map(float, parts[1:5]))
        conf = float(parts[5]) if len(parts) > 5 else None
        if has_conf and conf is not None and conf < conf_thres:
            continue
        rows.append({"line": line_no, "cls": cls, "box": box, "conf": conf})

    return rows


def xywh_to_xyxy(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x, y, w, h = box
    return x - w / 2, y - h / 2, x + w / 2, y + h / 2


def iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(box_a)
    bx1, by1, bx2, by2 = xywh_to_xyxy(box_b)

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom else 0.0


def class_name(cls: int) -> str:
    return CLASSES.get(cls, str(cls))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument("--pred", required=True, type=Path)
    parser.add_argument("--conf", default=0.25, type=float)
    parser.add_argument("--iou", default=0.5, type=float)
    parser.add_argument("--copy-bad-to", type=Path)
    args = parser.parse_args()

    image_names = sorted(
        p.stem for p in args.images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )

    summary = []
    for name in image_names:
        gt = read_yolo(args.gt / f"{name}.txt", has_conf=False, conf_thres=args.conf)
        pred = read_yolo(args.pred / f"{name}.txt", has_conf=True, conf_thres=args.conf)

        matched_gt: set[int] = set()
        matched_pred: set[int] = set()
        pairs = []
        for gt_idx, gt_box in enumerate(gt):
            for pred_idx, pred_box in enumerate(pred):
                if gt_box["cls"] == pred_box["cls"]:
                    pairs.append((iou(gt_box["box"], pred_box["box"]), gt_idx, pred_idx))

        matches = []
        for overlap, gt_idx, pred_idx in sorted(pairs, reverse=True):
            if overlap >= args.iou and gt_idx not in matched_gt and pred_idx not in matched_pred:
                matched_gt.add(gt_idx)
                matched_pred.add(pred_idx)
                matches.append((gt_idx, pred_idx, overlap))

        misses = []
        for gt_idx, gt_box in enumerate(gt):
            if gt_idx in matched_gt:
                continue
            best = max(
                ((iou(gt_box["box"], pred_box["box"]), pred_idx, pred_box) for pred_idx, pred_box in enumerate(pred)),
                default=(0.0, None, None),
                key=lambda item: item[0],
            )
            misses.append((gt_idx, gt_box, best))

        false_preds = []
        for pred_idx, pred_box in enumerate(pred):
            if pred_idx in matched_pred:
                continue
            best = max(
                ((iou(pred_box["box"], gt_box["box"]), gt_idx, gt_box) for gt_idx, gt_box in enumerate(gt)),
                default=(0.0, None, None),
                key=lambda item: item[0],
            )
            false_preds.append((pred_idx, pred_box, best))

        if gt or pred:
            bad_score = len(misses) * 3 + len(false_preds)
            summary.append((bad_score, name, gt, pred, matches, misses, false_preds))

    summary.sort(reverse=True, key=lambda item: (item[0], len(item[6]), len(item[5])))
    copied = []
    for _, name, _gt, _pred, _matches, misses, false_preds in summary:
        if not misses and not false_preds:
            continue
        if args.copy_bad_to:
            image_src = next(
                (p for p in args.images.iterdir() if p.stem == name and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}),
                None,
            )
            label_src = args.gt / f"{name}.txt"
            image_dst_dir = args.copy_bad_to / "images"
            label_dst_dir = args.copy_bad_to / "labels"
            image_dst_dir.mkdir(parents=True, exist_ok=True)
            label_dst_dir.mkdir(parents=True, exist_ok=True)
            if image_src:
                shutil.copy2(image_src, image_dst_dir / image_src.name)
            if label_src.exists():
                shutil.copy2(label_src, label_dst_dir / label_src.name)
            copied.append(name)
    if args.copy_bad_to:
        print(f"Done: {args.copy_bad_to}")
    else:
        print("Done.")


if __name__ == "__main__":
    main()
