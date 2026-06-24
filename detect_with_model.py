import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_YOLO_DIR = PROJECT_ROOT / "yolov5"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run YOLOv5 detect.py, and optionally run YOLOv5 val.py to generate "
            "overall precision/recall/mAP metrics and plots."
        )
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to the .pt weights file, for example runs/train/exp/weights/best.pt.",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Image, folder, video, or camera id for detect.py.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output folder for detect.py annotated images.",
    )
    parser.add_argument(
        "--yolo-dir",
        default=str(DEFAULT_YOLO_DIR),
        help="YOLOv5 project directory.",
    )
    parser.add_argument("--img", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="detect.py confidence threshold.")
    parser.add_argument("--device", default="0", help="Device, for example 0 or cpu.")
    parser.add_argument("--save-txt", action="store_true", help="Save detect.py prediction labels.")
    parser.add_argument("--save-conf", action="store_true", help="Save confidence in prediction labels.")
    parser.add_argument(
        "--preprocess",
        choices=["none", "gamma", "hsi_equalize", "yuv_equalize", "yuv_clahe", "gamma_hsi_equalize"],
        default="none",
        help="Optional image preprocessing for detect.py and val.py image inputs.",
    )
    parser.add_argument(
        "--preprocess-gamma",
        type=float,
        default=0.8,
        help="Gamma value for gamma preprocessing modes.",
    )
    parser.add_argument(
        "--preprocess-clahe-clip-limit",
        type=float,
        default=2.0,
        help="CLAHE clip limit for yuv_clahe preprocessing.",
    )
    parser.add_argument(
        "--preprocess-clahe-tile-size",
        type=int,
        default=8,
        help="CLAHE square tile grid size for yuv_clahe preprocessing.",
    )

    parser.add_argument(
        "--eval-data",
        default=None,
        help=(
            "Dataset yaml for YOLOv5 val.py. If set, val.py will generate overall "
            "P/R/mAP results, PR/P/R/F1 curves, and confusion matrix."
        ),
    )
    parser.add_argument(
        "--eval-split",
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split evaluated by val.py.",
    )
    parser.add_argument(
        "--eval-output",
        default=None,
        help="Output folder for val.py metrics and plots. Defaults to <output>_val.",
    )
    parser.add_argument("--eval-batch", type=int, default=16, help="Batch size for val.py.")
    parser.add_argument(
        "--eval-conf",
        type=float,
        default=0.001,
        help="val.py confidence threshold. YOLOv5 default is 0.001 for mAP calculation.",
    )
    parser.add_argument(
        "--eval-iou",
        type=float,
        default=0.6,
        help="val.py NMS IoU threshold. YOLOv5 default is 0.6.",
    )
    parser.add_argument("--eval-workers", type=int, default=0, help="Dataloader workers for val.py.")
    parser.add_argument("--eval-save-txt", action="store_true", help="Save val.py predictions as txt.")
    parser.add_argument("--eval-save-conf", action="store_true", help="Save confidence in val.py txt output.")
    return parser.parse_args()


def resolve_path(path_text):
    path = Path(path_text)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def quote_command(command):
    return " ".join(f'"{part}"' if " " in str(part) else str(part) for part in command)


def split_project_name(output):
    output_parent = output.parent
    output_name = output.name
    output_parent.mkdir(parents=True, exist_ok=True)
    return output_parent, output_name


def run_detect(args, yolo_dir, weights, output):
    detect_py = yolo_dir / "detect.py"
    if not detect_py.exists():
        raise FileNotFoundError(f"YOLOv5 detect.py was not found: {detect_py}")

    output_parent, output_name = split_project_name(output)
    command = [
        sys.executable,
        str(detect_py),
        "--weights",
        str(weights),
        "--source",
        args.source,
        "--img",
        str(args.img),
        "--conf",
        str(args.conf),
        "--device",
        args.device,
        "--preprocess",
        args.preprocess,
        "--preprocess-gamma",
        str(args.preprocess_gamma),
        "--preprocess-clahe-clip-limit",
        str(args.preprocess_clahe_clip_limit),
        "--preprocess-clahe-tile-size",
        str(args.preprocess_clahe_tile_size),
        "--project",
        str(output_parent),
        "--name",
        output_name,
        "--exist-ok",
    ]

    if args.save_txt:
        command.append("--save-txt")
    if args.save_conf:
        command.append("--save-conf")

    print("Running YOLOv5 detect.py...")
    print(quote_command(command))
    subprocess.run(command, cwd=str(yolo_dir), check=True)
    print(f"Detection results saved to: {output}")


def run_eval(args, yolo_dir, weights, detect_output):
    if not args.eval_data:
        return

    val_py = yolo_dir / "val.py"
    data_yaml = resolve_path(args.eval_data)
    eval_output = resolve_path(args.eval_output) if args.eval_output else detect_output.with_name(f"{detect_output.name}_val")

    if not val_py.exists():
        raise FileNotFoundError(f"YOLOv5 val.py was not found: {val_py}")
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset yaml for val.py was not found: {data_yaml}")

    output_parent, output_name = split_project_name(eval_output)
    command = [
        sys.executable,
        str(val_py),
        "--weights",
        str(weights),
        "--data",
        str(data_yaml),
        "--task",
        args.eval_split,
        "--img",
        str(args.img),
        "--batch-size",
        str(args.eval_batch),
        "--conf-thres",
        str(args.eval_conf),
        "--iou-thres",
        str(args.eval_iou),
        "--device",
        args.device,
        "--workers",
        str(args.eval_workers),
        "--preprocess",
        args.preprocess,
        "--preprocess-gamma",
        str(args.preprocess_gamma),
        "--preprocess-clahe-clip-limit",
        str(args.preprocess_clahe_clip_limit),
        "--preprocess-clahe-tile-size",
        str(args.preprocess_clahe_tile_size),
        "--project",
        str(output_parent),
        "--name",
        output_name,
        "--exist-ok",
        "--verbose",
    ]

    if args.eval_save_txt:
        command.append("--save-txt")
    if args.eval_save_conf:
        command.append("--save-conf")

    print("\nRunning YOLOv5 val.py for overall metrics and plots...")
    print(quote_command(command))
    subprocess.run(command, cwd=str(yolo_dir), check=True)
    print(f"Evaluation results saved to: {eval_output}")


def main():
    args = parse_args()

    yolo_dir = resolve_path(args.yolo_dir)
    weights = resolve_path(args.weights)
    output = resolve_path(args.output)

    if not weights.exists():
        raise FileNotFoundError(f"Weights file was not found: {weights}")

    run_detect(args, yolo_dir, weights, output)
    run_eval(args, yolo_dir, weights, output)


if __name__ == "__main__":
    main()
