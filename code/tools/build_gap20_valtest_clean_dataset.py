import csv
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(r"D:\Pythonfiles\dip\Project\Big_hw")
SOURCE_DATASET = ROOT / "Datasets" / "new_data"
OUTPUT_DATASET = ROOT / "Datasets" / "new_data_gap20_valtest_clean"
OUTPUT_YAML = ROOT / "Datasets" / "new_gap20_valtest_clean.yaml"
ASSIGNMENTS_CSV = OUTPUT_DATASET / "gap20_valtest_assignments.csv"
SUMMARY_TXT = OUTPUT_DATASET / "gap20_valtest_split_summary.txt"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = ("train", "val", "test")
GAP = 20
SEED = 20260614
TARGET_TEST = 132
TARGET_VAL = 76


@dataclass(frozen=True)
class Entry:
    old_split: str
    group: str
    frame: int
    image_path: Path
    label_path: Path

    @property
    def name(self):
        return self.image_path.name


def parse_group_frame(path):
    stem = path.stem
    if "_frame" not in stem:
        raise ValueError(f"Cannot parse group/frame from file name: {path.name}")
    group, frame_text = stem.split("_frame", 1)
    return group, int(frame_text)


def collect_entries(source_root):
    entries = []
    for split in SPLITS:
        image_dir = source_root / "images" / split
        label_dir = source_root / "labels" / split
        if not image_dir.exists():
            raise FileNotFoundError(f"Missing image split: {image_dir}")
        if not label_dir.exists():
            raise FileNotFoundError(f"Missing label split: {label_dir}")

        for image_path in sorted(image_dir.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            group, frame = parse_group_frame(image_path)
            label_path = label_dir / image_path.relative_to(image_dir).with_suffix(".txt")
            if not label_path.exists():
                raise FileNotFoundError(f"Missing label for {image_path}: {label_path}")
            entries.append(Entry(split, group, frame, image_path, label_path))
    return entries


def is_near(a, b, gap=GAP):
    return a.group == b.group and abs(a.frame - b.frame) <= gap


def near_any(entry, others, gap=GAP):
    return any(is_near(entry, other, gap) for other in others)


def exclusion_set(candidate, pool, gap=GAP):
    return {entry for entry in pool if is_near(candidate, entry, gap)}


def select_test_additions(base_test, pool, target_test, seed):
    selected = list(base_test)
    available = [entry for entry in pool if not near_any(entry, selected)]
    excluded = {entry for entry in pool if entry not in available}
    excluded_reasons = {entry: "within_gap20_of_old_val_test" for entry in excluded}

    needed = target_test - len(selected)
    if needed < 0:
        raise ValueError(f"Base test is larger than target: {len(selected)} > {target_test}")

    rng = random.Random(seed)
    group_order = list({entry.group for entry in available})
    rng.shuffle(group_order)
    selected_groups = set()

    while len(selected) < target_test:
        if not available:
            raise RuntimeError(f"Cannot fill test set to {target_test}; reached {len(selected)}")

        group_candidates = []
        for group in group_order:
            candidates = [entry for entry in available if entry.group == group]
            if candidates:
                group_candidates = candidates
                break
        if not group_candidates:
            group_order = list({entry.group for entry in available})
            rng.shuffle(group_order)
            continue

        # Prefer one test sample per group, and within a group choose the sample
        # that sacrifices the fewest neighboring training-pool samples.
        candidate = min(
            group_candidates,
            key=lambda entry: (
                group_candidates[0].group in selected_groups,
                len(exclusion_set(entry, available)),
                entry.old_split != "test",
                entry.frame,
                entry.name,
            ),
        )
        selected.append(candidate)
        selected_groups.add(candidate.group)

        blocked = exclusion_set(candidate, available)
        for entry in blocked:
            if entry != candidate:
                excluded.add(entry)
                excluded_reasons[entry] = f"within_gap20_of_added_test:{candidate.name}"
        available = [entry for entry in available if entry not in blocked]
        group_order = [group for group in group_order if group != candidate.group]

    return selected, available, excluded, excluded_reasons


def make_gap_blocks(entries):
    by_group = defaultdict(list)
    for entry in entries:
        by_group[entry.group].append(entry)

    blocks = []
    for group, group_entries in by_group.items():
        sorted_entries = sorted(group_entries, key=lambda entry: (entry.frame, entry.name))
        current = []
        previous = None
        for entry in sorted_entries:
            if previous is not None and entry.frame - previous.frame > GAP:
                blocks.append(current)
                current = []
            current.append(entry)
            previous = entry
        if current:
            blocks.append(current)
    return blocks


def assign_val_blocks(remaining_pool, target_val, seed):
    blocks = make_gap_blocks(remaining_pool)
    rng = random.Random(seed)
    rng.shuffle(blocks)
    blocks.sort(key=lambda block: (abs(len(block) - 3), len(block), block[0].group, block[0].frame))

    val_blocks = []
    val_count = 0
    for block in blocks:
        if val_count >= target_val:
            break
        if val_count + len(block) <= target_val or target_val - val_count >= len(block) / 2:
            val_blocks.append(block)
            val_count += len(block)

    val_entries = {entry for block in val_blocks for entry in block}
    train_entries = [entry for entry in remaining_pool if entry not in val_entries]
    return list(val_entries), train_entries, blocks


def validate_gap(assignments):
    by_group = defaultdict(list)
    for entry, split in assignments.items():
        by_group[entry.group].append((entry, split))

    violations = []
    nearest = None
    for group_entries in by_group.values():
        for i, (a, split_a) in enumerate(group_entries):
            for b, split_b in group_entries[i + 1 :]:
                if split_a == split_b:
                    continue
                distance = abs(a.frame - b.frame)
                if nearest is None or distance < nearest[0]:
                    nearest = (distance, a, split_a, b, split_b)
                if distance <= GAP:
                    violations.append((distance, a, split_a, b, split_b))
    return violations, nearest


def reset_output(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def copy_entries(assignments, output_root):
    for entry, split in assignments.items():
        target_image = output_root / "images" / split / entry.image_path.name
        target_label = output_root / "labels" / split / entry.label_path.name
        target_image.parent.mkdir(parents=True, exist_ok=True)
        target_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(entry.image_path, target_image)
        shutil.copy2(entry.label_path, target_label)


def count_output(output_root):
    counts = {}
    for split in SPLITS:
        image_count = sum(
            1
            for path in (output_root / "images" / split).rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        label_count = sum(1 for path in (output_root / "labels" / split).rglob("*.txt") if path.is_file())
        counts[split] = (image_count, label_count)
        if image_count != label_count:
            raise ValueError(f"{split} image/label mismatch: {image_count} != {label_count}")
    return counts


def write_yaml(path, dataset_root):
    content = f"""path: {dataset_root}
train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: litter
"""
    path.write_text(content, encoding="utf-8")


def write_reports(assignments, excluded_reasons, counts, violations, nearest, total_source):
    with ASSIGNMENTS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "label", "old_split", "new_split", "group", "frame", "status", "reason"])
        for entry, split in sorted(assignments.items(), key=lambda item: (item[1], item[0].group, item[0].frame, item[0].name)):
            writer.writerow([entry.name, entry.label_path.name, entry.old_split, split, entry.group, entry.frame, "kept", ""])
        for entry, reason in sorted(excluded_reasons.items(), key=lambda item: (item[0].group, item[0].frame, item[0].name)):
            writer.writerow([entry.name, entry.label_path.name, entry.old_split, "", entry.group, entry.frame, "excluded", reason])

    nearest_text = "none"
    if nearest is not None:
        distance, a, split_a, b, split_b = nearest
        nearest_text = f"{distance} frames: {split_a}/{a.name} vs {split_b}/{b.name}"

    lines = [
        "Gap20 val-as-test clean split summary",
        f"Source dataset: {SOURCE_DATASET}",
        f"Output dataset: {OUTPUT_DATASET}",
        f"Random seed: {SEED}",
        f"Gap threshold: <= {GAP} frames is treated as leakage",
        "",
        f"Source images: {total_source}",
        f"Kept images: {sum(image_count for image_count, _ in counts.values())}",
        f"Excluded images: {len(excluded_reasons)}",
        "",
        "Split counts:",
    ]
    for split in SPLITS:
        image_count, label_count = counts[split]
        lines.append(f"  {split}: {image_count} images, {label_count} labels")
    lines.extend(
        [
            "",
            f"Cross-split gap violations: {len(violations)}",
            f"Nearest cross-split pair: {nearest_text}",
            "",
            "Outputs:",
            f"  YAML: {OUTPUT_YAML}",
            f"  Assignments: {ASSIGNMENTS_CSV}",
        ]
    )
    SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    entries = collect_entries(SOURCE_DATASET)
    base_test = [entry for entry in entries if entry.old_split == "val"]
    pool = [entry for entry in entries if entry.old_split != "val"]

    test_entries, remaining_pool, excluded, excluded_reasons = select_test_additions(base_test, pool, TARGET_TEST, SEED)
    val_entries, train_entries, _ = assign_val_blocks(remaining_pool, TARGET_VAL, SEED + 1)

    assignments = {}
    assignments.update({entry: "test" for entry in test_entries})
    assignments.update({entry: "val" for entry in val_entries})
    assignments.update({entry: "train" for entry in train_entries})

    violations, nearest = validate_gap(assignments)
    if violations:
        examples = "\n".join(
            f"{distance}: {split_a}/{a.name} vs {split_b}/{b.name}"
            for distance, a, split_a, b, split_b in sorted(violations, key=lambda item: item[0])[:20]
        )
        raise RuntimeError(f"Cross-split gap violations found:\n{examples}")

    test_count = sum(1 for split in assignments.values() if split == "test")
    if test_count != TARGET_TEST:
        raise RuntimeError(f"Test count mismatch: {test_count} != {TARGET_TEST}")

    reset_output(OUTPUT_DATASET)
    copy_entries(assignments, OUTPUT_DATASET)
    counts = count_output(OUTPUT_DATASET)
    write_yaml(OUTPUT_YAML, OUTPUT_DATASET)
    write_reports(assignments, excluded_reasons, counts, violations, nearest, len(entries))

    print(f"Source images: {len(entries)}")
    print(f"Kept images: {sum(image_count for image_count, _ in counts.values())}")
    print(f"Excluded images: {len(excluded_reasons)}")
    for split in SPLITS:
        image_count, label_count = counts[split]
        print(f"{split}: images={image_count}, labels={label_count}")
    if nearest is not None:
        distance, a, split_a, b, split_b = nearest
        print(f"Nearest cross-split pair: {distance} frames ({split_a}/{a.name} vs {split_b}/{b.name})")
    print(f"YAML: {OUTPUT_YAML}")
    print(f"Summary: {SUMMARY_TXT}")
    print(f"Assignments: {ASSIGNMENTS_CSV}")


if __name__ == "__main__":
    main()
