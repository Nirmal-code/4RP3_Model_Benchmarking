import argparse
import csv
import json
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import matplotlib


META_COLUMNS = {
    "text",
    "id",
    "author",
    "subreddit",
    "link_id",
    "parent_id",
    "created_utc",
    "rater_id",
    "example_very_unclear",
}

labels = {"anger": 0, "disgust": 1, "fear": 2, "sadness": 3, "joy": 4, "love": 5, "surprise": 6, "optimism": 7}

num_annotators = 82

# annotator_pred: Dict[str, Dict[str, str]] = {"textId": {"annotatorId": "label"}} (something like this)
# model_pred: Dict[str, str] = {"textId": "label"}
# if no choosen_annotator, then we will count all annotators' labels; otherwise, only the choosen_annotator's label will be counted
def generate_confusion_matrix(annotator_pred: Dict[str, Dict[str, str]], model_pred: Dict[str, str], choosen_annotator=None):
    matrix = [[0 for _ in range(len(labels))] for _ in range(len(labels))]
    for text_id, annotators in annotator_pred.items():
        if choosen_annotator and choosen_annotator not in annotators:
            continue
        if text_id not in model_pred:
            print("Warning: text_id {} found in annotator predictions but not in model predictions; skipping.".format(text_id))
            continue
        model_label = model_pred[text_id]
        if (not choosen_annotator): 
            for annotator_label in annotators.values():
                matrix[labels[annotator_label]][labels[model_label]] += 1
        else:
            annotator_label = annotators.get(choosen_annotator)
            matrix[labels[annotator_label]][labels[model_label]] += 1
    
    return matrix


def compare_emotions(confusion_matrix: List[List[int]]) -> Dict[str, float]:
    label_order = list(labels.keys())
    n = len(label_order)
    f1_by_label: Dict[str, float] = {}

    for i, lab in enumerate(label_order):
        tp = confusion_matrix[i][i]
        fp = sum(confusion_matrix[r][i] for r in range(n) if r != i)
        fn = sum(confusion_matrix[i][c] for c in range(n) if c != i)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_by_label[lab] = f1

    return f1_by_label


def compute_overall_f1(confusion_matrix: List[List[int]]) -> float:
    n = len(confusion_matrix)
    tp = sum(confusion_matrix[i][i] for i in range(n))
    total = sum(sum(row) for row in confusion_matrix)
    if total == 0:
        return 0.0

    fp = total - tp
    fn = total - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def plot_emotion_scores(
    emotion_scores: Dict[str, float],
    output_path: str,
    title: str = "Emotion F1 Scores",
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for plotting") from exc

    label_order = [lab for lab in labels.keys() if lab in emotion_scores]
    labels_list = label_order or list(emotion_scores.keys())
    scores = [emotion_scores[lab] for lab in labels_list]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(labels_list, scores, color="#4c72b0")
    ax.set_ylim(0, max(scores) * 1.15 if scores else 1.0)
    ax.set_title(title)
    ax.set_ylabel("F1")
    ax.set_xlabel("Emotion")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.xticks(rotation=30, ha="right")

    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_annotator_scores(
    annotator_scores: Dict[str, float],
    output_path: str,
    title: str = "Annotator F1 Scores",
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for plotting") from exc

    def _sort_key(val: str) -> Tuple[int, str]:
        try:
            return (0, f"{int(val):06d}")
        except ValueError:
            return (1, val)

    labels_list = sorted(annotator_scores.keys(), key=_sort_key)
    scores = [annotator_scores[lab] for lab in labels_list]
    fig_width = max(10.0, 0.35 * len(labels_list))

    fig, ax = plt.subplots(figsize=(fig_width, 4.5))
    bars = ax.bar(labels_list, scores, color="#55a868")
    ax.set_ylim(0, max(scores) * 1.15 if scores else 1.0)
    ax.set_title(title)
    ax.set_ylabel("F1")
    ax.set_xlabel("Annotator")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.xticks(rotation=90, ha="center")

    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_annotator_emotion_scores(
    annotator_emotion_scores: Dict[str, Dict[str, float]],
    output_path: str,
    title: str = "Annotator Emotion F1 Scores",
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for plotting") from exc

    def _sort_key(val: str) -> Tuple[int, str]:
        try:
            return (0, f"{int(val):06d}")
        except ValueError:
            return (1, val)

    annotators = sorted(annotator_emotion_scores.keys(), key=_sort_key)
    if not annotators:
        raise ValueError("no annotator scores provided")

    all_emotions = [lab for lab in labels.keys()]
    present_emotions = [
        e for e in all_emotions
        if any(e in annotator_emotion_scores[a] for a in annotators)
    ]
    emotions = present_emotions or all_emotions

    group_count = len(annotators)
    series_count = len(emotions)
    if series_count == 0:
        raise ValueError("no emotion scores provided")

    group_spacing = 1.5
    group_positions = [i * group_spacing for i in range(group_count)]
    bar_width = min(0.6 / series_count, 0.06)
    bar_step = bar_width * 1.4
    fig_width = max(12.0, 0.75 * group_count)

    fig, ax = plt.subplots(figsize=(fig_width, 5.0))
    max_score = 0.0

    for idx, emotion in enumerate(emotions):
        offsets = [
            x - ((series_count - 1) / 2) * bar_step + idx * bar_step
            for x in group_positions
        ]
        scores = [annotator_emotion_scores[a].get(emotion, 0.0) for a in annotators]
        max_score = max(max_score, max(scores) if scores else 0.0)
        bars = ax.bar(offsets, scores, width=bar_width, label=emotion)
        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{score:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

    ax.set_xticks(group_positions)
    ax.set_xticklabels(annotators, rotation=90, ha="center")
    ax.set_ylim(0, max_score * 1.15 if max_score > 0 else 1.0)
    ax.set_title(title)
    ax.set_ylabel("F1")
    ax.set_xlabel("Annotator")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(title="Emotion", ncol=min(4, len(emotions)), bbox_to_anchor=(1.01, 1), loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def output_csv(
    confusion_matrices: List[Tuple[str, List[List[int]]]],
    output_path: str,
) -> None:
    label_order = list(labels.keys())
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for idx, (matrix_id, matrix) in enumerate(confusion_matrices):
            if len(matrix) != len(label_order):
                raise ValueError(f"matrix {matrix_id} has unexpected size")
            for i in range(len(label_order)):
                if len(matrix[i]) != len(label_order):
                    raise ValueError(f"matrix {matrix_id} row {i} has unexpected size")

            writer.writerow(["matrix_id", matrix_id])
            writer.writerow(["true\\pred"] + label_order)
            for i, true_lab in enumerate(label_order):
                writer.writerow([true_lab] + matrix[i])

            if idx < len(confusion_matrices) - 1:
                writer.writerow([])


def _positive_label_columns(row: Dict[str, str], label_columns: List[str]) -> List[str]:
    out: List[str] = []
    for col in label_columns:
        val = (row.get(col) or "").strip().lower()
        if val in {"1", "1.0", "true", "yes"}:
            out.append(col)
            continue
        try:
            if float(val) > 0:
                out.append(col)
        except ValueError:
            continue
    return out


def build_annotator_predictions(
    annotations_csv: str,
    id_field: str,
    annotator_field: str,
    label_columns_arg: List[str],
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, int]]:
    annotator_pred: Dict[str, Dict[str, str]] = defaultdict(dict)
    stats = {
        "rows": 0,
        "missing_id": 0,
        "missing_annotator": 0,
        "missing_label": 0,
        "multi_label": 0,
    }

    with open(annotations_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("annotations CSV has no header row")
        label_columns = infer_label_columns(reader.fieldnames, label_columns_arg)
        label_columns = [c for c in label_columns if c in labels]
        if not label_columns:
            raise ValueError("no label columns matched the labels mapping")

        for row in reader:
            stats["rows"] += 1
            pid = (row.get(id_field) or "").strip()
            if not pid:
                stats["missing_id"] += 1
                continue
            annotator_id = (row.get(annotator_field) or "").strip()
            if not annotator_id:
                stats["missing_annotator"] += 1
                continue

            row_labels = _positive_label_columns(row, label_columns)
            if not row_labels:
                stats["missing_label"] += 1
                continue
            if len(row_labels) > 1:
                stats["multi_label"] += 1

            annotator_pred[pid][annotator_id] = row_labels[0]

    return annotator_pred, stats


def load_for_confusion_matrix(
    annotations_csv: str,
    preds_csv: str,
    id_field: str = "id",
    annotator_field: str = "rater_id",
    pred_field: str = "pred_sentiment",
    label_columns: List[str] = None,
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, int]]:
    model_pred = load_predictions(preds_csv, id_field, pred_field)
    annotator_pred, stats = build_annotator_predictions(
        annotations_csv, id_field, annotator_field, label_columns
    )
    return annotator_pred, model_pred, stats



def load_predictions(path: str, id_field: str, pred_field: str) -> Tuple[Dict[str, str], int]:
    preds: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = (row.get(id_field) or "").strip()
            if not pid:
                continue
            if pid in preds:
                print(f"Warning: duplicate id {pid} found in predictions CSV; overwriting previous value.")
                continue
            preds[pid] = (row.get(pred_field) or "").strip()
    return preds


def infer_label_columns(header: List[str], label_columns_arg: List[str]) -> List[str]:
    if label_columns_arg:
        return [c for c in label_columns_arg if c in header]
    return [c for c in header if c not in META_COLUMNS]




def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations_csv", default="data/filtered_emotions.csv")
    ap.add_argument("--preds_csv", default="data/deepseek_preds.csv")
    ap.add_argument("--output_csv", default="outputs/pred_vs_annotator.csv")
    ap.add_argument("--id_field", default="id")
    ap.add_argument("--pred_field", default="pred_sentiment")
    ap.add_argument("--label_columns", default="")
    ap.add_argument("--compare_emotions", action="store_true")
    ap.add_argument("--compare_annotators", action="store_true")
    args = ap.parse_args()

    annotator_preds, model_preds, _ = load_for_confusion_matrix(args.annotations_csv, args.preds_csv, args.id_field, "rater_id", args.pred_field, list(labels.keys()))

    general_matrix = generate_confusion_matrix(annotator_preds, model_preds)
    

    if (args.compare_emotions and args.compare_annotators):
        annotator_matrices = []
        annotator_f1_emotion_scores = {}

        for annotator_idx in range(0, num_annotators):
            annotator_id = str(annotator_idx)
            matrix = generate_confusion_matrix(annotator_preds, model_preds, choosen_annotator=annotator_id)
            annotator_matrices.append((annotator_id, matrix))
            annotator_f1_emotion_scores[annotator_id] = compare_emotions(matrix)

        output_csv(
            [("general", general_matrix)] + annotator_matrices,
            args.output_csv.replace(".csv", "_all_matrices.csv"),
        )

        plot_annotator_emotion_scores(
            annotator_f1_emotion_scores,
            args.output_csv.replace(".csv", "_all_scores.png"),
            title="Model vs Annotators F1 Scores by Emotion",
        )
        return

    if (args.compare_emotions):
        emotion_scores = compare_emotions(general_matrix)
        print("F1 Scores by Emotion:")
        for emotion, f1 in emotion_scores.items():
            print(f"{emotion}: {f1:.4f}")

        output_csv([("general", general_matrix)], args.output_csv)


        plot_emotion_scores(
            emotion_scores,
            args.output_csv.replace(".csv", "_only_emotion_scores.png"),
            title="Model vs Annotators Emotion F1 Scores",
        )
    
    if (args.compare_annotators):
        annotator_matrices = []
        annotator_f1_scores = {}

        for annotator_idx in range(0, num_annotators):
            annotator_id = str(annotator_idx)
            matrix = generate_confusion_matrix(annotator_preds, model_preds, choosen_annotator=annotator_id)
            annotator_matrices.append((annotator_id, matrix))
            annotator_f1_scores[annotator_id] = compute_overall_f1(matrix)
            print(f"Annotator {annotator_id} Overall F1: {annotator_f1_scores[annotator_id]:.4f}")

        output_csv(
            [("general", general_matrix)] + annotator_matrices,
            args.output_csv.replace(".csv", "_all_matrices.csv"),
        )

        plot_annotator_scores(
            annotator_f1_scores,
            args.output_csv.replace(".csv", "_only_annotator_scores.png"),
            title="Model vs Annotators F1 Scores",
        )


    # print("General Confusion Matrix:")
    # print(general_matrix)

    # print("\nPer-Annotator Confusion Matrices:")
    # for annotator_id, matrix in annotator_matrices:
    #     print(f"Annotator {annotator_id}:")
    #     print(matrix)

if __name__ == "__main__":
    main()
