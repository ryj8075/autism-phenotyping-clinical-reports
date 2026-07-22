import os
import sys
import argparse
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTERMEDIATES_PATH = os.path.join(ROOT, "intermediates")
ANALYSIS_RESULTS_PATH = os.path.join(ROOT, "analysis", "analysis_results")

def np_sigmoid(x):
    return 1 / (1 + np.exp(-x))

def confusion_counts(y_true, y_pred):

    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    TP = int(np.sum((y_pred == 1) & (y_true == 1)))
    TN = int(np.sum((y_pred == 0) & (y_true == 0)))
    FP = int(np.sum((y_pred == 1) & (y_true == 0)))
    FN = int(np.sum((y_pred == 0) & (y_true == 1)))
    return TP, FP, TN, FN

def metrics_from_counts(TP, FP, TN, FN, probs=None, y_true=None):

    n = TP + FP + TN + FN
    acc = (TP + TN) / n if n else 0.0
    prec = TP / (TP + FP) if (TP + FP) else 0.0
    rec = TP / (TP + FN) if (TP + FN) else 0.0
    spec = TN / (TN + FP) if (TN + FP) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    auroc = np.nan
    if probs is not None and y_true is not None and roc_auc_score is not None:
        yt = np.asarray(y_true).astype(int).ravel()
        if len(np.unique(yt)) == 2:
            try:
                auroc = roc_auc_score(yt, np.asarray(probs).ravel())
            except Exception:
                auroc = np.nan
    return {
        "N": n, "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "accuracy": acc, "precision": prec, "recall": rec,
        "specificity": spec, "f1": f1, "auroc": auroc,
    }

def plot_confusion(TP, FP, TN, FN, title, out_file):

    mat = np.array([[TN, FP], [FN, TP]])  # [[True0, ...],[...,True1]]
    plt.figure(figsize=(5.5, 4.8))
    ax = sns.heatmap(mat, annot=True, fmt="d", cmap="Blues", cbar=False,
                     xticklabels=["Pred 0\n(Non-ASD)", "Pred 1\n(ASD)"],
                     yticklabels=["True 0\n(Non-ASD)", "True 1\n(ASD)"])
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()

def fmt_block(name, m):

    lines = [
        f"[{name}]  (N={m['N']})",
        f"               Pred=0    Pred=1",
        f"  True=0(Non)   TN={m['TN']:<5}  FP={m['FP']:<5}",
        f"  True=1(ASD)   FN={m['FN']:<5}  TP={m['TP']:<5}",
        f"  Acc={m['accuracy']:.4f}  Prec={m['precision']:.4f}  "
        f"Rec={m['recall']:.4f}  Spec={m['specificity']:.4f}  "
        f"F1={m['f1']:.4f}  AUROC={m['auroc']:.4f}",
    ]
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Compute confusion matrices directly from saved predictions")
    parser.add_argument("--experiment_name", required=True, type=str)
    parser.add_argument("--intermediates_path", default=None, type=str,
                        help="Default: ROOT/intermediates/{experiment_name}")
    parser.add_argument("--output_dir", default=None, type=str,
                        help="Default: analysis_results/{experiment_name}")
    parser.add_argument("--threshold", default=0.5, type=float,
                        help="Positive-class threshold [default: 0.5]. Recomputed when probs_np is available.")
    args = parser.parse_args()

    exp = args.experiment_name
    inter = args.intermediates_path or os.path.join(INTERMEDIATES_PATH, exp)
    out_dir = args.output_dir or os.path.join(ANALYSIS_RESULTS_PATH, exp)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(inter):
        print(f"[ERROR] Missing intermediates directory: {inter}")
        sys.exit(1)

    labels = np.load(os.path.join(inter, "labels_np.npy")).astype(int).ravel()

    probs = None
    probs_path = os.path.join(inter, "probs_np.npy")
    pred_path = os.path.join(inter, "predicted_labels_np.npy")
    if os.path.exists(probs_path):
        probs = np.load(probs_path).ravel()
    if abs(args.threshold - 0.5) > 1e-9 or not os.path.exists(pred_path):
        if probs is None:

            logits = np.load(os.path.join(inter, "logits_np.npy")).ravel()
            probs = np_sigmoid(logits)
        preds = (probs >= args.threshold).astype(int)
        thr_note = f"threshold={args.threshold}"
    else:
        preds = np.load(pred_path).astype(int).ravel()
        thr_note = "threshold=0.5 (predicted_labels_np)"

    fold = None
    fa_path = os.path.join(inter, "fold_assignment.npy")
    if os.path.exists(fa_path):
        fold = np.load(fa_path).astype(int).ravel()

    report_ids = None
    rid_path = os.path.join(inter, "report_id_array.npy")
    if os.path.exists(rid_path):
        report_ids = np.array([str(r) for r in np.load(rid_path, allow_pickle=True).ravel()])

    print("=" * 64)
    print(f"Confusion Matrix: {exp}")
    print(f"  {thr_note},  N={len(labels)}  "
          f"(positive {int(labels.sum())} / negative {len(labels)-int(labels.sum())})")
    print("=" * 64)

    csv_rows = ["group,N,TP,FP,TN,FN,accuracy,precision,recall,specificity,f1,auroc"]
    txt_blocks = [f"Confusion Matrix: {exp}  ({thr_note})", "=" * 64]

    def record(name, mask=None):
        if mask is None:
            yt, yp = labels, preds
            pr = probs
        else:
            yt, yp = labels[mask], preds[mask]
            pr = probs[mask] if probs is not None else None
        if len(yt) == 0:
            return
        TP, FP, TN, FN = confusion_counts(yt, yp)
        m = metrics_from_counts(TP, FP, TN, FN, probs=pr, y_true=yt)
        block = fmt_block(name, m)
        print("\n" + block)
        txt_blocks.append("\n" + block)
        csv_rows.append(f"{name},{m['N']},{m['TP']},{m['FP']},{m['TN']},{m['FN']},"
                        f"{m['accuracy']:.4f},{m['precision']:.4f},{m['recall']:.4f},"
                        f"{m['specificity']:.4f},{m['f1']:.4f},{m['auroc']:.4f}")
        return m

    m_overall = record("overall")
    TP, FP, TN, FN = m_overall["TP"], m_overall["FP"], m_overall["TN"], m_overall["FN"]
    plot_confusion(TP, FP, TN, FN,
                   f"Confusion Matrix - {exp}\n(N={m_overall['N']}, {thr_note})",
                   os.path.join(out_dir, f"confusion_matrix_{exp}.png"))

    if report_ids is not None and len(report_ids) == len(labels):
        rtype = np.array([rid.split('-')[-1][0] if '-' in rid and rid.split('-')[-1] else '?'
                          for rid in report_ids])
        for tchar, tname in [('A', 'A-type'), ('P', 'P-type')]:
            mask = rtype == tchar
            if mask.sum() > 0:
                m = record(f"{tchar}-type", mask)
                if m is not None:
                    plot_confusion(m["TP"], m["FP"], m["TN"], m["FN"],
                                   f"Confusion Matrix - {tname}\n(N={m['N']})",
                                   os.path.join(out_dir, f"confusion_matrix_{tchar}type_{exp}.png"))

    if fold is not None and len(fold) == len(labels):
        for f in sorted(set(fold.tolist())):
            if f == 0:
                continue
            record(f"fold{f}", fold == f)

    with open(os.path.join(out_dir, f"confusion_matrix_{exp}.csv"), "w", encoding="utf-8") as fcsv:
        fcsv.write("\n".join(csv_rows) + "\n")
    with open(os.path.join(out_dir, f"confusion_matrix_{exp}.txt"), "w", encoding="utf-8") as ftxt:
        ftxt.write("\n".join(txt_blocks) + "\n")

    print(f"\nSaved outputs to: {out_dir}")
    print(f"  - confusion_matrix_{exp}.png / .csv / .txt")
    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()
