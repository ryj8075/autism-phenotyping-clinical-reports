import os
import sys
import argparse
import numpy as np
import pandas as pd
from string import punctuation
from collections import Counter

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.batcher import create_batches
from custom_models.sentence_attention_base_pool_ko import SentenceAttentionBERTKorean

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle
from matplotlib.patheffects import withStroke

try:
    import ptitprince as pt
    from adjustText import adjust_text
    _HAS_RAINCLOUD = True
except Exception:
    _HAS_RAINCLOUD = False

def np_sigmoid(x):
    return 1 / (1 + np.exp(-x))

K_FOLDS = 5
BASE_MODEL_NAME = "klue/roberta-base"
HIDDEN_SIZE = 768
NUM_LAYERS = 13  # embedding + 12 transformer layers
RANDOM_STATE = 22

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data")
REPORT_PATH = os.path.join(DATA_PATH, "reports_txt")
MODEL_PATH = os.path.join(ROOT, "trained_models")
DSM_TXT_PATH = os.path.join(DATA_PATH, "DSM-5")

def decode_reports(input_tensor, tokenizer):

    decoded_reports = []

    for report in input_tensor:
        sentences = []
        for sent_tokens in report:

            tokens = sent_tokens[sent_tokens != tokenizer.pad_token_id]
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            sentences.append(text.strip())
        decoded_reports.append(sentences)

    return np.array(decoded_reports, dtype=object)

def run_fold_inference(experiment_name, input_tensor, attention_mask_tensor, label_tensor, pat_ids, device,
                       report_max_length=64, sentence_max_length=64):

    skf = StratifiedGroupKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    splits = list(skf.split(input_tensor, label_tensor, groups=pat_ids))

    n_samples = len(label_tensor)

    logits_np = np.zeros(n_samples)
    attention_matrices_np = np.zeros((n_samples, report_max_length, report_max_length))
    attention_weighted_sentence_embs_np = np.zeros((n_samples, report_max_length, HIDDEN_SIZE))
    lhs_embs_np = np.zeros((n_samples, report_max_length, sentence_max_length, HIDDEN_SIZE))
    layer_pooled_embs_np = np.zeros((n_samples, NUM_LAYERS, HIDDEN_SIZE))
    labels_np = label_tensor.numpy()
    fold_assignment = np.zeros(n_samples, dtype=int)

    for fold_idx, (train_idx, valid_idx) in enumerate(splits):
        print(f"\n  --- Fold {fold_idx + 1}/{K_FOLDS} ---")
        print(f"      Valid samples for inference: {len(valid_idx)}")

        model_file = os.path.join(MODEL_PATH, f"{experiment_name}_fold{fold_idx + 1}.pt")

        if not os.path.exists(model_file):
            print(f"      [WARNING] Missing model file: {model_file}")
            continue

        model = SentenceAttentionBERTKorean(
            base_model_name=BASE_MODEL_NAME,
            report_max_length=report_max_length
        )
        model.load_state_dict(torch.load(model_file, map_location=device))
        model = model.to(device)
        model.eval()

        valid_input = input_tensor[valid_idx].to(device)
        valid_mask = attention_mask_tensor[valid_idx].to(device)

        batch_size = 8
        valid_batches = create_batches(valid_input, batch_size)
        valid_mask_batches = create_batches(valid_mask, batch_size)

        fold_logits = []
        fold_attn_weights = []
        fold_attn_outputs = []
        fold_lhs_embs = []
        fold_layer_pooled = []

        with torch.no_grad():
            for batch_input, batch_mask in zip(valid_batches, valid_mask_batches):

                logits, attn_weights, attn_output, last_hidden_embs, layer_pooled_embs = model(
                    batch_input, attn_mask=batch_mask
                )

                fold_logits.append(logits.cpu().numpy())
                fold_attn_weights.append(attn_weights.cpu().numpy())
                fold_attn_outputs.append(attn_output.cpu().numpy())

                # last_hidden_embs: list of (num_sentences, seq_len, hidden_size) per batch item
                for lhs in last_hidden_embs:
                    fold_lhs_embs.append(lhs.cpu().numpy())

                # layer_pooled_embs: list of (num_layers, hidden_size) per batch item
                for lp in layer_pooled_embs:
                    fold_layer_pooled.append(lp.cpu().numpy())

        fold_logits = np.concatenate(fold_logits)
        fold_attn_weights = np.concatenate(fold_attn_weights)
        fold_attn_outputs = np.concatenate(fold_attn_outputs)
        fold_lhs_embs = np.stack(fold_lhs_embs, axis=0)
        fold_layer_pooled = np.stack(fold_layer_pooled, axis=0)

        for i, idx in enumerate(valid_idx):
            logits_np[idx] = fold_logits[i]
            attention_matrices_np[idx] = fold_attn_weights[i]
            attention_weighted_sentence_embs_np[idx] = fold_attn_outputs[i]
            lhs_embs_np[idx] = fold_lhs_embs[i]
            layer_pooled_embs_np[idx] = fold_layer_pooled[i]
            fold_assignment[idx] = fold_idx + 1

        print(f"      Inference complete")

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    return {
        'logits_np': logits_np,
        'attention_matrices_np': attention_matrices_np,
        'attention_weighted_sentence_embs_np': attention_weighted_sentence_embs_np,
        'lhs_embs_np': lhs_embs_np,
        'layer_pooled_embs_np': layer_pooled_embs_np,
        'labels_np': labels_np,
        'fold_assignment': fold_assignment,
    }

def analyze_layer_wise_prediction(layer_pooled_embs_np, labels_np, pat_ids, output_path, experiment_name):

    print("\n  Layer-wise diagnostic prediction analysis...")

    n_layers = layer_pooled_embs_np.shape[1]

    main_col = "#C73682"
    pal = sns.color_palette("blend:#d3d3d3," + main_col, n_colors=n_layers)

    plt.figure(figsize=(10, 8))
    ax = plt.subplot()

    layer_auc_list = []
    layer_results = []

    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for layer_idx in range(1, n_layers):
        print(f"    Layer {layer_idx}...")

        pooled_embs = layer_pooled_embs_np[:, layer_idx, :]

        layer_avg = []
        fold_probas = []
        y_true = []
        auc_list = []

        for train_index, test_index in skf.split(pooled_embs, labels_np, groups=pat_ids):
            X_train, X_test = pooled_embs[train_index], pooled_embs[test_index]
            y_train, y_test = labels_np[train_index], labels_np[test_index]

            clf = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000).fit(X_train, y_train)
            layer_avg.append(clf.score(X_test, y_test))
            fold_probas.extend(list(clf.predict_proba(X_test)[:, 1]))
            y_true.extend(y_test)
            auc_list.append(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))

        fpr, tpr, thresholds = roc_curve(y_true, fold_probas)
        ax.plot(fpr, tpr, label=f"Layer {layer_idx}", color=pal[layer_idx-1])

        mean_auc = np.mean(auc_list)
        std_auc = np.std(auc_list)

        layer_results.append({
            'layer': layer_idx,
            'mean_accuracy': np.mean(layer_avg),
            'mean_auc': mean_auc,
            'std_auc': std_auc,
            'ci_lower': np.percentile(auc_list, 2.5),
            'ci_upper': np.percentile(auc_list, 97.5),
        })
        layer_auc_list.append(auc_list)

        print(f"      AUC: {mean_auc:.4f} (+/- {std_auc:.4f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=0.6)
    ax.set_xlabel("False Positive Rate", fontsize=14)
    ax.set_ylabel("True Positive Rate", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_title("Layer-wise ROC Curves (Logistic Regression)", fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(output_path, f"ROC_layer_{experiment_name}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    layer_auc_list = np.array(layer_auc_list)
    mean_auc = np.mean(layer_auc_list, axis=1)
    lower_ci = np.percentile(layer_auc_list, 2.5, axis=1)
    upper_ci = np.percentile(layer_auc_list, 97.5, axis=1)

    plt.figure(figsize=(10, 6))
    for i in range(n_layers - 1):
        plt.errorbar(i + 1, mean_auc[i], fmt='o',
                     yerr=[[mean_auc[i] - lower_ci[i]], [upper_ci[i] - mean_auc[i]]],
                     color=pal[i], capsize=5, markersize=8)

    plt.plot(range(1, n_layers), mean_auc, linestyle='dashed', color="black", linewidth=0.8)
    plt.xlabel("Layer", fontsize=14)
    plt.ylabel("AUC", fontsize=14)
    plt.title("Layer-wise AUC with 95% CI", fontsize=14, fontweight='bold')
    plt.xticks(range(1, n_layers))
    plt.ylim(0.5, 1.0)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(output_path, f"AUC_layer_{experiment_name}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    results_df = pd.DataFrame(layer_results)
    results_df.to_csv(os.path.join(output_path, f"layer_wise_results_{experiment_name}.csv"), index=False)

    return results_df

def analyze_pca_embeddings(lhs_embs_np, labels_np, output_path, experiment_name, report_max_length=64):

    print("\n  PCA embedding visualization...")

    # lhs_embs_np shape: (n_samples, num_sentences, seq_len, hidden_size)

    sentence_embs = lhs_embs_np.mean(axis=2)  # (n_samples, report_max_length, 768)

    sentence_embs_flat = sentence_embs.reshape(-1, HIDDEN_SIZE)  # (n_samples * report_max_length, 768)

    targs = np.array([[item] * report_max_length for item in labels_np]).reshape(-1)

    # PCA
    pca = PCA(n_components=2)
    pca_embs = pca.fit_transform(sentence_embs_flat)

    ASD_col = "#23a2dc"
    nonASD_col = "#DC5D23"

    plt.figure(figsize=(12, 10))
    legend_map = {0: "Non-autism", 1: "Autism"}

    for label, color, name in [(0, nonASD_col, "Non-autism"), (1, ASD_col, "Autism")]:
        mask = targs == label
        plt.scatter(pca_embs[mask, 0], pca_embs[mask, 1],
                    c=color, s=2, alpha=0.5, label=name, rasterized=True)

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=14)
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=14)
    plt.title("Sentence Embeddings PCA (Mean-Pooled LHS)", fontsize=14, fontweight='bold')
    plt.legend(title='MD-Assessed Diagnosis', loc='upper right', fontsize=12,
               title_fontproperties={'size': 12, 'weight': 'bold'})
    plt.savefig(os.path.join(output_path, f"PCA_lhs_embs_{experiment_name}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    report_embs = sentence_embs.mean(axis=1)  # (n_samples, 768)
    pca_report = PCA(n_components=2)
    pca_report_embs = pca_report.fit_transform(report_embs)

    plt.figure(figsize=(10, 8))
    for label, color, name in [(0, nonASD_col, "Non-autism"), (1, ASD_col, "Autism")]:
        mask = labels_np == label
        plt.scatter(pca_report_embs[mask, 0], pca_report_embs[mask, 1],
                    c=color, s=30, alpha=0.7, label=name)

    plt.xlabel(f"PC1 ({pca_report.explained_variance_ratio_[0]*100:.1f}%)", fontsize=14)
    plt.ylabel(f"PC2 ({pca_report.explained_variance_ratio_[1]*100:.1f}%)", fontsize=14)
    plt.title("Report Embeddings PCA (Mean of Sentences)", fontsize=14, fontweight='bold')
    plt.legend(title='MD-Assessed Diagnosis', loc='upper right', fontsize=12)
    plt.savefig(os.path.join(output_path, f"PCA_report_embs_{experiment_name}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    return pca, pca_embs

# ============================================================================

# ============================================================================

def analyze_high_attention_sentences(attns, sentences, sample_idx, label_str, threshold=0.05, top_k=10):

    results = {
        'sample_idx': sample_idx,
        'label': label_str,
        'max_attention': float(np.max(attns)),
        'max_position': tuple(map(int, np.unravel_index(np.argmax(attns), attns.shape))),
        'high_attention_pairs': [],
        'key_sentence_ranking': [],
    }

    high_attn_positions = np.where(attns > threshold)
    for q, k in zip(high_attn_positions[0], high_attn_positions[1]):
        q_sent = sentences[q] if q < len(sentences) and sentences[q] else "[PADDING]"
        k_sent = sentences[k] if k < len(sentences) and sentences[k] else "[PADDING]"

        results['high_attention_pairs'].append({
            'query_idx': int(q),
            'key_idx': int(k),
            'attention_weight': float(attns[q, k]),
            'query_sentence': q_sent,
            'key_sentence': k_sent,
        })

    results['high_attention_pairs'].sort(key=lambda x: x['attention_weight'], reverse=True)

    key_attention_sum = attns.sum(axis=0)
    key_ranking_indices = np.argsort(key_attention_sum)[::-1][:top_k]

    for rank, idx in enumerate(key_ranking_indices):
        sent = sentences[idx] if idx < len(sentences) and sentences[idx] else "[PADDING]"
        results['key_sentence_ranking'].append({
            'rank': rank + 1,
            'sentence_idx': int(idx),
            'total_attention': float(key_attention_sum[idx]),
            'max_attention': float(np.max(attns[:, idx])),
            'sentence': sent,
        })

    return results

def save_attention_analysis(results, output_path, experiment_name, label_type):

    filename = os.path.join(output_path, f"attention_analysis_{label_type}_{experiment_name}.txt")

    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"ATTENTION WEIGHT DETAILED ANALYSIS - Sample {results['sample_idx']} ({results['label']})\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Maximum Attention Weight: {results['max_attention']:.6f}\n")
        f.write(f"Maximum value position: Query {results['max_position'][0]}, Key {results['max_position'][1]}\n\n")

        f.write("-" * 80 + "\n")
        f.write("High Attention Weight Pairs (Top 20)\n")
        f.write("-" * 80 + "\n\n")

        for i, pair in enumerate(results['high_attention_pairs'][:20]):
            f.write(f"[{i+1}] Query {pair['query_idx']:2d} → Key {pair['key_idx']:2d}: {pair['attention_weight']:.6f}\n")
            f.write(f"    Key sentence: {pair['key_sentence'][:150]}...\n\n")

        f.write("-" * 80 + "\n")
        f.write("Key sentence ranking (by total attention)\n")
        f.write("-" * 80 + "\n\n")

        for item in results['key_sentence_ranking']:
            f.write(f"#{item['rank']} [Index {item['sentence_idx']}]\n")
            f.write(f"   Total attention: {item['total_attention']:.4f}, Max: {item['max_attention']:.6f}\n")
            f.write(f"   Sentence: {item['sentence']}\n\n")

    return filename

def analyze_attention_heatmap(attention_matrices_np, decoded_reports, labels_np, logits_np,
                               report_id_array, output_path, experiment_name, colorbar="fixed"):

    print("\n  Attention heatmap visualization and detailed analysis...")

    if colorbar == "datarange":
        _vmin, _vmax = float(attention_matrices_np.min()), float(attention_matrices_np.max())
        print(f"    colorbar=datarange → vmin={_vmin:.6f}, vmax={_vmax:.6f}")
    else:
        _vmin, _vmax = 0, 1

    ASD_col = "#23a2dc"
    att_pal = sns.color_palette("blend:#f9f9f9," + ASD_col, as_cmap=True)

    asd_indices = np.where(labels_np == 1)[0]
    probs = np_sigmoid(logits_np)

    if len(asd_indices) > 0:

        asd_probs = probs[asd_indices]
        top_asd_idx = asd_indices[np.argmax(asd_probs)]

        attns = attention_matrices_np[top_asd_idx]
        sentences = decoded_reports[top_asd_idx]

        plt.figure(figsize=(12, 10))
        ax = sns.heatmap(attns, cmap=att_pal, vmax=_vmax, vmin=_vmin,
                         cbar_kws={'label': 'Attention Weight'})

        x_ticks_positions = np.arange(9, attns.shape[1], 10)
        x_ticks_positions = np.hstack((0, x_ticks_positions))
        y_ticks_positions = np.arange(9, attns.shape[0], 10)
        y_ticks_positions = np.hstack((0, y_ticks_positions))

        plt.xticks(x_ticks_positions, x_ticks_positions + 1)
        plt.yticks(y_ticks_positions, y_ticks_positions + 1, rotation=0)

        ax.xaxis.set_label_position('top')
        plt.gca().xaxis.tick_top()

        plt.xlabel("Key Sentence Index", fontsize=12)
        plt.ylabel("Query Sentence Index", fontsize=12)

        max_att = attns.sum(axis=0).argmax()
        ax.add_patch(Rectangle((max_att - 0.5, 0.15), 2, attns.shape[0] - 0.35,
                                fill=False, edgecolor="black", lw=1.5, alpha=1))

        plt.title(f"Attention Heatmap (Sample {top_asd_idx}, ASD)\n" +
                  f"Pred: {probs[top_asd_idx]:.3f}, True: ASD", fontsize=14, fontweight='bold')

        plt.savefig(os.path.join(output_path, f"attention_heatmap_ASD_{experiment_name}.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()

        asd_analysis = analyze_high_attention_sentences(
            attns, sentences, top_asd_idx, "ASD", threshold=0.05, top_k=10
        )

        print(f"\n    === ASD Sample {top_asd_idx} Attention Analysis ===")
        print(f"      Max Attention: {asd_analysis['max_attention']:.6f}")
        print(f"      Maximum value position: Query {asd_analysis['max_position'][0]} → Key {asd_analysis['max_position'][1]}")
        print(f"      Prediction: {probs[top_asd_idx]:.4f}")

        print(f"\n      Top attention-weighted sentences:")
        for i, pair in enumerate(asd_analysis['high_attention_pairs'][:5]):
            print(f"        [{i+1}] Q{pair['query_idx']}→K{pair['key_idx']}: {pair['attention_weight']:.6f}")
            print(f"            Sentence: {pair['key_sentence'][:80]}...")

        saved_file = save_attention_analysis(asd_analysis, output_path, experiment_name, "ASD")
        print(f"\n      Saved detailed analysis: {os.path.basename(saved_file)}")

    nonasd_indices = np.where(labels_np == 0)[0]
    if len(nonasd_indices) > 0:
        nonasd_probs = probs[nonasd_indices]
        top_nonasd_idx = nonasd_indices[np.argmin(nonasd_probs)]

        attns = attention_matrices_np[top_nonasd_idx]
        sentences = decoded_reports[top_nonasd_idx]

        nonASD_col = "#DC5D23"
        att_pal_ctl = sns.color_palette("blend:#f9f9f9," + nonASD_col, as_cmap=True)

        plt.figure(figsize=(12, 10))
        ax = sns.heatmap(attns, cmap=att_pal_ctl, vmax=_vmax, vmin=_vmin,
                         cbar_kws={'label': 'Attention Weight'})

        x_ticks_positions = np.arange(9, attns.shape[1], 10)
        x_ticks_positions = np.hstack((0, x_ticks_positions))
        y_ticks_positions = np.arange(9, attns.shape[0], 10)
        y_ticks_positions = np.hstack((0, y_ticks_positions))

        plt.xticks(x_ticks_positions, x_ticks_positions + 1)
        plt.yticks(y_ticks_positions, y_ticks_positions + 1, rotation=0)

        ax.xaxis.set_label_position('top')
        plt.gca().xaxis.tick_top()

        plt.xlabel("Key Sentence Index", fontsize=12)
        plt.ylabel("Query Sentence Index", fontsize=12)

        max_att = attns.sum(axis=0).argmax()
        ax.add_patch(Rectangle((max_att - 0.5, 0.15), 2, attns.shape[0] - 0.35,
                                fill=False, edgecolor="black", lw=1.5, alpha=1))

        plt.title(f"Attention Heatmap (Sample {top_nonasd_idx}, Non-ASD)\n" +
                  f"Pred: {probs[top_nonasd_idx]:.3f}, True: Non-ASD", fontsize=14, fontweight='bold')

        plt.savefig(os.path.join(output_path, f"attention_heatmap_NonASD_{experiment_name}.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()

        nonasd_analysis = analyze_high_attention_sentences(
            attns, sentences, top_nonasd_idx, "Non-ASD", threshold=0.05, top_k=10
        )

        print(f"\n    === Non-ASD Sample {top_nonasd_idx} Attention Analysis ===")
        print(f"      Max Attention: {nonasd_analysis['max_attention']:.6f}")
        print(f"      Maximum value position: Query {nonasd_analysis['max_position'][0]} → Key {nonasd_analysis['max_position'][1]}")
        print(f"      Prediction: {probs[top_nonasd_idx]:.4f}")

        saved_file = save_attention_analysis(nonasd_analysis, output_path, experiment_name, "NonASD")
        print(f"\n      Saved detailed analysis: {os.path.basename(saved_file)}")

    report_type_last = np.array([str(rid).split('-')[-1][0] for rid in report_id_array])
    type_configs = [
        ('A', 'Autism diagnostic report (A-type)', 'Asd'),
        ('P', 'Psychological evaluation report (P-type)', 'Psy'),
    ]

    for type_char, type_name_kr, type_suffix in type_configs:
        type_indices = np.where(report_type_last == type_char)[0]

        if len(type_indices) == 0:
            print(f"\n    {type_name_kr} samples not found; skipping heatmap")
            continue

        print(f"\n    === {type_name_kr} Attention Heatmap (v1.3) ===")
        print(f"    {type_char}-type sample count: {len(type_indices)}")

        type_probs = probs[type_indices]

        high_idx = type_indices[np.argmax(type_probs)]
        high_report_id = str(report_id_array[high_idx])
        high_true_label = "ASD" if labels_np[high_idx] == 1 else "Non-ASD"

        attns_h = attention_matrices_np[high_idx]
        sentences_h = decoded_reports[high_idx]

        plt.figure(figsize=(12, 10))
        ax = sns.heatmap(attns_h, cmap=att_pal, vmax=_vmax, vmin=_vmin,
                         cbar_kws={'label': 'Attention Weight'})

        x_ticks_positions = np.arange(9, attns_h.shape[1], 10)
        x_ticks_positions = np.hstack((0, x_ticks_positions))
        y_ticks_positions = np.arange(9, attns_h.shape[0], 10)
        y_ticks_positions = np.hstack((0, y_ticks_positions))

        plt.xticks(x_ticks_positions, x_ticks_positions + 1)
        plt.yticks(y_ticks_positions, y_ticks_positions + 1, rotation=0)

        ax.xaxis.set_label_position('top')
        plt.gca().xaxis.tick_top()

        plt.xlabel("Key Sentence Index", fontsize=12)
        plt.ylabel("Query Sentence Index", fontsize=12)

        max_att_h = attns_h.sum(axis=0).argmax()
        ax.add_patch(Rectangle((max_att_h - 0.5, 0.15), 2, attns_h.shape[0] - 0.35,
                                fill=False, edgecolor="black", lw=1.5, alpha=1))

        plt.title(f"Attention Heatmap - {type_char}-Report, Highest Prob (Sample {high_idx})\n" +
                  f"ID: {high_report_id}, Pred: {probs[high_idx]:.3f}, True: {high_true_label}",
                  fontsize=14, fontweight='bold')

        plt.savefig(os.path.join(output_path, f"attention_heatmap_HighProb_{type_suffix}_{experiment_name}.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()

        high_analysis = analyze_high_attention_sentences(
            attns_h, sentences_h, high_idx, f"HighProb ({type_char}-type)", threshold=0.05, top_k=10
        )
        saved_file = save_attention_analysis(high_analysis, output_path, experiment_name, f"HighProb_{type_suffix}")

        print(f"\n    Highest Prob {type_char}-type: Sample {high_idx} ({high_report_id})")
        print(f"      Prediction: {probs[high_idx]:.4f}, True: {high_true_label}")
        print(f"      Saved detailed analysis: {os.path.basename(saved_file)}")

        low_idx = type_indices[np.argmin(type_probs)]
        low_report_id = str(report_id_array[low_idx])
        low_true_label = "ASD" if labels_np[low_idx] == 1 else "Non-ASD"

        attns_l = attention_matrices_np[low_idx]
        sentences_l = decoded_reports[low_idx]

        nonASD_col_l = "#DC5D23"
        att_pal_ctl_l = sns.color_palette("blend:#f9f9f9," + nonASD_col_l, as_cmap=True)

        plt.figure(figsize=(12, 10))
        ax = sns.heatmap(attns_l, cmap=att_pal_ctl_l, vmax=_vmax, vmin=_vmin,
                         cbar_kws={'label': 'Attention Weight'})

        x_ticks_positions = np.arange(9, attns_l.shape[1], 10)
        x_ticks_positions = np.hstack((0, x_ticks_positions))
        y_ticks_positions = np.arange(9, attns_l.shape[0], 10)
        y_ticks_positions = np.hstack((0, y_ticks_positions))

        plt.xticks(x_ticks_positions, x_ticks_positions + 1)
        plt.yticks(y_ticks_positions, y_ticks_positions + 1, rotation=0)

        ax.xaxis.set_label_position('top')
        plt.gca().xaxis.tick_top()

        plt.xlabel("Key Sentence Index", fontsize=12)
        plt.ylabel("Query Sentence Index", fontsize=12)

        max_att_l = attns_l.sum(axis=0).argmax()
        ax.add_patch(Rectangle((max_att_l - 0.5, 0.15), 2, attns_l.shape[0] - 0.35,
                                fill=False, edgecolor="black", lw=1.5, alpha=1))

        plt.title(f"Attention Heatmap - {type_char}-Report, Lowest Prob (Sample {low_idx})\n" +
                  f"ID: {low_report_id}, Pred: {probs[low_idx]:.3f}, True: {low_true_label}",
                  fontsize=14, fontweight='bold')

        plt.savefig(os.path.join(output_path, f"attention_heatmap_LowProb_{type_suffix}_{experiment_name}.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()

        low_analysis = analyze_high_attention_sentences(
            attns_l, sentences_l, low_idx, f"LowProb ({type_char}-type)", threshold=0.05, top_k=10
        )
        saved_file = save_attention_analysis(low_analysis, output_path, experiment_name, f"LowProb_{type_suffix}")

        print(f"\n    Lowest Prob {type_char}-type: Sample {low_idx} ({low_report_id})")
        print(f"      Prediction: {probs[low_idx]:.4f}, True: {low_true_label}")
        print(f"      Saved detailed analysis: {os.path.basename(saved_file)}")

def analyze_phenotypes(attention_matrices_np, decoded_reports, labels_np, output_path, experiment_name):

    print("\n  Key phenotype analysis...")

    most_attended_sentences_asd = []
    most_attended_sentences_ctl = []
    n_asd = 0
    n_ctl = 0

    for i, attn in enumerate(attention_matrices_np):
        most_attended_idx = attn.sum(axis=0).argmax()
        sentence = decoded_reports[i][most_attended_idx]

        if labels_np[i] == 1:
            most_attended_sentences_asd.append(sentence)
            n_asd += 1
        else:
            most_attended_sentences_ctl.append(sentence)
            n_ctl += 1

    asd_freq_words = []
    for sent in most_attended_sentences_asd:
        words = [word.strip().strip(punctuation) for word in sent.split() if word.strip()]
        asd_freq_words.extend(words)
    asd_freq_words = Counter(asd_freq_words)
    for item in asd_freq_words:
        asd_freq_words[item] = asd_freq_words[item] / n_asd

    ctl_freq_words = []
    for sent in most_attended_sentences_ctl:
        words = [word.strip().strip(punctuation) for word in sent.split() if word.strip()]
        ctl_freq_words.extend(words)
    ctl_freq_words = Counter(ctl_freq_words)
    for item in ctl_freq_words:
        ctl_freq_words[item] = ctl_freq_words[item] / n_ctl

    asd_df = pd.DataFrame.from_dict(asd_freq_words, orient="index", columns=["asd_freq"])
    ctl_df = pd.DataFrame.from_dict(ctl_freq_words, orient="index", columns=["ctl_freq"])
    freq_df = pd.concat([asd_df, ctl_df], axis=1).fillna(0)

    freq_df["asd_ratio"] = freq_df["asd_freq"] / (freq_df["ctl_freq"] + 1e-6)
    freq_df["ctl_ratio"] = freq_df["ctl_freq"] / (freq_df["asd_freq"] + 1e-6)
    freq_df["diff"] = freq_df["asd_freq"] - freq_df["ctl_freq"]

    min_freq = 0.01
    freq_df_filtered = freq_df[(freq_df["asd_freq"] >= min_freq) | (freq_df["ctl_freq"] >= min_freq)]

    asd_enriched = freq_df_filtered.sort_values("asd_ratio", ascending=False).head(30)

    ctl_enriched = freq_df_filtered.sort_values("ctl_ratio", ascending=False).head(30)

    main_col = "#C73682"

    plt.figure(figsize=(12, 8))
    top_asd = asd_enriched.head(20)
    top_asd = top_asd[top_asd.index != '']

    plt.barh(range(len(top_asd)), top_asd['asd_ratio'], color=main_col, alpha=0.8)
    plt.yticks(range(len(top_asd)), top_asd.index)
    plt.xlabel("Frequency Ratio (ASD / Non-ASD)", fontsize=12)
    plt.ylabel("Word", fontsize=12)
    plt.title("Top ASD-Enriched Words in Most Attended Sentences", fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"phenotypes_ASD_{experiment_name}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    return {
        'asd_enriched': asd_enriched,
        'ctl_enriched': ctl_enriched,
        'full_freq_df': freq_df,
        'most_attended_asd': most_attended_sentences_asd,
        'most_attended_ctl': most_attended_sentences_ctl,
    }

# ============================================================================

# ============================================================================

def load_dsm5_criteria(tokenizer, report_max_length, sentence_max_length):

    if not os.path.isdir(DSM_TXT_PATH):
        return None, None
    criteria_dict = {}
    for file in os.listdir(DSM_TXT_PATH):
        if file.endswith(".txt"):
            with open(os.path.join(DSM_TXT_PATH, file), "r", encoding="utf-8") as f:
                criteria_dict[file.split(".")[0]] = [line.rstrip("\n") for line in f]
    if "A" not in criteria_dict or "B" not in criteria_dict:
        return None, None

    crit_inputs, crit_masks = {}, {}
    for key in ["A", "B"]:
        out = tokenizer(criteria_dict[key], truncation=True, padding="max_length",
                        max_length=sentence_max_length)
        ids, mask = out["input_ids"], out["attention_mask"]
        n_sent = len(ids)
        pad = np.full((report_max_length - n_sent, sentence_max_length), tokenizer.pad_token_id)
        apad = np.full((report_max_length - n_sent, sentence_max_length), 0)
        crit_inputs[key] = torch.tensor(np.vstack((ids, pad)))
        crit_masks[key] = torch.tensor(np.vstack((mask, apad)))
    return crit_inputs, crit_masks

def compute_fold_criteria_embeddings(experiment_name, crit_inputs, crit_masks, device, report_max_length):

    crit_embs_per_fold = {}
    for fold in range(1, K_FOLDS + 1):
        model_file = os.path.join(MODEL_PATH, f"{experiment_name}_fold{fold}.pt")
        if not os.path.exists(model_file):
            continue
        model = SentenceAttentionBERTKorean(base_model_name=BASE_MODEL_NAME,
                                            report_max_length=report_max_length)
        model.load_state_dict(torch.load(model_file, map_location=device))
        model = model.to(device); model.eval()
        fold_crit = {}
        with torch.no_grad():
            for key in ["A", "B"]:
                _, _, _, lhs, _ = model(crit_inputs[key].unsqueeze(0).to(device),
                                        attn_mask=crit_masks[key].unsqueeze(0).to(device))

                fold_crit[key] = lhs[0].cpu().numpy().mean(axis=1)
        crit_embs_per_fold[fold] = fold_crit
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    return crit_embs_per_fold

def _criteria_cosine_per_fold(lhs_embs_np, fold_assignment, crit_embs_per_fold):

    n, L = lhs_embs_np.shape[0], lhs_embs_np.shape[1]
    sim_A = np.full((n, 3, L), np.nan)
    sim_B = np.full((n, 4, L), np.nan)
    for i in range(n):
        f = int(fold_assignment[i])
        if f not in crit_embs_per_fold:
            continue
        sents = torch.tensor(lhs_embs_np[i].mean(axis=1))  # (L, hidden)
        cA, cB = crit_embs_per_fold[f]["A"], crit_embs_per_fold[f]["B"]
        for jj, j in enumerate([1, 2, 3]):
            sim_A[i, jj] = F.cosine_similarity(sents, torch.tensor(cA[j]).unsqueeze(0), dim=1).numpy()
        for kk, k in enumerate([1, 2, 3, 4]):
            sim_B[i, kk] = F.cosine_similarity(sents, torch.tensor(cB[k]).unsqueeze(0), dim=1).numpy()
    return sim_A, sim_B

def analyze_dsm5_criteria(lhs_embs_np, attention_matrices_np, labels_np, pat_ids,
                          fold_assignment, crit_embs_per_fold, pca, output_path,
                          experiment_name, report_max_length):

    print("\n  [Sections 6-8] DSM-5 criterion alignment analysis (per-fold criterion embeddings)...")

    main_col, ASD_col, nonASD_col = "#C73682", "#23a2dc", "#DC5D23"
    criteria_names = ["A1", "A2", "A3", "B1", "B2", "B3", "B4"]

    sim_A, sim_B = _criteria_cosine_per_fold(lhs_embs_np, fold_assignment, crit_embs_per_fold)

    n = lhs_embs_np.shape[0]
    idx = np.arange(n)
    most_attended = np.argmax(attention_matrices_np.mean(axis=1), axis=1)  # (n,)

    sim_A_att = sim_A[idx, :, most_attended]   # (n, 3)
    sim_B_att = sim_B[idx, :, most_attended]   # (n, 4)

    if _HAS_RAINCLOUD:
        df_A = pd.DataFrame(sim_A_att, columns=["A1", "A2", "A3"]); df_A["diagnosis"] = labels_np
        df_A = pd.melt(df_A, id_vars=["diagnosis"], value_vars=["A1", "A2", "A3"],
                       var_name="criteria", value_name="cos_sim")
        df_B = pd.DataFrame(sim_B_att, columns=["B1", "B2", "B3", "B4"]); df_B["diagnosis"] = labels_np
        df_B = pd.melt(df_B, id_vars=["diagnosis"], value_vars=["B1", "B2", "B3", "B4"],
                       var_name="criteria", value_name="cos_sim")
        all_att = pd.concat([df_A, df_B])

        f, ax = plt.subplots(figsize=(14, 7))
        ax = pt.RainCloud(x="criteria", y="cos_sim", data=all_att, hue="diagnosis",
                          palette=[nonASD_col, ASD_col], bw=0.05, width_viol=1, ax=ax,
                          orient="v", alpha=0.75, dodge=True, move=0, box_showfliers=False,
                          box_medianprops={"zorder": 11}, rain_alpha=0, offset=0.2,
                          scale="area", width_box=0.3, rasterized=True)
        plt.xlabel("DSM-5 Criterion", fontdict={"size": 12})
        plt.ylabel("Cosine Similarity to Most Attended Sentence", fontdict={"size": 12})
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(title="MD-Assessed Diagnosis", handles=handles, labels=["Non-autism", "Autism"],
                  loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=True, ncol=2, fontsize=12)
        plt.setp(ax.get_legend().get_title(), fontsize="12", weight="bold")
        ax.set_xlim(-0.7, 6.7); ax.autoscale(axis="y"); ax.use_sticky_edges = False
        plt.axhline(0, color="lightgrey", linewidth=1, linestyle="--")
        plt.tight_layout(); plt.subplots_adjust(bottom=0.18)
        f.savefig(os.path.join(output_path, f"cosine_dists_{experiment_name}.png"),
                  dpi=600, bbox_inches="tight")
        plt.close()
        print("    Section 6: saved cosine_dists")
    else:
        print("    [WARNING] ptitprince/adjustText not available; skipping RainCloud (section 6)")

    np.random.seed(RANDOM_STATE)
    rand_idx = np.random.randint(0, attention_matrices_np.shape[1], size=n)
    sim_A_rand = sim_A[idx, :, rand_idx]; sim_B_rand = sim_B[idx, :, rand_idx]

    sim_attended = np.concatenate((sim_A_att, sim_B_att), axis=1)   # (n, 7)
    sim_random = np.concatenate((sim_A_rand, sim_B_rand), axis=1)

    valid = np.isfinite(sim_attended).all(axis=1) & np.isfinite(sim_random).all(axis=1)
    sim_attended_v, sim_random_v = sim_attended[valid], sim_random[valid]
    y_v = np.asarray(labels_np)[valid]
    pat_v = np.asarray(pat_ids)[valid]

    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results_dict = {}
    for sim_type, sim_combined in [("attended", sim_attended_v), ("random", sim_random_v)]:
        coefs, avg_score, fold_probas, y_true = [], [], [], []
        for tr, te in skf.split(sim_combined, y_v, groups=pat_v):
            clf = LinearDiscriminantAnalysis(store_covariance=True).fit(sim_combined[tr], y_v[tr])
            scores = clf.transform(sim_combined[te]).squeeze()
            coefs.append([np.corrcoef(sim_combined[te][:, c], scores)[0, 1]
                          for c in range(sim_combined.shape[1])])
            avg_score.append(clf.score(sim_combined[te], y_v[te]))
            fold_probas.extend(clf.predict_proba(sim_combined[te])[:, 1]); y_true.extend(y_v[te])
        fpr, tpr, _ = roc_curve(y_true, fold_probas)
        results_dict[sim_type] = [fpr, tpr, np.array(coefs), roc_auc_score(y_true, fold_probas)]
        print(f"    Section 7 {sim_type}: Acc={np.mean(avg_score):.4f}, AUC={results_dict[sim_type][3]:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(results_dict["attended"][0], results_dict["attended"][1], color=main_col, label="Most Attended Sentence")
    plt.plot(results_dict["random"][0], results_dict["random"][1], color="#699296", label="Random Sentence")
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=0.6)
    plt.xlabel("False Positive Rate", fontsize=12); plt.ylabel("True Positive Rate", fontsize=12)
    plt.legend(loc="lower right", prop={"size": 12})
    plt.savefig(os.path.join(output_path, f"ROC_cos_sim_{experiment_name}.svg"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    coefs = results_dict["attended"][2]
    sns.swarmplot(data=coefs, color=main_col, size=7, alpha=1)
    plt.ylabel("Correlation with LDA Score", fontsize=12); plt.xlabel("DSM-5 Criteria", fontsize=12)
    if coefs.shape[1] == len(criteria_names):
        plt.xticks(np.arange(coefs.shape[1]), criteria_names)
    plt.axhline(y=0, color="k", linestyle="--")
    plt.savefig(os.path.join(output_path, f"LDA_cos_sim_bp_{experiment_name}.svg"), dpi=300, bbox_inches="tight")
    plt.close()
    print("    Section 7: saved ROC_cos_sim and LDA_cos_sim_bp")

    sentence_embs = lhs_embs_np.mean(axis=2).reshape(-1, HIDDEN_SIZE)
    pca_embs = pca.transform(sentence_embs)
    targs = np.array([[l] * report_max_length for l in labels_np]).reshape(-1)

    folds = sorted(crit_embs_per_fold.keys())
    A_stack = np.stack([crit_embs_per_fold[f]["A"][1:4] for f in folds]).mean(axis=0)  # (3, hidden)
    B_stack = np.stack([crit_embs_per_fold[f]["B"][1:5] for f in folds]).mean(axis=0)  # (4, hidden)
    crit_embs = np.concatenate((A_stack, B_stack), axis=0)  # (7, hidden)
    pca_crit = pca.transform(crit_embs)

    plt.figure(figsize=(6, 5.2))
    for label, color, name in [(0, nonASD_col, "Non-autism"), (1, ASD_col, "Autism")]:
        m = targs == label
        plt.scatter(pca_embs[m, 0], pca_embs[m, 1], s=2, c=color, alpha=0.5, label=name, rasterized=True)
    plt.legend(title="MD-Assessed Diagnosis", loc="upper right", prop={"size": 12},
               title_fontproperties={"size": 12, "weight": "bold"})
    plt.xlabel("PC1", fontdict={"size": 14}); plt.ylabel("PC2", fontdict={"size": 14})
    sc = plt.scatter(pca_crit[:, 0], pca_crit[:, 1], s=20, c=main_col, rasterized=True)
    sc.set_path_effects([withStroke(linewidth=2, foreground="white")])
    texts = []
    for i_c, txt in enumerate(criteria_names):
        t = plt.text(pca_crit[i_c, 0], pca_crit[i_c, 1], txt,
                     fontdict=dict(color=main_col, weight="bold", size=14))
        t.set_path_effects([withStroke(linewidth=1, foreground="white")]); texts.append(t)
    if _HAS_RAINCLOUD:
        adjust_text(texts)
    plt.title("Sentence PCA + DSM-5 Criteria (fold-avg, approx.)", fontsize=12, fontweight="bold")
    plt.savefig(os.path.join(output_path, f"pca_crit_embs_{experiment_name}.svg"), dpi=300, bbox_inches="tight")
    plt.close()
    print("    Section 8: saved pca_crit_embs (fold-average approximation)")

def main():
    parser = argparse.ArgumentParser(description="Korean clinical report phenotype analysis")
    parser.add_argument("--experiment_name", required=True, type=str,
                        help="Experiment name (model file prefix)")
    parser.add_argument("--output_dir", default=None, type=str,
                        help="Output directory")
    parser.add_argument("--use_saved_intermediates", action="store_true",
                        help="Use saved intermediate outputs and skip inference")
    parser.add_argument("--report_max_length", default=64, type=int,
                        help="Maximum number of sentences; must match the training setting")
    parser.add_argument("--sentence_max_length", default=64, type=int,
                        help="Maximum tokens per sentence; must match the training setting")
    parser.add_argument("--tokenized_path", default=None, type=str,
                        help="Tokenized tensor data path")
    parser.add_argument("--colorbar", default="fixed", choices=["fixed", "datarange"],
                        help="Attention heatmap colorbar range: fixed(0-1) or datarange(actual min-max)")
    args = parser.parse_args()

    experiment_name = args.experiment_name
    output_path = args.output_dir or os.path.join(os.path.dirname(__file__), "analysis_results", experiment_name)
    os.makedirs(output_path, exist_ok=True)

    report_max_length = args.report_max_length
    sentence_max_length = args.sentence_max_length

    print("=" * 60)
    print("Korean clinical report phenotype analysis")
    print("=" * 60)
    print(f"\nExperiment name: {experiment_name}")
    print(f"Output path: {output_path}")
    print(f"report_max_length: {report_max_length}")
    print(f"sentence_max_length: {sentence_max_length}")

    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False

    fm._load_fontmanager(try_read_cache=False)
    plt.rcParams['font.family'] = 'NanumBarunGothic'
    plt.rcParams['axes.unicode_minus'] = False

    print("\n[1/6] Loading data...")

    tokenized_path = args.tokenized_path or os.path.join(DATA_PATH, "reports_tokenized")
    print(f"  Tokenized data path: {tokenized_path}")

    input_tensor = torch.load(os.path.join(tokenized_path, "input_tensor"),
                              map_location=torch.device('cpu'), weights_only=False)
    attention_mask_tensor = torch.load(os.path.join(tokenized_path, "attention_mask_tensor"),
                                        map_location=torch.device('cpu'), weights_only=False)
    label_tensor = torch.load(os.path.join(tokenized_path, "label_tensor"),
                              map_location=torch.device('cpu'), weights_only=False)
    report_id_array = torch.load(os.path.join(tokenized_path, "report_id_array"),
                                  map_location=torch.device('cpu'), weights_only=False)

    pat_ids = [str(rid)[:12] for rid in report_id_array]

    print(f"  Total reports: {len(report_id_array)}")
    print(f"  Total patients: {len(set(pat_ids))}")
    print(f"  Label distribution: ASD={int(label_tensor.sum())}, Non-ASD={len(label_tensor) - int(label_tensor.sum())}")

    print("\n[2/6] Decoding report sentences...")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    decoded_reports = decode_reports(input_tensor, tokenizer)

    print(f"  Decoded reports: {len(decoded_reports)}")

    intermediates_path = os.path.join(ROOT, "intermediates", experiment_name)

    if args.use_saved_intermediates and os.path.exists(intermediates_path):
        print("\n[3/6] Loading saved intermediate outputs...")

        logits_np = np.load(os.path.join(intermediates_path, "logits_np.npy"))
        attention_matrices_np = np.load(os.path.join(intermediates_path, "attention_matrices_np.npy"))
        attention_weighted_sentence_embs_np = np.load(os.path.join(intermediates_path, "attention_weighted_sentence_embs_np.npy"))
        lhs_embs_np = np.load(os.path.join(intermediates_path, "lhs_embs_np.npy"))
        layer_pooled_embs_np = np.load(os.path.join(intermediates_path, "layer_pooled_embs_np.npy"))
        labels_np = np.load(os.path.join(intermediates_path, "labels_np.npy"))
        _fa = os.path.join(intermediates_path, "fold_assignment.npy")
        fold_assignment = np.load(_fa) if os.path.exists(_fa) else None

        print(f"  Loaded")
        print(f"    lhs_embs_np: {lhs_embs_np.shape}")
        print(f"    layer_pooled_embs_np: {layer_pooled_embs_np.shape}")
    else:
        print("\n[3/6] Running out-of-sample inference...")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Device: {device}")

        results = run_fold_inference(
            experiment_name, input_tensor, attention_mask_tensor,
            label_tensor, pat_ids, device,
            report_max_length=report_max_length,
            sentence_max_length=sentence_max_length
        )

        logits_np = results['logits_np']
        attention_matrices_np = results['attention_matrices_np']
        attention_weighted_sentence_embs_np = results['attention_weighted_sentence_embs_np']
        lhs_embs_np = results['lhs_embs_np']
        layer_pooled_embs_np = results['layer_pooled_embs_np']
        labels_np = results['labels_np']
        fold_assignment = results['fold_assignment']

    print("\n[4/6] Evaluating model performance...")

    probs = np_sigmoid(logits_np)
    preds = (probs >= 0.5).astype(int)
    accuracy = (preds == labels_np).mean()
    auc = roc_auc_score(labels_np, probs)

    print(f"  Out-of-sample performance:")
    print(f"    Accuracy: {accuracy:.4f}")
    print(f"    AUROC: {auc:.4f}")

    print("\n[5/6] Running analyses...")

    layer_results_df = analyze_layer_wise_prediction(
        layer_pooled_embs_np, labels_np, pat_ids, output_path, experiment_name
    )

    pca, pca_embs = analyze_pca_embeddings(
        lhs_embs_np, labels_np, output_path, experiment_name,
        report_max_length=report_max_length
    )

    analyze_attention_heatmap(
        attention_matrices_np, decoded_reports, labels_np, logits_np,
        report_id_array, output_path, experiment_name, colorbar=args.colorbar
    )

    phenotype_results = analyze_phenotypes(
        attention_matrices_np, decoded_reports, labels_np, output_path, experiment_name
    )

    asd_enriched = phenotype_results['asd_enriched']
    ctl_enriched = phenotype_results['ctl_enriched']

    print("\n  === Phenotypes enriched in the ASD group (top 20) ===")
    for i, (word, row) in enumerate(asd_enriched.head(20).iterrows()):
        if word:
            print(f"    {i+1:2d}. {word}: ASD freq={row['asd_freq']:.3f}, ratio={row['asd_ratio']:.2f}x")

    print("\n  === Phenotypes enriched in the Non-ASD group (top 20) ===")
    for i, (word, row) in enumerate(ctl_enriched.head(20).iterrows()):
        if word:
            print(f"    {i+1:2d}. {word}: Non-ASD freq={row['ctl_freq']:.3f}, ratio={row['ctl_ratio']:.2f}x")

    crit_inputs, crit_masks = load_dsm5_criteria(tokenizer, report_max_length, sentence_max_length)
    if crit_inputs is None:
        print(f"\n  [Sections 6-8 skip] DSM-5 criterion text not found: {DSM_TXT_PATH}")
        print(f"    (Add data/DSM-5/A.txt and B.txt to enable automatically)")
    elif fold_assignment is None:
        print("\n  [Sections 6-8 skip] fold_assignment is missing "
              "(run inference or provide intermediates/fold_assignment.npy)")
    else:
        device_c = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n  Computing DSM-5 criterion embeddings by fold, device={device_c})...")
        crit_embs_per_fold = compute_fold_criteria_embeddings(
            experiment_name, crit_inputs, crit_masks, device_c, report_max_length)
        if len(crit_embs_per_fold) == 0:
            print("  [Sections 6-8 skip] no fold models found")
        else:
            analyze_dsm5_criteria(
                lhs_embs_np, attention_matrices_np, labels_np, pat_ids,
                fold_assignment, crit_embs_per_fold, pca, output_path,
                experiment_name, report_max_length)

    print("\n[6/6] Saving results...")

    asd_enriched.to_csv(os.path.join(output_path, f"phenotypes_ASD_enriched_{experiment_name}.csv"))
    ctl_enriched.to_csv(os.path.join(output_path, f"phenotypes_NonASD_enriched_{experiment_name}.csv"))
    phenotype_results['full_freq_df'].to_csv(os.path.join(output_path, f"phenotypes_full_{experiment_name}.csv"))

    with open(os.path.join(output_path, f"most_attended_sentences_ASD_{experiment_name}.txt"), "w", encoding="utf-8") as f:
        for sent in phenotype_results['most_attended_asd']:
            f.write(sent + "\n")

    with open(os.path.join(output_path, f"most_attended_sentences_NonASD_{experiment_name}.txt"), "w", encoding="utf-8") as f:
        for sent in phenotype_results['most_attended_ctl']:
            f.write(sent + "\n")

    with open(os.path.join(output_path, f"performance_summary_{experiment_name}.txt"), "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"Experiment name: {experiment_name}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total reports: {len(report_id_array)}\n")
        f.write(f"Total patients: {len(set(pat_ids))}\n")
        f.write(f"ASD: {int(label_tensor.sum())}, Non-ASD: {len(label_tensor) - int(label_tensor.sum())}\n\n")
        f.write(f"Out-of-Sample Accuracy: {accuracy:.4f}\n")
        f.write(f"Out-of-Sample AUROC: {auc:.4f}\n\n")

        f.write("=== Layer-wise AUC ===\n")
        for _, row in layer_results_df.iterrows():
            f.write(f"Layer {int(row['layer'])}: AUC={row['mean_auc']:.4f} (+/- {row['std_auc']:.4f})\n")

        f.write("\n=== ASD Enriched Phenotypes (Top 20) ===\n")
        for i, (word, row) in enumerate(asd_enriched.head(20).iterrows()):
            if word:
                f.write(f"{i+1:2d}. {word}: freq={row['asd_freq']:.3f}, ratio={row['asd_ratio']:.2f}x\n")

        f.write("\n=== Non-ASD Enriched Phenotypes (Top 20) ===\n")
        for i, (word, row) in enumerate(ctl_enriched.head(20).iterrows()):
            if word:
                f.write(f"{i+1:2d}. {word}: freq={row['ctl_freq']:.3f}, ratio={row['ctl_ratio']:.2f}x\n")

    print(f"\nSaved files:")
    print(f"  - ROC_layer_{experiment_name}.png")
    print(f"  - AUC_layer_{experiment_name}.png")
    print(f"  - PCA_lhs_embs_{experiment_name}.png")
    print(f"  - PCA_report_embs_{experiment_name}.png")
    print(f"  - attention_heatmap_ASD_{experiment_name}.png")
    print(f"  - attention_heatmap_NonASD_{experiment_name}.png")
    print(f"  - attention_analysis_ASD_{experiment_name}.txt    ")
    print(f"  - attention_analysis_NonASD_{experiment_name}.txt ")
    print(f"  - attention_heatmap_HighProb_Asd_{experiment_name}.png  (if A-type reports exist)")
    print(f"  - attention_heatmap_LowProb_Asd_{experiment_name}.png   (if A-type reports exist)")
    print(f"  - attention_analysis_HighProb_Asd_{experiment_name}.txt (if A-type reports exist)")
    print(f"  - attention_analysis_LowProb_Asd_{experiment_name}.txt  (if A-type reports exist)")
    print(f"  - attention_heatmap_HighProb_Psy_{experiment_name}.png  (if P-type reports exist)")
    print(f"  - attention_heatmap_LowProb_Psy_{experiment_name}.png   (if P-type reports exist)")
    print(f"  - attention_analysis_HighProb_Psy_{experiment_name}.txt (if P-type reports exist)")
    print(f"  - attention_analysis_LowProb_Psy_{experiment_name}.txt  (if P-type reports exist)")
    print(f"  - phenotypes_ASD_{experiment_name}.png")
    print(f"  - phenotypes_ASD_enriched_{experiment_name}.csv")
    print(f"  - phenotypes_NonASD_enriched_{experiment_name}.csv")
    print(f"  - layer_wise_results_{experiment_name}.csv")
    print(f"  - performance_summary_{experiment_name}.txt")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
