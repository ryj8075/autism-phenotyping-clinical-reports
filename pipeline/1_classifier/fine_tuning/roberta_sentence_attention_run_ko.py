import os
import sys
import random
import numpy as np
import pandas as pd
import copy
from datetime import datetime
import json

import torch
from torch import nn

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from transformers import AutoTokenizer, AutoConfig
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score, accuracy_score, confusion_matrix
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.batcher import create_batches
from helpers.hierarchical_tokenizer_ko import hierarchical_tokenizer_korean
import glob

from custom_models.sentence_attention_base_pool_ko import SentenceAttentionBERTKorean

import wandb
import argparse

BASE_MODEL_NAME = "klue/roberta-base"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data")
# Raw report .txt corpus, used only when --data_mode txt; override with REPORT_TXT_DIR.
REPORT_PATH = os.environ.get(
    "REPORT_TXT_DIR", os.path.join(DATA_PATH, "489reports", "reports_txt_no_headings"))

def calculate_all_metrics(probs, labels):
    probs = np.array(probs)
    labels = np.array(labels)
    preds = (probs >= 0.5).astype(int)

    metrics = {}

    # AUROC
    try:
        metrics['auroc'] = roc_auc_score(labels, probs)
    except:
        metrics['auroc'] = 0.5

    # AUPRC
    try:
        metrics['auprc'] = average_precision_score(labels, probs)
    except:
        metrics['auprc'] = 0.5

    # Accuracy
    metrics['accuracy'] = accuracy_score(labels, preds)

    # Precision, Recall, F1
    metrics['precision'] = precision_score(labels, preds, zero_division=0)
    metrics['recall'] = recall_score(labels, preds, zero_division=0)
    metrics['f1'] = f1_score(labels, preds, zero_division=0)

    # Specificity (True Negative Rate)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics['TP'] = int(tp)
    metrics['FP'] = int(fp)
    metrics['TN'] = int(tn)
    metrics['FN'] = int(fn)

    return metrics

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
        self.best_attention_results = None

    def __call__(self, auroc, model, attention_results=None):
        if self.best_score is None:
            self.best_score = auroc
            self.save_checkpoint(model, attention_results)
        elif auroc < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = auroc
            self.save_checkpoint(model, attention_results)
            self.counter = 0

    def save_checkpoint(self, model, attention_results=None):
        if isinstance(model, nn.DataParallel):
            self.best_model_state = copy.deepcopy(model.module.state_dict())
        else:
            self.best_model_state = copy.deepcopy(model.state_dict())

        if attention_results is not None:
            self.best_attention_results = {
                'logits': attention_results['logits'].copy(),
                'attn_weights': attention_results['attn_weights'].copy(),
                'attn_outputs': attention_results['attn_outputs'].copy(),
                'lhs_embs': attention_results['lhs_embs'].copy(),
                'layer_pooled_embs': attention_results['layer_pooled_embs'].copy(),
            }

def load_data(data_mode, model_name, sentence_max_length, report_max_length,
              csv_path=None, meta_csv_path=None, tokenized_path=None):

    if data_mode == "tensor":
        if tokenized_path is None:
            tokenized_path = os.path.join(DATA_PATH, "reports_tokenized")
        input_tensor = torch.load(os.path.join(tokenized_path, "input_tensor"), weights_only=False)
        attention_mask_tensor = torch.load(os.path.join(tokenized_path, "attention_mask_tensor"), weights_only=False)
        label_tensor = torch.load(os.path.join(tokenized_path, "label_tensor"), weights_only=False)
        report_id_array = torch.load(os.path.join(tokenized_path, "report_id_array"), weights_only=False)
    elif data_mode == "txt":
        if meta_csv_path is None:
            raise ValueError("When data_mode='txt', --meta_csv_path must be provided.")
        report_files = glob.glob(os.path.join(REPORT_PATH, "*.txt"))
        meta_frame = pd.read_csv(meta_csv_path, encoding="utf-8")
        input_tensor, attention_mask_tensor, label_tensor, report_id_array = hierarchical_tokenizer_korean(
            report_files=report_files, meta_frame=meta_frame, tokenizer_name=model_name,
            sentence_max_length=sentence_max_length, sentence_min_length=30,
            report_max_length=report_max_length
        )
    else:
        raise ValueError(f"Unsupported data_mode: {data_mode}")

    return input_tensor, attention_mask_tensor, label_tensor, report_id_array

def train_single_fold(fold_idx, train_idx, valid_idx, input_tensor, attention_mask_tensor,
                      label_tensor, config, device, use_wandb=False):

    batch_size = config['batch_size']

    train_data = create_batches(input_tensor[train_idx], batch_size)
    train_mask = create_batches(attention_mask_tensor[train_idx], batch_size)
    valid_data = create_batches(input_tensor[valid_idx], batch_size)
    valid_mask = create_batches(attention_mask_tensor[valid_idx], batch_size)
    train_labels = create_batches(label_tensor[train_idx], batch_size)
    valid_labels = create_batches(label_tensor[valid_idx], batch_size)

    model = SentenceAttentionBERTKorean(
        base_model_name=config['model_name'],
        report_max_length=config['report_max_length'],
        ff_dropout=config['ff_dropout'],
        att_dropout=config['att_dropout'],
        class_dropout=config['class_dropout'],
        sentence_dropout=config['sentence_dropout']
    )

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    if config['optimizer'] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    loss_fn = nn.BCEWithLogitsLoss()

    early_stopping = EarlyStopping(patience=config['patience'])

    best_metrics = None
    best_epoch = 0

    for epoch in range(config['epochs']):
        model.train()
        train_losses = []
        train_probs_all = []
        train_labels_all = []

        for train_batch_idx in range(len(train_data)):
            model.zero_grad()
            current_batch = train_data[train_batch_idx]
            current_mask = train_mask[train_batch_idx]

            if isinstance(model, nn.DataParallel) and current_batch.size(0) < torch.cuda.device_count() * 2:
                output, _, _, _, _ = model.module(current_batch, attn_mask=current_mask)
            else:
                output, _, _, _, _ = model(current_batch, attn_mask=current_mask)

            output = torch.atleast_1d(output)
            train_labels_true = torch.atleast_1d(train_labels[train_batch_idx])
            loss = loss_fn(output, train_labels_true)

            train_losses.append(loss.item())
            train_probs_all.extend(torch.sigmoid(output).cpu().detach().numpy())
            train_labels_all.extend(train_labels_true.cpu().detach().numpy())

            loss.backward()
            optimizer.step()

        model.eval()
        valid_losses = []
        valid_probs_all = []
        valid_labels_all = []

        valid_logits_all = []
        valid_attn_weights_all = []
        valid_attn_outputs_all = []

        valid_lhs_embs_all = []
        valid_layer_pooled_embs_all = []

        with torch.no_grad():
            for valid_batch_idx in range(len(valid_data)):
                current_batch = valid_data[valid_batch_idx]
                current_mask = valid_mask[valid_batch_idx]

                if isinstance(model, nn.DataParallel):
                    output, attn_weights, attn_output, last_hidden_embs, all_layer_embs = model.module(current_batch, attn_mask=current_mask)
                else:
                    output, attn_weights, attn_output, last_hidden_embs, all_layer_embs = model(current_batch, attn_mask=current_mask)

                output = torch.atleast_1d(output)
                valid_labels_true = torch.atleast_1d(valid_labels[valid_batch_idx])
                loss = loss_fn(output, valid_labels_true)

                valid_losses.append(loss.item())
                valid_probs_all.extend(torch.sigmoid(output).cpu().detach().numpy())
                valid_labels_all.extend(valid_labels_true.cpu().detach().numpy())

                valid_logits_all.append(output.cpu().detach().numpy())
                valid_attn_weights_all.append(attn_weights.cpu().detach().numpy())
                valid_attn_outputs_all.append(attn_output.cpu().detach().numpy())

                # last_hidden_embs: list of (num_sentences, hidden_size) per batch item
                for lhs in last_hidden_embs:
                    valid_lhs_embs_all.append(lhs.cpu().detach().numpy())

                # layer_pooled_embs: list of (num_layers, hidden_size)
                for lp in all_layer_embs:
                    valid_layer_pooled_embs_all.append(lp.cpu().detach().numpy())

        train_metrics = calculate_all_metrics(train_probs_all, train_labels_all)
        train_metrics['loss'] = np.mean(train_losses)

        valid_metrics = calculate_all_metrics(valid_probs_all, valid_labels_all)
        valid_metrics['loss'] = np.mean(valid_losses)

        print(f"  Epoch {epoch+1}: Train AUROC={train_metrics['auroc']:.4f}, "
              f"Valid AUROC={valid_metrics['auroc']:.4f}, Valid Acc={valid_metrics['accuracy']:.4f}")

        if use_wandb:
            wandb.log({
                "train_loss": train_metrics['loss'],
                "train_auroc": train_metrics['auroc'],
                "valid_loss": valid_metrics['loss'],
                "valid_auroc": valid_metrics['auroc'],
                "valid_accuracy": valid_metrics['accuracy'],
            })

        attention_results = {
            'logits': np.concatenate(valid_logits_all),
            'attn_weights': np.concatenate(valid_attn_weights_all),
            'attn_outputs': np.concatenate(valid_attn_outputs_all),
            'lhs_embs': np.stack(valid_lhs_embs_all, axis=0),           # (num_valid_samples, num_sentences, seq_len, hidden_size)
            'layer_pooled_embs': np.stack(valid_layer_pooled_embs_all, axis=0),  # (num_valid_samples, num_layers, hidden_size)
        }

        early_stopping(valid_metrics['auroc'], model, attention_results)

        if early_stopping.best_score == valid_metrics['auroc']:
            best_metrics = valid_metrics.copy()
            best_epoch = epoch + 1

        if early_stopping.early_stop:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    if early_stopping.best_model_state is not None:
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(early_stopping.best_model_state)
        else:
            model.load_state_dict(early_stopping.best_model_state)

    return {
        'fold': fold_idx + 1,
        'best_epoch': best_epoch,
        'metrics': best_metrics,
        'model_state': early_stopping.best_model_state,
        'attention_results': early_stopping.best_attention_results,
    }

def get_patient_based_split(pat_ids, label_tensor, test_ratio, seed):

    from sklearn.model_selection import train_test_split

    unique_pats = np.unique(pat_ids)

    pat_labels = []
    for pat in unique_pats:
        pat_mask = np.array(pat_ids) == pat
        pat_label = int(label_tensor[pat_mask].float().mean() >= 0.5)
        pat_labels.append(pat_label)
    pat_labels = np.array(pat_labels)

    pats_trainval, pats_test = train_test_split(
        unique_pats, test_size=test_ratio, stratify=pat_labels, random_state=seed
    )

    trainval_idx = np.where(np.isin(pat_ids, pats_trainval))[0]
    test_idx = np.where(np.isin(pat_ids, pats_test))[0]

    return trainval_idx, test_idx

def train_model(config, use_wandb=False, verbose=True):
    seed = config.get('seed', 42)
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor, attention_mask_tensor, label_tensor, report_id_array = load_data(
        data_mode=config.get('data_mode', 'tensor'),
        model_name=config['model_name'],
        sentence_max_length=config.get('sentence_max_length', 64),
        report_max_length=config.get('report_max_length', 64),
        meta_csv_path=config.get('meta_csv_path'),
        tokenized_path=config.get('tokenized_path')
    )

    pat_ids = np.array([str(rid)[:12] for rid in report_id_array])

    if verbose:
        print(f"Total reports: {len(report_id_array)}, patients: {len(np.unique(pat_ids))}")

    test_ratio = config.get('test_ratio', 0.0)
    test_idx = None
    test_metrics = None

    report_id_array = np.asarray(report_id_array)

    if test_ratio > 0:
        trainval_idx, test_idx = get_patient_based_split(pat_ids, label_tensor, test_ratio, seed)
        if verbose:
            print(f"Hold-out split: Train+Valid={len(trainval_idx)}, Test={len(test_idx)}")

        input_tensor_cv = input_tensor[trainval_idx]
        attention_mask_tensor_cv = attention_mask_tensor[trainval_idx]
        label_tensor_cv = label_tensor[trainval_idx]
        pat_ids_cv = pat_ids[trainval_idx]
        report_id_array_cv = report_id_array[trainval_idx]
        cv_original_indices = trainval_idx
    else:
        input_tensor_cv = input_tensor
        attention_mask_tensor_cv = attention_mask_tensor
        label_tensor_cv = label_tensor
        pat_ids_cv = pat_ids
        report_id_array_cv = report_id_array
        cv_original_indices = np.arange(len(report_id_array))

    valid_sentence_mask_np = attention_mask_tensor_cv.cpu().numpy().sum(axis=-1) > 2
    n_valid_sentences_np = valid_sentence_mask_np.sum(axis=1).astype(np.int64)

    skf = StratifiedGroupKFold(n_splits=config.get('k_fold_CV', 5), shuffle=True, random_state=seed)
    splits = list(skf.split(input_tensor_cv, label_tensor_cv, groups=pat_ids_cv))

    fold_split_manifest = []
    for fold_idx, (train_idx, valid_idx) in enumerate(splits):
        fold_split_manifest.append({
            "fold": fold_idx + 1,
            "train_indices_cv": train_idx.tolist(),
            "valid_indices_cv": valid_idx.tolist(),
            "train_original_indices": cv_original_indices[train_idx].tolist(),
            "valid_original_indices": cv_original_indices[valid_idx].tolist(),
            "train_report_ids": report_id_array_cv[train_idx].astype(str).tolist(),
            "valid_report_ids": report_id_array_cv[valid_idx].astype(str).tolist(),
            "train_patient_ids": pat_ids_cv[train_idx].astype(str).tolist(),
            "valid_patient_ids": pat_ids_cv[valid_idx].astype(str).tolist(),
        })

    input_tensor_cv = input_tensor_cv.to(device)
    attention_mask_tensor_cv = attention_mask_tensor_cv.to(device)
    label_tensor_cv = label_tensor_cv.float().to(device)

    if test_ratio > 0:
        input_tensor_test = input_tensor[test_idx].to(device)
        attention_mask_test = attention_mask_tensor[test_idx].to(device)
        label_tensor_test = label_tensor[test_idx].float().to(device)

    all_fold_results = []
    best_fold_model_state = None
    best_fold_auroc = 0.0

    n_samples = len(input_tensor_cv)
    num_sentences = input_tensor_cv.shape[1]
    hidden_size = 768

    oos_logits = np.zeros(n_samples)
    oos_probs = np.zeros(n_samples)
    oos_attention_matrices = np.zeros((n_samples, num_sentences, num_sentences))
    oos_attention_weighted_embs = np.zeros((n_samples, num_sentences, hidden_size))

    num_layers = 13  # embedding + 12 transformer layers
    sentence_max_length = config.get('sentence_max_length', 64)
    oos_lhs_embs = np.zeros((n_samples, num_sentences, sentence_max_length, hidden_size))
    oos_layer_pooled_embs = np.zeros((n_samples, num_layers, hidden_size))

    oos_labels = label_tensor_cv.cpu().numpy()
    oos_fold_assignment = np.zeros(n_samples, dtype=int)

    model_dir = os.path.join(ROOT, "trained_models")
    os.makedirs(model_dir, exist_ok=True)

    fold_checkpoint_manifest = []

    for fold_idx, (train_idx, valid_idx) in enumerate(splits):
        if verbose:
            print(f"\n{'='*50}")
            print(f"Fold {fold_idx + 1}/{config.get('k_fold_CV', 5)}")
            print(f"{'='*50}")

        if use_wandb:
            wandb.init(
                project=config.get('experiment_name') or config.get('wandb_project', 'NLP-ASD-Korean'),
                name=f"GridSearch_Fold{fold_idx + 1}",
                group=f"lr={config['lr']}_dropout={config['ff_dropout']}",
                config=config,
                reinit=True
            )

        result = train_single_fold(
            fold_idx, train_idx, valid_idx,
            input_tensor_cv, attention_mask_tensor_cv, label_tensor_cv,
            config, device, use_wandb
        )
        all_fold_results.append(result)

        if result['metrics']['auroc'] > best_fold_auroc:
            best_fold_auroc = result['metrics']['auroc']
            best_fold_model_state = result['model_state']

        fold_model_file = None

        if config.get('save_fold_models', False) and result['model_state'] is not None:
            exp_name = config.get('experiment_name', 'model')
            fold_model_file = os.path.join(model_dir, f"{exp_name}_fold{fold_idx + 1}.pt")
            torch.save(result['model_state'], fold_model_file)
            if verbose:
                print(f"  Saved fold model: {fold_model_file}")

        fold_checkpoint_manifest.append({
            "fold": fold_idx + 1,
            "checkpoint_path": fold_model_file,
            "saved": fold_model_file is not None,
        })

        if config.get('save_intermediates', False) or config.get('save_fold_models', False):
            attn_results = result.get('attention_results')
            if attn_results is not None:

                for i, idx in enumerate(valid_idx):
                    oos_logits[idx] = attn_results['logits'][i]
                    oos_probs[idx] = 1 / (1 + np.exp(-attn_results['logits'][i]))  # sigmoid
                    oos_attention_matrices[idx] = attn_results['attn_weights'][i]
                    oos_attention_weighted_embs[idx] = attn_results['attn_outputs'][i]
                    oos_lhs_embs[idx] = attn_results['lhs_embs'][i]
                    oos_layer_pooled_embs[idx] = attn_results['layer_pooled_embs'][i]
                    oos_fold_assignment[idx] = fold_idx + 1

                if verbose:
                    print(f"  Saved best-epoch attention outputs: {len(valid_idx)} samples")

        if use_wandb:
            wandb.finish()

    metric_names = ['auroc', 'auprc', 'accuracy', 'precision', 'recall', 'specificity', 'f1', 'loss']
    summary = {}
    for m in metric_names:
        values = [r['metrics'][m] for r in all_fold_results]
        summary[f'avg_{m}'] = np.mean(values)
        summary[f'std_{m}'] = np.std(values)

    if test_ratio > 0 and best_fold_model_state is not None:
        if verbose:
            print(f"\n{'='*50}")
            print("Hold-out Test Set Evaluation")
            print(f"{'='*50}")

        model = SentenceAttentionBERTKorean(
            base_model_name=config['model_name'],
            report_max_length=config['report_max_length'],
            ff_dropout=config['ff_dropout'],
            att_dropout=config['att_dropout'],
            class_dropout=config['class_dropout'],
            sentence_dropout=config['sentence_dropout']
        )
        model.load_state_dict(best_fold_model_state)
        model = model.to(device)
        model.eval()

        test_data = create_batches(input_tensor_test, config['batch_size'])
        test_mask = create_batches(attention_mask_test, config['batch_size'])
        test_labels = create_batches(label_tensor_test, config['batch_size'])

        test_probs_all = []
        test_labels_all = []

        with torch.no_grad():
            for batch_idx in range(len(test_data)):
                current_batch = test_data[batch_idx]
                current_mask = test_mask[batch_idx]

                if isinstance(model, nn.DataParallel) and current_batch.size(0) < torch.cuda.device_count() * 2:
                    output, _, _, _, _ = model.module(current_batch, attn_mask=current_mask)
                else:
                    output, _, _, _, _ = model(current_batch, attn_mask=current_mask)

                output = torch.atleast_1d(output)
                test_probs_all.extend(torch.sigmoid(output).cpu().numpy())
                test_labels_all.extend(torch.atleast_1d(test_labels[batch_idx]).cpu().numpy())

        test_metrics = calculate_all_metrics(test_probs_all, test_labels_all)

        if verbose:
            print(f"Test AUROC: {test_metrics['auroc']:.4f}")
            print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
            print(f"Test F1: {test_metrics['f1']:.4f}")

    oos_results = None
    if config.get('save_intermediates', False) or config.get('save_fold_models', False):
        predicted_labels_np = (oos_probs >= 0.5).astype(np.int64)

        oos_results = {
            'logits_np': oos_logits,
            'probs_np': oos_probs,
            'predicted_labels_np': predicted_labels_np,
            'attention_matrices_np': oos_attention_matrices,
            'attention_weighted_sentence_embs_np': oos_attention_weighted_embs,
            'lhs_embs_np': oos_lhs_embs,
            'layer_pooled_embs_np': oos_layer_pooled_embs,
            'labels_np': oos_labels,
            'fold_assignment': oos_fold_assignment,
            "report_id_array": report_id_array_cv.astype(str),
            "patient_id_array": pat_ids_cv.astype(str),
            "cv_original_indices": cv_original_indices,
            "valid_sentence_mask_np": valid_sentence_mask_np,
            "n_valid_sentences_np": n_valid_sentences_np,
        }

    return {
        'fold_results': all_fold_results,
        'summary': summary,
        'config': config,
        'test_metrics': test_metrics,
        'best_model_state': best_fold_model_state,
        'splits': splits,
        'label_tensor_cv': label_tensor_cv.cpu(),
        'oos_results': oos_results,
        "fold_split_manifest": fold_split_manifest,
        "fold_checkpoint_manifest": fold_checkpoint_manifest,
    }

def main():
    parser = argparse.ArgumentParser(description='Korean Clinical Report Classification Training')

    parser.add_argument("--epochs", default=10, type=int, help="Maximum number of training epochs")
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size")
    parser.add_argument("--lr", default=1e-5, type=float, help="Learning rate")
    parser.add_argument("--optimizer", default="AdamW", type=str, help="Optimizer")
    parser.add_argument("--patience", default=3, type=int, help="Early stopping patience")

    parser.add_argument("--ff_dropout", default=0.1, type=float)
    parser.add_argument("--att_dropout", default=0.1, type=float)
    parser.add_argument("--class_dropout", default=0.1, type=float)
    parser.add_argument("--sentence_dropout", default=0.0, type=float)

    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--job_id", default="0", type=str)
    parser.add_argument("--k_fold_CV", default=5, type=int)
    parser.add_argument("--model_name", default=BASE_MODEL_NAME, type=str)
    parser.add_argument("--report_max_length", default=64, type=int)
    parser.add_argument("--sentence_max_length", default=64, type=int)

    parser.add_argument("--data_mode", default="tensor", type=str)
    parser.add_argument("--meta_csv_path", default=None, type=str)
    parser.add_argument("--tokenized_path", default=None, type=str,
                        help="Tokenized tensor data path used when data_mode='tensor'")

    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--test_ratio", default=0.0, type=float,
                        help="Hold-out test set ratio. Use 0 to run K-fold CV without a test set.")
    parser.add_argument("--experiment_name", default=None, type=str,
                        help="Experiment name used in result filenames")
    parser.add_argument("--save_fold_models", action="store_true",
                        help="Save each fold model as a separate file for out-of-sample analysis")
    parser.add_argument("--save_intermediates", action="store_true",
                        help="Save out-of-sample inference intermediates for downstream pipeline analyses")

    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="NLP-ASD-Korean", type=str)

    args = parser.parse_args()

    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'optimizer': args.optimizer,
        'patience': args.patience,
        'ff_dropout': args.ff_dropout,
        'att_dropout': args.att_dropout,
        'class_dropout': args.class_dropout,
        'sentence_dropout': args.sentence_dropout,
        'weight_decay': args.weight_decay,
        'k_fold_CV': args.k_fold_CV,
        'model_name': args.model_name,
        'report_max_length': args.report_max_length,
        'sentence_max_length': args.sentence_max_length,
        'data_mode': args.data_mode,
        'meta_csv_path': args.meta_csv_path,
        'tokenized_path': args.tokenized_path,
        'seed': args.seed,
        'test_ratio': args.test_ratio,
        'experiment_name': args.experiment_name,
        'save_fold_models': args.save_fold_models,
        'save_intermediates': args.save_intermediates,
        'wandb_project': args.wandb_project,
    }

    print("=" * 60)
    print("NLP-ASD Korean Model Training (v3)")
    print("=" * 60)
    print(f"\nConfiguration: {config}\n")

    results = train_model(config, use_wandb=not args.no_wandb, verbose=True)

    print("\n" + "=" * 60)
    print("=== Final Results (mean +/- standard deviation) ===")
    print("=" * 60)

    summary = results['summary']
    print(f"AUROC:       {summary['avg_auroc']:.4f} (+/- {summary['std_auroc']:.4f})")
    print(f"AUPRC:       {summary['avg_auprc']:.4f} (+/- {summary['std_auprc']:.4f})")
    print(f"Accuracy:    {summary['avg_accuracy']:.4f} (+/- {summary['std_accuracy']:.4f})")
    print(f"Precision:   {summary['avg_precision']:.4f} (+/- {summary['std_precision']:.4f})")
    print(f"Recall:      {summary['avg_recall']:.4f} (+/- {summary['std_recall']:.4f})")
    print(f"Specificity: {summary['avg_specificity']:.4f} (+/- {summary['std_specificity']:.4f})")
    print(f"F1:          {summary['avg_f1']:.4f} (+/- {summary['std_f1']:.4f})")
    print(f"Loss:        {summary['avg_loss']:.4f} (+/- {summary['std_loss']:.4f})")

    print("\n--- Results by Fold ---")
    for r in results['fold_results']:
        m = r['metrics']
        print(f"Fold {r['fold']} (Best Epoch={r['best_epoch']}): "
              f"AUROC={m['auroc']:.4f}, Acc={m['accuracy']:.4f}, F1={m['f1']:.4f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    if config.get('experiment_name'):
        results_file = os.path.join(results_dir, f"{config['experiment_name']}.txt")
    else:
        results_file = os.path.join(results_dir, f"training_results_v3_{timestamp}.txt")

    with open(results_file, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("NLP-ASD Korean Model Training Results (v3)\n")
        f.write("=" * 60 + "\n\n")

        f.write("### Configuration ###\n")
        for k, v in config.items():
            f.write(f"{k}: {v}\n")

        f.write("\n### CV Results (mean +/- standard deviation) ###\n")
        f.write(f"AUROC:       {summary['avg_auroc']:.4f} (+/- {summary['std_auroc']:.4f})\n")
        f.write(f"AUPRC:       {summary['avg_auprc']:.4f} (+/- {summary['std_auprc']:.4f})\n")
        f.write(f"Accuracy:    {summary['avg_accuracy']:.4f} (+/- {summary['std_accuracy']:.4f})\n")
        f.write(f"Precision:   {summary['avg_precision']:.4f} (+/- {summary['std_precision']:.4f})\n")
        f.write(f"Recall:      {summary['avg_recall']:.4f} (+/- {summary['std_recall']:.4f})\n")
        f.write(f"Specificity: {summary['avg_specificity']:.4f} (+/- {summary['std_specificity']:.4f})\n")
        f.write(f"F1:          {summary['avg_f1']:.4f} (+/- {summary['std_f1']:.4f})\n")
        f.write(f"Loss:        {summary['avg_loss']:.4f} (+/- {summary['std_loss']:.4f})\n")

        if results.get('test_metrics'):
            test_m = results['test_metrics']
            f.write("\n### Hold-out Test Results ###\n")
            f.write(f"Test AUROC:       {test_m['auroc']:.4f}\n")
            f.write(f"Test AUPRC:       {test_m['auprc']:.4f}\n")
            f.write(f"Test Accuracy:    {test_m['accuracy']:.4f}\n")
            f.write(f"Test Precision:   {test_m['precision']:.4f}\n")
            f.write(f"Test Recall:      {test_m['recall']:.4f}\n")
            f.write(f"Test Specificity: {test_m['specificity']:.4f}\n")
            f.write(f"Test F1:          {test_m['f1']:.4f}\n")

        f.write("\n### Results by Fold ###\n")
        for r in results['fold_results']:
            m = r['metrics']
            f.write(f"Fold {r['fold']} (Best Epoch={r['best_epoch']}): "
                    f"AUROC={m['auroc']:.4f}, AUPRC={m['auprc']:.4f}, "
                    f"Acc={m['accuracy']:.4f}, Prec={m['precision']:.4f}, "
                    f"Rec={m['recall']:.4f}, Spec={m['specificity']:.4f}, F1={m['f1']:.4f}\n")

        f.write("\n### Class Distribution and Confusion Matrix by Fold ###\n")
        splits = results['splits']
        label_tensor_cv = results['label_tensor_cv']

        total_TP, total_FP, total_TN, total_FN = 0, 0, 0, 0

        for fold_idx, (train_idx, valid_idx) in enumerate(splits):
            r = results['fold_results'][fold_idx]
            m = r['metrics']

            train_labels = label_tensor_cv[train_idx]
            valid_labels = label_tensor_cv[valid_idx]
            train_pos = int(train_labels.sum().item())
            train_neg = len(train_labels) - train_pos
            valid_pos = int(valid_labels.sum().item())
            valid_neg = len(valid_labels) - valid_pos

            # Confusion Matrix
            TP, FP, TN, FN = m['TP'], m['FP'], m['TN'], m['FN']
            total_TP += TP
            total_FP += FP
            total_TN += TN
            total_FN += FN

            f.write(f"\nFold {fold_idx + 1}:\n")
            f.write(f"  Train: {len(train_idx)} samples (Pos={train_pos}, Neg={train_neg}, Pos ratio={train_pos/len(train_idx):.1%})\n")
            f.write(f"  Valid: {len(valid_idx)} samples (Pos={valid_pos}, Neg={valid_neg}, Pos ratio={valid_pos/len(valid_idx):.1%})\n")
            f.write(f"  Confusion Matrix: TP={TP}, FP={FP}, TN={TN}, FN={FN}\n")

            if TP == 0 and FP == 0:
                f.write(f"  [Warning] All samples predicted as Negative\n")
            elif TN == 0 and FN == 0:
                f.write(f"  [Warning] All samples predicted as Positive\n")

        f.write(f"\nOverall Confusion Matrix Totals:\n")
        f.write(f"  TP={total_TP}, FP={total_FP}, TN={total_TN}, FN={total_FN}\n")
        if total_TP + total_FP > 0:
            f.write(f"  Overall Precision: {total_TP / (total_TP + total_FP):.4f}\n")
        if total_TP + total_FN > 0:
            f.write(f"  Overall Recall: {total_TP / (total_TP + total_FN):.4f}\n")
        if total_TN + total_FP > 0:
            f.write(f"  Overall Specificity: {total_TN / (total_TN + total_FP):.4f}\n")
        total_samples = total_TP + total_FP + total_TN + total_FN
        if total_samples > 0:
            f.write(f"  Overall Accuracy: {(total_TP + total_TN) / total_samples:.4f}\n")

        f.write(f"\nSaved at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"\nSaved results: {results_file}")

    if config.get('test_ratio', 0) > 0 and results.get('best_model_state'):
        model_dir = os.path.join(ROOT, "trained_models")
        os.makedirs(model_dir, exist_ok=True)
        if config.get('experiment_name'):
            model_file = os.path.join(model_dir, f"{config['experiment_name']}.pt")
        else:
            model_file = os.path.join(model_dir, f"best_model_{timestamp}.pt")
        torch.save(results['best_model_state'], model_file)
        print(f"Saved model: {model_file}")

    if results.get('oos_results') is not None:
        exp_name = config.get('experiment_name', f'experiment_{timestamp}')
        intermediates_dir = os.path.join(ROOT, "intermediates", exp_name)
        os.makedirs(intermediates_dir, exist_ok=True)

        oos = results['oos_results']
        np.save(os.path.join(intermediates_dir, "logits_np.npy"), oos['logits_np'])
        np.save(os.path.join(intermediates_dir, "probs_np.npy"), oos['probs_np'])
        np.save(os.path.join(intermediates_dir, "predicted_labels_np.npy"), oos["predicted_labels_np"])
        np.save(os.path.join(intermediates_dir, "attention_matrices_np.npy"), oos['attention_matrices_np'])
        np.save(os.path.join(intermediates_dir, "attention_weighted_sentence_embs_np.npy"), oos['attention_weighted_sentence_embs_np'])
        np.save(os.path.join(intermediates_dir, "lhs_embs_np.npy"), oos['lhs_embs_np'])
        np.save(os.path.join(intermediates_dir, "layer_pooled_embs_np.npy"), oos['layer_pooled_embs_np'])
        np.save(os.path.join(intermediates_dir, "labels_np.npy"), oos['labels_np'])
        np.save(os.path.join(intermediates_dir, "fold_assignment.npy"), oos['fold_assignment'])
        np.save(os.path.join(intermediates_dir, "report_id_array.npy"), oos["report_id_array"])
        np.save(os.path.join(intermediates_dir, "patient_id_array.npy"), oos["patient_id_array"])
        np.save(os.path.join(intermediates_dir, "cv_original_indices.npy"), oos["cv_original_indices"])
        np.save(os.path.join(intermediates_dir, "valid_sentence_mask_np.npy"), oos["valid_sentence_mask_np"])
        np.save(os.path.join(intermediates_dir, "n_valid_sentences_np.npy"), oos["n_valid_sentences_np"])

        with open(os.path.join(intermediates_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        with open(os.path.join(intermediates_dir, "fold_split_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(results["fold_split_manifest"], f, ensure_ascii=False, indent=2)

        with open(os.path.join(intermediates_dir, "fold_checkpoint_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(results["fold_checkpoint_manifest"], f, ensure_ascii=False, indent=2)

        print(f"\nSaved intermediates: {intermediates_dir}")
        print(f"  - logits_np.npy: {oos['logits_np'].shape}")
        print(f"  - probs_np.npy: {oos['probs_np'].shape}")
        print(f"  - attention_matrices_np.npy: {oos['attention_matrices_np'].shape}")
        print(f"  - attention_weighted_sentence_embs_np.npy: {oos['attention_weighted_sentence_embs_np'].shape}")
        print(f"  - lhs_embs_np.npy: {oos['lhs_embs_np'].shape}")
        print(f"  - layer_pooled_embs_np.npy: {oos['layer_pooled_embs_np'].shape}")
        print(f"  - labels_np.npy: {oos['labels_np'].shape}")
        print(f"  - fold_assignment.npy: {oos['fold_assignment'].shape}")

    print("\nTraining complete!")

if __name__ == "__main__":
    main()
