import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoModelForSequenceClassification, AutoConfig

# ============================================================================

# ============================================================================
#

#

#

# ============================================================================

class SelfAttentionAverage(nn.Module):

    def __init__(self, embed_dim=768, weight_dim=768, dropout=0.0, classifier_dropout=0.0):

        super(SelfAttentionAverage, self).__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads=1,
            kdim=weight_dim,
            vdim=weight_dim,
            batch_first=True,
            dropout=dropout
        )

        self.dropout_layer = nn.Dropout(p=classifier_dropout)

        self.linear = nn.Linear(weight_dim, 1)

    def forward(self, x, attn_mask=None, key_padding_mask=None, average_attn_weights=False):

        attn_output, attn_weights = self.attn(
            query=x,
            key=x,
            value=x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            average_attn_weights=average_attn_weights
        )

        if key_padding_mask is not None:

            mask_expanded = key_padding_mask.unsqueeze(-1)  # (batch, num_sentences, 1)
            attn_output = attn_output.masked_fill(mask_expanded, 0.0)

        if key_padding_mask is not None:

            valid_mask = ~key_padding_mask
            valid_count = valid_mask.sum(dim=1, keepdim=True).float()  # (batch, 1)
            valid_count = valid_count.clamp(min=1.0)

            pooled_output = attn_output.sum(dim=1) / valid_count.squeeze(-1).unsqueeze(-1)  # (batch, embed_dim)
        else:

            pooled_output = torch.mean(attn_output, dim=1)

        pooled_output_drop = self.dropout_layer(pooled_output)

        log_reg = self.linear(pooled_output_drop).squeeze(-1)

        return log_reg, attn_weights, attn_output

class SentenceAttentionBERTKorean(nn.Module):

    def __init__(
        self,
        base_model_name="klue/roberta-base",
        sentence_embed_dim=768,
        sentence_weight_dim=768,
        word_embed_dim=768,
        word_weight_dim=768,
        sentence_dropout=0.0,
        ff_dropout=0.1,
        att_dropout=0.1,
        report_max_length=64,
        class_dropout=0.0
    ):

        super(SentenceAttentionBERTKorean, self).__init__()

        config = AutoConfig.from_pretrained(base_model_name)

        self.hidden_size = config.hidden_size

        config.hidden_dropout_prob = ff_dropout
        config.attention_probs_dropout_prob = att_dropout

        config.num_labels = sentence_embed_dim

        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            config=config,
            ignore_mismatched_sizes=True
        )

        self.sentence_attention = SelfAttentionAverage(
            embed_dim=sentence_embed_dim,
            weight_dim=sentence_weight_dim,
            dropout=sentence_dropout,
            classifier_dropout=class_dropout
        )

        self.report_max_length = report_max_length

    def _create_sentence_padding_mask(self, attn_mask):

        valid_token_count = attn_mask.sum(dim=-1)  # (batch, num_sentences)

        sentence_padding_mask = valid_token_count <= 2

        return sentence_padding_mask

    def forward(self, input, attn_mask=None, sentence_attn_mask=None):

        if attn_mask is not None:
            sentence_padding_mask = self._create_sentence_padding_mask(attn_mask)
        else:
            sentence_padding_mask = None

        batch_cls = []
        batch_hidden_states = []
        batch_layer_pooled = []

        for report_ind in range(input.shape[0]):
            report_cls = []
            report_hidden_states = []

            for micro_batch_inds in range(int(input.shape[1] / self.report_max_length)):
                start_idx = micro_batch_inds * self.report_max_length
                end_idx = (micro_batch_inds + 1) * self.report_max_length

                single_report_input = input[report_ind, start_idx:end_idx, :].squeeze()
                single_report_mask = attn_mask[report_ind, start_idx:end_idx, :].squeeze() if attn_mask is not None else None

                base_outputs = self.base_model(
                    single_report_input,
                    attention_mask=single_report_mask,
                    output_hidden_states=True
                )

                base_cls = base_outputs.logits

                hidden_states = base_outputs.hidden_states

                report_cls.append(base_cls)
                report_hidden_states.append(hidden_states)

            report_cls = torch.cat(report_cls, dim=0)
            batch_cls.append(report_cls)

            num_layers = len(report_hidden_states[0])
            merged_hidden_full = []
            layer_pooled_list = []

            for layer_idx in range(num_layers):
                layer_tensors = [hs[layer_idx] for hs in report_hidden_states]
                merged_layer = torch.cat(layer_tensors, dim=0)

                merged_hidden_full.append(merged_layer)

                pooled = merged_layer.mean(dim=0).mean(dim=0)
                layer_pooled_list.append(pooled)

            batch_hidden_states.append(merged_hidden_full)
            batch_layer_pooled.append(torch.stack(layer_pooled_list, dim=0))

        batch_cls = torch.stack(batch_cls, dim=0)

        last_hidden_embs = [hs[-1] for hs in batch_hidden_states]

        layer_pooled_embs = batch_layer_pooled

        logits, sentence_attn_weights, attn_output = self.sentence_attention(
            batch_cls,
            attn_mask=sentence_attn_mask,
            key_padding_mask=sentence_padding_mask
        )

        return logits, sentence_attn_weights, attn_output, last_hidden_embs, layer_pooled_embs

SentenceAttentionBERT = SentenceAttentionBERTKorean
