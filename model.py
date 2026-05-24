import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


class AttentionPooling(nn.Module):
    def __init__(self, embedding_dim):
        super(AttentionPooling, self).__init__()
        self.attention_weights = nn.Linear(embedding_dim, 1)

    def forward(self, x, mask=None):
        attn_scores = self.attention_weights(x).squeeze(-1)

        if mask is not None:
            # Emmascarem els valors de padding perquè no afectin l'atenció
            attn_scores = attn_scores.masked_fill(mask, -1e9)

        # Convertim els scores en probabilitats d'atenció
        attn_probs = F.softmax(attn_scores, dim=1)

        # Fem la suma ponderada
        weighted_sum = torch.sum(x * attn_probs.unsqueeze(-1), dim=1)

        return weighted_sum


class LogTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, num_heads=4, num_layers=3, dropout=0.2):
        super(LogTransformer, self).__init__()

        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 2,
            dropout=dropout,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.attention_pooling = AttentionPooling(embedding_dim)

        # Classificador dual: Attention Pooling + Mean Pooling
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim * 2),
            nn.Linear(embedding_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        src_key_padding_mask = (x == 0)

        x_emb = self.embedding(x) * math.sqrt(self.embedding_dim)
        x_pos = self.pos_encoder(x_emb)

        transformer_out = self.transformer_encoder(
            x_pos,
            src_key_padding_mask=src_key_padding_mask
        )

        # 1. Extracció de característiques per Attention Pooling
        attn_out = self.attention_pooling(
            transformer_out,
            mask=src_key_padding_mask
        )

        # 2. Extracció de característiques globals per Mean Pooling
        transformer_out_masked = transformer_out.masked_fill(
            src_key_padding_mask.unsqueeze(-1),
            0.0
        )

        sum_out = torch.sum(transformer_out_masked, dim=1)

        lengths = (~src_key_padding_mask).sum(dim=1, keepdim=True).float()
        lengths = torch.clamp(lengths, min=1.0)

        mean_out = sum_out / lengths

        # 3. Concatenació: Attention Pooling + Mean Pooling
        combined_out = torch.cat([attn_out, mean_out], dim=1)

        output = self.classifier(combined_out)

        return output