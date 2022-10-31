from typing import Callable, Dict, List, Optional

import torch
from torch import nn
from torch.nn import functional as F


class MultiGrainSemanticUnit(nn.Module):
    def __init__(self, hiddne_dim=20):
        super().__init__()
        self.dp_attn = DotProductAttention()
        self.trm = nn.TransformerEncoderLayer(
            d_model=hiddne_dim, nhead=4, dim_feedforward=hiddne_dim
        )

    def forward(self, query_sources, q_his):
        """
        query_sources is expected to be [q_gram_1, q_gram_2, q_seq]
        """

        pooled_sources = []
        for query_source in query_sources:
            pooled_sources.append(torch.mean(query_source, dim=1).unsqueeze(1))

        q_seq_his = self.dp_attn(pooled_sources[-1], q_his, q_his.shape[0])
        q_seq_seq = torch.mean(self.trm(q_seq), 1, keepdim=True)
        q_mix = (
            q_seq_his
            + q_seq_seq
            + torch.sum(torch.cat(pooled_sources, 1), 1, keepdim=True)
        )
        q_msg = torch.cat([*pooled_sources, q_seq_seq, q_seq_his, q_mix], 2)

        return q_msg


class UserTower(nn.Module):
    def __init__(self, hidden_dim=20):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.msgu = MultiGrainSemanticUnit()

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.mh_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)

        self.u_dp_attn = UserDotProductAttention()

        self.cls_token = nn.Embedding(1, 20)

    def forward(self, query_feautures, user_features):
        """
        query_feautures is expected to be  [[q_gram_1, q_gram_2, q_seq],q_his]
        user_features = is expected to be [real, short, long]-time features
        """

        assert len(query_feautures) == 2 and len(query_feautures[0]) == 3

        q_msg = self.msgu(query_feautures[0], query_feautures[1])

        real, short, long = user_features

        real_part = self.real_time_part(q_msg, real)
        short_part = self.short_time_part(q_msg, short)
        long_part = self.long_time_part(q_msg, long)

        cls_token = self.cls_token(torch.tensor([[0]])).expand(
            long_part.shape[0], 1, -1
        )

        sequence = torch.cat(
            [
                cls_token,
                q_msg.view(-1, 6, self.hidden_dim),
                real_part,
                short_part,
                long_part,
            ],
            1,
        )

        x, _ = self.mh_attn(sequence, sequence, sequence)

        return x[:, 0]

    def real_time_part(self, q_msg, user_real_features):
        x, (hn, cn) = self.lstm(user_real_time)

        x, _ = self.mh_attn(x, x, x)
        x = torch.cat([torch.zeros(x.shape[0], 1, self.hidden_dim), x], dim=1)
        return torch.cat(self.u_dp_attn(q_msg, x), 1)

    def short_time_part(self, q_msg, user_short_features):
        x, _ = self.mh_attn(
            user_short_features, user_short_features, user_short_features
        )
        x = torch.cat([torch.zeros(x.shape[0], 1, self.hidden_dim), x], dim=1)
        return torch.cat(self.u_dp_attn(q_msg, x), 1)

    def long_time_part(self, q_msg, user_long_features):
        """
        user_long_features is expected to be [4x[clicked, bought, collected]]
        """
        h_attributes = []
        for attribute_index in range(user_long_features.shape[1]):
            x = torch.mean(user_long_features[:, attribute_index], dim=2)
            x = torch.cat([torch.zeros(x.shape[0], 1, self.hidden_dim), x], dim=1)
            h_attributes.append(torch.cat(self.u_dp_attn(q_msg, x), 1))
        return torch.cat(h_attributes, 1)
