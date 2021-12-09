import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import rnn


class HighwayLayer(nn.Module):
    def __init__(self, hidden_size, layer_num):
        super(HighwayLayer, self).__init__()
        self.layer_num = layer_num
        self.linears = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU()) for _ in range(layer_num)]
        )
        self.gates = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU()) for _ in range(layer_num)]
        )

    def forward(self, x):
        for i in range(self.layer_num):
            h = self.linears[i](x)
            t = self.gates[i](x)
            x = h * t + (1 - t) * x
        return x


class AttentionFlowLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionFlowLayer, self).__init__()
        self.w = nn.Linear(hidden_size * 3, 1)

    def forward(self, H, U):
        S = []
        for i in range(U.shape[1]):
            u = U[:, i, :].unsqueeze(1)
            u = u.expand(H.shape)
            hu = H * u
            hu = torch.cat((H, u, hu), dim=2)
            s = self.w(hu)
            S.append(s)
        S = torch.stack(S, dim=2)
        S = S.squeeze(3)

        c2q_attn = F.softmax(S, dim=2)
        c2q = torch.bmm(c2q_attn, U)

        q2c_attn = F.softmax(torch.max(S, dim=2)[0], dim=1).unsqueeze(1)
        q2c = torch.bmm(q2c_attn, H).squeeze()

        if H.shape[0] == 1:
            q2c = q2c.unsqueeze(0)
        q2c = q2c.unsqueeze(1).expand(-1, H.shape[1], -1)

        out = torch.cat([H, c2q, H * c2q, H * q2c], dim=2)
        return out


class BiDAF(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size, inp_dropout_rate, hid_dropout_rate,
                 embed_fix=False, use_bigram=False, bigram_vocab_size=0, bigram_embed_size=0):
        super(BiDAF, self).__init__()
        self.use_bigram = use_bigram

        inp_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)

        if use_bigram:
            inp_size += bigram_embed_size * 2
            self.bigram_embedding = nn.Embedding(bigram_vocab_size, bigram_embed_size)

        if embed_fix:
            for param in self.embedding.parameters():
                param.requires_grad = False

            if use_bigram:
                for param in self.bigram_embedding.parameters():
                    param.requires_grad = False

        self.embed_highway = HighwayLayer(inp_size, 2)

        self.contextual = nn.LSTM(input_size=inp_size,
                                  hidden_size=hidden_size,
                                  num_layers=1,
                                  dropout=hid_dropout_rate,
                                  batch_first=True,
                                  bidirectional=True)

        self.attention = AttentionFlowLayer(hidden_size * 2)

        self.modeling = nn.LSTM(input_size=hidden_size * 8,
                                hidden_size=hidden_size,
                                num_layers=2,
                                dropout=hid_dropout_rate,
                                batch_first=True,
                                bidirectional=True)

        self.decoding = nn.LSTM(input_size=hidden_size * 2,
                                hidden_size=hidden_size,
                                num_layers=1,
                                dropout=hid_dropout_rate,
                                batch_first=True,
                                bidirectional=True)

        self.p1 = nn.Linear(hidden_size * 10, 1)
        self.p2 = nn.Linear(hidden_size * 10, 1)

        self.in_dropout = nn.Dropout(inp_dropout_rate)
        self.hid_dropout = nn.Dropout(hid_dropout_rate)
        self.ce_loss = nn.CrossEntropyLoss()

    def init_embedding(self, pretrained_embeddings):
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

    def init_bigram_embedding(self, pretrained_embeddings):
        self.bigram_embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

    def forward(self, question, context, question_bigram=None, ans_ctx_bigram=None, decode=True, tags=None):
        context_len = [len(ctx) for ctx in context]
        context = rnn.pad_sequence(context, batch_first=True)
        ans_ctx_bigram = rnn.pad_sequence(ans_ctx_bigram, batch_first=True)

        q_embed = self.embedding(question)
        if self.use_bigram:
            q_embed_bi = torch.cat(
                [self.bigram_embedding(question_bigram[:, :, i]) for i in range(question_bigram.size()[2])], dim=2)
            q_embed = torch.cat((q_embed, q_embed_bi), dim=2)

        c_embed = self.embedding(context)
        if self.use_bigram:
            c_embed_bi = torch.cat(
                [self.bigram_embedding(ans_ctx_bigram[:, :, i]) for i in range(ans_ctx_bigram.size()[2])], dim=2)
            c_embed = torch.cat((c_embed, c_embed_bi), dim=2)

        q_embed = self.embed_highway(q_embed)
        c_embed = self.embed_highway(c_embed)

        q_embed = self.in_dropout(q_embed)
        c_embed = self.in_dropout(c_embed)

        c_embed = rnn.pack_padded_sequence(c_embed, context_len, batch_first=True)
        q_rep, _ = self.contextual(q_embed)
        c_rep, _ = self.contextual(c_embed)
        c_rep, _ = rnn.pad_packed_sequence(c_rep, batch_first=True)

        G = self.attention(c_rep, q_rep)

        M, _ = self.modeling(G)
        start = self.hid_dropout(torch.cat([G, M], dim=2))
        start = self.p1(start)

        M, _ = self.decoding(M)
        end = self.hid_dropout(torch.cat([G, M], dim=2))
        end = self.p2(end)

        start = start.squeeze(2)
        end = end.squeeze(2)

        if decode:
            start = torch.argmax(start, dim=1)
            end = torch.argmax(end, dim=1)
            return start, end
        else:
            tags_start = tags[:, 0]
            tags_end = tags[:, 1]

            loss_start = self.ce_loss(start, tags_start)
            loss_end = self.ce_loss(end, tags_end)

            loss = loss_start + loss_end
            return loss
