import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import rnn


class MatchLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MatchLayer, self).__init__()
        self.wq = nn.Linear(input_size, hidden_size, bias=False)
        self.wk = nn.Linear(input_size, hidden_size, bias=False)
        self.wv = nn.Linear(input_size, hidden_size, bias=False)

    def forward(self, q, c):
        Q = self.wq(c)
        K = self.wk(q).permute(0, 2, 1)
        V = self.wv(q)

        attn = torch.matmul(Q, K)
        attn = F.softmax(attn, dim=2)
        out = torch.matmul(attn, V)
        return out


class PointerLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PointerLayer, self).__init__()
        self.w = nn.Linear(input_size, hidden_size)
        self.u = nn.Linear(input_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, H, h):
        attn = torch.tanh(self.w(H) + self.u(h).unsqueeze(1))
        attn = F.softmax(self.v(attn), dim=1)
        out = attn.mul(H).sum(1)
        return attn, out


class MatchLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, inp_dropout_rate, hid_dropout_rate,
                 embed_fix=False, use_bigram=False, bigram_vocab_size=0, bigram_embed_size=0):
        super(MatchLSTM, self).__init__()
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

        self.encoding = nn.LSTM(input_size=inp_size,
                                hidden_size=hidden_size,
                                num_layers=1,
                                dropout=hid_dropout_rate,
                                batch_first=True,
                                bidirectional=True)

        self.match_layer = MatchLayer(hidden_size * 2, hidden_size * 2)

        self.aggregate = nn.LSTM(input_size=hidden_size * 4,
                                 hidden_size=hidden_size,
                                 num_layers=1,
                                 dropout=hid_dropout_rate,
                                 batch_first=True,
                                 bidirectional=True)

        self.decoding = nn.LSTMCell(input_size=hidden_size * 2,
                                    hidden_size=hidden_size * 2)

        self.pointer = PointerLayer(hidden_size * 2, hidden_size * 2)
        self.in_dropout = nn.Dropout(inp_dropout_rate)
        self.ce_loss = nn.CrossEntropyLoss()

    def init_embedding(self, pretrained_embeddings):
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

    def init_bigram_embedding(self, pretrained_embeddings):
        self.bigram_embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

    def forward(self, question, ans_ctx, question_bigram=None, ans_ctx_bigram=None, decode=True, tags=None):
        ans_ctx_len = [len(ctx) for ctx in ans_ctx]
        ans_ctx = rnn.pad_sequence(ans_ctx, batch_first=True)
        ans_ctx_bigram = rnn.pad_sequence(ans_ctx_bigram, batch_first=True)

        q_embed = self.embedding(question)
        if self.use_bigram:
            q_embed_bi = torch.cat(
                [self.bigram_embedding(question_bigram[:, :, i]) for i in range(question_bigram.size()[2])], dim=2)
            q_embed = torch.cat((q_embed, q_embed_bi), dim=2)

        c_embed = self.embedding(ans_ctx)
        if self.use_bigram:
            c_embed_bi = torch.cat(
                [self.bigram_embedding(ans_ctx_bigram[:, :, i]) for i in range(ans_ctx_bigram.size()[2])], dim=2)
            c_embed = torch.cat((c_embed, c_embed_bi), dim=2)

        q_embed = self.in_dropout(q_embed)
        c_embed = self.in_dropout(c_embed)

        c_embed = rnn.pack_padded_sequence(c_embed, ans_ctx_len, batch_first=True)
        q_rep, _ = self.encoding(q_embed)
        c_rep, _ = self.encoding(c_embed)
        c_rep, _ = rnn.pad_packed_sequence(c_rep, batch_first=True)

        match_rep = self.match_layer(q_rep, c_rep)
        c_rep = torch.cat((c_rep, match_rep), dim=2)

        c_rep = rnn.pack_padded_sequence(c_rep, ans_ctx_len, batch_first=True)
        c_rep, _ = self.aggregate(c_rep)
        c_rep, _ = rnn.pad_packed_sequence(c_rep, batch_first=True)

        a_h = torch.zeros([c_rep.shape[0], c_rep.shape[2]])
        a_c = torch.zeros([c_rep.shape[0], c_rep.shape[2]])
        a_h = a_h.to(c_rep.device)
        a_c = a_c.to(c_rep.device)

        start, a_cep = self.pointer(c_rep, a_h)
        a_h, _ = self.decoding(a_cep, (a_h, a_c))
        end, _ = self.pointer(c_rep, a_h)

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
