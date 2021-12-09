import torch
import torch.nn as nn
from torch.nn.utils import rnn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionLayer, self).__init__()
        self.wq = nn.Linear(input_size, hidden_size)
        self.wp = nn.Linear(hidden_size, hidden_size)
        self.wv = nn.Linear(hidden_size, hidden_size)
        self.ws = nn.Linear(hidden_size, 1)

    def forward(self, H, p, v=None):
        attn = self.wq(H) + self.wp(p).unsqueeze(1)
        if v is not None:
            attn = attn + self.wv(v).unsqueeze(1)
        attn = self.ws(torch.tanh(attn))
        attn = F.softmax(attn.squeeze(2), dim=1)
        out = torch.bmm(attn.unsqueeze(1), H)
        out = out.squeeze(1)
        return out, attn


class GatedRecurrentLayer(nn.Module):
    def __init__(self, hidden_size):
        super(GatedRecurrentLayer, self).__init__()

        self.rnn = nn.GRUCell(input_size=hidden_size,
                              hidden_size=hidden_size)

        self.attention = AttentionLayer(hidden_size, hidden_size)
        self.wg = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, P, Q):
        P_len = P.shape[1]
        vp = torch.zeros([P.shape[0], P.shape[2]])
        vp = vp.to(P.device)

        out = []
        for i in range(P_len):
            p = P[:, i, :].squeeze(1)
            c, _ = self.attention(H=Q, p=p, v=vp)
            g = F.sigmoid(self.wg(torch.cat([p, c], dim=1)))
            c = torch.mul(g, c)
            vp = self.rnn(c, vp)
            out.append(vp)
        out = torch.stack(out, dim=1)
        return out


class SelfMatchingLayer(nn.Module):
    def __init__(self, hidden_size):
        super(SelfMatchingLayer, self).__init__()

        self.rnn_l = nn.GRUCell(input_size=hidden_size,
                                hidden_size=hidden_size)
        self.rnn_r = nn.GRUCell(input_size=hidden_size,
                                hidden_size=hidden_size)

        self.attention = AttentionLayer(hidden_size, hidden_size)
        self.wg = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, P):
        P_len = P.shape[1]
        hp_l = torch.zeros([P.shape[0], P.shape[2]])
        hp_r = torch.zeros([P.shape[0], P.shape[2]])
        hp_l = hp_l.to(P.device)
        hp_r = hp_r.to(P.device)

        out_l = []
        out_r = []
        c_list = []
        for i in range(P_len):
            p = P[:, i, :].squeeze(1)
            c, _ = self.attention(H=P, p=p)
            g = F.sigmoid(self.wg(torch.cat([p, c], dim=1)))
            c = torch.mul(g, c)
            c_list.append(c)

            hp_l = self.rnn_l(c, hp_l)
            out_l.append(hp_l)

        for i in range(P_len):
            c = c_list[P_len - i - 1]
            hp_r = self.rnn_r(c, hp_r)
            out_r.insert(0, hp_r)

        out_l = torch.stack(out_l, dim=1)
        out_r = torch.stack(out_r, dim=1)
        out = torch.cat([out_l, out_r], dim=2)
        return out


class PointerLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PointerLayer, self).__init__()

        self.attention = AttentionLayer(input_size, hidden_size)

        self.rnn = nn.GRUCell(input_size=input_size,
                              hidden_size=hidden_size)

    def forward(self, P, Q):
        ha = torch.mean(Q, dim=1)
        c, start = self.attention(P, ha)
        ha = self.rnn(c, ha)
        _, end = self.attention(P, ha)
        return start, end


class RNet(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, inp_dropout_rate, hid_dropout_rate,
                 embed_fix=False, use_bigram=False, bigram_vocab_size=0, bigram_embed_size=0):
        super(RNet, self).__init__()
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

        self.encoder = nn.GRU(input_size=inp_size,
                              hidden_size=hidden_size,
                              num_layers=3,
                              dropout=hid_dropout_rate,
                              batch_first=True,
                              bidirectional=True)

        self.gate_attention = GatedRecurrentLayer(hidden_size * 2)
        self.self_matching = SelfMatchingLayer(hidden_size * 2)
        self.pointer = PointerLayer(hidden_size * 4, hidden_size * 2)

        self.in_dropout = nn.Dropout(inp_dropout_rate)
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

        q_embed = self.in_dropout(q_embed)
        c_embed = self.in_dropout(c_embed)

        c_embed = rnn.pack_padded_sequence(c_embed, context_len, batch_first=True)
        q_rep, _ = self.encoder(q_embed)
        c_rep, _ = self.encoder(c_embed)
        c_rep, _ = rnn.pad_packed_sequence(c_rep, batch_first=True)

        c_rep = self.gate_attention(c_rep, q_rep)
        c_rep = self.self_matching(c_rep)

        start, end = self.pointer(c_rep, q_rep)

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
