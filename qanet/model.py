import math
import torch
import torch.nn as nn
from torch.nn.utils import rnn
import torch.nn.functional as F


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


class PosEncoder(nn.Module):
    def __init__(self, min_timescale=1.0, max_timescale=1.0e4):
        super(PosEncoder, self).__init__()
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale

    def forward(self, x):
        _, length, channels = x.size()
        signal = self.get_pos_encoding(length, channels, self.min_timescale, self.max_timescale)
        signal = signal.to(x.device)
        out = x + signal
        return out

    def get_pos_encoding(self, length, channels, min_timescale=1.0, max_timescale=1.0e4):
        position = torch.arange(length).type(torch.float32)
        num_timescales = channels // 2
        log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales).type(torch.float32) * -log_timescale_increment)
        scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
        m = nn.ZeroPad2d((0, (channels % 2), 0, 0))
        signal = m(signal)
        signal = signal.view(1, length, channels)
        return signal


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, bias=True):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k_size, padding=k_size // 2,
                                        bias=bias, groups=in_ch)
        self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0,
                                        bias=bias)

    def forward(self, x):
        return F.relu(self.pointwise_conv(self.depthwise_conv(x)))


class EncoderBlock(nn.Module):
    def __init__(self, conv_kernel_num, conv_kernel_size, conv_layer_num, attn_head_num, hid_dropout_rate):
        super(EncoderBlock, self).__init__()
        self.conv_layer_num = conv_layer_num

        self.pos_encode = PosEncoder()

        self.norm_re = nn.ModuleList(
            [nn.LayerNorm(conv_kernel_num) for _ in range(conv_layer_num)]
        )
        self.conv_re = nn.ModuleList(
            [DepthwiseSeparableConv(conv_kernel_num, conv_kernel_num, conv_kernel_size) for _ in range(conv_layer_num)]
        )

        self.norm_attn = nn.LayerNorm(conv_kernel_num)
        self.self_attn = nn.MultiheadAttention(conv_kernel_num, attn_head_num, dropout=hid_dropout_rate)

        self.norm_ff = nn.LayerNorm(conv_kernel_num)
        self.ff = nn.Linear(conv_kernel_num, conv_kernel_num)

    def forward(self, x, masks):
        out = self.pos_encode(x)

        for i in range(self.conv_layer_num):
            out_conv = self.norm_re[i](out)
            out_conv = out_conv.permute(0, 2, 1)
            out_conv = self.conv_re[i](out_conv)
            out_conv = out_conv.permute(0, 2, 1)
            out = out + out_conv

        out_attn = self.norm_attn(out)
        out_attn = out_attn.permute(1, 0, 2)
        out_attn, _ = self.self_attn(out_attn, out_attn, out_attn, key_padding_mask=masks)
        out_attn = out_attn.permute(1, 0, 2)
        out = out + out_attn

        out_ff = self.norm_ff(out)
        out_ff = self.ff(out_ff)
        out = out + out_ff
        return out


class ContextQueryAttentionLayer(nn.Module):
    def __init__(self, conv_kernel_num, hid_dropout_rate):
        super(ContextQueryAttentionLayer, self).__init__()
        w = torch.empty(conv_kernel_num * 3)
        lim = 1 / conv_kernel_num
        nn.init.uniform_(w, -math.sqrt(lim), math.sqrt(lim))
        self.w = nn.Parameter(w)
        self.dropout = nn.Dropout(hid_dropout_rate)

    def forward(self, c_rep, q_rep, c_mask, q_mask):
        c_mask = c_mask.unsqueeze(2).int()
        q_mask = q_mask.unsqueeze(1).int()
        shape = (c_rep.size(0), c_rep.size(1), q_rep.size(1), c_rep.size(2))

        ct = c_rep.unsqueeze(2).expand(shape)
        qt = q_rep.unsqueeze(1).expand(shape)
        cq = torch.mul(ct, qt)

        S = torch.cat([ct, qt, cq], dim=3)
        S = torch.matmul(S, self.w)

        S1 = F.softmax(self.mask_logits(S, q_mask), dim=2)
        S2 = F.softmax(self.mask_logits(S, c_mask), dim=1)
        A = torch.bmm(S1, q_rep)
        B = torch.bmm(torch.bmm(S1, S2.transpose(1, 2)), c_rep)

        out = torch.cat([c_rep, A, torch.mul(c_rep, A), torch.mul(c_rep, B)], dim=2)
        out = self.dropout(out)

        return out

    def mask_logits(self, target, mask):
        return target * (1 - mask) + mask * (-1e30)


class ModelEncoderLayer(nn.Module):
    def __init__(self, conv_kernel_num, conv_kernel_size, conv_layer_num, attn_head_num, hid_dropout_rate):
        super(ModelEncoderLayer, self).__init__()
        self.blocks = nn.ModuleList(
            [EncoderBlock(conv_kernel_num, conv_kernel_size, conv_layer_num, attn_head_num, hid_dropout_rate)
             for _ in range(7)]
        )

    # def forward(self, c_rep, c_mask):
    #     out = c_rep
    #     for block in self.blocks:
    #         out = block(out, c_mask)
    #
    #     M_list = []
    #     for block in self.blocks:
    #         out = block(out, c_mask)
    #         M_list.append(out)
    #
    #     return M_list[0], M_list[1], M_list[2]

    # def forward(self, c_rep, c_mask):
    #     out = c_rep
    #     M_list = []
    #     for i in range(len(self.blocks)):
    #         out = self.blocks[i](out, c_mask)
    #         if i >= (len(self.blocks) - 3):
    #             M_list.append(out)
    #
    #     return M_list[0], M_list[1], M_list[2]

    def forward(self, c_rep, c_mask):
        M0 = c_rep
        for block in self.blocks:
            M0 = block(M0, c_mask)

        M1 = M0
        for block in self.blocks:
            M1 = block(M1, c_mask)

        M2 = M1
        for block in self.blocks:
            M2 = block(M2, c_mask)

        return M0, M1, M2


class PointerLayer(nn.Module):
    def __init__(self, hidden_size):
        super(PointerLayer, self).__init__()
        self.ws = nn.Linear(hidden_size, 1)
        self.we = nn.Linear(hidden_size, 1)

    def forward(self, M0, M1, M2):
        start = torch.cat([M0, M1], dim=2)
        end = torch.cat([M0, M2], dim=2)

        start = self.ws(start)
        end = self.we(end)

        start = start.squeeze(2)
        end = end.squeeze(2)

        return start, end


class QANet(nn.Module):
    def __init__(self, vocab_size, embed_size, conv_kernel_num, conv_kernel_size, conv_layer_num, attn_head_num,
                 inp_dropout_rate, hid_dropout_rate,
                 embed_fix=False, use_bigram=False, bigram_vocab_size=0, bigram_embed_size=0):
        super(QANet, self).__init__()
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
        self.embed_conv = DepthwiseSeparableConv(inp_size, conv_kernel_num, conv_kernel_size)

        self.q_encoder = EncoderBlock(conv_kernel_num, conv_kernel_size, conv_layer_num, attn_head_num,
                                      hid_dropout_rate)
        self.c_encoder = EncoderBlock(conv_kernel_num, conv_kernel_size, conv_layer_num, attn_head_num,
                                      hid_dropout_rate)

        self.cq_attn = ContextQueryAttentionLayer(conv_kernel_num, hid_dropout_rate)
        self.attn_conv = DepthwiseSeparableConv(conv_kernel_num * 4, conv_kernel_num, conv_kernel_size)

        self.model_encoder = ModelEncoderLayer(conv_kernel_num, conv_kernel_size - 2, conv_layer_num - 2, attn_head_num,
                                               hid_dropout_rate)

        self.pointer = PointerLayer(conv_kernel_num * 2)

        self.in_dropout = nn.Dropout(inp_dropout_rate)
        self.ce_loss = nn.CrossEntropyLoss()

    def init_embedding(self, pretrained_embeddings):
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

    def init_bigram_embedding(self, pretrained_embeddings):
        self.bigram_embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

    def forward(self, question, context, question_bigram=None, ans_ctx_bigram=None,
                decode=True, tags=None):
        context = rnn.pad_sequence(context, batch_first=True)
        ans_ctx_bigram = rnn.pad_sequence(ans_ctx_bigram, batch_first=True)
        q_mask = question == 0
        c_mask = context == 0

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

        q_embed = self.embed_highway(q_embed)
        c_embed = self.embed_highway(c_embed)

        q_embed = q_embed.permute(0, 2, 1)
        q_embed = self.embed_conv(q_embed)
        q_embed = q_embed.permute(0, 2, 1)

        c_embed = c_embed.permute(0, 2, 1)
        c_embed = self.embed_conv(c_embed)
        c_embed = c_embed.permute(0, 2, 1)

        q_rep = self.q_encoder(q_embed, q_mask)
        c_rep = self.c_encoder(c_embed, c_mask)

        c_rep = self.cq_attn(c_rep, q_rep, c_mask, q_mask)
        c_rep = c_rep.permute(0, 2, 1)
        c_rep = self.attn_conv(c_rep)
        c_rep = c_rep.permute(0, 2, 1)

        M0, M1, M2 = self.model_encoder(c_rep, c_mask)

        start, end = self.pointer(M0, M1, M2)

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
