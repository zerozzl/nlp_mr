import io
import json
import codecs
import logging
import jieba
from torch.utils.data import Dataset

TOKEN_PAD = '[PAD]'
TOKEN_UNK = '[UNK]'
TOKEN_CLS = '[CLS]'
TOKEN_SEP = '[SEP]'
TOKEN_EDGES_START = '<s>'
TOKEN_EDGES_END = '</s>'


class MrDataset(Dataset):
    def __init__(self, data_path, token_type, tokenizer, use_bigram=False, bigram_tokenizer=None,
                 max_context_len=0, max_question_len=0, do_pad=False, pad_token=TOKEN_PAD,
                 do_to_id=False, do_sort=False, do_train=False, for_bert=False, debug=False):
        super(MrDataset, self).__init__()
        self.data = []

        documents = json.load(io.open(data_path))
        documents = documents['data']
        for document in documents:
            paragraphs = document['paragraphs']
            for paragraph in paragraphs:
                context = self.dbc_to_sbc(paragraph['context'])

                qas = paragraph['qas']
                for qa in qas:
                    question = self.dbc_to_sbc(qa['question'])
                    if token_type == 'char':
                        question = [ch for ch in question]
                    elif token_type == 'word':
                        question = list(jieba.cut(question))

                    if max_question_len > 0:
                        question = question[:max_question_len]

                    segment = []
                    if for_bert:
                        question = [TOKEN_CLS] + question + [TOKEN_SEP]
                        segment = [1] * len(question)

                    question_bigram = []
                    if use_bigram:
                        question_bigram = [TOKEN_EDGES_START] + question + [TOKEN_EDGES_END]
                        question_bigram = [[question_bigram[i - 1] + question_bigram[i]] + [
                            question_bigram[i] + question_bigram[i + 1]] for i in
                                           range(1, len(question_bigram) - 1)]

                    if do_pad and (max_question_len > 0):
                        if use_bigram:
                            question_bigram = question_bigram + [[pad_token, pad_token]] * (
                                    max_question_len - len(question))
                        question = question + [pad_token] * (max_question_len - len(question))

                    if do_to_id:
                        question = tokenizer.convert_tokens_to_ids(question)
                        if use_bigram:
                            question_bigram = bigram_tokenizer.convert_tokens_to_ids(question_bigram)

                    answers = qa['answers']
                    if do_train:
                        ans_text = self.dbc_to_sbc(answers[0]['text'])
                        ans_start = int(answers[0]['answer_start'])

                        ans_ctx = [context[:ans_start], context[ans_start:ans_start + len(ans_text)],
                                   context[ans_start + len(ans_text):]]

                        if token_type == 'char':
                            ans_ctx = [[ch for ch in piece] for piece in ans_ctx]
                            ans_text = [ch for ch in ans_text]
                        elif token_type == 'word':
                            ans_ctx = [list(jieba.cut(piece)) for piece in ans_ctx]
                            ans_text = list(jieba.cut(ans_text))

                        ans_tags = [len(ans_ctx[0]), len(ans_ctx[0]) + (len(ans_ctx[1]) - 1)]
                        ans_ctx = ans_ctx[0] + ans_ctx[1] + ans_ctx[2]

                        if max_context_len > 0:
                            if for_bert:
                                if ((ans_tags[0] + len(question)) >= max_context_len) or \
                                        ((ans_tags[1] + len(question)) >= max_context_len):
                                    break
                                ans_ctx = ans_ctx[:(max_context_len - len(question))]
                                segment = segment + [0] * len(ans_ctx)
                            else:
                                if (ans_tags[0] > max_context_len) or (ans_tags[1] > max_context_len):
                                    break
                                ans_ctx = ans_ctx[:max_context_len]

                        ans_ctx_len = len(ans_ctx)

                        ans_ctx_bigram = []
                        if use_bigram:
                            ans_ctx_bigram = [TOKEN_EDGES_START] + ans_ctx + [TOKEN_EDGES_END]
                            ans_ctx_bigram = [[ans_ctx_bigram[i - 1] + ans_ctx_bigram[i]] + [
                                ans_ctx_bigram[i] + ans_ctx_bigram[i + 1]] for i in range(1, len(ans_ctx_bigram) - 1)]

                        # if do_pad and (max_context_len > 0):
                        #     if use_bigram:
                        #         ans_ctx_bigram = ans_ctx_bigram + [[pad_token, pad_token]] * (
                        #                 max_context_len - len(ans_ctx))
                        #
                        #     ans_ctx = ans_ctx + [pad_token] * (max_context_len - len(ans_ctx))

                        if do_to_id:
                            ans_ctx = tokenizer.convert_tokens_to_ids(ans_ctx)
                            if use_bigram:
                                ans_ctx_bigram = bigram_tokenizer.convert_tokens_to_ids(ans_ctx_bigram)

                        self.data.append([question, ans_ctx, ans_tags, ans_ctx_len, [''.join(ans_text)],
                                          question_bigram, ans_ctx_bigram, segment])
                    else:
                        ans_text = [self.dbc_to_sbc(ans['text']) for ans in answers]

                        if token_type == 'char':
                            ans_ctx = [ch for ch in context]
                        elif token_type == 'word':
                            ans_ctx = list(jieba.cut(context))

                        if max_context_len > 0:
                            if for_bert:
                                ans_ctx = ans_ctx[:(max_context_len - len(question))]
                                segment = segment + [0] * len(ans_ctx)
                            else:
                                ans_ctx = ans_ctx[:max_context_len]

                        ans_ctx_len = len(ans_ctx)

                        ans_ctx_bigram = []
                        if use_bigram:
                            ans_ctx_bigram = [TOKEN_EDGES_START] + ans_ctx + [TOKEN_EDGES_END]
                            ans_ctx_bigram = [[ans_ctx_bigram[i - 1] + ans_ctx_bigram[i]] + [
                                ans_ctx_bigram[i] + ans_ctx_bigram[i + 1]] for i in range(1, len(ans_ctx_bigram) - 1)]

                        # if do_pad and (max_context_len > 0):
                        #     if use_bigram:
                        #         ans_ctx_bigram = ans_ctx_bigram + [[pad_token, pad_token]] * (
                        #                 max_context_len - len(ans_ctx))
                        #
                        #     ans_ctx = ans_ctx + [pad_token] * (max_context_len - len(ans_ctx))

                        if do_to_id:
                            ans_ctx = tokenizer.convert_tokens_to_ids(ans_ctx)
                            if use_bigram:
                                ans_ctx_bigram = bigram_tokenizer.convert_tokens_to_ids(ans_ctx_bigram)

                        self.data.append([question, ans_ctx, [], ans_ctx_len, ans_text,
                                          question_bigram, ans_ctx_bigram, segment])

                    if debug:
                        if len(self.data) >= 10:
                            break
                if debug:
                    if len(self.data) >= 10:
                        break
            if debug:
                if len(self.data) >= 10:
                    break

        if do_sort:
            self.data = sorted(self.data, key=lambda x: x[3], reverse=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def dbc_to_sbc(self, ustring):
        rstring = ''
        for uchar in ustring:
            inside_code = ord(uchar)
            if inside_code == 0x3000:
                inside_code = 0x0020
            else:
                inside_code -= 0xfee0
            if not (0x0021 <= inside_code and inside_code <= 0x7e):
                rstring += uchar
                continue
            rstring += chr(inside_code)
        return rstring

    @staticmethod
    def statistics(datapath):
        logging.info('========== statistics %s ==========' % datapath)
        context_len_dict = {128: 0, 256: 0, 384: 0, 400: 0, 512: 0, 768: 0, 1024: 0, 2048: 0, 4096: 0}
        question_len_dict = {16: 0, 24: 0, 32: 0, 64: 0, 128: 0, 256: 0}

        data_granularity = ['char', 'word']
        data_split = ['train', 'dev']
        for dg in data_granularity:
            for ds in data_split:
                logging.info('%s level %s data count' % (dg, ds))
                data = json.load(io.open('%s/%s.json' % (datapath, ds)))
                documents = data['data']
                for document in documents:
                    paragraphs = document['paragraphs']
                    for paragraph in paragraphs:
                        context = paragraph['context']

                        if dg == 'char':
                            context_len = len(context)
                        elif dg == 'word':
                            context_len = len(list(jieba.cut(context)))
                        for cl in context_len_dict:
                            if context_len <= cl:
                                context_len_dict[cl] = context_len_dict[cl] + 1
                                break

                        qas = paragraph['qas']
                        for qa in qas:
                            question = qa['question']

                            if dg == 'char':
                                que_len = len(question)
                            elif dg == 'word':
                                que_len = len(list(jieba.cut(question)))
                            for ql in question_len_dict:
                                if que_len <= ql:
                                    question_len_dict[ql] = question_len_dict[ql] + 1
                                    break

                logging.info('context length: %s' % str(context_len_dict))
                logging.info('question length: %s' % str(question_len_dict))

                for cl in context_len_dict:
                    context_len_dict[cl] = 0
                for ql in question_len_dict:
                    question_len_dict[ql] = 0


class Tokenizer:
    def __init__(self, token_to_id):
        self.token_to_id = token_to_id
        self.id_to_token = {v: k for k, v in token_to_id.items()}

    def convert_tokens_to_ids(self, tokens, unk_token=TOKEN_UNK):
        ids = []
        for token in tokens:
            if isinstance(token, str):
                ids.append(self.token_to_id.get(token, self.token_to_id[unk_token]))
            else:
                ids.append([self.token_to_id.get(t, self.token_to_id[unk_token]) for t in token])
        return ids

    def convert_ids_to_tokens(self, ids, max_sent_len=0):
        tokens = [self.id_to_token[i] for i in ids]
        if max_sent_len > 0:
            tokens = tokens[:max_sent_len]
        return tokens


def load_pretrain_embedding(filepath, has_meta=False,
                            add_pad=False, pad_token=TOKEN_PAD, add_unk=False, unk_token=TOKEN_UNK, debug=False):
    with codecs.open(filepath, 'r', 'utf-8', errors='ignore') as fin:
        token_to_id = {}
        embed = []

        if has_meta:
            meta_info = fin.readline().strip().split()

        first_line = fin.readline().strip().split()
        embed_size = len(first_line) - 1

        if add_pad:
            token_to_id[pad_token] = len(token_to_id)
            embed.append([0.] * embed_size)

        if add_unk:
            token_to_id[unk_token] = len(token_to_id)
            embed.append([0.] * embed_size)

        token_to_id[first_line[0]] = len(token_to_id)
        embed.append([float(x) for x in first_line[1:]])

        for line in fin:
            line = line.split()

            if len(line) != embed_size + 1:
                continue
            if line[0] in token_to_id:
                continue

            token_to_id[line[0]] = len(token_to_id)
            embed.append([float(x) for x in line[1:]])

            if debug:
                if len(embed) >= 1000:
                    break

    return token_to_id, embed


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)

    MrDataset.statistics('../data/datasets/dureader')
    MrDataset.statistics('../data/datasets/cmrc2018')
