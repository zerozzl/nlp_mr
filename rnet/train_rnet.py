import os
import logging
from argparse import ArgumentParser
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from rnet.model import RNet
from utils.dataloader import Tokenizer, MrDataset, load_pretrain_embedding
from utils.logger import Logger
from utils import model_utils, evaluator


def get_dataset(args, tokenizer, bigram_tokenizer):
    train_dataset = MrDataset('%s/%s/train.json' % (args.data_path, args.task), args.token_type, tokenizer,
                              use_bigram=args.use_bigram, bigram_tokenizer=bigram_tokenizer,
                              max_context_len=args.max_context_len, max_question_len=args.max_question_len,
                              do_pad=True, do_to_id=True, do_sort=True, do_train=True, debug=args.debug)
    test_dataset = MrDataset('%s/%s/dev.json' % (args.data_path, args.task), args.token_type, tokenizer,
                             use_bigram=args.use_bigram, bigram_tokenizer=bigram_tokenizer,
                             max_context_len=args.max_context_len, max_question_len=args.max_question_len,
                             do_pad=True, do_to_id=True, do_sort=True, do_train=False, debug=args.debug)

    return train_dataset, test_dataset


def data_collate_fn(data):
    data = np.array(data)

    question = torch.LongTensor(np.array(data[:, 0].tolist()))

    ans_ctx = data[:, 1].tolist()
    ans_ctx = [torch.LongTensor(np.array(s)) for s in ans_ctx]

    ans_tags = torch.LongTensor(np.array(data[:, 2].tolist()))

    ans_text = data[:, 4].tolist()

    question_bigram = torch.LongTensor(np.array(data[:, 5].tolist()))

    ans_ctx_bigram = data[:, 6].tolist()
    ans_ctx_bigram = [torch.LongTensor(np.array(s)) for s in ans_ctx_bigram]

    return question, ans_ctx, ans_tags, ans_text, question_bigram, ans_ctx_bigram


def train(args, dataset, dataloader, model, optimizer, lr_scheduler):
    model.train()
    loss_sum = 0
    for batch, data in enumerate(dataloader):
        optimizer.zero_grad()
        question, ans_ctx, ans_tags, _, question_bigram, ans_ctx_bigram = data

        question = question.cpu() if args.use_cpu else question.cuda()
        ans_ctx = [item.cpu() if args.use_cpu else item.cuda() for item in ans_ctx]
        ans_tags = ans_tags.cpu() if args.use_cpu else ans_tags.cuda()
        question_bigram = question_bigram.cpu() if args.use_cpu else question_bigram.cuda()
        ans_ctx_bigram = [item.cpu() if args.use_cpu else item.cuda() for item in ans_ctx_bigram]

        loss = model(question, ans_ctx, question_bigram=question_bigram, ans_ctx_bigram=ans_ctx_bigram,
                     decode=False, tags=ans_tags)
        loss = loss.mean()

        loss_sum += loss.item()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    loss_sum = loss_sum / len(dataset)
    return loss_sum


def evaluate(args, dataloader, model, tokenizer):
    pred_answers = []
    gold_answers = []

    model.eval()

    if args.multi_gpu:
        model = model.module

    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            question, ans_ctx, _, ans_text, question_bigram, ans_ctx_bigram = data

            question = question.cpu() if args.use_cpu else question.cuda()
            ans_ctx = [item.cpu() if args.use_cpu else item.cuda() for item in ans_ctx]
            question_bigram = question_bigram.cpu() if args.use_cpu else question_bigram.cuda()
            ans_ctx_bigram = [item.cpu() if args.use_cpu else item.cuda() for item in ans_ctx_bigram]

            starts, ends = model(question, ans_ctx, question_bigram=question_bigram, ans_ctx_bigram=ans_ctx_bigram)
            starts = starts.cpu().numpy()
            ends = ends.cpu().numpy()

            for i in range(len(ans_ctx)):
                ctx = ans_ctx[i].cpu().numpy()
                start = starts[i]
                end = ends[i]

                if end >= start:
                    ctx = tokenizer.convert_ids_to_tokens(ctx)
                    pred_answers.append(''.join(ctx[start:end + 1]))
                else:
                    pred_answers.append('')

                gold_answers.append(ans_text[i])

    f1_score, em_score, _, _ = evaluator.evaluate(gold_answers, pred_answers)
    return f1_score, em_score


def main(args):
    logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)

    if args.debug:
        args.batch_size = 3

    if args.multi_gpu:
        logging.info("run on multi GPU")
        torch.distributed.init_process_group(backend="nccl")

    model_utils.setup_seed(0)

    output_path = '%s/%s/%s_%s' % (args.output_path, args.task, args.token_type, args.embed_type)
    if args.use_bigram:
        output_path += '_bigram'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    model_logger = Logger(data_path=output_path)

    logging.info("loading embedding")
    token_to_id, pretrain_embed = load_pretrain_embedding(args.pretrained_emb_path,
                                                          has_meta=True if (args.token_type == 'word') else False,
                                                          add_pad=True, add_unk=True, debug=args.debug)
    tokenizer = Tokenizer(token_to_id)

    bigram_tokenizer = None
    bigram_to_id = {}
    if args.use_bigram:
        bigram_to_id, pretrain_bigram_embed = load_pretrain_embedding(args.pretrained_bigram_emb_path,
                                                                      add_pad=True, add_unk=True, debug=args.debug)
        bigram_tokenizer = Tokenizer(bigram_to_id)

    logging.info("loading dataset")
    train_dataset, test_dataset = get_dataset(args, tokenizer, bigram_tokenizer)

    if args.multi_gpu:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=data_collate_fn,
                                      sampler=DistributedSampler(train_dataset, shuffle=False))
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=data_collate_fn,
                                     sampler=DistributedSampler(test_dataset, shuffle=False))
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=data_collate_fn,
                                      shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=data_collate_fn,
                                     shuffle=False)

    best_f1 = 0
    epoch = 0

    if args.pretrained_model_path is not None:
        logging.info("loading pretrained model")
        model, optimizer, epoch, best_f1 = model_utils.load(args.pretrained_model_path)
        model = model.cpu() if args.use_cpu else model.cuda()
    else:
        logging.info("creating model")
        model = RNet(len(token_to_id), args.embed_size, args.hidden_size,
                     args.input_dropout_rate, args.hidden_dropout_rate,
                     embed_fix=True if args.embed_type == 'static' else False,
                     use_bigram=args.use_bigram, bigram_vocab_size=len(bigram_to_id),
                     bigram_embed_size=args.embed_size)

        if args.embed_type in ['pretrain', 'static']:
            model.init_embedding(np.array(pretrain_embed))
            if args.use_bigram:
                model.init_bigram_embedding(np.array(pretrain_bigram_embed))

        model = model.cpu() if args.use_cpu else model.cuda()

        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.multi_gpu:
        model = DistributedDataParallel(model, find_unused_parameters=True)

    num_train_steps = int(len(train_dataset) / args.batch_size * args.epoch_size)
    num_warmup_steps = int(num_train_steps * args.lr_warmup_proportion)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=num_warmup_steps, gamma=args.lr_decay_gamma)

    logging.info("begin training")
    while epoch < args.epoch_size:
        epoch += 1

        train_loss = train(args, train_dataset, train_dataloader, model, optimizer, lr_scheduler)
        test_f1, test_em = evaluate(args, test_dataloader, model, tokenizer)

        logging.info('epoch[%s/%s], train loss: %s' % (epoch, args.epoch_size, train_loss))
        logging.info('epoch[%s/%s], test f1: %s, em: %s' % (epoch, args.epoch_size, test_f1, test_em))
        model_utils.save(output_path, 'last.pth', model, optimizer, epoch, test_f1)

        remark = ''
        if test_f1 > best_f1:
            best_f1 = test_f1
            remark = 'best'
            model_utils.save(output_path, 'best.pth', model, optimizer, epoch, best_f1)

        model_logger.write(epoch, train_loss, test_f1, test_em, remark)

    logging.info("complete training")
    model_logger.draw_plot()


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--task', type=str, choices=['dureader', 'cmrc2018'],
                        default='dureader')
    parser.add_argument('--data_path', type=str,
                        default='../data/datasets/')
    parser.add_argument('--pretrained_emb_path', type=str,
                        help='[gigaword_chn.all.a2b.uni.ite50.vec],[news_tensite.pku.words.w2v50]',
                        default='../data/embeddings/gigaword_chn.all.a2b.uni.ite50.vec')
    parser.add_argument('--pretrained_bigram_emb_path', type=str,
                        default='../data/embeddings/gigaword_chn.all.a2b.bi.ite50.vec')
    parser.add_argument('--pretrained_model_path', type=str,
                        default=None)
    parser.add_argument('--output_path', type=str,
                        default='../runtime/rnet/')
    parser.add_argument('--token_type', type=str, choices=['char', 'word'],
                        default='char')
    parser.add_argument('--embed_type', type=str, choices=['rand', 'pretrain', 'static'],
                        default='rand')
    parser.add_argument('--use_bigram', type=bool,
                        default=False)
    parser.add_argument('--embed_size', type=int,
                        default=50)
    parser.add_argument('--max_question_len', type=int,
                        help='24 for dureader, 64 for cmrc2018',
                        default=24)
    parser.add_argument('--max_context_len', type=int,
                        default=1024)
    parser.add_argument('--hidden_size', type=int,
                        default=75)
    parser.add_argument('--input_dropout_rate', type=float,
                        default=0.5)
    parser.add_argument('--hidden_dropout_rate', type=float,
                        default=0.2)
    parser.add_argument('--batch_size', type=int,
                        default=32)
    parser.add_argument('--epoch_size', type=int,
                        default=30)
    parser.add_argument('--learning_rate', type=float,
                        default=0.1)
    parser.add_argument('--lr_warmup_proportion', type=float,
                        default=0.1)
    parser.add_argument('--lr_decay_gamma', type=float,
                        default=0.9)
    parser.add_argument('--use_cpu', type=bool,
                        default=False)
    parser.add_argument('--multi_gpu', type=bool,
                        help='run with: -m torch.distributed.launch',
                        default=False)
    parser.add_argument('--local_rank', type=int,
                        default=0)
    parser.add_argument('--debug', type=bool,
                        default=False)

    args = parser.parse_args()

    main(args)
