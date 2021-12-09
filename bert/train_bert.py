import os
import logging
from argparse import ArgumentParser
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from transformers import BertTokenizer

from bert.model import Bert
from utils.dataloader import MrDataset
from utils.logger import Logger
from utils import model_utils, evaluator


def get_dataset(args, tokenizer):
    train_dataset = MrDataset('%s/%s/train.json' % (args.data_path, args.task), 'char', tokenizer,
                              max_context_len=args.max_context_len, max_question_len=args.max_question_len,
                              for_bert=True, do_to_id=True, do_train=True, debug=args.debug)
    test_dataset = MrDataset('%s/%s/dev.json' % (args.data_path, args.task), 'char', tokenizer,
                             max_context_len=args.max_context_len, max_question_len=args.max_question_len,
                             for_bert=True, do_to_id=True, do_train=False, debug=args.debug)

    return train_dataset, test_dataset


def data_collate_fn(data):
    data = np.array(data)

    contexts = []
    ans_tags = []
    ans_texts = []
    segments = []
    for item in data:
        question = item[0]
        ans_ctx = item[1]
        ans_tag = item[2]
        ans_text = item[4]
        segment = item[7]

        contexts.append(question + ans_ctx)
        ans_tags.append([tag + len(question) for tag in ans_tag])
        ans_texts.append(ans_text)
        segments.append(segment)

    contexts = [torch.LongTensor(np.array(s)) for s in contexts]
    ans_tags = torch.LongTensor(ans_tags)
    segments = [torch.LongTensor(np.array(s)) for s in segments]

    return contexts, segments, ans_tags, ans_texts


def train(args, dataset, dataloader, model, optimizer, lr_scheduler):
    model.train()
    loss_sum = 0
    for batch, data in enumerate(dataloader):
        optimizer.zero_grad()
        contexts, segments, ans_tags, _ = data

        contexts = [item.cpu() if args.use_cpu else item.cuda() for item in contexts]
        segments = [item.cpu() if args.use_cpu else item.cuda() for item in segments]
        ans_tags = ans_tags.cpu() if args.use_cpu else ans_tags.cuda()

        loss = model(contexts, segments, decode=False, tags=ans_tags)
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
            contexts, segments, _, ans_text = data

            contexts = [item.cpu() if args.use_cpu else item.cuda() for item in contexts]
            segments = [item.cpu() if args.use_cpu else item.cuda() for item in segments]

            starts, ends = model(contexts, segments)
            starts = starts.cpu().numpy()
            ends = ends.cpu().numpy()

            for i in range(len(contexts)):
                ctx = contexts[i].cpu().numpy()
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

    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    if args.debug:
        args.batch_size = 3

    if args.multi_gpu:
        logging.info("run on multi GPU")
        torch.distributed.init_process_group(backend="nccl")

    model_utils.setup_seed(0)

    output_path = '%s/%s' % (args.output_path, args.task)
    if args.bert_freeze:
        output_path += '_freeze'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    model_logger = Logger(data_path=output_path)

    logging.info("loading embedding")
    tokenizer = BertTokenizer.from_pretrained('%s/vocab.txt' % args.pretrained_bert_path)

    logging.info("loading dataset")
    train_dataset, test_dataset = get_dataset(args, tokenizer)

    if args.multi_gpu:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=data_collate_fn,
                                      sampler=DistributedSampler(train_dataset, shuffle=True))
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=data_collate_fn,
                                     sampler=DistributedSampler(test_dataset, shuffle=False))
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=data_collate_fn,
                                      shuffle=True)
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
        model = Bert('%s/config.json' % args.pretrained_bert_path, '%s/pytorch_model.bin' % args.pretrained_bert_path,
                     args.bert_freeze)
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
    parser.add_argument('--pretrained_bert_path', dest='pretrained_bert_path',
                        default='../data/bert/bert-base-chinese/')
    parser.add_argument('--pretrained_model_path', dest='pretrained_model_path',
                        default=None)
    parser.add_argument('--output_path', type=str,
                        default='../runtime/bert/')
    parser.add_argument('--bert_freeze', dest='bert_freeze', type=bool,
                        default=False)
    parser.add_argument('--max_question_len', type=int,
                        help='24 for dureader, 64 for cmrc2018',
                        default=24)
    parser.add_argument('--max_context_len', type=int,
                        default=512)
    parser.add_argument('--batch_size', type=int,
                        default=32)
    parser.add_argument('--epoch_size', type=int,
                        default=30)
    parser.add_argument('--learning_rate', type=float,
                        default=5e-5)
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
