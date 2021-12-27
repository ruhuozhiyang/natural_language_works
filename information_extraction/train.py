import argparse
import logging
import os

import torch
import torch.nn as nn
import torch.optim as optima
from torch.optim.lr_scheduler import LambdaLR
from tqdm import trange

import utils.utils as utils
import cnn_net as net
from utils.data_loader import DataLoader
from evaluate import evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/SemEval2010_task8')
parser.add_argument('--embedding_file', default='data/embeddings/word2vector.txt', )
parser.add_argument('--result_dir', default='experiments/result', )
parser.add_argument('--params_file', default='./params.json')


def train(cnn_net, data_iterator, opt, schedule, params, steps_num):
  loss_avg = utils.RunningAverage()
  t = trange(steps_num)
  for _ in t:
    batch_data, batch_labels = next(data_iterator)
    batch_output = cnn_net(batch_data)
    loss = cnn_net.loss(batch_output, batch_labels)
    loss.backward()
    nn.utils.clip_grad_norm_(cnn_net.parameters(), params.clip_grad)
    opt.step()
    loss_avg.update(loss.item())
    t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
  schedule.step()
  return loss_avg()


def train_and_evaluate(cnn_net, tra_data, valid_data, opt,
                       schedule, params, labels, result_dir):
  cnn_net.train()
  best_val_f1 = 0.0
  p_c = 0

  for epoch in range(1, params.epoch_num + 1):
    cnn_net.zero_grad()
    logging.info("Epoch {}/{}".format(epoch, params.epoch_num))

    train_steps_num = params.train_size // params.batch_size
    val_steps_num = params.val_size // params.batch_size

    train_data_iterator = data_loader.data_iterator(tra_data, params.batch_size, shuffle='True')
    train_loss = train(cnn_net, train_data_iterator, opt, schedule, params, train_steps_num)

    train_data_iterator = data_loader.data_iterator(tra_data, params.batch_size)
    val_data_iterator = data_loader.data_iterator(valid_data, params.batch_size)
    train_metrics = evaluate(cnn_net, train_data_iterator, train_steps_num, labels)
    train_metrics['loss'] = train_loss
    train_metrics_str = "; ".join("{}: {:05.2f}".format(k, v) for k, v in train_metrics.items())
    logging.info("Train metrics: " + train_metrics_str)

    val_metrics = evaluate(cnn_net, val_data_iterator, val_steps_num, labels)
    val_metrics_str = "; ".join("{}: {:05.2f}".format(k, v) for k, v in val_metrics.items())
    logging.info("Validate metrics: " + val_metrics_str)

    val_f1 = val_metrics['f1']
    improve_f1 = val_f1 - best_val_f1

    utils.save_checkpoint(
      {
        'epoch': epoch + 1,
        'state_dict': cnn_net.state_dict(),
        'optima_dict': opt.state_dict()
      },
      is_best=improve_f1 > 0,
      checkpoint=result_dir
    )
    if improve_f1 > 0:
      logging.info("New improved F1")
      best_val_f1 = val_f1
      p_c = (p_c + 1) if improve_f1 < params.patience else 0
    else:
      p_c += 1

    if p_c >= params.patience_num and epoch > params.min_epoch_num:
      logging.info("Stop early and best validate f1: {:05.2f}".format(best_val_f1))
      break


if __name__ == '__main__':
  args = parser.parse_args()
  json_path = args.params_file
  assert os.path.isfile(json_path), "json file not found at {}".format(json_path)
  params = utils.Params(json_path)

  torch.manual_seed(230)

  utils.set_logger(os.path.join(args.result_dir, 'train.log'))
  logging.info("Loading the datasets...")

  data_loader = DataLoader(data_dir=args.data_dir,
                           embedding_file=args.embedding_file,
                           word_emb_dim=params.word_emb_dim,
                           max_len=params.max_len,
                           pos_dis_limit=params.pos_dis_limit,
                           pad_word='<pad>',
                           unk_word='<unk>',
                           other_label='Other')
  data_loader.load_embeddings_and_unique_words(emb_delimiter=' ')
  metric_labels = data_loader.metric_labels

  train_data = data_loader.load_data('train')
  val_data = data_loader.load_data('test')

  params.train_size = train_data['size']  # 训练集有多少行句子.
  params.val_size = val_data['size']  # 验证集有多少行句子.
  logging.info("Loaded the datasets successfully.")

  model = net.CnnNet(data_loader, params)
  optimizer = optima.Adam(model.parameters(), lr=params.learning_rate,
                          weight_decay=params.weight_decay)

  scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (1 + 0.05 * epoch))

  logging.info("Starting training for {} epoch(s)".format(params.epoch_num))
  train_and_evaluate(cnn_net=model,
                     tra_data=train_data,
                     valid_data=val_data,
                     opt=optimizer,
                     schedule=scheduler,
                     params=params,
                     labels=metric_labels,
                     result_dir=args.result_dir)
