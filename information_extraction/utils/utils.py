import json
import logging
import os
import shutil
import torch
import numpy as np


class Params:
  def __init__(self, json_path):
    with open(json_path) as f:
      params = json.load(f)
      self.__dict__.update(params)

  def save(self, json_path):
    with open(json_path, 'w') as f:
      json.dump(self.__dict__, f, indent=4)

  def update(self, json_path):
    with open(json_path) as f:
      params = json.load(f)
      self.__dict__.update(params)

  def dict(self):
    return self.__dict__


class RunningAverage:
  def __init__(self):
    self.steps = 0
    self.total = 0

  def update(self, val):
    self.total += val
    self.steps += 1

  def __call__(self):
    return self.total / float(self.steps)


def set_logger(log_path):
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)

  if not logger.handlers:
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
  with open(json_path, 'w') as f:
    d = {k: float(v) for k, v in d.items()}
    json.dump(d, f, indent=4)


def save_checkpoint(state, is_best, checkpoint):
  filepath = os.path.join(checkpoint, 'last.pth.tar')
  if not os.path.exists(checkpoint):
    os.mkdir(checkpoint)
  torch.save(state, filepath)
  if is_best:
    shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, model, optimizer=None):
  if not os.path.exists(checkpoint):
    raise ("File doesn't exist {}".format(checkpoint))
  checkpoint = torch.load(checkpoint)
  model.load_state_dict(checkpoint['state_dict'])

  if optimizer:
    optimizer.load_state_dict(checkpoint['optima_dict'])

  return checkpoint


def generate_zero_vector(embedding_dim):
  return [0] * embedding_dim


def generate_random_vector(embedding_dim):
  return np.random.uniform(-np.sqrt(3.0 / embedding_dim),
                           np.sqrt(3.0 / embedding_dim),
                           embedding_dim).tolist()
