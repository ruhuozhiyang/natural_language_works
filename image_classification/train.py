import argparse
import os
from keras.utils.vis_utils import plot_model
from dataset import TrainData, ValidateData, TestData
from model import Image_DNN, Image_CNN
from utils.cyclic_lr import CyclicLR
from utils.tools import DrawImg, ParamsHandle

parser = argparse.ArgumentParser()
parser.add_argument('--params_file', default='./params.json')


def train(config, clr, f):
  model = ''
  if f == 0:
    model = Image_DNN(config).get_model_dnn()
  elif f == 1:
    model = Image_CNN(config).get_model_cnn()

  """
  compile用于配置训练模型; metrics为模型评估标准.
  """
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

  """
  训练模型。
  epochs为训练模型迭代轮次，一个轮次是在整个 x 和 y 上的一轮迭代。
  validation用来评估当前模型损失，模型将不会在这个数据上进行训练。
  """
  result = model.fit(
    TrainData(config).get_data(),
    epochs=config.epochs,
    validation_data=ValidateData(config).get_data(),
    callbacks=clr)

  """
  测试模型、评估分数
  """
  score = model.evaluate(TestData(config).get_data(), steps=5)
  print('测试分数：' + str(score))

  DrawImg(result).draw_img()
  record_model_info(config, model)


def record_model_info(config, model):
  if not os.path.exists(config.result_dir):
    os.mkdir(config.result_dir)

  plot_model(model, to_file=config.result_dir + '/model.png')

  with open(config.result_dir + '/model_architecture.json', 'w') as f:
    f.write(model.to_json())


if __name__ == '__main__':
  args = parser.parse_args()
  params_path = args.params_file
  assert os.path.isfile(params_path), "params file {} does not exist".format(params_path)
  params = ParamsHandle(params_path)

  CLR = CyclicLR(base_lr=0.0005, max_lr=0.001, step_size=2000, mode='triangular')

  """
  0表示使用DNN模型；1表示使用CNN模型；
  """
  train(params, CLR, 0)
