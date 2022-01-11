import argparse
import os
from keras.utils.vis_utils import plot_model

from image_classification.dataset import TrainData, ValidateData, TestData
from image_classification.model import Image_CNN
from image_classification.utils.cyclic_lr import CyclicLR
from image_classification.utils.draw_img import DrawImg
from image_classification.utils.tools import ParamsHandle

parser = argparse.ArgumentParser()
parser.add_argument('--params_file', default='./params.json')


train_iteration_count = 200
val_iteration_count = 20


clr = CyclicLR(base_lr=0.0005, max_lr=0.001, step_size=2000, mode='triangular')


def train(config):
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
    steps_per_epoch=train_iteration_count,
    epochs=config.epochs,
    validation_data=ValidateData(config).get_data(),
    validation_steps=val_iteration_count,
    callbacks=clr)

  """
  测试模型、评估分数
  """
  score = model.evaluate(TestData(config).get_data(), steps=5)
  print('测试分数：' + str(score))

  DrawImg(result).draw_img()
  record_model_info(config, model)


def record_model_info(config, model):
  plot_model(model, to_file=config.result_dir + '/model.png')

  with open(config.result_dir + '/model_architecture.json', 'w') as f:
    f.write(model.to_json())


if __name__ == '__main__':
  args = parser.parse_args()
  params_path = args.params_file
  assert os.path.isfile(params_path), "params file {} does not exist".format(params_path)
  params = ParamsHandle(params_path)

  train(params)
