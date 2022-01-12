# -*- coding: utf-8 -*-
import json
import os
import shutil
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')


class ParamsHandle:
  """
  Python下一切皆对象，每个对象都有多个属性(attribute)，Python对属性有一套统一的管理方案。
  self.__dict__是用来存储 对象属性 的一个字典，其键为属性名，值为属性的值。
  """

  def __init__(self, params_path):
    with open(params_path) as f:
      params = json.load(f)
      self.__dict__.update(params)


class CpImage:
  """
  该类是用来复制图片移动到目标文件夹下的。
  此处是为了使用更多的kaggle数据集中图像来训练。
  """

  def __init__(self, s_path, p1, t1_path, p2, t2_path):
    self.path = 'r' + s_path
    self.p1 = p1
    self.p2 = p2
    self.t1_path = 'r' + t1_path
    self.t2_path = 'r' + t2_path

  def move_image(self, file_name, target_path):
    image_path = os.path.join(self.path, file_name)
    shutil.copy(image_path, target_path)

  def cp(self):
    assert os.path.exists(self.path)

    for filename in os.listdir(self.path):
      if not os.path.isdir(filename):
        if filename.find(self.p1) > -1:
          self.move_image(filename, self.t1_path)
        elif filename.find(self.p2) > -1:
          self.move_image(filename, self.t2_path)


class DrawImg:
  def __init__(self, result) -> None:
    self.result = result

  def draw_img(self):
    plt.figure(1)
    plt.title('accuracy')
    plt.legend(['train', 'val'], loc='upper right')
    plt.plot(self.result.history['accuracy'])
    plt.plot(self.result.history['val_accuracy'])
    plt.ylabel('accuracy')
    plt.xlabel('epochs')

    plt.figure(2)
    plt.title('loss')
    plt.legend(['train', 'val'], loc='upper right')
    plt.plot(self.result.history['loss'])
    plt.plot(self.result.history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.show()
