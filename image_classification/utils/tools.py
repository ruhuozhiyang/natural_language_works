# -*- coding: utf-8 -*-
import json


class ParamsHandle:
  """
  Python下一切皆对象，每个对象都有多个属性(attribute)，Python对属性有一套统一的管理方案。
  self.__dict__是用来存储 对象属性 的一个字典，其键为属性名，值为属性的值。
  """
  def __init__(self, params_path):
    with open(params_path) as f:
      params = json.load(f)
      self.__dict__.update(params)
