from keras_preprocessing.image import ImageDataGenerator


class TrainData:
  def __init__(self, config):
    self.batch_size = config.batch_size
    self.img_width = config.img_width
    self.img_height = config.img_height
    self.train_data_dir = config.train_data_dir

  def get_data(self):
    """
    对图像数据进行增强以及生成数据
    :return: train data generator
    """
    train_augment = ImageDataGenerator(rescale=1. / 255, shear_range=0.2,
                                       zoom_range=0.2, horizontal_flip=True)
    train_generator = train_augment.flow_from_directory(
      directory=self.train_data_dir,
      target_size=(self.img_width, self.img_height),
      batch_size=self.batch_size,
      class_mode='categorical')

    return train_generator


class ValidateData:
  def __init__(self, config):
    self.batch_size = config.batch_size
    self.img_width = config.img_width
    self.img_height = config.img_height
    self.val_data_dir = config.val_data_dir

  def get_data(self):
    """
    对图像数据进行增强以及生成数据
    :return: validation data generator.
    """
    val_augment = ImageDataGenerator(rescale=1. / 255)
    val_generator = val_augment.flow_from_directory(
      self.val_data_dir,
      target_size=(self.img_width, self.img_height),
      batch_size=self.batch_size,
      class_mode='categorical')

    return val_generator


class TestData:
  def __init__(self, config):
    self.batch_size = config.batch_size
    self.img_width = config.img_width
    self.img_height = config.img_height
    self.test_data_dir = config.test_data_dir

  def get_data(self):
    """
    对图像数据进行增强以及生成数据
    :return: test data generator.
    """
    test_augment = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_augment.flow_from_directory(
      directory=self.test_data_dir,
      target_size=(self.img_width, self.img_height),
      batch_size=self.batch_size,
      class_mode='categorical')

    return test_generator
