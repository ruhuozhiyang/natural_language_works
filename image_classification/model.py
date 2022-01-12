from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dropout, Dense


class Image_CNN:
	"""
	这是模型CNN的结构定义，用来进行图像分类。
	"""
	def __init__(self, config):
		self.img_width = config.img_width
		self.img_height = config.img_height

	def get_model_cnn(self):
		"""
		最简单的模型是 Sequential 顺序模型，它由多个网络层线性堆叠。
		对于更复杂的结构，你应该使用 Keras 函数式 API，它允许构建任意的神经网络图。
		"""
		model = Sequential()
		"""
		卷积层作为模型第一层时候，必须提供input_shape参数.
		"""
		model.add(Conv2D(16, (3, 3), padding="same", input_shape=(self.img_width, self.img_height, 3)))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(32, (3, 3), padding="same"))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Flatten())  # 扁平层
		model.add(Dropout(0.25))
		model.add(Dense(1024, activation='relu'))  # 全连接
		model.add(Dense(2, activation='softmax'))

		return model


class Image_DNN:
	"""
	这是模型DNN的结构定义，用来进行图像分类。
	"""
	def __init__(self, config):
		self.img_width = config.img_width
		self.img_height = config.img_height

	def get_model_dnn(self):
		model = Sequential()
		model.add(Flatten(data_format='channels_last', input_shape=(self.img_width, self.img_height, 3)))
		model.add(Dense(1024, activation='relu'))
		model.add(Dense(512, activation='relu'))
		model.add(Dense(256, activation='relu'))
		model.add(Dense(2, activation='softmax'))

		return model


class Image_RNN:
	"""
	这是模型RNN的结构定义，用来进行图像分类。
	"""
	def __init__(self):
		pass
