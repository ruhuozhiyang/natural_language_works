import matplotlib
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout

matplotlib.use('TkAgg')

# 数据路径
result_dir = './result'

# 图像大小
img_width, img_height = 128, 128
input_shape = (img_height, img_width, 3)

model = Sequential()
model.add(Flatten(input_shape=input_shape))
model.add(Dropout(0.25))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(512, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.summary()
plot_model(model, to_file=result_dir + '/model.png')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit()
