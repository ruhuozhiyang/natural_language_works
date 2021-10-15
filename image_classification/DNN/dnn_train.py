import matplotlib
import matplotlib.pyplot as plt
from utils.cyclic_lr import CyclicLR
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
matplotlib.use('TkAgg')

# 数据路径
train_data_dir = '../data/train'
validation_data_dir = '../data/validation'
test_data_dir = '../data/test'
result_dir = './result'

# 载入图像大小
img_width, img_height = 128, 128

# 超参数配置 epochs/batch_size很关键的两个参数.
epochs = 100
batch_size = 25
train_iteration_count = 200
val_iteration_count = 20

# 模型的具体内容
model = Sequential()

model.add(Flatten())  # 扁平层
model.add(Dropout(0.25))
model.add(Dense(1024, activation='relu'))  # 全连接
model.add(Dropout(0.25))
model.add(Dense(1024, activation='relu'))  # 全连接
model.add(Dropout(0.25))
model.add(Dense(1024, activation='relu'))  # 全连接
model.add(Dense(2, activation='softmax'))

# compile用于配置训练模型: adam的默认学习率为0.001、loss配置损失函数、metrics为模型评估标准.
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
plot_model(model, to_file=result_dir + '/model.png')

# 对图像数据进行增强以及生成数据
train_augment = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
val_augment = ImageDataGenerator(rescale=1. / 255)
test_augment = ImageDataGenerator(rescale=1. / 255)
train_generator = train_augment.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
validation_generator = val_augment.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
test_generator = test_augment.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

clr = CyclicLR(base_lr=0.0005, max_lr=0.001, step_size=2000, mode='triangular')

# 训练模型
# epochs为训练模型迭代轮次.一个轮次是在整个 x 和 y 上的一轮迭代.
# validation为用来评估当前模型损失，模型将不会在这个数据上进行训练.
result = model.fit(
    train_generator,
    steps_per_epoch=train_iteration_count,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=val_iteration_count,
    callbacks=clr)

# 测试模型、评估分数
score = model.evaluate(test_generator, steps=5)
print('测试分数：' + str(score))

# 绘图并将模型json/weight信息导出
plt.plot(result.history['accuracy'])
plt.plot(result.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(result.history['loss'])
plt.plot(result.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

json_string = model.to_json()
open(result_dir + '/model_architecture.json', 'w').write(json_string)