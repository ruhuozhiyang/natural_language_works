import argparse
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from image_classification.model import Image_CNN
from image_classification.utils.cyclic_lr import CyclicLR
from image_classification.utils.draw_img import DrawImg

parser = argparse.ArgumentParser()
parser.add_argument('--params_file', default='./params.json')
train_data_dir = '../data/train'
validation_data_dir = '../data/validation'
test_data_dir = '../data/test'
result_dir = './result'

# 载入图像大小
img_width, img_height = 128, 128

# 超参数配置 epochs/batch_size很关键的两个参数.
epochs = 1
batch_size = 25
train_iteration_count = 200
val_iteration_count = 20

model = Image_CNN().get_model_cnn()

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

DrawImg(result).draw_img()

json_string = model.to_json()
open(result_dir + '/model_architecture.json', 'w').write(json_string)