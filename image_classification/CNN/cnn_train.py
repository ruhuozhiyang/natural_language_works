from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model

# 全局常量
base_dir = '/Users/foiunclekay/Documents/GitHub/natural_language_works/image_classification'
train_data_dir = base_dir + '/data/train'
validation_data_dir = base_dir + '/data/validation'
test_data_dir = base_dir + '/data/test'
result_dir = base_dir + '/CNN/result'

train_iteration_count = 100
val_iteration_count = 25
epochs = 10
batch_size = 20
img_width, img_height = 50, 50
input_shape = (img_width, img_height, 3)

# 模型的具体内容
model = Sequential()
model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25));
model.add(Flatten())
model.add(Dense(1024, activation='relu'))  # 全连接
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
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

# 训练模型
# epochs为训练模型迭代轮次.一个轮次是在整个 x 和 y 上的一轮迭代.
# validation为用来评估当前模型损失，模型将不会在这个数据上进行训练.
result = model.fit(
    train_generator,
    steps_per_epoch=train_iteration_count,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=val_iteration_count)

# 测试模型、评估分数
score = model.evaluate(test_generator, steps=2)
print('测试分数：' + str(score))

# 绘图并将模型json/weight信息导出
plt.plot(result.history['accuracy'])
plt.plot(result.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(result.history['loss'])
plt.plot(result.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

json_string = model.to_json()
open(result_dir + '/model_architecture.json', 'w').write(json_string)

model.save_weights(result_dir + '/model_weights.h5')
