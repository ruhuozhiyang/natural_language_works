import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')


class DrawImg:
  def __init__(self, result) -> None:
    self.result = result

  def draw_img(self):
    plt.plot(self.result.history['accuracy'])
    plt.plot(self.result.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.plot(self.result.history['loss'])
    plt.plot(self.result.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
