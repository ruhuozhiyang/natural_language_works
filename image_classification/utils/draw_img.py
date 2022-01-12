import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')


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
