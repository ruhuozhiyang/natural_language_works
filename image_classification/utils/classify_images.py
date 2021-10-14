import os
import shutil

path = r'C:\Users\ChenR\Desktop\kaggle\train\train'
cat_path = r'D:\GitHub_Projects\natural_language_works\image_classification\data\train\cat'
dog_path = r'D:\GitHub_Projects\natural_language_works\image_classification\data\train\dog'


def move_image(file_name, target_path):
    image_path = os.path.join(path, file_name)
    print(image_path)
    shutil.copy(image_path, target_path)


for filename in os.listdir(path):
    if not os.path.isdir(filename):
        print(filename)
        if filename.find('cat') > -1:
            move_image(filename, cat_path)
        else:
            move_image(filename, dog_path)

