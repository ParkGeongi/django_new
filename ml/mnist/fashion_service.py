import os

import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from sklearn import datasets
from tensorflow import keras


class FashionService(object):

    def __init__(self):
        global class_names
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    def service_model(self,i)->[]:
        #model = load_model(os.path.join(os.path.abspath("save"), "fashion_model.h5"))
        model = load_model(r'C:\Users\AIA\project\jdango_new\ml\mnist\save\fashion_model.h5')
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

        predictions = model.predict(test_images)
        predictions_array, true_label, img = predictions[i], test_labels[i], test_images[i]
        predicted_label = np.argmax(predictions_array)
        '''plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img, cmap = plt.cm.binary)
        predicted_label = np.argmax(predictions_array)
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'

        plt.xlabel('{} {:2.0f}% ({})'.format(
            class_names[predicted_label],
            100 * np.max(predictions_array),
            class_names[true_label]
        ), color=color)
        plt.show()'''
        return f'예측 : {class_names[predicted_label]}+ argmax : {100 * np.max(predictions_array)}+ 실제 : {class_names[true_label]}'
@staticmethod
def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = \
        predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10),
                       predictions_array,
                       color='#777777')
    plt.ylim([0,1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


'''
 --- 1.Shape ---
(150, 6)
 --- 2.Features ---
Index(['Id', 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm',
       'Species'],
'''
fashion_menu = ["Exit",  # 0
             "hook",  # 1
             ]
fashion_lambda = {
    "1": lambda x: x.service_model(20)

}
if __name__ == '__main__':
    fashion = FashionService()

    while True:
        [print(f"{i}. {j}") for i, j in enumerate(fashion_menu)]
        menu = input('메뉴선택: ')
        if menu == '0':
            print("종료")
            break
        else:
            try:
                fashion_lambda[menu](fashion)
            except KeyError as e:
                if 'some error message' in str(e):
                    print('Caught error message')
                else:
                    print("Didn't catch error message")
