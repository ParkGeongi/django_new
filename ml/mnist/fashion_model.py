import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from keras.layers import Dense
from keras import Sequential
from tensorflow import keras


class FashionModel(object):


    def hook(self):
        self.spec()
        #self.create_model()


    def spec(self):
        pass




    def create_model(self):
        (train_images,train_labels),(test_images,test_labels) = keras.datasets.fashion_mnist.load_data()
        plt.figure()
        plt.imshow(train_images[10])
        plt.colorbar()
        plt.grid(False)
        plt.show()
        model = Sequential([
            keras.layers.Flatten(input_shape=(28,28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer = 'adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(train_images, train_labels, epochs=10)
        test_loss, test_acc = model.evaluate(test_images, test_labels)
        print(f'Test accuracy is {test_acc}')
        file_name = './save/fashion_model.h5'
        model.save(file_name)
        print(f'Model Save in {file_name}')



'''
 --- 1.Shape ---
(150, 6)
 --- 2.Features ---
Index(['Id', 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm',
       'Species'],
'''
fashion_menu = ["Exit", #0
                "hook",#1
             ]
fashion_lambda = {
    "1" : lambda x: x.create_model(),
}

if __name__ == '__main__':
    fashion = FashionModel()


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
