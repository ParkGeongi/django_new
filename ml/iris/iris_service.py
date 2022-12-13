import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from keras.layers import Dense
import tensorflow as tf
from keras.models import load_model
from tensorflow.python.framework.ops import get_default_graph


class IrisService(object):

    def __init__(self):

        #self.model = load_model('./save/iris_model_h5')
        self.graph = get_default_graph()
        self.target_names = datasets.load_iris().target_names
        self.SepalLengthCm = None
        self.SepalWidthCm = None
        self.PetalLengthCm = None
        self.PetalWidthCm = None
    def hook(self):

        self.service_model()


    def service_model(self):
        pass

    def print(self):

        print(self.PetalWidthCm)



'''
 --- 1.Shape ---
(150, 6)
 --- 2.Features ---
Index(['Id', 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm',
       'Species'],
'''
iris_menu = ["Exit", #0
                "hook",#1
             ]
iris_lambda = {
    "1" : lambda x: x.hook(),

}
if __name__ == '__main__':
    iris = IrisService()


    while True:
        [print(f"{i}. {j}") for i, j in enumerate(iris_menu)]
        menu = input('메뉴선택: ')
        if menu == '0':
            print("종료")
            break
        else:
            try:
                iris_lambda[menu](iris)
            except KeyError as e:
                if 'some error message' in str(e):
                    print('Caught error message')
                else:
                    print("Didn't catch error message")
