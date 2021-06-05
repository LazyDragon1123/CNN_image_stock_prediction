from PIL import Image
import pandas as pd
import os, glob
import numpy as np
from sklearn import model_selection


class IMAGN:
    base_path = './images/'
    image_size = 50
    def __init__(self,ticker_names):
        self.ticker_names = ticker_names # list
        ticker = ticker_names[0]
        df_temp = pd.read_csv(self.base_path + ticker + '/target.csv',index_col=0)
        self.date_index = df_temp.index.to_list()


    def numerize(self):
        X = []
        Y = []
        for ticker in self.ticker_names:
            df_target = pd.read_csv(self.base_path + ticker + '/target.csv',index_col=0)
            for date in self.date_index:
                file = self.base_path + ticker + '/' + str(date) +  '.png'
                image = Image.open(file)
                image = image.convert("RGB")
                image = image.resize((self.image_size, self.image_size))
                data = np.asarray(image)
                X.append(data)
                target = df_target.loc[date,'target']
                Y.append(target)
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y)
        xy = (X_train, X_test, y_train, y_test)
        self.xy = xy
    def save(self):
        np.save("dataset.npy", self.xy)