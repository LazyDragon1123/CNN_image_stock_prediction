import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
import np_utils
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
import numpy as np

class CNN_simple:
        
    classes = ["sell","buy"]
    num_classes = len(classes)
    image_size = 50

    def __init__(self,data_path):
        self.data_path = data_path
        X_train, X_test, y_train, y_test = np.load(data_path, allow_pickle=True)
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        self.X_train = X_train.astype("float32") / 256
        self.X_test = X_test.astype("float32") / 256
        self.y_train = to_categorical(y_train, self.num_classes)
        self.y_test = to_categorical(y_test, self.num_classes)

    def train(self):
        def model_train(X, y):
            
            model = Sequential()
            model.add(Conv2D(16,(3,3), padding='same',input_shape=X.shape[1:]))
            model.add(Activation('relu'))
            model.add(Conv2D(16,(3,3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2,2)))
            model.add(Dropout(0.25))

            model.add(Conv2D(32,(3,3), padding='same'))
            model.add(Activation('relu'))
            model.add(Conv2D(32,(3,3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2,2)))
            model.add(Dropout(0.25))

            model.add(Flatten())
            model.add(Dense(256))
            model.add(Activation('relu'))
            model.add(Dropout(0.5))
            model.add(Dense(self.num_classes))
            model.add(Activation('softmax'))

            opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
            model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
            model.summary()
            model.fit(X, y, batch_size=32, epochs=100)

            model.save('trained_model.h5')

            return model

        model = model_train(self.X_train, self.y_train)
        self.model = model

    def eval(self):
        def model_eval(model, X, y):
            scores = model.evaluate(X, y, verbose=1)
            print('Test Loss: ', scores[0])
            print('Test Accuracy: ', scores[1])
        model_eval(self.model, self.X_test, self.y_test)

