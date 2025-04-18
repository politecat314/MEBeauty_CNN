import numpy as np
import os
import pickle
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x = x_set
        self.y = y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, iter):
        batch_x = self.x[iter * self.batch_size:(iter + 1) * self.batch_size]
        batch_y = self.y[iter * self.batch_size:(iter + 1) * self.batch_size]
        
        #p = np.random.permutation(len(batch_x))
        #np.take(batch_x, p, axis=0, out=batch_x)
        #np.take(batch_y, p, axis=0,out=batch_y)

        #return batch_x[p], batch_y[p]

        return batch_x, batch_y
    
    #def on_epoch_end(self):
        #p = np.random.permutation(len(self.x))
        #np.take(self.x, p, axis=0, out=self.x)
        #np.take(self.y, p, axis=0,out=self.y)
    
class Dataset():
    def __init__(
            self,
            feature_extractor,
            directory,
            image_path,
            train_path,
            test_path,
            ratings_path,
            predict, 
            batch_size,
            val_path=None
        ):

        self.directory = directory
        self.image_path = image_path
        self.ratings = pd.read_csv(ratings_path)

        self.train_files = self.path_to_filenames(train_path)
        self.test_files = self.path_to_filenames(test_path)
        self.val_files = self.path_to_filenames(val_path)

        self.predict = predict
        self.batch_size = batch_size
        self.feature_extractor = feature_extractor
        
    @property
    def feature_extractor(self):
        return self._feature_extractor
    
    @feature_extractor.setter
    def feature_extractor(self, model):
        self._feature_extractor = model
        self.update_feature_path()

    def update_feature_path(self):
        self.model_dir = self.directory+self.feature_extractor._name+"/"

        if self.predict:
            self.feature_path = self.model_dir+str(self.feature_extractor.output_shape[1:])+"/"
        else:
            self.feature_path = self.model_dir+str(self.feature_extractor.input_shape[1:])+"/"

        if os.path.exists(self.feature_path):
            with open(self.feature_path+"train.pkl","rb") as f:
                self.train = pickle.load(f)
            with open(self.feature_path+"test.pkl","rb") as f:
                self.test = pickle.load(f)

            if os.path.exists(self.feature_path+"val.pkl"):
                with open(self.feature_path+"val.pkl","rb") as f:
                    self.val = pickle.load(f)

        else:
            os.makedirs(self.feature_path)
            self.generate()

    def path_to_filenames(self, path):
        if not path:
            return 0
        
        with open(path, "r") as f:
            return [l.split(".jpg")[0]+".jpg" for l in f.readlines()]
    
    def shuffle(self, test_size=0.2, val_size=0.1):
        X_train, self.test.x, \
        y_train, self.test.y = train_test_split(
            np.concatenate((
                self.train.x[:len(self.train_files)],
                self.val.x,
                self.test.x
            )),
            np.concatenate((
                self.train.y[:len(self.train_files)],
                self.val.y,
                self.test.y
            )),
            shuffle=True,
            test_size=test_size
        )

        if not isinstance(self.val_files, int):
            X_train, self.val.x, \
            y_train, self.val.y = train_test_split(
                X_train,
                y_train,
                test_size=val_size
            )
    
    def preprocess(self, X):
        X = X[..., ::-1] #Convert from RGB to BGR

        #Zero center each channel w.r.t training dataset
        if isinstance(self.mean, int):
            self.mean = np.array([
                np.mean(X[..., 0]),
                np.mean(X[..., 1]),
                np.mean(X[..., 2])],
                np.float32
            )
            self.std = np.array([
                np.std(X[..., 0]),
                np.std(X[..., 1]),
                np.std(X[..., 2])],
                np.float32
            ) 

        X -= self.mean
        if self.feature_extractor._name == "vgg16": X = X / self.std
        if self.predict: X = self.feature_extractor.predict(X)

        return X

    def generate(self):
        self.mean = 0
        self.std = 0

        X_train, y_train = self.load(self.train_files)
        X_test, y_test = self.load(self.test_files)

        X_train = self.preprocess(X_train)
        X_test = self.preprocess(X_test)

        print(X_train.shape, X_test.shape)
        print(y_train.shape, y_test.shape)

        if not isinstance(self.val_files, int):
            X_val, y_val = self.load(self.val_files)
            X_val = self.preprocess(X_val)
            print(X_val.shape, y_val.shape)

            self.val = DataGenerator(X_val, y_val, self.batch_size)
            with open(self.feature_path+"val.pkl", "wb") as f:pickle.dump(self.val, f)
            print(f"Val generator saved to {self.feature_path+'val.pkl'}")

        self.train = DataGenerator(X_train, y_train, self.batch_size)
        self.test = DataGenerator(X_test, y_test, self.batch_size)

        with open(self.feature_path+"train.pkl", "wb") as f: pickle.dump(self.train, f)
        with open(self.feature_path+"test.pkl", "wb") as f: pickle.dump(self.test, f)
        print(f"Train generator saved to {self.feature_path+'train.pkl'}")
        print(f"Test generator saved to {self.feature_path+'test.pkl'}")

    def load(self, files):
        X = []
        y = []
        for file in files:
            if os.path.exists(self.image_path+file) and file in self.ratings["filename"].to_numpy():
                label = np.asarray(
                    self.ratings.loc[self.ratings["filename"]==file]
                    .to_numpy()[0][2:-1], 
                    dtype=np.float32
                )
                label /= np.sum(label)
                y.append(label)

                load_img, img_to_array

                X.append(img_to_array(load_img(
                    self.image_path+file,
                    target_size=(
                        self.feature_extractor.input_shape[1],
                        self.feature_extractor.input_shape[2]
                    ),
                    interpolation="bicubic"
                )))

        X = np.array(X, np.float32)
        y = np.array(y, np.float32)

        return X, y

class SCUTFBP5500(Dataset):
    def __init__(
            self,
            base_model,
            image_path="./SCUT-FBP5500/mediapipe/",
            ratings_path="./SCUT-FBP5500/distribution.csv",
            train_path="./SCUT-FBP5500/train.txt",
            test_path="./SCUT-FBP5500/test.txt",
            predict=True,
            batch_size=32
        ):

        super().__init__(
            base_model,
            "./SCUT-FBP5500/",
            image_path,
            train_path,
            test_path,
            ratings_path,
            predict,
            batch_size,
        )

        self.n = 5

class MEBeauty(Dataset):
    def __init__(
            self,
            base_model,
            image_path="./MEBeauty/mediapipe/",
            ratings_path="./MEBeauty/distribution.csv",
            train_path="./MEBeauty/train_2023.txt",
            test_path="./MEBeauty/test_2023.txt",
            val_path="./MEBeauty/val_2023.txt",
            predict=True,
            batch_size=32
        ):

        super().__init__(
            base_model,
            "./MEBeauty/",
            image_path,
            train_path,
            test_path,
            ratings_path,
            predict,
            batch_size,
            val_path=val_path,
        )

        self.n = 10
