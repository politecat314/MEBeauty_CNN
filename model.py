import tensorflow as tf
#import tensorflow_addons as tfa
from livelossplot import PlotLossesKerasTF
import timeit
import os
import losses
import numpy as np
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class Model():
    def __init__(self, get_cnns, dataset):
        self.get_cnns = get_cnns
        self.dataset = dataset
        self.mlp_path = self.dataset.model_dir+"mlp.hdf5" 

        print("model mlp path: ", self.mlp_path)

    def construct(
            self,
            train=0,
            augment=False,
            supress=False,
            mlp_loss = "emd",
        ):

        tf.keras.backend.clear_session()
        base_cnn, train_cnn, full_cnn = self.get_cnns() 
        
        for layer in base_cnn.layers[:int(len(base_cnn.layers)*(1-train))]:
            layer.trainable = False

        #MLP
        print("train cnn boolean", train_cnn)
        print(train_cnn.output_shape[1:])
        print(base_cnn.output_shape[1:])

        inputs = tf.keras.layers.Input(
            train_cnn.output_shape[1:] \
            if train_cnn else \
            base_cnn.output_shape[1:]
        )
        x = tf.keras.layers.Flatten()(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs = tf.keras.layers.Dense(self.dataset.n, activation="softmax")(x)
        
        mlp = tf.keras.Model(inputs, outputs)
        mlp._name = "MLP"
        if not supress: mlp.summary()

        if not os.path.exists(self.mlp_path):
            self.dataset.feature_extractor = full_cnn
            self.model = mlp
            self.compile(
                mlp_loss,
                metrics=[losses.PearsonCorrelation(self.dataset.n)]
            )
            self.train()
            self.dataset.feature_extractor = base_cnn
        
        mlp.load_weights(self.mlp_path)

        inputs = tf.keras.layers.Input(
            base_cnn.output_shape[1:] \
            if self.dataset.predict else \
            base_cnn.input_shape[1:]
        )
        x = inputs
        if augment: 
            x = tf.keras.layers.RandomRotation(0.1)(x) #5/360
            x = base_cnn(x)
        else:
            x = train_cnn(x)
            if not supress: train_cnn.summary()
        outputs = mlp(x)

        self.model = tf.keras.Model(inputs, outputs)

    def compile(self, loss_str, metrics=None, learning_rate=0.0001):
        if loss_str == "emd": loss = losses.EarthMoversDistance()
        elif loss_str == "kld": loss = tf.keras.losses.KLDivergence()
        elif loss_str == "cce": loss = tf.keras.losses.CategoricalCrossentropy()
        elif loss_str == "l2": loss = tf.keras.losses.MeanSquaredError()

        self.pred_path = self.dataset.model_dir+loss_str+"/"
        if not os.path.exists(self.pred_path): os.makedirs(self.pred_path)

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=metrics
        )

    def train(
            self,
            save=True,
            predict=True,
            callbacks=None, 
            epochs=1000, 
            patience=50, 
            verbose=1,
            monitor="val_correlation", 
            iteration=0,
        ):

        mode = "max" if monitor == "val_correlation" else "min"

        if callbacks:
            callbacks += [PlotLossesKerasTF()]
        else:
            callbacks = [PlotLossesKerasTF()]

        if patience:
            stopping = tf.keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=patience,
                restore_best_weights=True,
                mode=mode
            )

            callbacks += [stopping]

        start = timeit.default_timer()
        self.model.fit(
            self.dataset.train,
            epochs=epochs,
            validation_data=self.dataset.test,
            callbacks=callbacks,
            verbose=verbose
        )
        time = (timeit.default_timer() - start)/60
        print(f"Time: {np.round(time, 4)}")

        #MLP
        if not os.path.exists(self.mlp_path):
            self.model.save_weights(self.mlp_path)
            return
        
        if predict:
            y_pred = self.model.predict(self.dataset.test.x)
            np.save(self.pred_path+f"/y_pred_{iteration}", y_pred)

        if save:
            self.model.save_weights(self.pred_path+f"/weights_{iteration}.hdf5")

        return y_pred, time