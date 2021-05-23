import tensorflow as tf
import sys
from datetime import datetime
from utils.train_utils import get_parsed_in, create_folders, Logger, get_data
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, ReLU, add, BatchNormalization, Layer
from tensorflow.keras import Model, Input, datasets
import tensorflow_addons as tfa

import os


class LearningRateLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self._supports_tf_logs = True

    def on_epoch_end(self, epoch, logs=None):
        if logs is None or "lr" in logs:
            return
        logs["lr"] = self.model.optimizer.lr

        if logs is None or "learning_rate" in logs:
            return
        logs["learning_rate"] = self.model.optimizer.learning_rate

        # if logs is None or "weight_decay" in logs:
        #     return
        # logs["weight_decay"] = self.model.optimizer.weight_decay


class group_norm(Layer):
    def __init__(self, last=False, G=8, eps=0.00001): #  G=32
        super(group_norm, self).__init__()
        self.eps = eps
        self.G = G
        self.last = last

    def build(self, input_shape):
        # input_shape: N,H,W,C
        # gamma, beta: scale and offset, with shape [1,C,1,1]
        # G: number of groups for GN

        # NHWC
        if self.last:
            self.gamma = tf.Variable(tf.zeros(shape=(1, 1, 1, input_shape[-1])), name="Gamma")
        else:
            self.gamma = tf.Variable(tf.ones(shape=(1, 1, 1, input_shape[-1])), name="Gamma")
        self.beta = tf.Variable(tf.zeros(shape=(1, 1, 1, input_shape[-1])), name="Gamma")

        # NCHW
        # if self.last:
        #     self.gamma = tf.Variable(tf.zeros(shape=(1, input_shape[-1], 1, 1)), name="Gamma")
        # else:
        #     self.gamma = tf.Variable(tf.ones(shape=(1, input_shape[-1], 1, 1)), name="Gamma")
        # self.beta = tf.Variable(tf.zeros(shape=(1, input_shape[-1], 1, 1)), name="Gamma")
        # self.N, self.H, self.W, self.C = input_shape

    def call(self, x):
        # Watch channel last vs channel first(paper)
        N = tf.shape(x)[0]
        H = tf.shape(x)[1]
        W = tf.shape(x)[2]
        C = tf.shape(x)[3]

        x = tf.reshape(x, [N, H, W, self.G, C // self.G])
        mean, var = tf.nn.moments(x, [1, 2, 4], keepdims=True)
        x = (x - mean) / tf.sqrt(var + self.eps)

        x = tf.reshape(x, [N, H, W, C])

        return x * self.gamma + self.beta

    # def call(self, x):
    #     # Watch channel last vs channel first(paper)
    #     # From NHWC to NCHW
    #     x = tf.transpose(x, [0, 3, 1, 2])
    #     # x: input features with shape [N,C,H,W]
    #     x = tf.reshape(x, [self.N, self.G, self.C // self.G, self.H, self.W])
    #     mean, var = tf.nn.moments(x, [2, 3, 4], keepdims=True)
    #     x = (x - mean) / tf.sqrt(var + self.eps)
    #     x = tf.reshape(x, [self.N, self.C, self.H, self.W])
    #     x = x * self.gamma + self.beta
    #     # From NCHW to NHWC
    #     return tf.transpose(x, [0, 2, 3, 1])


def residual_block(input_layer, nb_filters, arguments, last=False, projection_shortcut=False):
    skip_connection = input_layer

    if projection_shortcut:
        strides = 2  # downsample
    else:
        strides = 1

    y = Conv2D(nb_filters, (3, 3), strides=strides, padding='same')(input_layer)
    if arguments['norm'] == 'BN':
        y = BatchNormalization()(y)
    else:
        y = group_norm(last=last)(y)

    y = ReLU()(y)

    y = Conv2D(nb_filters, (3, 3), strides=(1, 1), padding='same')(y)

    if arguments['norm'] == 'BN':
        y = BatchNormalization()(y)
    else:
        y = group_norm(last=last)(y)

    if projection_shortcut:
        skip_connection = Conv2D(nb_filters, kernel_size=(1, 1), strides=(2, 2))(skip_connection)  # projection shortcut
    y = add([skip_connection, y])
    y = ReLU()(y)
    return y


def get_data_aug():
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
        # tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ])
    return data_augmentation


def get_uncompiled_resnet(inputs, arguments):
    data_augmentation = get_data_aug()
    num_classes = 10

    y = data_augmentation(inputs)
    # initial convolution
    y = Conv2D(16, (3, 3), strides=(1, 1), padding='same')(y)
    if arguments['norm'] == 'BN':
        y = BatchNormalization()(y)
    else:
        y = group_norm(last=False)(y)
    y = ReLU()(y)

    #y = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(y)

    res_block = residual_block

    # Block 1
    y = res_block(y, nb_filters=16, arguments=arguments)
    y = res_block(y, nb_filters=16, arguments=arguments)
    if arguments["ResNet"] == 32:
        y = res_block(y, nb_filters=16, arguments=arguments)
        y = res_block(y, nb_filters=16, arguments=arguments)
    y = res_block(y, nb_filters=16, arguments=arguments, last=True)

    # Block 2
    y = res_block(y, nb_filters=32, projection_shortcut=True, arguments=arguments)
    y = res_block(y, nb_filters=32, arguments=arguments)
    if arguments["ResNet"] == 32:
        y = res_block(y, nb_filters=32, arguments=arguments)
        y = res_block(y, nb_filters=32, arguments=arguments)
    y = res_block(y, nb_filters=32, arguments=arguments, last=True)

    # Block 3
    y = res_block(y, nb_filters=64, projection_shortcut=True, arguments=arguments)
    y = res_block(y, nb_filters=64, arguments=arguments)
    if arguments["ResNet"] == 32:
        y = res_block(y, nb_filters=64, arguments=arguments)
        y = res_block(y, nb_filters=64, arguments=arguments)
    y = res_block(y, nb_filters=64, arguments=arguments, last=True)


    # Classifier
    y = GlobalAveragePooling2D()(y)
    outputs = Dense(num_classes, activation='softmax')(y)

    if arguments["ResNet"] == 32:
        model = Model(inputs=inputs, outputs=outputs, name="ResNet32")
    else:
        model = Model(inputs=inputs, outputs=outputs, name="ResNet18")

    return model


def scheduler(epoch, lr):

    if epoch > 0 and epoch % 15 == 0: #30
        lr= lr / 10

    print("epoch :", epoch, " lr :", lr)
    return lr



def main():
    # Get command line arguments
    arguments = get_parsed_in()

    # Create needed folder structure
    log_path, models_path = create_folders(arguments)

    # Log output
    sys.stdout = Logger(log_path)
    # sys.stdout = open(f'{log_path}/output_log.log', 'w')
    # sys.stderr = open(f'{log_path}/output_log.log', 'w')
    print("Training started: " + str(datetime.now()))

    # Turn memory growth an to run multiple scripts in parallel on the same GPU
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
    print("Tensorflow version: ", tf.__version__)
    print("Training config: " + str(arguments))

    # Get data
    train_ds,valid_ds = get_data(arguments)
    #Get model and plot summary
    model = get_uncompiled_resnet(inputs=Input(shape=(32, 32, 3), batch_size=arguments['batch_size']), arguments=arguments)
    model.summary()

    model = get_uncompiled_resnet(inputs=Input(shape=(32, 32, 3)), arguments=arguments)


    lr = 0.01 * (arguments['batch_size'] / 32)  # modified from 0.1 to get convergence on ResNet20

    optimizer = tfa.optimizers.AdamW(weight_decay=0.0001, learning_rate=lr)

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # callbakcs
    scheduler_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=10, write_graph=False, profile_batch=0)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=models_path + '/best_model',
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1)

    model.fit(train_ds, epochs=50, validation_data=valid_ds, callbacks=[scheduler_callback, LearningRateLogger(), model_checkpoint_callback, tensorboard])



if __name__ == "__main__":
    main()
