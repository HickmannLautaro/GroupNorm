import pickle

import tensorflow as tf
import tensorflow_addons as tfa
import os


class group_normalization_layer(tf.keras.layers.Layer):
    """
    Implementation of GroupNorm ("Group Normalization" - https://arxiv.org/abs/1803.08494)
    This layer is based on the modified code proposed in the paper, so that it can be used in channel last configuration
    """

    def __init__(self, gamma_inint='ones', G=8, eps=0.00001):  # G=32
        super(group_normalization_layer, self).__init__()
        self.eps = eps
        # G: number of groups for GN
        self.G = G
        # gamma, beta: scale and offset, with shape [1,1,1, C]
        self.gamma_initializer = gamma_inint

    def build(self, input_shape):
        # input_shape: N,H,W,C
        # gamma, beta: scale and offset, with shape [1,1,1, C]

        C = input_shape[-1]  # Channels last

        # Custom layer add weights
        self.gamma = self.add_weight("Gamma", shape=[1, 1, 1, C], initializer=self.gamma_initializer)
        self.beta = self.add_weight("Beta", shape=[1, 1, 1, C], initializer='zeros')

    def call(self, inp, **kwargs):
        # Watch channel last vs channel first(paper)
        N, H, W, C = inp.shape
        inp = tf.reshape(inp, [-1, H, W, self.G, C // self.G])
        mean, var = tf.nn.moments(inp, [1, 2, 4], keepdims=True)
        inp = (inp - mean) / tf.sqrt(var + self.eps)

        inp = tf.reshape(inp, [-1, H, W, C])

        return inp * self.gamma + self.beta


# https://www.tensorflow.org/tutorials/customization/custom_layers
class resnet_block_type_a(tf.keras.Model):
    """
    Implementation of single resnet block as described in Deep Residual Learning for Image Recognition (https://arxiv.org/pdf/1512.03385.pdf)
    """

    def __init__(self, filters, stride, normalization, kernel_size=3):
        super(resnet_block_type_a, self).__init__(name='')
        self.stride = stride
        self.conv2a = tf.keras.layers.Conv2D(filters, kernel_size, strides=stride, padding='same', kernel_initializer=tf.keras.initializers.HeNormal(), bias_initializer=tf.keras.initializers.Zeros())
        if normalization == 'BN':
            self.norm2a = tf.keras.layers.BatchNormalization()
        else:
            self.norm2a = group_normalization_layer()

        self.conv2b = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', kernel_initializer=tf.keras.initializers.HeNormal(), bias_initializer=tf.keras.initializers.Zeros())
        if normalization == 'BN':
            self.norm2b = tf.keras.layers.BatchNormalization()
        else:
            self.norm2b = group_normalization_layer(gamma_inint='zeros')

        self.conv_downsample = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), strides=(2, 2), kernel_initializer=tf.keras.initializers.HeNormal(), bias_initializer=tf.keras.initializers.Zeros())

    def call(self, input_tensor, training=False, **kwargs):
        x = self.conv2a(input_tensor)
        x = self.norm2a(x, training=training)
        x = tf.keras.layers.ReLU()(x)

        x = self.conv2b(x)
        x = self.norm2b(x, training=training)

        if self.stride == 2:
            x = tf.keras.layers.add([x, self.conv_downsample(input_tensor)])
        else:
            x = tf.keras.layers.add([x , input_tensor])

        x = tf.keras.layers.ReLU()(x)
        return x

    def summary(self, **kwargs):
        x = tf.keras.layers.Input(shape=(32, 32, 16))
        model = tf.keras.Model(inputs=[x], outputs=self.call(x))
        return model.summary()

    def build_graph(self, **kwargs):
        x = tf.keras.layers.Input(shape=(32, 32, 16))
        model = tf.keras.Model(inputs=[x], outputs=self.call(x))
        return model


class res_net(tf.keras.Model):
    """
   Impleentation of the ResNet 6n+2, e.g.with n=3 ResNet20 and n=5 ResNet32 as described in Deep Residual Learning for Image Recognition (https://arxiv.org/pdf/1512.03385.pdf)
   :param norm: What normalization to use BN: for batch normalization or GN: for group normalization
   :return: Compiled ResNet20 model
   """

    def __init__(self, n, normalization, classification_classes, expe_data_aug, **kwargs):
        super(res_net, self).__init__(**kwargs)
        self.norm = normalization
        self.n = n
        self.out_class = classification_classes
        self.conv2init = tf.keras.layers.Conv2D(16, kernel_size=3, strides=1, padding='same', kernel_initializer=tf.keras.initializers.HeNormal(), bias_initializer=tf.keras.initializers.Zeros())

        self.model = []
        self.expe_data_aug = expe_data_aug
        if expe_data_aug:
            self.RF = tf.keras.layers.experimental.preprocessing.RandomFlip(mode='horizontal')
            self.pad = tf.keras.layers.ZeroPadding2D(padding=4)
            self.RC = tf.keras.layers.experimental.preprocessing.RandomCrop(32, 32)

        filters = [16, 32, 64]  # Taken from the ResNet paper
        for block_nr, filter in enumerate(filters):
            for layer_in_block in range(n):
                if layer_in_block == 0 and block_nr != 0:  # First layer in block and if not first filter block use projection shortcut
                    self.model.append(resnet_block_type_a(filters=filter, stride=2, normalization=self.norm))
                else:  # Second to las layers in block no projection shortcut
                    self.model.append(resnet_block_type_a(filters=filter, stride=1, normalization=self.norm))

        self.global_pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.classifier = tf.keras.layers.Dense(classification_classes)


    def call(self, inputs, training=None, **kwargs):
        if self.expe_data_aug:  # Data augmentation
            y = self.RF(inputs)
            y = self.pad(y)
            inputs = self.RC(y)

        y = self.conv2init(inputs)  # Initial convolution

        for res_layer in self.model:  # All resnet layers
            y = res_layer(y)

        global_pooled = self.global_pooling(y)  # poole out
        classified = self.classifier(global_pooled)  # classified out, no softmax since sparse categorical crossentropy works best without it.
        return classified

    def summary(self, **kwargs):
        x = tf.keras.layers.Input(shape=(32,32,3))
        model = tf.keras.Model( inputs=[x], outputs= self.call(x))
        return model.summary()

    def build_graph(self, **kwargs):
        x = tf.keras.layers.Input(shape=(32,32,3))
        model = tf.keras.Model( inputs=[x], outputs= self.call(x))
        return model




def get_resnet_n(arguments):
    """
    Implementation of the ResNet 20 as described in Deep Residual Learning for Image Recognition (https://arxiv.org/pdf/1512.03385.pdf)
    arguments['norm']: What normalization to use BN: for batch normalization or GN: for group normalization
    arguments['weight_decay']: use weight decay for training, i.e. choose between adam and adamW
    arguments['ResNet']: number of resnet blocks, e.g n=3 for Resnet20 and n=5 for ResNet32 as defined in Deep Residual Learning for Image Recognition (https://arxiv.org/pdf/1512.03385.pdf)
    :return: Compiled ResNet20 model
    """

    norm = arguments['norm']
    model = res_net(n=arguments['ResNet'], normalization=norm, classification_classes=10, expe_data_aug=arguments['experimental_data_aug'])

    if arguments['weight_decay']:
        optimizer = tfa.optimizers.SGDW(weight_decay=0.0001, momentum=0.9)
    else:
        optimizer = tf.keras.optimizers.Adam()

    print("ResNet block summary with projection shortcut, i.e. the las convolution is only used when projection shortcut is used")
    res_block = resnet_block_type_a(filters=16, stride=2, normalization=norm)
    res_block.summary()
    print("ResNet block summary without projection shortcut, i.e. the las convolution is only used when projection shortcut is used")
    res_block = resnet_block_type_a(filters=16, stride=2, normalization=norm)
    res_block.summary()

    print("\n \nSummary of complete ResNet model")
    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['sparse_categorical_accuracy'])
    model.summary()
    init_epoch = 0
    return model, init_epoch


def save_model(path, model, epoch):
    path = path + f"/epoch_{epoch:04d}"
    if not os.path.exists(path):
        os.makedirs(path)
    model.save_weights(path + '/weights.h5')
    symbolic_weights = getattr(model.optimizer, 'weights')
    weight_values = tf.keras.backend.batch_get_value(symbolic_weights)
    with open(path + "/optimizer.pkl", 'wb') as f:
        pickle.dump(weight_values, f)


def load_model(arguments, path):
    model, init_ep = get_resnet_n(arguments)
    model.load_weights(path + '/weights.h5')
    # model._make_train_function()
    with open(path + '/optimizer.pkl', 'rb') as f:
        weight_values = pickle.load(f)
    zero_grads = [tf.zeros_like(w) for w in model.trainable_variables]
    saved_vars = [tf.identity(w) for w in model.trainable_variables]
    model.optimizer.apply_gradients(zip(zero_grads, model.trainable_variables))

    [x.assign(y) for x, y in zip(model.trainable_variables, saved_vars)]
    model.optimizer.set_weights(weight_values)

    return model
