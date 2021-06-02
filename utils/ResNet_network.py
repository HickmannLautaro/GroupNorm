import os
import pickle

import tensorflow as tf
import tensorflow_addons as tfa


class group_normalization_layer(tf.keras.layers.Layer):
    """
    Implementation of GroupNorm ("Group Normalization" - https://arxiv.org/abs/1803.08494)
    This layer is based on the modified code proposed in the paper, so that it can be used in channel last configuration
    """

    def __init__(self, gamma_initialization='ones', groups=8, eps=0.00001):
        """
        Layer initialization
        @param gamma_initialization: how to initialize the gamma values. Specifically for the last layer they should be init with 0s otherwise with 1s
        @param groups: in how many groups to divide the input channels. Per default 8, since in the paper for 64 filters 32 groups were used and ResNet20 uses 16 filters, i.e. groups = filters/2
        @param eps: small constant to avoid sqrt of 0
        """
        super(group_normalization_layer, self).__init__()
        self.eps = eps
        # G: number of groups for GN
        self.G = groups
        # gamma, beta: scale and offset, with shape [1,1,1, C]
        self.gamma_initializer = gamma_initialization

    def build(self, input_shape):
        """
        Initializes gamma and betta depending on the input shape. This can't be done in the initialization since the input shape is not known
        @param input_shape: tensor containing the input dimensions
        """
        # input_shape: N,H,W,C
        # gamma, beta: scale and offset, with shape [1,1,1, C]

        C = input_shape[-1]  # Channels last

        # Custom layer add weights
        self.gamma = self.add_weight("Gamma", shape=[1, 1, 1, C], initializer=self.gamma_initializer)
        self.beta = self.add_weight("Beta", shape=[1, 1, 1, C], initializer='zeros')

    def call(self, inp, **kwargs):
        """
        Forward normalizing pass of GN, as described in the paper, modified to accommodate channels last.
        @param inp: input tensor
        @return: normalized input tensor
        """
        N, H, W, C = inp.shape
        inp = tf.reshape(inp, [-1, H, W, self.G, C // self.G])
        mean, var = tf.nn.moments(inp, [1, 2, 4], keepdims=True)
        inp = (inp - mean) / tf.sqrt(var + self.eps)

        inp = tf.reshape(inp, [-1, H, W, C])

        return inp * self.gamma + self.beta


# https://www.tensorflow.org/tutorials/customization/custom_layers
class resnet_block_type_a(tf.keras.Model):
    """
    Implementation of single resnet block of type A as described in Deep Residual Learning for Image Recognition (https://arxiv.org/pdf/1512.03385.pdf) in section 4.2. CIFAR-10 and Analysis.
    """

    def __init__(self, filters, stride, normalization, kernel_size=3):
        """
        Initialization of the class
        @param filters: filters for the block
        @param stride: 1 skip connection, 2 shortcut connection
        @param normalization: what normalization layer to use
        @param kernel_size: convolution kernel size set to 3 per default, and not changed/
        """
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
            self.norm2b = group_normalization_layer(gamma_initialization='zeros')

        # Downsampler for connection shortcut. It could be made conditional to save space but it would make all saved models useless
        self.conv_downsample = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), strides=(2, 2), kernel_initializer=tf.keras.initializers.HeNormal(), bias_initializer=tf.keras.initializers.Zeros())

    def call(self, input_tensor, training=False, **kwargs):
        """
        Forward pass of a resnet block
        @param input_tensor: input data
        @param training: For batch norm this is needed, set automatically by the keras fit() and evaluate() functions.
        @return: results of the input tensor passed by a resnet block
        """
        x = self.conv2a(input_tensor)
        x = self.norm2a(x, training=training)
        x = tf.keras.layers.ReLU()(x)

        x = self.conv2b(x)
        x = self.norm2b(x, training=training)

        if self.stride == 2:  # Projection shortcut
            x = tf.keras.layers.add([x, self.conv_downsample(input_tensor)])
        else:  # Skip connection
            x = tf.keras.layers.add([x, input_tensor])

        x = tf.keras.layers.ReLU()(x)
        return x

    def summary(self, **kwargs):
        """
        Function needed to get a summary when using a subclassing model and calling resnet_block_type_a.summary().
        """
        x = tf.keras.layers.Input(shape=(32, 32, 16))
        model = tf.keras.Model(inputs=[x], outputs=self.call(x))
        return model.summary()

    def build_graph(self):
        """
        Function needed to get a model plot when using a subclassing model and calling tf.keras.utils.plot_model .
        """
        x = tf.keras.layers.Input(shape=(32, 32, 16))
        model = tf.keras.Model(inputs=[x], outputs=self.call(x))
        return model


class res_net(tf.keras.Model):
    """
    Implementation of the ResNet 6n+2, e.g.with n=3 ResNet20 and n=5 ResNet32 as described in Deep Residual Learning for Image Recognition (https://arxiv.org/pdf/1512.03385.pdf)
    """

    def __init__(self, n, normalization, classification_classes, experimental_data_aug, old_struc, **kwargs):
        """
        Initialize resnet model
        @param n: with n=3 ResNet20 and n=5 ResNet32 as described in Deep Residual Learning for Image Recognition (https://arxiv.org/pdf/1512.03385.pdf)
        @param normalization: what normalization to use BN: for batch normalization or GN: for group normalization
        @param classification_classes: output classes 10 for CIFAR-10
        @param experimental_data_aug: test using keras experimental layer for preprocessing
        @param old_struc: I changed the layer order and some saved models have different order. To be able to load them both structures can be made. In the model architecture nothing changes just the order in this method.
        """
        super(res_net, self).__init__(**kwargs)
        self.norm = normalization
        self.n = n
        self.out_class = classification_classes
        self.conv2init = tf.keras.layers.Conv2D(16, kernel_size=3, strides=1, padding='same', kernel_initializer=tf.keras.initializers.HeNormal(), bias_initializer=tf.keras.initializers.Zeros())
        if old_struc:  # When saving the weights keras takes into account the initialization order, this variable enables to load the old models.
            self.global_pooling = tf.keras.layers.GlobalAveragePooling2D()
            self.classifier = tf.keras.layers.Dense(classification_classes)
        self.model = []
        self.experimental_data_aug = experimental_data_aug
        if experimental_data_aug:
            self.RF = tf.keras.layers.experimental.preprocessing.RandomFlip(mode='horizontal')
            self.pad = tf.keras.layers.ZeroPadding2D(padding=4)
            self.RC = tf.keras.layers.experimental.preprocessing.RandomCrop(32, 32)

        filters = [16, 32, 64]  # Taken from the ResNet paper

        for block_nr, filter_block in enumerate(filters):
            for layer_in_block in range(n):
                if layer_in_block == 0 and block_nr != 0:  # First layer in block and if not first filter block use projection shortcut
                    self.model.append(resnet_block_type_a(filters=filter_block, stride=2, normalization=self.norm))
                else:  # Second to las layers in block no projection shortcut
                    self.model.append(resnet_block_type_a(filters=filter_block, stride=1, normalization=self.norm))

        if not old_struc:
            self.global_pooling = tf.keras.layers.GlobalAveragePooling2D()
            self.classifier = tf.keras.layers.Dense(classification_classes)

    def call(self, inputs, training=None, **kwargs):
        """
        Forward pass over the resnet
        @param inputs: input tensor
        @param training: For batch norm this is needed, set automatically by the keras fit() and evaluate() functions.
        @return: Classifications of input tensor
        """
        if self.experimental_data_aug:  # Data augmentation
            y = self.RF(inputs)
            y = self.pad(y)
            inputs = self.RC(y)

        y = self.conv2init(inputs)  # Initial convolution

        for res_layer in self.model:  # All resnet layers
            y = res_layer(y)

        global_pooled = self.global_pooling(y)  # poole out
        classified = self.classifier(global_pooled)  # classified out, no softmax since sparse categorical cross entropy works best without it.
        return classified

    def summary(self, **kwargs):
        """
        Function needed to get a summary when using a subclassing model and calling res_net.summary().
        """
        x = tf.keras.layers.Input(shape=(32, 32, 3))
        model = tf.keras.Model(inputs=[x], outputs=self.call(x))
        return model.summary()

    def build_graph(self):
        """
        Function needed to get a model plot when using a subclassing model and calling tf.keras.utils.plot_model .
        """
        x = tf.keras.layers.Input(shape=(32, 32, 3))
        model = tf.keras.Model(inputs=[x], outputs=self.call(x))
        return model


def get_resnet_n(arguments, print_info=True, old_struc=False):
    """
    Implementation of the ResNet as described in Deep Residual Learning for Image Recognition (https://arxiv.org/pdf/1512.03385.pdf)
      @param    arguments: arguments['norm']: What normalization to use BN: for batch normalization or GN: for group normalization
                arguments['weight_decay']: use weight decay for training, i.e. choose between adam and adamW
                arguments['ResNet']: number of resnet blocks, e.g n=3 for Resnet20 and n=5 for ResNet32 as defined in Deep Residual Learning for Image Recognition (https://arxiv.org/pdf/1512.03385.pdf)
    @param print_info: If plotting dont show the model summaries.
    @param old_struc: I changed the layer order and some saved models have different order. To be able to load them both structures can be made. In the model architecture nothing changes just the order in this method.
    @return:  Compiled ResNet20 model
    """

    norm = arguments['norm']

    if arguments['weight_decay']:
        optimizer = tfa.optimizers.SGDW(weight_decay=0.0001, momentum=0.9)
    else:
        optimizer = tf.keras.optimizers.Adam()

    if print_info:
        print("ResNet block summary with projection shortcut, i.e. the las convolution is only used when projection shortcut is used")
        res_block = resnet_block_type_a(filters=16, stride=2, normalization=norm)
        res_block.summary()
        print("ResNet block summary without projection shortcut, i.e. the las convolution is only used when projection shortcut is used")
        res_block = resnet_block_type_a(filters=16, stride=2, normalization=norm)
        res_block.summary()

    # Create model
    model = res_net(n=arguments['ResNet'], normalization=norm, classification_classes=10, experimental_data_aug=arguments['experimental_data_aug'], old_struc=old_struc)
    # Compile model
    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['sparse_categorical_accuracy'])

    if print_info:
        print("\n \nSummary of complete ResNet model")
        model.summary()

    init_epoch = 0  # Since new model is created start training form 0
    return model, init_epoch


def save_model(path, model, epoch):
    """
    Custom save function, saves model weights as .h5 and optimizer as .pkl
    @param path: where to save to
    @param model: model object
    @param epoch: current epoch
    """
    # Create new folder for current epoch
    path = path + f"/epoch_{epoch:04d}"
    if not os.path.exists(path):
        os.makedirs(path)
    # Save model weights
    model.save_weights(path + '/weights.h5')
    # Get and save optimizer
    symbolic_weights = getattr(model.optimizer, 'weights')
    weight_values = tf.keras.backend.batch_get_value(symbolic_weights)
    with open(path + "/optimizer.pkl", 'wb') as f:
        pickle.dump(weight_values, f)


def load_model_with_optimizer(arguments, path):
    """
    Custom function to create model and, load weights and optimizer from file
    @param arguments: arguments to create the resnet model
    @param path: where to find the files
    @return: compiled model with loaded optimizer state and weights
    """
    # Create and build model
    model, init_ep = get_resnet_n(arguments)
    model.build(input_shape=(None, 32, 32, 3))  # Build for visualization
    # Load weights
    model.load_weights(path + '/weights.h5')

    # Initialize optimizer values
    zero_grads = [tf.zeros_like(w) for w in model.trainable_variables]
    saved_vars = [tf.identity(w) for w in model.trainable_variables]
    model.optimizer.apply_gradients(zip(zero_grads, model.trainable_variables))
    [x.assign(y) for x, y in zip(model.trainable_variables, saved_vars)]

    # Load optimizer values
    with open(path + '/optimizer.pkl', 'rb') as f:
        weight_values = pickle.load(f)
    model.optimizer.set_weights(weight_values)

    return model


def load_model_weights(arguments, path, old_struc):
    """
    Custom function to create model and, load weights  from file
    @param arguments: arguments to create the resnet model
    @param path: where to find the files
    @param old_struc: I changed the layer order and some saved models have different order. To be able to load them both structures can be made. In the model architecture nothing changes just the order in this method.
    @return: compiled model with loaded weights
    """
    model, init_ep = get_resnet_n(arguments, print_info=False, old_struc=old_struc)
    model.build(input_shape=(None, 32, 32, 3))  # Build for visualization
    model.load_weights(path + '/weights.h5')
    return model
