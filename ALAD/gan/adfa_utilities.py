import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

"""Class for ADFA GAN architecture.

Generator and discriminator.

"""


learning_rate = 0.00001
batch_size = 1024
layer = 1
latent_dim = 32
dis_inter_layer_dim = 128
init_kernel = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")

def generator(z_inp, is_training=False, getter=None, reuse=False):
    """ Generator architecture in tensorflow

    Generates data from the latent space

    Args:
        z_inp (tensor): variable in the latent space
        reuse (bool): sharing variables or not

    Returns:
        (tensor): last activation layer of the generator

    """
    with tf.compat.v1.variable_scope('generator', reuse=reuse, custom_getter=getter):

        name_net = 'layer_1'
        with tf.compat.v1.variable_scope(name_net):
            net = tf.compat.v1.layers.dense(z_inp,
                                  units=64,
                                  kernel_initializer=init_kernel,
                                  name='fc')
            net = tf.nn.relu(net, name='relu')

        name_net = 'layer_2'
        with tf.compat.v1.variable_scope(name_net):
            net = tf.compat.v1.layers.dense(net,
                                  units=128,
                                  kernel_initializer=init_kernel,
                                  name='fc')
            net = tf.nn.relu(net, name='relu')

        name_net = 'layer_4'
        with tf.compat.v1.variable_scope(name_net):
            net = tf.compat.v1.layers.dense(net,
                                  units=3,
                                  kernel_initializer=init_kernel,
                                  name='fc')

    return net

def discriminator(x_inp, is_training=False, getter=None, reuse=False):
    """ Discriminator architecture in tensorflow

    Discriminates between real data and generated data

    Args:
        x_inp (tensor): input data for the encoder.
        reuse (bool): sharing variables or not

    Returns:
        logits (tensor): last activation layer of the discriminator (shape 1)
        intermediate_layer (tensor): intermediate layer for feature matching

    """
    with tf.compat.v1.variable_scope('discriminator', reuse=reuse, custom_getter=getter):

        name_net = 'layer_1'
        with tf.compat.v1.variable_scope(name_net):
            net = tf.compat.v1.layers.dense(x_inp,
                                  units=256,
                                  kernel_initializer=init_kernel,
                                  name='fc')
            net = leakyReLu(net)
            net = tf.compat.v1.layers.dropout(net, rate=0.2, name='dropout',
                                  training=is_training)

        name_net = 'layer_2'
        with tf.compat.v1.variable_scope(name_net):
            net = tf.compat.v1.layers.dense(net,
                                  units=128,
                                  kernel_initializer=init_kernel,
                                  name='fc')
            net = leakyReLu(net)
            net = tf.compat.v1.layers.dropout(net, rate=0.2, name='dropout',
                                  training=is_training)

        name_net = 'layer_3'
        with tf.compat.v1.variable_scope(name_net):
            net = tf.compat.v1.layers.dense(net,
                                  units=dis_inter_layer_dim,
                                  kernel_initializer=init_kernel,
                                  name='fc')
            net = leakyReLu(net)
            net = tf.compat.v1.layers.dropout(net,
                                    rate=0.2,
                                    name='dropout',
                                    training=is_training)

        intermediate_layer = net

        name_net = 'layer_4'
        with tf.compat.v1.variable_scope(name_net):
            net = tf.compat.v1.layers.dense(net,
                                  units=1,
                                  kernel_initializer=init_kernel,
                                  name='fc')

        net = tf.squeeze(net)
        
        return net, intermediate_layer

def leakyReLu(x, alpha=0.1, name=None):
    if name:
        with tf.compat.v1.variable_scope(name):
            return _leakyReLu_impl(x, alpha)
    else:
        return _leakyReLu_impl(x, alpha)

def _leakyReLu_impl(x, alpha):
    return tf.nn.relu(x) - (alpha * tf.nn.relu(-x))
