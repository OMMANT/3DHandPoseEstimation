import tensorflow as tf

class Conv(tf.keras.layers.Layer):
    def __init__(self, name, kernel_size, stride, filters, trainable=True):
        super(Conv, self).__init__(name=name)
        self.kernel_size = kernel_size
        self.filters = filters
        self.strides = [1, stride, stride, 1]
        self.trainable = trainable

    def build(self, input_shape):
        init = tf.keras.initializers.GlorotUniform()
        kernel_shape = [self.kernel_size, self.kernel_size, input_shape[3], self.filters]
        self.kernel = tf.Variable(name='kernel', initial_value=init(shape=kernel_shape,
                                                     dtype=tf.float32), trainable=self.trainable)
        self.biases = tf.Variable(name='biases', initial_value=tf.keras.initializers.Constant(1e-4)(shape=[kernel_shape[3]]),
                                  trainable=self.trainable)
    def call(self, inputs):
        tmp_result = tf.nn.conv2d(inputs, self.kernel, self.strides, padding='SAME')
        return tf.nn.bias_add(tmp_result, self.biases, name='out')


class Conv_relu(tf.keras.layers.Layer):
    def __init__(self, name, kernel_size, stride, filters, trainable=True):
        super(Conv_relu, self).__init__(name=name)
        self.kernel_size = kernel_size
        self.filters = filters
        self.strides = [1, stride, stride, 1]
        self.trainable = trainable

    def build(self, input_shape):
        init = tf.keras.initializers.GlorotUniform()
        kernel_shape = [self.kernel_size, self.kernel_size, input_shape[3], self.filters]
        self.kernel = tf.Variable(name='kernel', initial_value=init(shape=kernel_shape,
                                                     dtype=tf.float32), trainable=self.trainable)
        self.biases = tf.Variable(name='biases', initial_value=tf.keras.initializers.Constant(1e-4)(shape=[kernel_shape[3]]),
                                  trainable=self.trainable)
    def call(self, inputs):
        tmp_result = tf.nn.conv2d(inputs, self.kernel, self.strides, padding='SAME')
        tensor = tf.nn.bias_add(tmp_result, self.biases, name='conv_out')
        return tf.maximum(tensor, tensor*.01, name='relu_out')

class Max_pool(tf.keras.layers.Layer):
    def __init__(self, name):
        super(Max_pool, self).__init__(name=name)

    def call(self, inputs):
        return tf.nn.max_pool(inputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool')