import tensorflow as tf

"""
This is a list of possible combination of supported operators for backbone design.
Note that the limitation of design:
1. hardware does not support structure like maxpooling after maxpooling. See readme file for details
2. software does not support keras conv right now. You can try it but not promise working.

Fullhan Copy Right Reserved.

"""
def conv(input_tensor, filter_shape, stride, padding='SAME', conv_scope='conv'):
    """
    conv
    :param input_tensor:
    :param filter_shape: [filter_height, filter_width, in_channels, out_channels]
    :param stride:
    :param padding:
    :return: output
    """
    initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
    with tf.variable_scope(conv_scope):
        filter_kernel = tf.get_variable(name='weight', shape=filter_shape, initializer=initializer, dtype=tf.float32)
    output = tf.nn.conv2d(input_tensor, filter_kernel, strides=[1, stride, stride, 1], padding=padding)
    return output

def conv_bn_relu6(input_tensor, filter_shape, stride, padding='SAME', conv_scope='conv', bn_scope='bn'):
    """
    conv + batchnorm + relu6
    :param input_tensor:
    :param filter_shape: [filter_height, filter_width, in_channels, out_channels]
    :param stride:
    :param padding:
    :return: output
    """
    initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
    with tf.variable_scope(conv_scope):
        filter_kernel = tf.get_variable(name='weight', shape=filter_shape, initializer=initializer, dtype=tf.float32)
        conv_layer = tf.nn.conv2d(input_tensor, filter_kernel, strides=[1, stride, stride, 1], padding=padding)
    #bn_layer = tf.layers.batch_normalization(conv_layer, name='bn')
    initializer_mean = tf.constant_initializer(0.0)
    initializer_variance = tf.constant_initializer(1.0)
    initializer_scale = tf.constant_initializer(1.0)
    initializer_offset = tf.constant_initializer(0.0)
    with tf.variable_scope(bn_scope):
        mean = tf.get_variable('running_mean', shape=filter_shape[-1], initializer=initializer_mean, dtype=tf.float32)
        variance  = tf.get_variable('running_var',  shape=filter_shape[-1], initializer=initializer_variance, dtype=tf.float32)
        scale  = tf.get_variable('weight', shape=filter_shape[-1], initializer=initializer, dtype=tf.float32)
        offset = tf.get_variable('bias', shape=filter_shape[-1], initializer=initializer, dtype=tf.float32)  
        bn_layer, _, _ = tf.nn.fused_batch_norm(conv_layer, scale, offset, mean, variance, is_training=False, epsilon=1e-05)
    output = tf.nn.relu6(bn_layer)
    return output

def conv_bn(input_tensor, filter_shape, stride, padding='SAME', conv_scope='conv', bn_scope='bn'):
    """
    conv + batchnorm
    :param input_tensor:
    :param filter_shape: [filter_height, filter_width, in_channels, out_channels]
    :param stride:
    :param padding:
    :return: output
    """
    initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
    with tf.variable_scope(conv_scope):
        filter_kernel = tf.get_variable(name='weight', shape=filter_shape, initializer=initializer, dtype=tf.float32)
    conv_layer = tf.nn.conv2d(input_tensor, filter_kernel, strides=[1, stride, stride, 1], padding=padding)
    #output = tf.layers.batch_normalization(conv_layer, name='bn')
    with tf.variable_scope(bn_scope):
        mean = tf.get_variable('running_mean', shape=filter_shape[-1], initializer=initializer_mean, dtype=tf.float32)
        variance  = tf.get_variable('running_var',  shape=filter_shape[-1], initializer=initializer_variance, dtype=tf.float32)
        scale  = tf.get_variable('weight', shape=filter_shape[-1], initializer=initializer, dtype=tf.float32)
        offset = tf.get_variable('bias', shape=filter_shape[-1], initializer=initializer, dtype=tf.float32)  
    output, _, _ = tf.nn.fused_batch_norm(conv_layer, scale, offset, mean, variance, is_training=False, epsilon=1e-05)    
    return output

def conv_ba(input_tensor, filter_shape, stride, padding='SAME', conv_scope='conv'):
    """
    conv + bias
    :param input_tensor:
    :param filter_shape: [filter_height, filter_width, in_channels, out_channels]
    :param stride:
    :param padding:
    :return: output
    """
    initializer = tf.glorot_uniform_initializer()
    with tf.variable_scope(conv_scope):
        filter_kernel = tf.get_variable(name='weight', shape=filter_shape, initializer=initializer, dtype=tf.float32)
        biases = tf.get_variable('bias', shape=filter_shape[-1], initializer=initializer, dtype=tf.float32)
        conv_layer = tf.nn.conv2d(input_tensor, filter_kernel, strides=[1, stride, stride, 1], padding=padding)
        output = tf.nn.bias_add(conv_layer, biases)
    return output

def conv_ba_bn(input_tensor, filter_shape, stride, padding='SAME', conv_scope='conv', bn_scope='bn'):
    """
    conv + bias + batchnorm
    :param input_tensor:
    :param filter_shape: [filter_height, filter_width, in_channels, out_channels]
    :param stride:
    :param padding:
    :return: output
    """
    initializer = tf.glorot_uniform_initializer()
    with tf.variable_scope(conv_scope):
        filter_kernel = tf.get_variable(name='weight', shape=filter_shape, initializer=initializer, dtype=tf.float32)
        biases = tf.get_variable('bias', shape=filter_shape[-1], initializer=initializer, dtype=tf.float32)
    conv_layer = tf.nn.conv2d(input_tensor, filter_kernel, strides=[1, stride, stride, 1], padding=padding)
    conv_bias_layer = tf.nn.bias_add(conv_layer, biases)
    #bn_layer = tf.layers.batch_normalization(conv_bias_layer, name='bn')
    initializer_mean = tf.constant_initializer(0.0)
    initializer_variance = tf.constant_initializer(1.0)
    initializer_scale = tf.constant_initializer(1.0)
    initializer_offset = tf.constant_initializer(0.0)
    with tf.variable_scope(bn_scope):
        mean = tf.get_variable('running_mean', shape=filter_shape[-1], initializer=initializer_mean, dtype=tf.float32)
        variance  = tf.get_variable('running_var',  shape=filter_shape[-1], initializer=initializer_variance, dtype=tf.float32)
        scale  = tf.get_variable('weight', shape=filter_shape[-1], initializer=initializer, dtype=tf.float32)
        offset = tf.get_variable('bias', shape=filter_shape[-1], initializer=initializer, dtype=tf.float32)    
    bn_layer, _, _ = tf.nn.fused_batch_norm(conv_bias_layer, scale, offset, mean, variance, is_training=False, epsilon=1e-05)
    return bn_layer

def conv_ba_relu6(input_tensor, filter_shape, stride, padding='SAME', conv_scope='conv'):
    """
    conv + bias + relu6
    :param input_tensor:
    :param filter_shape: [filter_height, filter_width, in_channels, out_channels]
    :param stride:
    :param padding:
    :return: output
    """
    initializer = tf.glorot_uniform_initializer()
    with tf.variable_scope(conv_scope):
        filter_kernel = tf.get_variable(name='weight', shape=filter_shape, initializer=initializer, dtype=tf.float32)
        biases = tf.get_variable('bias', shape=filter_shape[-1], initializer=initializer, dtype=tf.float32)
    conv_layer = tf.nn.conv2d(input_tensor, filter_kernel, strides=[1, stride, stride, 1], padding=padding)
    conv_bias_layer = tf.nn.bias_add(conv_layer, biases)
    output = tf.nn.relu6(conv_bias_layer)
    return output

def conv_ba_bn_relu6(input_tensor, filter_shape, stride, padding='SAME', conv_scope='conv', bn_scope='bn'):
    """
    conv + bias + batchnorm + relu6
    :param input_tensor:
    :param filter_shape: [filter_height, filter_width, in_channels, out_channels]
    :param stride:
    :param padding:
    :return: output
    """
    initializer = tf.glorot_uniform_initializer()
    with tf.variable_scope(conv_scope):
        filter_kernel = tf.get_variable(name='weight', shape=filter_shape, initializer=initializer, dtype=tf.float32)
        biases = tf.get_variable('bias', shape=filter_shape[-1], initializer=initializer, dtype=tf.float32)
    conv_layer = tf.nn.conv2d(input_tensor, filter_kernel, strides=[1, stride, stride, 1], padding=padding)
    conv_bias_layer = tf.nn.bias_add(conv_layer, biases)
    #bn_layer = tf.layers.batch_normalization(conv_bias_layer, name='bn')
    initializer_mean = tf.constant_initializer(0.0)
    initializer_variance = tf.constant_initializer(1.0)
    initializer_scale = tf.constant_initializer(1.0)
    initializer_offset = tf.constant_initializer(0.0)
    with tf.variable_scope(bn_scope):
        mean = tf.get_variable('running_mean', shape=filter_shape[-1], initializer=initializer_mean, dtype=tf.float32)
        variance  = tf.get_variable('running_var',  shape=filter_shape[-1], initializer=initializer_variance, dtype=tf.float32)
        scale  = tf.get_variable('weight', shape=filter_shape[-1], initializer=initializer, dtype=tf.float32)
        offset = tf.get_variable('bias', shape=filter_shape[-1], initializer=initializer, dtype=tf.float32)    
    bn_layer, _, _ = tf.nn.fused_batch_norm(conv_bias_layer, scale, offset, mean, variance, is_training=False, epsilon=1e-05)    
    output = tf.nn.relu6(bn_layer)
    return output

def conv_ba_relu(input_tensor, filter_shape, stride, padding='SAME', conv_scope='conv'):
    """
    conv + bias + relu
    :param input_tensor:
    :param filter_shape: [filter_height, filter_width, in_channels, out_channels]
    :param stride:
    :param padding:
    :return: output
    """
    initializer = tf.glorot_uniform_initializer()
    with tf.variable_scope(conv_scope):
        filter_kernel = tf.get_variable(name='weight', shape=filter_shape, initializer=initializer, dtype=tf.float32)
        biases = tf.get_variable('bias', shape=filter_shape[-1], initializer=initializer, dtype=tf.float32)
    conv_layer = tf.nn.conv2d(input_tensor, filter_kernel, strides=[1, stride, stride, 1], padding=padding)
    conv_bias_layer = tf.nn.bias_add(conv_layer, biases)
    output = tf.nn.relu(conv_bias_layer)
    return output

def conv_ba_bn_relu(input_tensor, filter_shape, stride, padding='SAME', conv_scope='conv', bn_scope='bn'):
    """
    conv + bias + batchnorm + relu
    :param input_tensor:
    :param filter_shape: [filter_height, filter_width, in_channels, out_channels]
    :param stride:
    :param padding:
    :return: output
    """
    initializer = tf.glorot_uniform_initializer()
    with tf.variable_scope(conv_scope):
        filter_kernel = tf.get_variable(name='weight', shape=filter_shape, initializer=initializer, dtype=tf.float32)
        biases = tf.get_variable('bias', shape=filter_shape[-1], initializer=initializer, dtype=tf.float32)
    conv_layer = tf.nn.conv2d(input_tensor, filter_kernel, strides=[1, stride, stride, 1], padding=padding)
    conv_bias_layer = tf.nn.bias_add(conv_layer, biases)
    #bn_layer = tf.layers.batch_normalization(conv_bias_layer, name='bn')
    initializer_mean = tf.constant_initializer(0.0)
    initializer_variance = tf.constant_initializer(1.0)
    initializer_scale = tf.constant_initializer(1.0)
    initializer_offset = tf.constant_initializer(0.0)
    with tf.variable_scope(bn_scope):
        mean = tf.get_variable('running_mean', shape=filter_shape[-1], initializer=initializer_mean, dtype=tf.float32)
        variance  = tf.get_variable('running_var',  shape=filter_shape[-1], initializer=initializer_variance, dtype=tf.float32)
        scale  = tf.get_variable('weight', shape=filter_shape[-1], initializer=initializer, dtype=tf.float32)
        offset = tf.get_variable('bias', shape=filter_shape[-1], initializer=initializer, dtype=tf.float32)    
    bn_layer, _, _ = tf.nn.fused_batch_norm(conv_bias_layer, scale, offset, mean, variance, is_training=False, epsilon=1e-05)      
    output = tf.nn.relu(bn_layer)
    return output

def conv_bn_relu(input_tensor, filter_shape, stride, padding='SAME', conv_scope='conv', bn_scope='bn'):
    """
    conv + batchnorm + relu
    :param input_tensor:
    :param filter_shape: [filter_height, filter_width, in_channels, out_channels]
    :param stride:
    :param padding:
    :return: output
    """
    initializer = tf.glorot_uniform_initializer()
    with tf.variable_scope(conv_scope):
        filter_kernel = tf.get_variable(name='weight', shape=filter_shape, initializer=initializer, dtype=tf.float32)
    conv_layer = tf.nn.conv2d(input_tensor, filter_kernel, strides=[1, stride, stride, 1], padding=padding)
    #bn_layer = tf.layers.batch_normalization(conv_layer, name='bn')
    initializer_mean = tf.constant_initializer(0.0)
    initializer_variance = tf.constant_initializer(1.0)
    initializer_scale = tf.constant_initializer(1.0)
    initializer_offset = tf.constant_initializer(0.0)
    with tf.variable_scope(bn_scope):
        mean = tf.get_variable('running_mean', shape=filter_shape[-1], initializer=initializer_mean, dtype=tf.float32)
        variance  = tf.get_variable('running_var',  shape=filter_shape[-1], initializer=initializer_variance, dtype=tf.float32)
        scale  = tf.get_variable('weight', shape=filter_shape[-1], initializer=initializer, dtype=tf.float32)
        offset = tf.get_variable('bias', shape=filter_shape[-1], initializer=initializer, dtype=tf.float32)    
    bn_layer, _, _ = tf.nn.fused_batch_norm(conv_layer, scale, offset, mean, variance, is_training=False, epsilon=1e-05)     
    output = tf.nn.relu(bn_layer)
    return output


def maxpool2x2(input):
    output = tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return output

def print_layer_info(name, module, input_tensor, output_tensor):
    # 提取卷积层的参数
    conv_layer = module.conv  # 假设你的 QMConv 内部卷积对象叫 self.conv
    k = conv_layer.kernel_size[0]
    s = conv_layer.stride[0]

    print(f"Layer: {name:10} | "
          f"In: {str(list(input_tensor.shape)):18} | "
          f"Out: {str(list(output_tensor.shape)):18} | "
          f"K: {k} | S: {s}")

