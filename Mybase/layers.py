import copy
import math
import numpy as np
import tensorflow as tf
#from tensorflow.python import control_flow_ops

#tf.GraphKeys.REGULARIZATION_LOSSES 
#tf.GraphKeys.TRAINABLE_VARIABLES
#tf.GraphKeys.GLOBAL_VARIABLES

def get_shape(x, rank=None):
    """Returns the dimensions of a Tensor as list of integers or scale tensors.
    Args:
      x: N-d Tensor;
      rank: Rank of the Tensor. If None, will try to guess it.
    Returns:
      A list of `[d1, d2, ..., dN]` corresponding to the dimensions of the
        input tensor.  Dimensions that are statically known are python integers,
        otherwise they are integer scalar tensors.
    """
    if x.get_shape().is_fully_defined():
        return x.get_shape().as_list()
    else:
        static_shape = x.get_shape()
        if rank is None:
            static_shape = static_shape.as_list()
            rank = len(static_shape)
        else:
            static_shape = x.get_shape().with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(x), rank)
        return [s if s is not None else d for s, d in zip(static_shape, dynamic_shape)]

    
def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())


def relu1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    alpha = params["relu"]["alpha"]
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
        
    with tf.variable_scope("relu1_"+str(layer)) as scope:
        if alpha > 0:
            tensor_out = tf.nn.leaky_relu(tensor_in, alpha)
        else:
            tensor_out = tf.nn.relu(tensor_in)
        print_activations(tensor_out)
    return tensor_out


def conv1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    reg       = params["com"]["reg"]
    wscale    = params["com"]["wscale"]
    dtype     = params["com"]["dtype"]
    reuse     = params["com"]["reuse"]
    is_train  = params["com"]["is_train"]
    trainable = params["com"]["trainable"]
    
    number    = params["conv"]["number"]
    shape     = params["conv"]["shape"]
    rate      = params["conv"]["rate"]
    stride    = params["conv"]["stride"]
    padding   = params["conv"]["padding"]
    use_bias  = params["conv"]["use_bias"]
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    #x_shape = tensor_in.get_shape().as_list()
    x_shape = get_shape(tensor_in)
    kernel_shape  = [shape[0], shape[1], x_shape[3], number]
    kernel_stride = [1, stride[0], stride[1], 1]
    
    with tf.variable_scope("conv1_"+str(layer), reuse=reuse) as scope:
        kernel = tf.get_variable(name="weights", shape=kernel_shape, dtype=dtype, \
                                 #initializer=tf.truncated_normal_initializer(stddev=wscale), \
                                 initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,mode='FAN_AVG',uniform=True), 
                                 #initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32),
                                 regularizer=tf.contrib.layers.l2_regularizer(reg), \
                                 trainable=trainable)
        if use_bias:
            biases = tf.get_variable(name="biases", shape=[number], dtype=dtype, \
                                     initializer=tf.constant_initializer(0.0), \
                                     trainable=trainable)
        if rate == 1:
            conv = tf.nn.conv2d(tensor_in, kernel, kernel_stride, padding=padding)
        else:
            conv = tf.nn.atrous_conv2d(tensor_in, kernel, rate, padding=padding)
        if use_bias:
            tensor_out = tf.nn.bias_add(conv, biases)
        else:
            tensor_out = conv
        #tf.summary.histogram("conv", tensor_out)
        print_activations(tensor_out)
    return tensor_out 


def deconv1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    reg       = params["com"]["reg"]
    wscale    = params["com"]["wscale"]
    dtype     = params["com"]["dtype"]
    reuse     = params["com"]["reuse"]
    is_train  = params["com"]["is_train"]
    trainable = params["com"]["trainable"]
    
    number    = params["deconv"]["number"]
    shape     = params["deconv"]["shape"]
    out_shape = params["deconv"]["out_shape"]
    rate      = params["deconv"]["rate"]
    stride    = params["deconv"]["stride"]
    padding   = params["deconv"]["padding"]
    use_bias  = params["deconv"]["use_bias"]
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    #x_shape = tensor_in.get_shape().as_list()
    x_shape = get_shape(tensor_in)
    kernel_shape  = [shape[0], shape[1], number, x_shape[3]]
    kernel_stride = [1, stride[0], stride[1], 1]
    y_shape = [x_shape[0], out_shape[0], out_shape[1], number]

    with tf.variable_scope("deconv1_"+str(layer), reuse=reuse) as scope:
        kernel = tf.get_variable(name="weights", shape=kernel_shape, dtype=dtype, \
                                 #initializer=tf.truncated_normal_initializer(stddev=wscale), \
                                 initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,mode='FAN_AVG',uniform=True), 
                                 regularizer=tf.contrib.layers.l2_regularizer(reg), \
                                 trainable=trainable)
        if use_bias:
            biases = tf.get_variable(name="biases", shape=[number], dtype=dtype, \
                                     initializer=tf.constant_initializer(0.0), \
                                     trainable=trainable)
        if rate == 1:
            deconv = tf.nn.conv2d_transpose(tensor_in, kernel, y_shape, kernel_stride, padding=padding)
        else:
            deconv = tf.nn.atrous_conv2d_transpose(tensor_in, kernel, y_shape, rate, padding=padding)
        if use_bias:
            tensor_out = tf.nn.bias_add(deconv, biases)
        else:
            tensor_out = deconv
        #tf.summary.histogram("conv", tensor_out)
        print_activations(tensor_out)
    return tensor_out


def conv_relu1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    with tf.variable_scope("conv_relu1_"+str(layer)) as scope:
        bias = conv1(tensor_in, 0, params, mtrain)
        tensor_out = relu1(bias, 0, params, mtrain) 
    return tensor_out 

'''
def conv_sigmoid1(tensor_in=None, layer=0, params=None, mtrain=None):
    with tf.variable_scope("conv_sigmoid1_"+str(layer)) as scope:
        bias = conv1(tensor_in, 0, params, mtrain)
        tensor_out = tf.sigmoid(bias)
        print_activations(tensor_out)
    return tensor_out 
'''

def deconv_relu1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    with tf.variable_scope("deconv_relu1_"+str(layer)) as scope:
        bias = deconv1(tensor_in, 0, params, mtrain)
        tensor_out = relu1(bias, 0, params, mtrain)
    return tensor_out

'''
def deconv_sigmoid1(tensor_in=None, layer=0, params=None, mtrain=None):
    with tf.variable_scope("deconv_sigmoid1_"+str(layer)) as scope:
        bias = deconv1(tensor_in, 0, params, mtrain)
        tensor_out = tf.sigmoid(bias) 
        print_activations(tensor_out)
    return tensor_out
'''

def dropout1(tensor_in=None, layer=0, params=None, mtrain=None):
    '''
    Args:
        keep_prob:   A scalar Tensor with the same type as x. The probability that each element is kept.
        noise_shape: A 1-D Tensor of type int32, representing the shape for randomly generated keep/drop flags.
    Notes:
        With probability keep_prob, outputs the input element scaled up by 1 / keep_prob, otherwise outputs 0. 
        The scaling is so that the expected sum is unchanged.
        By default, each element is kept or dropped independently. If noise_shape is specified, it must be 
        broadcastable to the shape of x, and only dimensions with noise_shape[i] == shape(x)[i] will make independent 
        decisions. For example, if shape(x) = [k, l, m, n] and noise_shape = [k, 1, 1, n], each batch and channel 
        component will be kept independently and each row and column will be kept or not kept together.
    '''
    keep_p = params["dropout"]["keep_p"]
    shape  = params["dropout"]["shape" ]

    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
        
    with tf.variable_scope("dropout1_"+str(layer)) as scope:
        tensor_out = tf.cond(mtrain, lambda: tf.nn.dropout(tensor_in, keep_p, shape), lambda: tf.identity(tensor_in))
        print_activations(tensor_out)    
    return tensor_out


def conv_relu_dropout1(tensor_in=None, layer=0, params=None, mtrain=None):

    with tf.variable_scope("conv_relu_dropout1_"+str(layer)) as scope:
        relu = conv_relu1(tensor_in, 0, params, mtrain)
        tensor_out = dropout1(relu,  0, params, mtrain)
    return tensor_out


def max_pool1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    shape  = params["max_pool"]["shape"]
    stride = params["max_pool"]["stride"]
    padding= params["max_pool"]["padding"]
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    kernel_shape  = [1,  shape[0],  shape[1], 1]
    kernel_stride = [1, stride[0], stride[1], 1]
    
    with tf.variable_scope("max_pool1_"+str(layer)) as scope:
        tensor_out = tf.nn.max_pool(tensor_in, ksize=kernel_shape, strides=kernel_stride, padding=padding)
        print_activations(tensor_out)
    return tensor_out 


def avg_pool1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    shape  = params["avg_pool"]["shape"]
    stride = params["avg_pool"]["stride"]
    padding= params["avg_pool"]["padding"]
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    kernel_shape  = [1,  shape[0],  shape[1], 1]
    kernel_stride = [1, stride[0], stride[1], 1]
    
    with tf.variable_scope("avg_pool1_"+str(layer)) as scope:
        tensor_out = tf.nn.avg_pool(tensor_in, ksize=kernel_shape, strides=kernel_stride, padding=padding)
        print_activations(tensor_out)
    return tensor_out



def glb_pool1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    axis  = params["glb_pool"]["axis"]
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    with tf.variable_scope("glb_pool1_"+str(layer)) as scope:
        tensor_out = tf.reduce_mean(tensor_in, axis=axis, keepdims=True)
        print_activations(tensor_out)
    return tensor_out



def conv_relu_max_pool1(tensor_in=None, layer=0, params=None, mtrain=None):

    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
        
    with tf.variable_scope("conv_relu_max_pool1_"+str(layer)) as scope:
        relu = conv_relu1(tensor_in, 0, params, mtrain)
        tensor_out = max_pool1(relu, 0, params, mtrain)
    return tensor_out


def bilstm1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    reg       = params["com"]["reg"]
    wscale    = params["com"]["wscale"]
    dtype     = params["com"]["dtype"]
    reuse     = params["com"]["reuse"]
    is_train  = params["com"]["is_train"]
    trainable = params["com"]["trainable"]
    
    num_h = params["bilstm"]["num_h"]
    num_o = params["bilstm"]["num_o"]
    fbias = params["bilstm"]["fbias"]
    tmajr = params["bilstm"]["tmajr"]

    #为了把sequence_length变量传进来
    if isinstance(tensor_in, tuple):
        slens = tensor_in[1]
        tensor_in = tensor_in[0]
    else:
        slens = None

    #shape = tf.shape(tensor_in)
    x_shape = get_shape(tensor_in) #[T, N, C]
    T, N, C = x_shape[0], x_shape[1], x_shape[2]
    
    #initializer=tf.truncated_normal_initializer(stddev=wscale)
    #regularizer=tf.contrib.layers.l2_regularizer(reg)
    
    with tf.variable_scope("bilstm1_"+str(layer), reuse=reuse) as scope:
        
        lstm_fw_cell = tf.contrib.rnn.LSTMCell(num_units=num_h, forget_bias=fbias, state_is_tuple=True)
        lstm_bw_cell = tf.contrib.rnn.LSTMCell(num_units=num_h, forget_bias=fbias, state_is_tuple=True)
        
        lstm_out, last_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell, \
                                                               tensor_in, sequence_length=slens, \
                                                               dtype=dtype, parallel_iterations=32, \
                                                               swap_memory=True, time_major=tmajr)
        lstm_out = tf.concat(lstm_out, axis=-1)
        lstm_out = tf.reshape(lstm_out, [T, N, 2*num_h])
        if num_o != None:
            params["conv"] = {"number": num_o, "shape": [1, 1], "rate": 1, "stride": [1, 1], "padding": "SAME"}
            lstm_out = tf.expand_dims(lstm_out, axis=0)
            lstm_out = conv1(lstm_out, 0, params, mtrain)
            lstm_out = tf.squeeze(lstm_out, axis=[0])
            """
            params["affine"]["dim"] = num_o
            lstm_out = affine1(lstm_out, 0, params, mtrain)
            lstm_out = tf.reshape(lstm_out, [T, N, num_o])
            """
        print_activations(lstm_out)
        if slens != None:
            tensor_out = [lstm_out, slens]
        return tensor_out

    
def batchnorm1(tensor_in=None, layer=0, params=None, mtrain=None):

    reg       = params["com"]["reg"]
    wscale    = params["com"]["wscale"]
    dtype     = params["com"]["dtype"]
    reuse     = params["com"]["reuse"]
    is_train  = params["com"]["is_train"]
    trainable = params["com"]["trainable"]
    
    eps   = params["bn"]['eps'] #1e-3
    decay = params["bn"]["decay"] #0.9
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    with tf.variable_scope("batchnorm1_"+str(layer), reuse=reuse) as scope:
        
        tensor_out = tf.contrib.layers.batch_norm(
            tensor_in, decay=decay, center=True, scale=True, epsilon=eps,
            activation_fn=None,
            param_initializers=None,
            param_regularizers=None,#tf.contrib.layers.l2_regularizer(reg),
            updates_collections=tf.GraphKeys.UPDATE_OPS,
            is_training=is_train, trainable=trainable, fused=False)
        print_activations(tensor_out) 
    return tensor_out


def conv_bn1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    params["conv"]["use_bias"] = False
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    with tf.variable_scope("conv_bn1_"+str(layer)) as scope:
        bias = conv1(tensor_in, 0, params, mtrain)
        tensor_out = batchnorm1(bias, 0, params, mtrain)
    return tensor_out


def deconv_bn1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    params["deconv"]["use_bias"] = False
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    with tf.variable_scope("deconv_bn1_"+str(layer)) as scope:
        bias = deconv1(tensor_in, 0, params, mtrain)
        tensor_out = batchnorm1(bias, 0, params, mtrain)
    return tensor_out


def conv_bn_relu1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    with tf.variable_scope("conv_bn_relu1_"+str(layer)) as scope:
        bn = conv_bn1(tensor_in, 0, params, mtrain)
        tensor_out = relu1(bn, 0, params, mtrain)
    return tensor_out 

#full pre-activation
def bn_relu1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    with tf.variable_scope("bn_relu1_"+str(layer)) as scope:
        bn = batchnorm1(tensor_in, 0, params, mtrain)
        tensor_out = relu1(bn, 0, params, mtrain)
    return tensor_out


def bn_relu_conv1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    with tf.variable_scope("bn_relu_conv1_"+str(layer)) as scope:
        relu = bn_relu1(tensor_in, 0, params, mtrain)
        tensor_out = conv1(relu, 0, params, mtrain)
    return tensor_out


def deconv_bn_relu1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
        
    with tf.variable_scope("deconv_bn_relu1_"+str(layer)) as scope:
        bn = deconv_bn1(tensor_in, 0, params, mtrain)
        tensor_out = relu1(bn, 0, params, mtrain)
    return tensor_out 


def conv_bn_relu_max_pool1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    with tf.variable_scope("conv_bn_relu_max_pool1_"+str(layer)) as scope:
        relu = conv_bn_relu1(tensor_in, 0, params, mtrain)
        tensor_out = max_pool1(relu, 0, params, mtrain)
    return tensor_out


def resnet_unit1(tensor_in=None, layer=0, params=None, mtrain=None):
    """
    第一种类型的resnet_unit
    Residual unit with 2 sub layers, using Plan A for shortcut connection.
    """
    params["conv"]["padding"] = "SAME"
    stride = params["conv"]["stride"]
    params["avg_pool"]["shape"]  = [stride[0], stride[1]]
    params["avg_pool"]["stride"] = [stride[0], stride[1]]
    params["avg_pool"]["padding"] = "SAME"
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    x_shape    = tensor_in.get_shape()
    in_filter  = int(x_shape[3])
    out_filter = params["conv"]["number"]
    
    with tf.variable_scope("resnet_unit1_"+str(layer)) as scope:
        
        relu = conv_bn_relu1(tensor_in, 0, params, mtrain)
        params["conv"]["stride"] = [1, 1]
        bn = conv_bn1(relu, 0, params, mtrain)
        orig_x = avg_pool1(tensor_in, 0, params, mtrain)
        
        if in_filter != out_filter:
            pad1 = (out_filter - in_filter) // 2
            pad2 = out_filter - in_filter - pad1
            orig_x = tf.pad(orig_x, [[0, 0], [0, 0], [0, 0], [pad1, pad2]])
        tensor_out = relu1(tf.add(bn, orig_x), 0, params, mtrain)
    return tensor_out


def resnet_unit2(tensor_in=None, layer=0, params=None, mtrain=None):
    
    depth_output = params["resnet_unit"]["depth_output"]
    depth_bottle = params["resnet_unit"]["depth_bottle"]
    use_branch   = params["resnet_unit"]["use_branch"]
    shape        = params["resnet_unit"]["shape"]
    stride       = params["resnet_unit"]["stride"]
    rate         = params["resnet_unit"]["rate"]
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    x_shape = tensor_in.get_shape().as_list()
    depth_input = x_shape[-1]
    
    with tf.variable_scope("resnet_unit2_"+str(layer)) as scope:

        if use_branch:
            params["conv"] = {"number":depth_output, "shape":shape, "rate":1, "stride":stride, "padding":"VALID"}
            shortcut = conv_bn1(tensor_in, 0, params, mtrain)
        else:
            shortcut = tensor_in
        params["conv"] = {"number":depth_bottle, "shape":shape,  "rate":1,    "stride":stride, "padding":"VALID"}
        residual = conv_bn_relu1(tensor_in, 0, params, mtrain)
        params["conv"] = {"number":depth_bottle, "shape":[3, 3], "rate":rate, "stride":[1, 1], "padding":"SAME" }
        residual = conv_bn_relu1(residual,  1, params, mtrain)
        params["conv"] = {"number":depth_output, "shape":[1, 1], "rate":1,    "stride":[1, 1], "padding":"VALID"}
        residual = conv_bn1(residual, 1, params, mtrain)
        tensor_out = relu1(shortcut+residual, 0, params, mtrain)
        return tensor_out

    
def resnet_unit3(tensor_in=None, layer=0, params=None, mtrain=None):
    """
    第三种类型的resnet_unit
    """
    depth_output = params["resnet_unit"]["depth_output"]
    depth_bottle = params["resnet_unit"]["depth_bottle"]
    stride = params["resnet_unit"]["stride"]
    rate = params["resnet_unit"]["rate"]
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    x_shape = tensor_in.get_shape().as_list()
    depth_input = x_shape[-1]
    
    with tf.variable_scope("resnet_unit3_"+str(layer)) as scope:
        
        if depth_output == depth_input:
            if stride == 1:
                shortcut = tensor_in
            else:
                params["max_pool"] = {"shape": [1, 1], "stride": [stride, stride], "padding": "VALID"}
                shortcut = max_pool1(tensor_in, 0, params, mtrain)
            params["conv"] = {"number": depth_bottle, "shape": [1, 1], "rate": 1, "stride": [stride, stride], "padding": "VALID"}
            residual = bn_relu_conv1(tensor_in, 0, params, mtrain)
        else:
            tensor_in = bn_relu1(tensor_in, 0, params, mtrain)
            params["conv"] = {"number": depth_output, "shape": [1, 1], "rate": 1, "stride": [stride, stride], "padding": "VALID"}
            shortcut = conv1(tensor_in, 0, params, mtrain)
            params["conv"] = {"number": depth_bottle, "shape": [1, 1], "rate": 1, "stride": [stride, stride], "padding": "VALID"}
            residual = conv1(tensor_in, 1, params, mtrain)
        
        params["conv"] = {"number": depth_bottle, "shape": [3, 3], "rate": rate, "stride": [1, 1], "padding": "SAME"}
        residual = bn_relu_conv1(residual,  1, params, mtrain)
        params["conv"] = {"number": depth_output, "shape": [1, 1], "rate": 1, "stride": [1, 1], "padding": "VALID"}
        residual = bn_relu_conv1(residual, 2, params, mtrain)
        tensor_out = shortcut + residual
        print_activations(tensor_out)
        return tensor_out
    
    
def resnet_block2(tensor_in=None, layer=0, params=None, mtrain=None):
    
    block_setting = params["resnet_block"]["block_setting"]
    output_stride = params["resnet_block"]["output_stride"]

    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    rate           = 1
    current_stride = 1
    tensor_out     = tensor_in
    out_list       = []
    for i, block in enumerate(block_setting):
        
        depth_output, depth_bottle, shape, stride, unit_number, unit_trainable = block
        params["com"]["trainable"] = unit_trainable
        
        with tf.variable_scope("resnet_block2_"+str(layer)+"_"+str(i)) as scope:
            
            for j in range(unit_number):
                if j == 0: #the first unit in the block
                    if current_stride == output_stride:
                        stride = [1, 1]
                        rate *= stride[0]
                    else:
                        rate = 1
                        current_stride *= stride[0]
                    params["resnet_unit"] = {"depth_output": depth_output, "depth_bottle": depth_bottle, "use_branch": True, \
                                             "shape": shape, "stride": stride, "rate": rate}
                else: #identity mapping
                    params["resnet_unit"] = {"depth_output": depth_output, "depth_bottle": depth_bottle, "use_branch": False, \
                                             "shape": [1, 1], "stride": [1, 1], "rate": rate}
                tensor_out = resnet_unit2(tensor_out, j, params, mtrain)
        out_list.append(tensor_out)

    if output_stride is not None and current_stride != output_stride:
        raise ValueError('The target output_stride cannot be reached.')
    return out_list


def resnet_block3(tensor_in=None, layer=0, params=None, mtrain=None):

    block_setting = params["resnet_block"]["block_setting"]
    output_stride = params["resnet_block"]["output_stride"]

    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    current_stride = 1
    rate = 1
    tensor_out = tensor_in
    out_list = []
    
    for i, block in enumerate(block_setting):
        
        with tf.variable_scope("resnet_block3_"+str(layer)+"_"+str(i)) as scope:
            
            for j, unit in enumerate(block):

                if output_stride is not None and current_stride > output_stride:
                    raise ValueError('The target output_stride cannot be reached.')

                unit_depth_output, unit_depth_bottle, unit_stride, unit_trainable = unit
                params["com"]["trainable"] = unit_trainable

                if output_stride is not None and current_stride == output_stride:
                    params["resnet_unit"] = {"depth_output": unit_depth_output, "depth_bottle": unit_depth_bottle, 
                                             "stride": 1, "rate": rate}
                    tensor_out = resnet_unit3(tensor_out, j, params, mtrain)
                    rate *= unit_stride  #完成之后做记录
                else:
                    params["resnet_unit"] = {"depth_output": unit_depth_output, "depth_bottle": unit_depth_bottle, 
                                             "stride": unit_stride, "rate": 1}
                    tensor_out = resnet_unit3(tensor_out, j, params, mtrain)
                    current_stride *= unit_stride  #完成之后做记录
        out_list.append(bn_relu1(tensor_out, 0, params, mtrain))

    if output_stride is not None and current_stride != output_stride:
        raise ValueError('The target output_stride cannot be reached.')
    return out_list


def pyramid1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    depth = params["pyramid"]["depth"]
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    num_layers = len(tensor_in)
    pyramid    = []
    with tf.variable_scope("pyramid1_"+str(layer)) as scope:
        
        params["conv"] = {"number": depth, "shape": [1, 1], "rate": 1, "stride": [1, 1], "padding": "SAME"}
        pyramid.append(conv_bn_relu1(tensor_in[-1], 0, params, mtrain))

        for i in range(num_layers-2, -1, -1):

            j = num_layers - 2 - i
            p, c = pyramid[j], tensor_in[i]
            
            c_shape = tf.shape(c)
            p = tf.image.resize_bilinear(p, [c_shape[1], c_shape[2]])
            
            params["conv"]["shape"] = [1, 1]
            c = conv_bn_relu1(c, 1+j, params, mtrain)
            
            p = tf.add(p, c)
            pyramid.append(p)
        
        for i in range(num_layers):
            p = pyramid[i]
            params["conv"]["shape"] = [3, 3]
            p = conv_bn_relu1(p, num_layers+i, params, mtrain)
            pyramid[i] = p
            
        pyramid = pyramid[::-1]
    return pyramid


def pyramid2(tensor_in=None, layer=0, params=None, mtrain=None):
    
    depth = params["pyramid"]["depth"]
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
        
    num_layers = len(tensor_in)
    pyramid = []
    with tf.variable_scope("pyramid2_"+str(layer)) as scope:

        pyramid.append(tensor_in[-1])
        
        params["conv"] = {"rate": 1, "stride": [1, 1], "padding": "SAME"}
        
        for i in range(num_layers-2, -1, -1):

            j = num_layers - 2 - i
            p, c = pyramid[j], tensor_in[i]
            c_shape = tf.shape(c)
            p = tf.image.resize_bilinear(p, [c_shape[1], c_shape[2]])
            p = tf.concat([p, c], axis=-1)
            
            params["conv"] = {"shape": [1, 1], "number": depth[j], "rate": 1, "stride": [1, 1], "padding": "SAME"}
            p = conv_bn_relu1(p, 0+2*j, params, mtrain)
            params["conv"] = {"shape": [3, 3], "number": depth[j], "rate": 1, "stride": [1, 1], "padding": "SAME"}
            p = conv_bn_relu1(p, 1+2*j, params, mtrain)
            pyramid.append(p)
        #pyramid = list(reversed(pyramid))
        p = pyramid[-1]
        params["conv"] = {"shape": [3, 3], "number": depth[-1], "rate": 1, "stride": [1, 1], "padding": "SAME"}
        tensor_out = conv_bn_relu1(p, 2*(num_layers-1), params, mtrain)
    return tensor_out


def pyramid3(tensor_in=None, layer=0, params=None, mtrain=None):
    """
    第三种类型的pyramid #DSSD
    """
    depth = params["pyramid"]["depth"]
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    num_layers = len(tensor_in)
    pyramid    = []
    with tf.variable_scope("pyramid3_"+str(layer)) as scope:
        
        params["conv"]   = {"number": depth, "shape": [3, 3], "rate": 1, "stride": [1, 1], "padding": "SAME"}
        params["deconv"] = {"number": depth, "shape": [2, 2], "rate": 1, "stride": [2, 2], "padding": "SAME", "out_shape":[28, 28]}
        p = tensor_in[-1]
        pyramid.append(p)
        
        for i in range(num_layers-2, -1, -1):

            j = num_layers - 2 - i
            p, c = pyramid[j], tensor_in[i]

            c_shape = c.get_shape()
            params["deconv"]["out_shape"] = [int(c_shape[1]), int(c_shape[2])]
            p = deconv1(p, j, params, mtrain)
            p = conv_bn1(p, 0+2*j, params, mtrain)
            
            c = conv_bn_relu1(c, j, params, mtrain)
            c = conv_bn1(c, 1+2*j, params, mtrain)
            
            p = tf.multiply(p, c)
            #p = tf.add(p, c)
            p = relu1(p, 0, params, mtrain)
            pyramid.append(p)
        pyramid = pyramid[::-1]
    return pyramid


def pyramid4(tensor_in=None, layer=0, params=None, mtrain=None):
    """
    第四种类型的pyramid #DSSD
    """
    depth = params["pyramid"]["depth"]
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    num_layers = len(tensor_in)
    pyramid    = []
    with tf.variable_scope("pyramid4_"+str(layer)) as scope:
        
        params["conv"]   = {"number": depth, "shape": [3, 3], "rate": 1, "stride": [1, 1], "padding": "SAME"}
        params["deconv"] = {"number": depth, "shape": [2, 2], "rate": 1, "stride": [2, 2], "padding": "SAME", "out_shape":[28, 28]}
        p = tensor_in[-1]
        p = conv_bn_relu1(p, 0, params, mtrain)
        p = conv_bn_relu1(p, 1, params, mtrain)
        pyramid.append(p)
        
        for i in range(num_layers-2, -1, -1):

            j = num_layers - 2 - i
            p, c = pyramid[j], tensor_in[i]

            c_shape = c.get_shape()
            params["deconv"]["out_shape"] = [int(c_shape[1]), int(c_shape[2])]
            p = deconv_bn1(p, j, params, mtrain)
            
            c = conv_bn_relu1(c, 2+2*j, params, mtrain)
            c = conv_bn1(c, j, params, mtrain)
            
            p = tf.add(p, c)
            p = relu1(p, 0, params, mtrain)
            p = conv_bn_relu1(p, 3+2*j, params, mtrain)
            pyramid.append(p)
        pyramid = list(reversed(pyramid))
    return pyramid


def affine1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    reg       = params["com"]["reg"]
    wscale    = params["com"]["wscale"]
    dtype     = params["com"]["dtype"]
    reuse     = params["com"]["reuse"]
    is_train  = params["com"]["is_train"]
    trainable = params["com"]["trainable"]
    
    dim       = params["affine"]["dim"]
    use_bias  = params["affine"]["use_bias"]
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    #x_shape = tensor_in.get_shape().as_list()
    x_shape = get_shape(tensor_in)
    L = len(x_shape)
    nodes = 1
    for i in range(1,L):
        nodes *= x_shape[i]
    x_reshaped = tf.reshape(tensor_in, [-1,nodes])
    #r_shape = x_reshaped.get_shape().as_list()
    r_shape = get_shape(x_reshaped)
    affine_dim = [r_shape[1], dim]

    with tf.variable_scope("affine1_"+str(layer), reuse=reuse) as scope:
        weights = tf.get_variable(name="weights", shape=affine_dim, dtype=dtype, \
                                  initializer=tf.truncated_normal_initializer(stddev=wscale), \
                                  #initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,mode='FAN_AVG',uniform=True),
                                  regularizer=tf.contrib.layers.l2_regularizer(reg), \
                                  trainable=trainable)
        if use_bias:
            biases  = tf.get_variable(name="biases", shape=[dim], dtype=dtype, \
                                      initializer=tf.constant_initializer(0.0), \
                                      trainable=trainable)
        affine = tf.matmul(x_reshaped, weights)
        if use_bias:
            tensor_out = affine + biases
        else:
            tensor_out = affine
        print_activations(tensor_out)
    return tensor_out


def affine_softmax1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    with tf.variable_scope("affine_softmax1_"+str(layer)) as scope:
        bias = affine1(tensor_in, 0, params, mtrain)
        tensor_out = tf.nn.softmax(bias)
        print_activations(tensor_out)
    return tensor_out    


def affine_relu1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    with tf.variable_scope("affine_relu1_"+str(layer)) as scope:
        bias = affine1(tensor_in, 0, params, mtrain)
        tensor_out = relu1(bias, 0, params, mtrain)
    return tensor_out


def affine_relu_dropout1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
        
    with tf.variable_scope("affine_relu_dropout1_"+str(layer)) as scope:
        relu = affine_relu1(tensor_in, 0, params, mtrain)
        tensor_out = dropout1(relu)
    return tensor_out    


def affine_bn1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
        
    with tf.variable_scope("affine_bn1_"+str(layer)) as scope:
        bias = affine1(tensor_in, 0, params, mtrain)
        tensor_out = batchnorm1(bias, 0, params, mtrain)
    return tensor_out


def affine_bn_relu1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
        
    with tf.variable_scope("affine_bn_relu1_"+str(layer)) as scope:
        bn = affine_bn1(tensor_in, 0, params, mtrain)
        tensor_out = relu1(bn, 0, params, mtrain)
    return tensor_out


def affine_bn_relu_dropout1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
        
    with tf.variable_scope("affine_bn_relu_dropout1_"+str(layer)) as scope:
        relu = affine_bn_relu1(tensor_in, 0, params, mtrain)
        tensor_out = dropout1(relu, 0, params, mtrain)
    return tensor_out


def pad1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    shape  = params["pad"]["shape"] #(1,1)
    mode   = params["pad"]["mode"] #'CONSTANT'
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    with tf.variable_scope("pad1_"+str(layer)) as scope:
        paddings = [[0, 0], [shape[0], shape[0]], [shape[1], shape[1]], [0, 0]]
        tensor_out = tf.pad(tensor_in, paddings, mode=mode)
        print_activations(tensor_out) 
    return tensor_out

    
def reshape1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    shape  = params["reshape"]["shape"]
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    with tf.variable_scope("reshape1_"+str(layer)) as scope:
        tensor_out = tf.reshape(tensor_in, shape=shape)
        print_activations(tensor_out)
    return tensor_out
    

def squeeze1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    axis  = params["squeeze"]["axis"]
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    with tf.variable_scope("squeeze1_"+str(layer)) as scope:
        tensor_out = tf.squeeze(tensor_in, axis=axis)
        print_activations(tensor_out)
    return tensor_out
    
    
def transpose1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    perm = params["transpose"]["perm"]
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    with tf.variable_scope("transpose1_"+str(layer)) as scope:
        tensor_out = tf.transpose(tensor_in, perm=perm)
        print_activations(tensor_out)
    return tensor_out


def concat1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    axis = params["concat"]["axis"]
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    with tf.variable_scope("concat1_"+str(layer)) as scope:
        tensor_out = tf.concat(tensor_in, axis)
        print_activations(tensor_out)
    return tensor_out


def split1(tensor_in=None, layer=0, params=None, mtrain=None):
    
    axis = params["split"]["axis"]
    number = params["split"]["number"]
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    with tf.variable_scope("split1_"+str(layer)) as scope:
        tensor_out = tf.split(tensor_in, number, axis)
        print_activations(tensor_out)
    return tensor_out

    
def l2_norm1(tensor_in=None, layer=0, params=None, mtrain=None):
    """
    第一种类型的l2_norm(特征之间的l2 normalization)
    """
    dtype = params["com"]["dtype"]
    reuse = params["com"]["reuse"]
    trainable = params["com"]["trainable"]
    eps   = params["l2_norm"]['eps'] #1e-12
    
    if isinstance(tensor_in, tuple):
        tensor_in = tensor_in[0]
    
    x_shape = tensor_in.get_shape().as_list()
    params_shape = [x_shape[-1]]
    x_rank = len(x_shape)
    axis = list(range(x_rank-1, x_rank))
    
    with tf.variable_scope("l2_norm1_"+str(layer), reuse=reuse) as scope:
        
        l2_norm = tf.nn.l2_normalize(tensor_in, axis, epsilon=eps)
        gamma = tf.get_variable(name="gamma", shape=params_shape, dtype=dtype, \
                                initializer=tf.constant_initializer(1.0), \
                                regularizer=tf.contrib.layers.l2_regularizer(reg), \
                                trainable=trainable)
        tensor_out = tf.multiply(l2_norm, gamma)
        print_activations(tensor_out) 
    return tensor_out
