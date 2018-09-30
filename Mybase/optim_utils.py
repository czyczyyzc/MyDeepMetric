import tensorflow as tf
import numpy as np
import pickle
import os
from . import optim

def average_gradients(tower_grads, clip_norm=5.0):
    average_grads = []
    #枚举所有的变量和变量在不同GPU上计算得出的梯度
    for grad_and_vars in zip(*tower_grads):
        #计算所有GPU上的梯度平均值
        grads = []
        for g,_ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        if clip_norm != None:
            grad = tf.clip_by_norm(grad, clip_norm, axes=None)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    #返回所有变量的平均梯度，这将被用于变量更新
    return average_grads

def get_variables_to_train(trainable_scopes = None):
    """
    Returns:
        A list of variables to train by the optimizer.
    """
    if trainable_scopes is None:
        return tf.trainable_variables()
    
    else:
        scopes = [scope.strip() for scope in trainable_scopes.split(',')]

    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    
    return variables_to_train

def update_rule(config = None, global_step = None):
    
    if config is None: config = {}
    
    decay_rule = config.get('decay_rule', 'exponential')
    optim_rule = config.get('optim_rule', 'momentum')
    
    decay_rule = decay_rule + "_decay"
    optim_rule = optim_rule + "_optim"
    
    if not hasattr(optim, decay_rule):
        raise ValueError('Invalid decay_rule "%s"' % decay_rule)
    
    if not hasattr(optim, optim_rule):
        raise ValueError('Invalid optim_rule "%s"' % optim_rule)

    decay_rule = getattr(optim, decay_rule)
    optim_rule = getattr(optim, optim_rule)
    
    learning_rate = decay_rule(config, global_step)
    
    config["learning_rate"] = learning_rate
    
    train_step = optim_rule(config, global_step)
    
    return train_step

"""
def distort_color(image, color_ordering = 0):
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta = 32.)
        image = tf.image.random_saturation(image, lower = 0.5, upper = 1.5)
        image = tf.image.random_hue(image, max_delta = 0.2)
        image = tf.image.random_contrast(image, lower = 0.5, upper = 1.5)
    elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower = 0.5, upper = 1.5)
        image = tf.image.random_brightness(image, max_delta = 32.)
        image = tf.image.random_contrast(image, lower = 0.5, upper = 1.5)
        image = tf.image.random_hue(image, max_delta = 0.2)
    #elif color_odering == 2:
    return tf.clip_by_value(image, 0.0, 1.0)

def preprocess_for_train(mtrain = None, image = None, height = None, width = None, bbox = None, min_object_covered = 0.5):
    if image.dtype != tf.float32:
        #image = tf.image.convert_image_dtype(image, dtype = tf.float32)
        distorted_image = tf.cast(image, tf.float32)
    if mtrain == True:
        
        # if bbox is None:
        #     bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype = tf.float32, shape = [1,1,4])
        # bbox_begin, bbox_size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
        #     tf.shape(image), bounding_boxes = bbox, min_object_covered = min_object_covered)
        # distorted_image = tf.slice(distorted_image, bbox_begin, bbox_size)
        # distorted_image = tf.image.resize_images(distorted_image, (height, width), method = np.random.randint(4))
        # distorted_image = tf.image.random_flip_left_right(distorted_image)
        # distorted_image = distort_color(distorted_image, np.random.randint(2))
        
        distorted_image = tf.image.resize_image_with_crop_or_pad(distorted_image, int(height*1.25), int(width*1.25))
        distorted_image = tf.random_crop(distorted_image, [height, width, 3])
        distorted_image = tf.image.random_flip_left_right(distorted_image)
    else:
        distorted_image = tf.image.resize_images(distorted_image, (height, width), method = np.random.randint(4))
    distorted_image = tf.image.per_image_standardization(distorted_image)
    return distorted_image


def get_input(mtrain = None, image_size = 32, channels = 3, batch_size = 128, filename = None, min_after_dequeue = None, min_object_covered = None, num_threads = 16):
    #创建文件列表，并通过文件列表创建输入文件队列。
    #在调用输入数据处理流程前，需要统一所有原始数据的格式并将它们存储到TFRecord文件中
    #文件列表应该包含所有提供训练数据的TFRecord文件
    #filename = "mnist/tfrecords/data.tfrecords-*"
    files = tf.train.match_filenames_once(filename)
    filename_queue = tf.train.string_input_producer(files, shuffle = mtrain)
    
    #解析TFRecord文件里的数据
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features = {
            "image_raw": tf.FixedLenFeature([], tf.string),
            "label":  tf.FixedLenFeature([], tf.int64),
            "height": tf.FixedLenFeature([], tf.int64),
            "width":  tf.FixedLenFeature([], tf.int64),
            "channels": tf.FixedLenFeature([], tf.int64),
        }
    )

    image_raw, label = features["image_raw"],  features["label"]
    #height, width = features["height"], features["width"]
    #channels = features["channels"]
    #从原始图像数据解析出像素矩阵，并根据图像尺寸还原图像
    decoded_image = tf.decode_raw(image_raw, tf.uint8)
    decoded_image = tf.reshape(decoded_image,[image_size, image_size, channels])
    distorted_image = preprocess_for_train(mtrain, decoded_image, image_size, image_size, None)
    
    capacity = min_after_dequeue + 3 * batch_size
    #tf.train.shuffle_batch_join
    
    # if mtrain == True:
    #     images, labels =  tf.train.shuffle_batch(
    #         [distorted_image, label], batch_size = batch_size, num_threads = num_threads,
    #         capacity = capacity, min_after_dequeue = min_after_dequeue
    #     )
    # else:
    #     images, labels =  tf.train.batch(
    #         [distorted_image, label], batch_size = batch_size, num_threads = num_threads, capacity = capacity
    #     )
    
    images, labels =  tf.train.shuffle_batch(
        [distorted_image, label], batch_size = batch_size, num_threads = num_threads,
        capacity = capacity, min_after_dequeue = min_after_dequeue
    )
    return images, tf.reshape(labels, [batch_size])
    
def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list = tf.train.FloatList(value=[value]))

def make_input(images = None, labels = None, height = None, width = None, channels = None, data_path = None, num_shards = 3, instances_per_shard = 6):
    index = 0
    for i in range(num_shards):
        filename = (os.path.join(data_path,"data.tfrecords-%.5d-of-%.5d") %(i, num_shards))
        writer = tf.python_io.TFRecordWriter(filename)
        
        for j in range(instances_per_shard):
            image_raw = images[index].tostring()
            example = tf.train.Example(features = tf.train.Features(feature = {
                "image_raw": _bytes_feature(image_raw),
                "label":  _int64_feature(labels[index]),
                "height": _int64_feature(height),
                "width":  _int64_feature(width),
                "channels": _int64_feature(channels)
            }))
            writer.write(example.SerializeToString())
            index += 1
        writer.close()

def load_CIFAR_batch(filename):
    with tf.gfile.Open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='bytes')
        X = datadict[b'data']
        Y = datadict[b'labels']
        num_X = X.shape[0]
        X = X.reshape(num_X, 3, 32, 32).transpose(0,2,3,1)
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte
        
def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=10000):
    cifar10_dir = 'Mybase/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]
    make_input(X_train, y_train, 32, 32, 3, "Mybase/tfrecords/train", 1, num_training)
    make_input(X_val, y_val, 32, 32, 3, "Mybase/tfrecords/val", 1, num_validation)
    make_input(X_test, y_test, 32, 32, 3, "Mybase/tfrecords/test", 1, num_test)
"""