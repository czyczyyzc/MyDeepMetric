import numpy as np
import tensorflow as tf

def tensor_update_py(bottom_data = None, bottom_inds = None, new_data = None):
    
    #assert len(bottom_inds.shape) == 2, "bottom_inds shape wrong"
    #assert bottom_inds.shape[0] == new_data.shape[0], "bottom_inds and new_data shape wrong"
    #assert bottom_inds.shape[0] != 0, "shape wrong"
    #assert len(bottom_inds) != 0, "len wrong"
    
    copy_data = bottom_data.copy()
    
    if bottom_inds.shape[0] == 0:
        return copy_data

    bottom_inds = np.transpose(bottom_inds)
    bottom_inds = list(bottom_inds)
    copy_data[bottom_inds] = new_data
    
    return copy_data


def tensor_update(bottom_data = None, bottom_inds = None, new_data = None):
    
    bottom_data_shape = tf.shape(bottom_data)
    bottom_data_dtype = bottom_data.dtype
    top_data = tf.py_func(tensor_update_py, [bottom_data, bottom_inds, new_data], bottom_data_dtype)
    top_data = tf.reshape(top_data, bottom_data_shape)
    
    return top_data
