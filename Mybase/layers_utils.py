import copy
import numpy as np
import tensorflow as tf
from . import layers
from .layers import *
"""
def layers_init(com_params = None):
    decay_mv = com_params["com"]["decay_mv"]
    decay_bn = com_params["bn"]['decay']#0.9
    ema_mv = tf.train.ExponentialMovingAverage(decay = decay_mv)
    ema_bn = tf.train.ExponentialMovingAverage(decay = decay_bn)
    return ema_mv, ema_bn
"""    
def layers_module1(tensor_in=None, layer=0, com_params=None, operations=None, mtrain=None):
    """
        com_params = {
            "com": {"dtype": self.dtype, "wscale": self.wscale},
            "conv":{"number": 64, "shape": [3, 3], "stride": 1, "padding": "SAME"},
            "max_pool":{"shape": [2, 2], "stride": 2, "padding": "VALID"},
            "bn":{"eps": 1e-5, "decay": 0.9},
            "dropout":{"keep_p": self.drop_out},
            "affine":{"dim": 4096}
        }
        
        operations = {
            "op": [
                {"op": "conv_relu_max_pool1", "loop": 1, "params":{"conv":{"number": 64}}},
                {"op": "conv_relu_max_pool1", "loop": 1, "params":{"conv":{"number":192}}},
                {"op": "conv_relu1", "loop": 1, "params":{"conv":{"number": 384}}},
                {"op": "conv_relu1", "loop": 1, "params":{"conv":{"number": 256}}},
                {"op": "conv_relu_max_pool1",  "loop": 1, "params":{"conv":{"number": 256}}},
                {"op": "affine_relu_dropout1", "loop": 2, "params":{}},
                {"op": "affine1", "loop": 1, "params":{"affine":{"dim":1000}}}
            ],
            "loop": 1
        }
    """
    #参数替换
    op = []
    params = []
    funs = []
    L = len(operations["op"])
    #i代表第几个操作
    #操作里包含操作名称和对应参数
    for i in range(L):
        #params[op] = {k: com_params[k] for k in op["params"]}
        op.append(operations["op"][i]["op"])
        params.append(copy.deepcopy(com_params))
        for m in operations["op"][i]["params"]:
            for n in operations["op"][i]["params"][m]:
                #params[i][m][n] = copy.deepcopy(operations["op"][i]["params"][m][n])
                params[i][m][n] = operations["op"][i]["params"][m][n]
                
        #print(operations["op"][i]["params"])
        #print(params[i])
        if not hasattr(layers, op[i]):
            raise ValueError('Invalid operation "%s"' %(op[i]))
        else:
            funs.append(getattr(layers, op[i]))
    
    #构建module
    with tf.variable_scope("layers_module1_"+str(layer)) as scope:
        l = 0
        tensor_out = tensor_in    
        loop1 = operations["loop"]
        for lp1 in range(loop1):
            for i in range(L):
                loop2  = operations["op"][i]["loop"]
                for lp2 in range(loop2):
                    tensor_out = funs[i](tensor_out, l, params[i], mtrain)
                    l += 1
    return tensor_out
