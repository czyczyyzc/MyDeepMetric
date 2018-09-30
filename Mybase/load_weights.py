import h5py
import pickle
import numpy as np

def load_weights():
    fff = h5py.File('Mybase/mask_rcnn_coco.h5','r')   #打开h5文件  
    #print(list(f.keys()))
    mydict = {}
    mydict['global_step:0'] = 1000
    ########res1########
    dset = fff['conv1']
    a = dset['conv1']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn_conv1']
    a = dset['bn_conv1']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_0/conv_bn_relu1_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_0/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_0/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    ########res2########
    dset = fff['res2a_branch1']
    a = dset['res2a_branch1']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn2a_branch1']
    a = dset['bn2a_branch1']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res2a_branch2a']
    a = dset['res2a_branch2a']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn2a_branch2a']
    a = dset['bn2a_branch2a']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_0/conv_bn_relu1_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_0/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_0/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res2a_branch2b']
    a = dset['res2a_branch2b']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn2a_branch2b']
    a = dset['bn2a_branch2b']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_0/conv_bn_relu1_1/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_0/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_0/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res2a_branch2c']
    a = dset['res2a_branch2c']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn2a_branch2c']
    a = dset['bn2a_branch2c']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_0/conv_bn1_1/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_0/conv_bn1_1/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_0/conv_bn1_1/batchnorm1_0/BatchNorm/beta:0' ] = h
    ################################
    dset = fff['res2b_branch2a']
    a = dset['res2b_branch2a']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn2b_branch2a']
    a = dset['bn2b_branch2a']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_1/conv_bn_relu1_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_1/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_1/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res2b_branch2b']
    a = dset['res2b_branch2b']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn2b_branch2b']
    a = dset['bn2b_branch2b']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_1/conv_bn_relu1_1/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_1/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_1/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res2b_branch2c']
    a = dset['res2b_branch2c']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn2b_branch2c']
    a = dset['bn2b_branch2c']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_1/conv_bn1_1/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_1/conv_bn1_1/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_1/conv_bn1_1/batchnorm1_0/BatchNorm/beta:0' ] = h
    ############################
    dset = fff['res2c_branch2a']
    a = dset['res2c_branch2a']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn2c_branch2a']
    a = dset['bn2c_branch2a']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_2/conv_bn_relu1_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_2/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_2/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res2c_branch2b']
    a = dset['res2c_branch2b']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn2c_branch2b']
    a = dset['bn2c_branch2b']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_2/conv_bn_relu1_1/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_2/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_2/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res2c_branch2c']
    a = dset['res2c_branch2c']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn2c_branch2c']
    a = dset['bn2c_branch2c']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_2/conv_bn1_1/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_2/conv_bn1_1/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_2/conv_bn1_1/batchnorm1_0/BatchNorm/beta:0' ] = h
    ########res3########
    dset = fff['res3a_branch1']
    a = dset['res3a_branch1']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn3a_branch1']
    a = dset['bn3a_branch1']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res3a_branch2a']
    a = dset['res3a_branch2a']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn3a_branch2a']
    a = dset['bn3a_branch2a']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_0/conv_bn_relu1_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_0/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_0/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res3a_branch2b']
    a = dset['res3a_branch2b']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn3a_branch2b']
    a = dset['bn3a_branch2b']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_0/conv_bn_relu1_1/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_0/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_0/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res3a_branch2c']
    a = dset['res3a_branch2c']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn3a_branch2c']
    a = dset['bn3a_branch2c']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_0/conv_bn1_1/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_0/conv_bn1_1/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_0/conv_bn1_1/batchnorm1_0/BatchNorm/beta:0' ] = h
    ################################
    dset = fff['res3b_branch2a']
    a = dset['res3b_branch2a']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn3b_branch2a']
    a = dset['bn3b_branch2a']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_1/conv_bn_relu1_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_1/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_1/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res3b_branch2b']
    a = dset['res3b_branch2b']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn3b_branch2b']
    a = dset['bn3b_branch2b']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_1/conv_bn_relu1_1/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_1/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_1/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res3b_branch2c']
    a = dset['res3b_branch2c']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn3b_branch2c']
    a = dset['bn3b_branch2c']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_1/conv_bn1_1/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_1/conv_bn1_1/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_1/conv_bn1_1/batchnorm1_0/BatchNorm/beta:0' ] = h
    ############################
    dset = fff['res3c_branch2a']
    a = dset['res3c_branch2a']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn3c_branch2a']
    a = dset['bn3c_branch2a']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_2/conv_bn_relu1_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_2/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_2/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res3c_branch2b']
    a = dset['res3c_branch2b']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn3c_branch2b']
    a = dset['bn3c_branch2b']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_2/conv_bn_relu1_1/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_2/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_2/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res3c_branch2c']
    a = dset['res3c_branch2c']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn3c_branch2c']
    a = dset['bn3c_branch2c']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_2/conv_bn1_1/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_2/conv_bn1_1/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_2/conv_bn1_1/batchnorm1_0/BatchNorm/beta:0' ] = h
    ############################
    dset = fff['res3d_branch2a']
    a = dset['res3d_branch2a']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn3d_branch2a']
    a = dset['bn3d_branch2a']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_3/conv_bn_relu1_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_3/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_3/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res3d_branch2b']
    a = dset['res3d_branch2b']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn3d_branch2b']
    a = dset['bn3d_branch2b']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_3/conv_bn_relu1_1/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_3/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_3/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res3d_branch2c']
    a = dset['res3d_branch2c']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn3d_branch2c']
    a = dset['bn3d_branch2c']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_3/conv_bn1_1/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_3/conv_bn1_1/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_3/conv_bn1_1/batchnorm1_0/BatchNorm/beta:0' ] = h
    ########res4########
    dset = fff['res4a_branch1']
    a = dset['res4a_branch1']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4a_branch1']
    a = dset['bn4a_branch1']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res4a_branch2a']
    a = dset['res4a_branch2a']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4a_branch2a']
    a = dset['bn4a_branch2a']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_0/conv_bn_relu1_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_0/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_0/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res4a_branch2b']
    a = dset['res4a_branch2b']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4a_branch2b']
    a = dset['bn4a_branch2b']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_0/conv_bn_relu1_1/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_0/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_0/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res4a_branch2c']
    a = dset['res4a_branch2c']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4a_branch2c']
    a = dset['bn4a_branch2c']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_0/conv_bn1_1/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_0/conv_bn1_1/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_0/conv_bn1_1/batchnorm1_0/BatchNorm/beta:0' ] = h
    ################################
    dset = fff['res4b_branch2a']
    a = dset['res4b_branch2a']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4b_branch2a']
    a = dset['bn4b_branch2a']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_1/conv_bn_relu1_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_1/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_1/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res4b_branch2b']
    a = dset['res4b_branch2b']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4b_branch2b']
    a = dset['bn4b_branch2b']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_1/conv_bn_relu1_1/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_1/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_1/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res4b_branch2c']
    a = dset['res4b_branch2c']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4b_branch2c']
    a = dset['bn4b_branch2c']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_1/conv_bn1_1/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_1/conv_bn1_1/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_1/conv_bn1_1/batchnorm1_0/BatchNorm/beta:0' ] = h
    ############################
    dset = fff['res4c_branch2a']
    a = dset['res4c_branch2a']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4c_branch2a']
    a = dset['bn4c_branch2a']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_2/conv_bn_relu1_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_2/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_2/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res4c_branch2b']
    a = dset['res4c_branch2b']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4c_branch2b']
    a = dset['bn4c_branch2b']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_2/conv_bn_relu1_1/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_2/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_2/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res4c_branch2c']
    a = dset['res4c_branch2c']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4c_branch2c']
    a = dset['bn4c_branch2c']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_2/conv_bn1_1/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_2/conv_bn1_1/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_2/conv_bn1_1/batchnorm1_0/BatchNorm/beta:0' ] = h
    ############################
    dset = fff['res4d_branch2a']
    a = dset['res4d_branch2a']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4d_branch2a']
    a = dset['bn4d_branch2a']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_3/conv_bn_relu1_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_3/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_3/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res4d_branch2b']
    a = dset['res4d_branch2b']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4d_branch2b']
    a = dset['bn4d_branch2b']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_3/conv_bn_relu1_1/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_3/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_3/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res4d_branch2c']
    a = dset['res4d_branch2c']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4d_branch2c']
    a = dset['bn4d_branch2c']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_3/conv_bn1_1/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_3/conv_bn1_1/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_3/conv_bn1_1/batchnorm1_0/BatchNorm/beta:0' ] = h
    ############################
    dset = fff['res4e_branch2a']
    a = dset['res4e_branch2a']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4e_branch2a']
    a = dset['bn4e_branch2a']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_4/conv_bn_relu1_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_4/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_4/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res4e_branch2b']
    a = dset['res4e_branch2b']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4e_branch2b']
    a = dset['bn4e_branch2b']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_4/conv_bn_relu1_1/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_4/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_4/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res4e_branch2c']
    a = dset['res4e_branch2c']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4e_branch2c']
    a = dset['bn4e_branch2c']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_4/conv_bn1_1/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_4/conv_bn1_1/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_4/conv_bn1_1/batchnorm1_0/BatchNorm/beta:0' ] = h
    ############################
    dset = fff['res4f_branch2a']
    a = dset['res4f_branch2a']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4f_branch2a']
    a = dset['bn4f_branch2a']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_5/conv_bn_relu1_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_5/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_5/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res4f_branch2b']
    a = dset['res4f_branch2b']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4f_branch2b']
    a = dset['bn4f_branch2b']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_5/conv_bn_relu1_1/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_5/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_5/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res4f_branch2c']
    a = dset['res4f_branch2c']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4f_branch2c']
    a = dset['bn4f_branch2c']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_5/conv_bn1_1/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_5/conv_bn1_1/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_5/conv_bn1_1/batchnorm1_0/BatchNorm/beta:0' ] = h
    ############################
    dset = fff['res4g_branch2a']
    a = dset['res4g_branch2a']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4g_branch2a']
    a = dset['bn4g_branch2a']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_6/conv_bn_relu1_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_6/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_6/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res4g_branch2b']
    a = dset['res4g_branch2b']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4g_branch2b']
    a = dset['bn4g_branch2b']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_6/conv_bn_relu1_1/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_6/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_6/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res4g_branch2c']
    a = dset['res4g_branch2c']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4g_branch2c']
    a = dset['bn4g_branch2c']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_6/conv_bn1_1/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_6/conv_bn1_1/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_6/conv_bn1_1/batchnorm1_0/BatchNorm/beta:0' ] = h
    ############################
    dset = fff['res4h_branch2a']
    a = dset['res4h_branch2a']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4h_branch2a']
    a = dset['bn4h_branch2a']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_7/conv_bn_relu1_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_7/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_7/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res4h_branch2b']
    a = dset['res4h_branch2b']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4h_branch2b']
    a = dset['bn4h_branch2b']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_7/conv_bn_relu1_1/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_7/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_7/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res4h_branch2c']
    a = dset['res4h_branch2c']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4h_branch2c']
    a = dset['bn4h_branch2c']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_7/conv_bn1_1/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_7/conv_bn1_1/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_7/conv_bn1_1/batchnorm1_0/BatchNorm/beta:0' ] = h
    ############################
    dset = fff['res4i_branch2a']
    a = dset['res4i_branch2a']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4i_branch2a']
    a = dset['bn4i_branch2a']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_8/conv_bn_relu1_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_8/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_8/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res4i_branch2b']
    a = dset['res4i_branch2b']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4i_branch2b']
    a = dset['bn4i_branch2b']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_8/conv_bn_relu1_1/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_8/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_8/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res4i_branch2c']
    a = dset['res4i_branch2c']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4i_branch2c']
    a = dset['bn4i_branch2c']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_8/conv_bn1_1/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_8/conv_bn1_1/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_8/conv_bn1_1/batchnorm1_0/BatchNorm/beta:0' ] = h
    ############################
    dset = fff['res4j_branch2a']
    a = dset['res4j_branch2a']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4j_branch2a']
    a = dset['bn4j_branch2a']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_9/conv_bn_relu1_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_9/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_9/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res4j_branch2b']
    a = dset['res4j_branch2b']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4j_branch2b']
    a = dset['bn4j_branch2b']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_9/conv_bn_relu1_1/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_9/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_9/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res4j_branch2c']
    a = dset['res4j_branch2c']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4j_branch2c']
    a = dset['bn4j_branch2c']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_9/conv_bn1_1/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_9/conv_bn1_1/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_9/conv_bn1_1/batchnorm1_0/BatchNorm/beta:0' ] = h
    ############################
    dset = fff['res4k_branch2a']
    a = dset['res4k_branch2a']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4k_branch2a']
    a = dset['bn4k_branch2a']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_10/conv_bn_relu1_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_10/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_10/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res4k_branch2b']
    a = dset['res4k_branch2b']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4k_branch2b']
    a = dset['bn4k_branch2b']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_10/conv_bn_relu1_1/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_10/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_10/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res4k_branch2c']
    a = dset['res4k_branch2c']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4k_branch2c']
    a = dset['bn4k_branch2c']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_10/conv_bn1_1/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_10/conv_bn1_1/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_10/conv_bn1_1/batchnorm1_0/BatchNorm/beta:0' ] = h
    ############################
    dset = fff['res4l_branch2a']
    a = dset['res4l_branch2a']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4l_branch2a']
    a = dset['bn4l_branch2a']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_11/conv_bn_relu1_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_11/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_11/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res4l_branch2b']
    a = dset['res4l_branch2b']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4l_branch2b']
    a = dset['bn4l_branch2b']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_11/conv_bn_relu1_1/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_11/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_11/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res4l_branch2c']
    a = dset['res4l_branch2c']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4l_branch2c']
    a = dset['bn4l_branch2c']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_11/conv_bn1_1/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_11/conv_bn1_1/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_11/conv_bn1_1/batchnorm1_0/BatchNorm/beta:0' ] = h
    ############################
    dset = fff['res4m_branch2a']
    a = dset['res4m_branch2a']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4m_branch2a']
    a = dset['bn4m_branch2a']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_12/conv_bn_relu1_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_12/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_12/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res4m_branch2b']
    a = dset['res4m_branch2b']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4m_branch2b']
    a = dset['bn4m_branch2b']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_12/conv_bn_relu1_1/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_12/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_12/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res4m_branch2c']
    a = dset['res4m_branch2c']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4m_branch2c']
    a = dset['bn4m_branch2c']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_12/conv_bn1_1/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_12/conv_bn1_1/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_12/conv_bn1_1/batchnorm1_0/BatchNorm/beta:0' ] = h
    ############################
    dset = fff['res4n_branch2a']
    a = dset['res4n_branch2a']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4n_branch2a']
    a = dset['bn4n_branch2a']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_13/conv_bn_relu1_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_13/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_13/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res4n_branch2b']
    a = dset['res4n_branch2b']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4n_branch2b']
    a = dset['bn4n_branch2b']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_13/conv_bn_relu1_1/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_13/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_13/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res4n_branch2c']
    a = dset['res4n_branch2c']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4n_branch2c']
    a = dset['bn4n_branch2c']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_13/conv_bn1_1/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_13/conv_bn1_1/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_13/conv_bn1_1/batchnorm1_0/BatchNorm/beta:0' ] = h
    ############################
    dset = fff['res4o_branch2a']
    a = dset['res4o_branch2a']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4o_branch2a']
    a = dset['bn4o_branch2a']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_14/conv_bn_relu1_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_14/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_14/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res4o_branch2b']
    a = dset['res4o_branch2b']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4o_branch2b']
    a = dset['bn4o_branch2b']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_14/conv_bn_relu1_1/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_14/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_14/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res4o_branch2c']
    a = dset['res4o_branch2c']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4o_branch2c']
    a = dset['bn4o_branch2c']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_14/conv_bn1_1/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_14/conv_bn1_1/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_14/conv_bn1_1/batchnorm1_0/BatchNorm/beta:0' ] = h
    ############################
    dset = fff['res4p_branch2a']
    a = dset['res4p_branch2a']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4p_branch2a']
    a = dset['bn4p_branch2a']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_15/conv_bn_relu1_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_15/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_15/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res4p_branch2b']
    a = dset['res4p_branch2b']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4p_branch2b']
    a = dset['bn4p_branch2b']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_15/conv_bn_relu1_1/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_15/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_15/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res4p_branch2c']
    a = dset['res4p_branch2c']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4p_branch2c']
    a = dset['bn4p_branch2c']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_15/conv_bn1_1/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_15/conv_bn1_1/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_15/conv_bn1_1/batchnorm1_0/BatchNorm/beta:0' ] = h
    ############################
    dset = fff['res4q_branch2a']
    a = dset['res4q_branch2a']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4q_branch2a']
    a = dset['bn4q_branch2a']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_16/conv_bn_relu1_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_16/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_16/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res4q_branch2b']
    a = dset['res4q_branch2b']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4q_branch2b']
    a = dset['bn4q_branch2b']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_16/conv_bn_relu1_1/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_16/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_16/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res4q_branch2c']
    a = dset['res4q_branch2c']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4q_branch2c']
    a = dset['bn4q_branch2c']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_16/conv_bn1_1/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_16/conv_bn1_1/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_16/conv_bn1_1/batchnorm1_0/BatchNorm/beta:0' ] = h
    ############################
    dset = fff['res4r_branch2a']
    a = dset['res4r_branch2a']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4r_branch2a']
    a = dset['bn4r_branch2a']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_17/conv_bn_relu1_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_17/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_17/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res4r_branch2b']
    a = dset['res4r_branch2b']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4r_branch2b']
    a = dset['bn4r_branch2b']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_17/conv_bn_relu1_1/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_17/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_17/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res4r_branch2c']
    a = dset['res4r_branch2c']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4r_branch2c']
    a = dset['bn4r_branch2c']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_17/conv_bn1_1/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_17/conv_bn1_1/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_17/conv_bn1_1/batchnorm1_0/BatchNorm/beta:0' ] = h
    ############################
    dset = fff['res4s_branch2a']
    a = dset['res4s_branch2a']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4s_branch2a']
    a = dset['bn4s_branch2a']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_18/conv_bn_relu1_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_18/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_18/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res4s_branch2b']
    a = dset['res4s_branch2b']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4s_branch2b']
    a = dset['bn4s_branch2b']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_18/conv_bn_relu1_1/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_18/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_18/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res4s_branch2c']
    a = dset['res4s_branch2c']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4s_branch2c']
    a = dset['bn4s_branch2c']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_18/conv_bn1_1/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_18/conv_bn1_1/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_18/conv_bn1_1/batchnorm1_0/BatchNorm/beta:0' ] = h
    ############################
    dset = fff['res4t_branch2a']
    a = dset['res4t_branch2a']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4t_branch2a']
    a = dset['bn4t_branch2a']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_19/conv_bn_relu1_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_19/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_19/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res4t_branch2b']
    a = dset['res4t_branch2b']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4t_branch2b']
    a = dset['bn4t_branch2b']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_19/conv_bn_relu1_1/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_19/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_19/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res4t_branch2c']
    a = dset['res4t_branch2c']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4t_branch2c']
    a = dset['bn4t_branch2c']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_19/conv_bn1_1/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_19/conv_bn1_1/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_19/conv_bn1_1/batchnorm1_0/BatchNorm/beta:0' ] = h
    ############################
    dset = fff['res4u_branch2a']
    a = dset['res4u_branch2a']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4u_branch2a']
    a = dset['bn4u_branch2a']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_20/conv_bn_relu1_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_20/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_20/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res4u_branch2b']
    a = dset['res4u_branch2b']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4u_branch2b']
    a = dset['bn4u_branch2b']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_20/conv_bn_relu1_1/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_20/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_20/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res4u_branch2c']
    a = dset['res4u_branch2c']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4u_branch2c']
    a = dset['bn4u_branch2c']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_20/conv_bn1_1/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_20/conv_bn1_1/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_20/conv_bn1_1/batchnorm1_0/BatchNorm/beta:0' ] = h
    ############################
    dset = fff['res4v_branch2a']
    a = dset['res4v_branch2a']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4v_branch2a']
    a = dset['bn4v_branch2a']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_21/conv_bn_relu1_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_21/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_21/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res4v_branch2b']
    a = dset['res4v_branch2b']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4v_branch2b']
    a = dset['bn4v_branch2b']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_21/conv_bn_relu1_1/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_21/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_21/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res4v_branch2c']
    a = dset['res4v_branch2c']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4v_branch2c']
    a = dset['bn4v_branch2c']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_21/conv_bn1_1/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_21/conv_bn1_1/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_21/conv_bn1_1/batchnorm1_0/BatchNorm/beta:0' ] = h
    ############################
    dset = fff['res4w_branch2a']
    a = dset['res4w_branch2a']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4w_branch2a']
    a = dset['bn4w_branch2a']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_22/conv_bn_relu1_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_22/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_22/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res4w_branch2b']
    a = dset['res4w_branch2b']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4w_branch2b']
    a = dset['bn4w_branch2b']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_22/conv_bn_relu1_1/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_22/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_22/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res4w_branch2c']
    a = dset['res4w_branch2c']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn4w_branch2c']
    a = dset['bn4w_branch2c']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_22/conv_bn1_1/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_22/conv_bn1_1/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_22/conv_bn1_1/batchnorm1_0/BatchNorm/beta:0' ] = h
    ########res5########
    dset = fff['res5a_branch1']
    a = dset['res5a_branch1']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn5a_branch1']
    a = dset['bn5a_branch1']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res5a_branch2a']
    a = dset['res5a_branch2a']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn5a_branch2a']
    a = dset['bn5a_branch2a']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_0/conv_bn_relu1_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_0/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_0/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res5a_branch2b']
    a = dset['res5a_branch2b']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn5a_branch2b']
    a = dset['bn5a_branch2b']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_0/conv_bn_relu1_1/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_0/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_0/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res5a_branch2c']
    a = dset['res5a_branch2c']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn5a_branch2c']
    a = dset['bn5a_branch2c']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_0/conv_bn1_1/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_0/conv_bn1_1/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_0/conv_bn1_1/batchnorm1_0/BatchNorm/beta:0' ] = h
    ################################
    dset = fff['res5b_branch2a']
    a = dset['res5b_branch2a']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn5b_branch2a']
    a = dset['bn5b_branch2a']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_1/conv_bn_relu1_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_1/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_1/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res5b_branch2b']
    a = dset['res5b_branch2b']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn5b_branch2b']
    a = dset['bn5b_branch2b']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_1/conv_bn_relu1_1/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_1/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_1/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res5b_branch2c']
    a = dset['res5b_branch2c']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn5b_branch2c']
    a = dset['bn5b_branch2c']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_1/conv_bn1_1/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_1/conv_bn1_1/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_1/conv_bn1_1/batchnorm1_0/BatchNorm/beta:0' ] = h
    ############################
    dset = fff['res5c_branch2a']
    a = dset['res5c_branch2a']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn5c_branch2a']
    a = dset['bn5c_branch2a']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_2/conv_bn_relu1_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_2/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_2/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res5c_branch2b']
    a = dset['res5c_branch2b']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn5c_branch2b']
    a = dset['bn5c_branch2b']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_2/conv_bn_relu1_1/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_2/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_2/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    #########
    dset = fff['res5c_branch2c']
    a = dset['res5c_branch2c']
    b = np.array(a['kernel:0'], dtype=np.float32)
    c = np.array(a['bias:0'  ], dtype=np.float32)
    dset = fff['bn5c_branch2c']
    a = dset['bn5c_branch2c']
    d = np.array(a['beta:0' ], dtype=np.float32)
    e = np.array(a['gamma:0'], dtype=np.float32)
    f = np.array(a['moving_mean:0'    ], dtype=np.float32)
    g = np.array(a['moving_variance:0'], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_2/conv_bn1_1/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_2/conv_bn1_1/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_2/conv_bn1_1/batchnorm1_0/BatchNorm/beta:0' ] = h
    return mydict