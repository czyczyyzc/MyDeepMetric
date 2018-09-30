import h5py
import pickle
import numpy as np

def load_weights():
    fff = h5py.File('Mybase/resnet50_weights_tf_dim_ordering_tf_kernels.h5','r')   #打开h5文件  
    #print(list(f.keys()))
    mydict = {}
    mydict['global_step:0'] = 2000
    ########res1########
    dset = fff['conv1']
    ##a = dset['conv1']
    b = np.array(dset['conv1_W:0'], dtype=np.float32)
    c = np.array(dset['conv1_b:0'], dtype=np.float32)
    dset = fff['bn_conv1']
    #a = dset['bn_conv1']
    d = np.array(dset['bn_conv1_beta:0' ], dtype=np.float32)
    e = np.array(dset['bn_conv1_gamma:0'], dtype=np.float32)
    f = np.array(dset['bn_conv1_running_mean:0'], dtype=np.float32)
    g = np.array(dset['bn_conv1_running_std:0' ], dtype=np.float32)
    h = ((c - f) / g) * e + d
    mydict['layers_module1_0/conv_bn_relu1_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_0/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_0/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = h
    ########res2########
    dset = fff['res2a_branch1']
    #a = dset['res2a_branch1']
    b = np.array(dset['res2a_branch1_W:0'], dtype=np.float32)
    dset = fff['bn2a_branch1']
    #a = dset['bn2a_branch1']
    d = np.array(dset['bn2a_branch1_beta:0' ], dtype=np.float32)
    e = np.array(dset['bn2a_branch1_gamma:0'], dtype=np.float32)
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = d
    #########
    dset = fff['res2a_branch2a']
    #a = dset['res2a_branch2a']
    b = np.array(dset['res2a_branch2a_W:0'], dtype=np.float32)
    dset = fff['bn2a_branch2a']
    #a = dset['bn2a_branch2a']
    d = np.array(dset['bn2a_branch2a_beta:0' ], dtype=np.float32)
    e = np.array(dset['bn2a_branch2a_gamma:0'], dtype=np.float32)
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_0/conv_bn_relu1_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_0/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_0/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = d
    #########
    dset = fff['res2a_branch2b']
    #a = dset['res2a_branch2b']
    b = np.array(dset['res2a_branch2b_W:0'], dtype=np.float32)
    dset = fff['bn2a_branch2b']
    #a = dset['bn2a_branch2b']
    d = np.array(dset['bn2a_branch2b_beta:0' ], dtype=np.float32)
    e = np.array(dset['bn2a_branch2b_gamma:0'], dtype=np.float32)
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_0/conv_bn_relu1_1/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_0/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_0/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = d
    #########
    dset = fff['res2a_branch2c']
    #a = dset['res2a_branch2c']
    b = np.array(dset['res2a_branch2c_W:0'], dtype=np.float32)
    dset = fff['bn2a_branch2c']
    #a = dset['bn2a_branch2c']
    d = np.array(dset['bn2a_branch2c_beta:0' ], dtype=np.float32)
    e = np.array(dset['bn2a_branch2c_gamma:0'], dtype=np.float32)
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_0/conv_bn1_1/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_0/conv_bn1_1/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_0/conv_bn1_1/batchnorm1_0/BatchNorm/beta:0' ] = d
    ################################
    dset = fff['res2b_branch2a']
    #a = dset['res2b_branch2a']
    b = np.array(dset['res2b_branch2a_W:0'], dtype=np.float32)
    dset = fff['bn2b_branch2a']
    #a = dset['bn2b_branch2a']
    d = np.array(dset['bn2b_branch2a_beta:0' ], dtype=np.float32)
    e = np.array(dset['bn2b_branch2a_gamma:0'], dtype=np.float32)
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_1/conv_bn_relu1_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_1/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_1/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = d
    #########
    dset = fff['res2b_branch2b']
    #a = dset['res2b_branch2b']
    b = np.array(dset['res2b_branch2b_W:0'], dtype=np.float32)
    dset = fff['bn2b_branch2b']
    #a = dset['bn2b_branch2b']
    d = np.array(dset['bn2b_branch2b_beta:0' ], dtype=np.float32)
    e = np.array(dset['bn2b_branch2b_gamma:0'], dtype=np.float32)
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_1/conv_bn_relu1_1/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_1/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_1/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = d
    #########
    dset = fff['res2b_branch2c']
    #a = dset['res2b_branch2c']
    b = np.array(dset['res2b_branch2c_W:0'], dtype=np.float32)
    dset = fff['bn2b_branch2c']
    #a = dset['bn2b_branch2c']
    d = np.array(dset['bn2b_branch2c_beta:0' ], dtype=np.float32)
    e = np.array(dset['bn2b_branch2c_gamma:0'], dtype=np.float32)
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_1/conv_bn1_1/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_1/conv_bn1_1/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_1/conv_bn1_1/batchnorm1_0/BatchNorm/beta:0' ] = d
    ############################
    dset = fff['res2c_branch2a']
    #a = dset['res2c_branch2a']
    b = np.array(dset['res2c_branch2a_W:0'], dtype=np.float32)
    dset = fff['bn2c_branch2a']
    #a = dset['bn2c_branch2a']
    d = np.array(dset['bn2c_branch2a_beta:0' ], dtype=np.float32)
    e = np.array(dset['bn2c_branch2a_gamma:0'], dtype=np.float32)
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_2/conv_bn_relu1_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_2/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_2/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = d
    #########
    dset = fff['res2c_branch2b']
    #a = dset['res2c_branch2b']
    b = np.array(dset['res2c_branch2b_W:0'], dtype=np.float32)
    dset = fff['bn2c_branch2b']
    #a = dset['bn2c_branch2b']
    d = np.array(dset['bn2c_branch2b_beta:0' ], dtype=np.float32)
    e = np.array(dset['bn2c_branch2b_gamma:0'], dtype=np.float32)
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_2/conv_bn_relu1_1/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_2/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_2/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = d
    #########
    dset = fff['res2c_branch2c']
    #a = dset['res2c_branch2c']
    b = np.array(dset['res2c_branch2c_W:0'], dtype=np.float32)
    dset = fff['bn2c_branch2c']
    #a = dset['bn2c_branch2c']
    d = np.array(dset['bn2c_branch2c_beta:0' ], dtype=np.float32)
    e = np.array(dset['bn2c_branch2c_gamma:0'], dtype=np.float32)
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_2/conv_bn1_1/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_2/conv_bn1_1/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_0/resnet_unit2_2/conv_bn1_1/batchnorm1_0/BatchNorm/beta:0' ] = d
    ########res3########
    dset = fff['res3a_branch1']
    #a = dset['res3a_branch1']
    b = np.array(dset['res3a_branch1_W:0'], dtype=np.float32)
    dset = fff['bn3a_branch1']
    #a = dset['bn3a_branch1']
    d = np.array(dset['bn3a_branch1_beta:0' ], dtype=np.float32)
    e = np.array(dset['bn3a_branch1_gamma:0'], dtype=np.float32)
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = d
    #########
    dset = fff['res3a_branch2a']
    #a = dset['res3a_branch2a']
    b = np.array(dset['res3a_branch2a_W:0'], dtype=np.float32)
    dset = fff['bn3a_branch2a']
    #a = dset['bn3a_branch2a']
    d = np.array(dset['bn3a_branch2a_beta:0' ], dtype=np.float32)
    e = np.array(dset['bn3a_branch2a_gamma:0'], dtype=np.float32)
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_0/conv_bn_relu1_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_0/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_0/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = d
    #########
    dset = fff['res3a_branch2b']
    #a = dset['res3a_branch2b']
    b = np.array(dset['res3a_branch2b_W:0'], dtype=np.float32)
    dset = fff['bn3a_branch2b']
    #a = dset['bn3a_branch2b']
    d = np.array(dset['bn3a_branch2b_beta:0' ], dtype=np.float32)
    e = np.array(dset['bn3a_branch2b_gamma:0'], dtype=np.float32)
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_0/conv_bn_relu1_1/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_0/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_0/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = d
    #########
    dset = fff['res3a_branch2c']
    #a = dset['res3a_branch2c']
    b = np.array(dset['res3a_branch2c_W:0'], dtype=np.float32)
    dset = fff['bn3a_branch2c']
    #a = dset['bn3a_branch2c']
    d = np.array(dset['bn3a_branch2c_beta:0' ], dtype=np.float32)
    e = np.array(dset['bn3a_branch2c_gamma:0'], dtype=np.float32)
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_0/conv_bn1_1/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_0/conv_bn1_1/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_0/conv_bn1_1/batchnorm1_0/BatchNorm/beta:0' ] = d
    ################################
    dset = fff['res3b_branch2a']
    #a = dset['res3b_branch2a']
    b = np.array(dset['res3b_branch2a_W:0'], dtype=np.float32)
    dset = fff['bn3b_branch2a']
    #a = dset['bn3b_branch2a']
    d = np.array(dset['bn3b_branch2a_beta:0' ], dtype=np.float32)
    e = np.array(dset['bn3b_branch2a_gamma:0'], dtype=np.float32)
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_1/conv_bn_relu1_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_1/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_1/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = d
    #########
    dset = fff['res3b_branch2b']
    #a = dset['res3b_branch2b']
    b = np.array(dset['res3b_branch2b_W:0'], dtype=np.float32)
    dset = fff['bn3b_branch2b']
    #a = dset['bn3b_branch2b']
    d = np.array(dset['bn3b_branch2b_beta:0' ], dtype=np.float32)
    e = np.array(dset['bn3b_branch2b_gamma:0'], dtype=np.float32)
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_1/conv_bn_relu1_1/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_1/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_1/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = d
    #########
    dset = fff['res3b_branch2c']
    #a = dset['res3b_branch2c']
    b = np.array(dset['res3b_branch2c_W:0'], dtype=np.float32)
    dset = fff['bn3b_branch2c']
    #a = dset['bn3b_branch2c']
    d = np.array(dset['bn3b_branch2c_beta:0' ], dtype=np.float32)
    e = np.array(dset['bn3b_branch2c_gamma:0'], dtype=np.float32)
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_1/conv_bn1_1/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_1/conv_bn1_1/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_1/conv_bn1_1/batchnorm1_0/BatchNorm/beta:0' ] = d
    ############################
    dset = fff['res3c_branch2a']
    #a = dset['res3c_branch2a']
    b = np.array(dset['res3c_branch2a_W:0'], dtype=np.float32)
    dset = fff['bn3c_branch2a']
    #a = dset['bn3c_branch2a']
    d = np.array(dset['bn3c_branch2a_beta:0' ], dtype=np.float32)
    e = np.array(dset['bn3c_branch2a_gamma:0'], dtype=np.float32)
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_2/conv_bn_relu1_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_2/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_2/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = d
    #########
    dset = fff['res3c_branch2b']
    #a = dset['res3c_branch2b']
    b = np.array(dset['res3c_branch2b_W:0'], dtype=np.float32)
    dset = fff['bn3c_branch2b']
    #a = dset['bn3c_branch2b']
    d = np.array(dset['bn3c_branch2b_beta:0' ], dtype=np.float32)
    e = np.array(dset['bn3c_branch2b_gamma:0'], dtype=np.float32)
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_2/conv_bn_relu1_1/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_2/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_2/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = d
    #########
    dset = fff['res3c_branch2c']
    #a = dset['res3c_branch2c']
    b = np.array(dset['res3c_branch2c_W:0'], dtype=np.float32)
    dset = fff['bn3c_branch2c']
    #a = dset['bn3c_branch2c']
    d = np.array(dset['bn3c_branch2c_beta:0' ], dtype=np.float32)
    e = np.array(dset['bn3c_branch2c_gamma:0'], dtype=np.float32)
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_2/conv_bn1_1/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_2/conv_bn1_1/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_2/conv_bn1_1/batchnorm1_0/BatchNorm/beta:0' ] = d
    ############################
    dset = fff['res3d_branch2a']
    #a = dset['res3d_branch2a']
    b = np.array(dset['res3d_branch2a_W:0'], dtype=np.float32)
    dset = fff['bn3d_branch2a']
    #a = dset['bn3d_branch2a']
    d = np.array(dset['bn3d_branch2a_beta:0' ], dtype=np.float32)
    e = np.array(dset['bn3d_branch2a_gamma:0'], dtype=np.float32)
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_3/conv_bn_relu1_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_3/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_3/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = d
    #########
    dset = fff['res3d_branch2b']
    #a = dset['res3d_branch2b']
    b = np.array(dset['res3d_branch2b_W:0'], dtype=np.float32)
    dset = fff['bn3d_branch2b']
    #a = dset['bn3d_branch2b']
    d = np.array(dset['bn3d_branch2b_beta:0' ], dtype=np.float32)
    e = np.array(dset['bn3d_branch2b_gamma:0'], dtype=np.float32)
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_3/conv_bn_relu1_1/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_3/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_3/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = d
    #########
    dset = fff['res3d_branch2c']
    #a = dset['res3d_branch2c']
    b = np.array(dset['res3d_branch2c_W:0'], dtype=np.float32)
    dset = fff['bn3d_branch2c']
    #a = dset['bn3d_branch2c']
    d = np.array(dset['bn3d_branch2c_beta:0' ], dtype=np.float32)
    e = np.array(dset['bn3d_branch2c_gamma:0'], dtype=np.float32)
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_3/conv_bn1_1/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_3/conv_bn1_1/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_1/resnet_unit2_3/conv_bn1_1/batchnorm1_0/BatchNorm/beta:0' ] = d
    ########res4########
    dset = fff['res4a_branch1']
    #a = dset['res4a_branch1']
    b = np.array(dset['res4a_branch1_W:0'], dtype=np.float32)
    dset = fff['bn4a_branch1']
    #a = dset['bn4a_branch1']
    d = np.array(dset['bn4a_branch1_beta:0' ], dtype=np.float32)
    e = np.array(dset['bn4a_branch1_gamma:0'], dtype=np.float32)
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = d
    #########
    dset = fff['res4a_branch2a']
    #a = dset['res4a_branch2a']
    b = np.array(dset['res4a_branch2a_W:0'], dtype=np.float32)
    dset = fff['bn4a_branch2a']
    #a = dset['bn4a_branch2a']
    d = np.array(dset['bn4a_branch2a_beta:0' ], dtype=np.float32)
    e = np.array(dset['bn4a_branch2a_gamma:0'], dtype=np.float32)
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_0/conv_bn_relu1_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_0/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_0/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = d
    #########
    dset = fff['res4a_branch2b']
    #a = dset['res4a_branch2b']
    b = np.array(dset['res4a_branch2b_W:0'], dtype=np.float32)
    dset = fff['bn4a_branch2b']
    #a = dset['bn4a_branch2b']
    d = np.array(dset['bn4a_branch2b_beta:0' ], dtype=np.float32)
    e = np.array(dset['bn4a_branch2b_gamma:0'], dtype=np.float32)
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_0/conv_bn_relu1_1/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_0/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_0/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = d
    #########
    dset = fff['res4a_branch2c']
    #a = dset['res4a_branch2c']
    b = np.array(dset['res4a_branch2c_W:0'], dtype=np.float32)
    dset = fff['bn4a_branch2c']
    #a = dset['bn4a_branch2c']
    d = np.array(dset['bn4a_branch2c_beta:0' ], dtype=np.float32)
    e = np.array(dset['bn4a_branch2c_gamma:0'], dtype=np.float32)
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_0/conv_bn1_1/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_0/conv_bn1_1/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_0/conv_bn1_1/batchnorm1_0/BatchNorm/beta:0' ] = d
    ################################
    dset = fff['res4b_branch2a']
    #a = dset['res4b_branch2a']
    b = np.array(dset['res4b_branch2a_W:0'], dtype=np.float32)
    dset = fff['bn4b_branch2a']
    #a = dset['bn4b_branch2a']
    d = np.array(dset['bn4b_branch2a_beta:0' ], dtype=np.float32)
    e = np.array(dset['bn4b_branch2a_gamma:0'], dtype=np.float32)
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_1/conv_bn_relu1_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_1/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_1/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = d
    #########
    dset = fff['res4b_branch2b']
    #a = dset['res4b_branch2b']
    b = np.array(dset['res4b_branch2b_W:0'], dtype=np.float32)
    dset = fff['bn4b_branch2b']
    #a = dset['bn4b_branch2b']
    d = np.array(dset['bn4b_branch2b_beta:0' ], dtype=np.float32)
    e = np.array(dset['bn4b_branch2b_gamma:0'], dtype=np.float32)
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_1/conv_bn_relu1_1/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_1/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_1/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = d
    #########
    dset = fff['res4b_branch2c']
    #a = dset['res4b_branch2c']
    b = np.array(dset['res4b_branch2c_W:0'], dtype=np.float32)
    dset = fff['bn4b_branch2c']
    #a = dset['bn4b_branch2c']
    d = np.array(dset['bn4b_branch2c_beta:0' ], dtype=np.float32)
    e = np.array(dset['bn4b_branch2c_gamma:0'], dtype=np.float32)
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_1/conv_bn1_1/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_1/conv_bn1_1/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_1/conv_bn1_1/batchnorm1_0/BatchNorm/beta:0' ] = d
    ############################
    dset = fff['res4c_branch2a']
    #a = dset['res4c_branch2a']
    b = np.array(dset['res4c_branch2a_W:0'], dtype=np.float32)
    dset = fff['bn4c_branch2a']
    #a = dset['bn4c_branch2a']
    d = np.array(dset['bn4c_branch2a_beta:0' ], dtype=np.float32)
    e = np.array(dset['bn4c_branch2a_gamma:0'], dtype=np.float32)
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_2/conv_bn_relu1_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_2/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_2/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = d
    #########
    dset = fff['res4c_branch2b']
    #a = dset['res4c_branch2b']
    b = np.array(dset['res4c_branch2b_W:0'], dtype=np.float32)
    dset = fff['bn4c_branch2b']
    #a = dset['bn4c_branch2b']
    d = np.array(dset['bn4c_branch2b_beta:0' ], dtype=np.float32)
    e = np.array(dset['bn4c_branch2b_gamma:0'], dtype=np.float32)
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_2/conv_bn_relu1_1/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_2/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_2/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = d
    #########
    dset = fff['res4c_branch2c']
    #a = dset['res4c_branch2c']
    b = np.array(dset['res4c_branch2c_W:0'], dtype=np.float32)
    dset = fff['bn4c_branch2c']
    #a = dset['bn4c_branch2c']
    d = np.array(dset['bn4c_branch2c_beta:0' ], dtype=np.float32)
    e = np.array(dset['bn4c_branch2c_gamma:0'], dtype=np.float32)
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_2/conv_bn1_1/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_2/conv_bn1_1/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_2/conv_bn1_1/batchnorm1_0/BatchNorm/beta:0' ] = d
    ############################
    dset = fff['res4d_branch2a']
    #a = dset['res4d_branch2a']
    b = np.array(dset['res4d_branch2a_W:0'], dtype=np.float32)
    dset = fff['bn4d_branch2a']
    #a = dset['bn4d_branch2a']
    d = np.array(dset['bn4d_branch2a_beta:0' ], dtype=np.float32)
    e = np.array(dset['bn4d_branch2a_gamma:0'], dtype=np.float32)
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_3/conv_bn_relu1_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_3/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_3/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = d
    #########
    dset = fff['res4d_branch2b']
    #a = dset['res4d_branch2b']
    b = np.array(dset['res4d_branch2b_W:0'], dtype=np.float32)
    dset = fff['bn4d_branch2b']
    #a = dset['bn4d_branch2b']
    d = np.array(dset['bn4d_branch2b_beta:0' ], dtype=np.float32)
    e = np.array(dset['bn4d_branch2b_gamma:0'], dtype=np.float32)
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_3/conv_bn_relu1_1/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_3/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_3/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = d
    #########
    dset = fff['res4d_branch2c']
    #a = dset['res4d_branch2c']
    b = np.array(dset['res4d_branch2c_W:0'], dtype=np.float32)
    dset = fff['bn4d_branch2c']
    #a = dset['bn4d_branch2c']
    d = np.array(dset['bn4d_branch2c_beta:0' ], dtype=np.float32)
    e = np.array(dset['bn4d_branch2c_gamma:0'], dtype=np.float32)
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_3/conv_bn1_1/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_3/conv_bn1_1/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_3/conv_bn1_1/batchnorm1_0/BatchNorm/beta:0' ] = d
    ############################
    dset = fff['res4e_branch2a']
    #a = dset['res4e_branch2a']
    b = np.array(dset['res4e_branch2a_W:0'], dtype=np.float32)
    dset = fff['bn4e_branch2a']
    #a = dset['bn4e_branch2a']
    d = np.array(dset['bn4e_branch2a_beta:0' ], dtype=np.float32)
    e = np.array(dset['bn4e_branch2a_gamma:0'], dtype=np.float32)
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_4/conv_bn_relu1_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_4/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_4/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = d
    #########
    dset = fff['res4e_branch2b']
    #a = dset['res4e_branch2b']
    b = np.array(dset['res4e_branch2b_W:0'], dtype=np.float32)
    dset = fff['bn4e_branch2b']
    #a = dset['bn4e_branch2b']
    d = np.array(dset['bn4e_branch2b_beta:0' ], dtype=np.float32)
    e = np.array(dset['bn4e_branch2b_gamma:0'], dtype=np.float32)
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_4/conv_bn_relu1_1/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_4/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_4/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = d
    #########
    dset = fff['res4e_branch2c']
    #a = dset['res4e_branch2c']
    b = np.array(dset['res4e_branch2c_W:0'], dtype=np.float32)
    dset = fff['bn4e_branch2c']
    #a = dset['bn4e_branch2c']
    d = np.array(dset['bn4e_branch2c_beta:0' ], dtype=np.float32)
    e = np.array(dset['bn4e_branch2c_gamma:0'], dtype=np.float32)
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_4/conv_bn1_1/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_4/conv_bn1_1/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_4/conv_bn1_1/batchnorm1_0/BatchNorm/beta:0' ] = d
    ############################
    dset = fff['res4f_branch2a']
    #a = dset['res4f_branch2a']
    b = np.array(dset['res4f_branch2a_W:0'], dtype=np.float32)
    dset = fff['bn4f_branch2a']
    #a = dset['bn4f_branch2a']
    d = np.array(dset['bn4f_branch2a_beta:0' ], dtype=np.float32)
    e = np.array(dset['bn4f_branch2a_gamma:0'], dtype=np.float32)
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_5/conv_bn_relu1_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_5/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_5/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = d
    #########
    dset = fff['res4f_branch2b']
    #a = dset['res4f_branch2b']
    b = np.array(dset['res4f_branch2b_W:0'], dtype=np.float32)
    dset = fff['bn4f_branch2b']
    #a = dset['bn4f_branch2b']
    d = np.array(dset['bn4f_branch2b_beta:0' ], dtype=np.float32)
    e = np.array(dset['bn4f_branch2b_gamma:0'], dtype=np.float32)
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_5/conv_bn_relu1_1/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_5/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_5/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = d
    #########
    dset = fff['res4f_branch2c']
    #a = dset['res4f_branch2c']
    b = np.array(dset['res4f_branch2c_W:0'], dtype=np.float32)
    dset = fff['bn4f_branch2c']
    #a = dset['bn4f_branch2c']
    d = np.array(dset['bn4f_branch2c_beta:0' ], dtype=np.float32)
    e = np.array(dset['bn4f_branch2c_gamma:0'], dtype=np.float32)
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_5/conv_bn1_1/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_5/conv_bn1_1/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_2/resnet_unit2_5/conv_bn1_1/batchnorm1_0/BatchNorm/beta:0' ] = d
    ########res5########
    dset = fff['res5a_branch1']
    #a = dset['res5a_branch1']
    b = np.array(dset['res5a_branch1_W:0'], dtype=np.float32)
    dset = fff['bn5a_branch1']
    #a = dset['bn5a_branch1']
    d = np.array(dset['bn5a_branch1_beta:0' ], dtype=np.float32)
    e = np.array(dset['bn5a_branch1_gamma:0'], dtype=np.float32)
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = d
    #########
    dset = fff['res5a_branch2a']
    #a = dset['res5a_branch2a']
    b = np.array(dset['res5a_branch2a_W:0'], dtype=np.float32)
    dset = fff['bn5a_branch2a']
    #a = dset['bn5a_branch2a']
    d = np.array(dset['bn5a_branch2a_beta:0' ], dtype=np.float32)
    e = np.array(dset['bn5a_branch2a_gamma:0'], dtype=np.float32)
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_0/conv_bn_relu1_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_0/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_0/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = d
    #########
    dset = fff['res5a_branch2b']
    #a = dset['res5a_branch2b']
    b = np.array(dset['res5a_branch2b_W:0'], dtype=np.float32)
    dset = fff['bn5a_branch2b']
    #a = dset['bn5a_branch2b']
    d = np.array(dset['bn5a_branch2b_beta:0' ], dtype=np.float32)
    e = np.array(dset['bn5a_branch2b_gamma:0'], dtype=np.float32)
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_0/conv_bn_relu1_1/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_0/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_0/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = d
    #########
    dset = fff['res5a_branch2c']
    #a = dset['res5a_branch2c']
    b = np.array(dset['res5a_branch2c_W:0'], dtype=np.float32)
    dset = fff['bn5a_branch2c']
    #a = dset['bn5a_branch2c']
    d = np.array(dset['bn5a_branch2c_beta:0' ], dtype=np.float32)
    e = np.array(dset['bn5a_branch2c_gamma:0'], dtype=np.float32)
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_0/conv_bn1_1/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_0/conv_bn1_1/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_0/conv_bn1_1/batchnorm1_0/BatchNorm/beta:0' ] = d
    ################################
    dset = fff['res5b_branch2a']
    #a = dset['res5b_branch2a']
    b = np.array(dset['res5b_branch2a_W:0'], dtype=np.float32)
    dset = fff['bn5b_branch2a']
    #a = dset['bn5b_branch2a']
    d = np.array(dset['bn5b_branch2a_beta:0' ], dtype=np.float32)
    e = np.array(dset['bn5b_branch2a_gamma:0'], dtype=np.float32)
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_1/conv_bn_relu1_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_1/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_1/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = d
    #########
    dset = fff['res5b_branch2b']
    #a = dset['res5b_branch2b']
    b = np.array(dset['res5b_branch2b_W:0'], dtype=np.float32)
    dset = fff['bn5b_branch2b']
    #a = dset['bn5b_branch2b']
    d = np.array(dset['bn5b_branch2b_beta:0' ], dtype=np.float32)
    e = np.array(dset['bn5b_branch2b_gamma:0'], dtype=np.float32)
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_1/conv_bn_relu1_1/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_1/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_1/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = d
    #########
    dset = fff['res5b_branch2c']
    #a = dset['res5b_branch2c']
    b = np.array(dset['res5b_branch2c_W:0'], dtype=np.float32)
    dset = fff['bn5b_branch2c']
    #a = dset['bn5b_branch2c']
    d = np.array(dset['bn5b_branch2c_beta:0' ], dtype=np.float32)
    e = np.array(dset['bn5b_branch2c_gamma:0'], dtype=np.float32)
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_1/conv_bn1_1/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_1/conv_bn1_1/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_1/conv_bn1_1/batchnorm1_0/BatchNorm/beta:0' ] = d
    ############################
    dset = fff['res5c_branch2a']
    #a = dset['res5c_branch2a']
    b = np.array(dset['res5c_branch2a_W:0'], dtype=np.float32)
    dset = fff['bn5c_branch2a']
    #a = dset['bn5c_branch2a']
    d = np.array(dset['bn5c_branch2a_beta:0' ], dtype=np.float32)
    e = np.array(dset['bn5c_branch2a_gamma:0'], dtype=np.float32)
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_2/conv_bn_relu1_0/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_2/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_2/conv_bn_relu1_0/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = d
    #########
    dset = fff['res5c_branch2b']
    #a = dset['res5c_branch2b']
    b = np.array(dset['res5c_branch2b_W:0'], dtype=np.float32)
    dset = fff['bn5c_branch2b']
    #a = dset['bn5c_branch2b']
    d = np.array(dset['bn5c_branch2b_beta:0' ], dtype=np.float32)
    e = np.array(dset['bn5c_branch2b_gamma:0'], dtype=np.float32)
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_2/conv_bn_relu1_1/conv_bn1_0/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_2/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_2/conv_bn_relu1_1/conv_bn1_0/batchnorm1_0/BatchNorm/beta:0' ] = d
    #########
    dset = fff['res5c_branch2c']
    #a = dset['res5c_branch2c']
    b = np.array(dset['res5c_branch2c_W:0'], dtype=np.float32)
    dset = fff['bn5c_branch2c']
    #a = dset['bn5c_branch2c']
    d = np.array(dset['bn5c_branch2c_beta:0' ], dtype=np.float32)
    e = np.array(dset['bn5c_branch2c_gamma:0'], dtype=np.float32)
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_2/conv_bn1_1/conv1_0/weights:0'] = b
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_2/conv_bn1_1/batchnorm1_0/BatchNorm/gamma:0'] = e
    mydict['layers_module1_1/resnet_block2_0_3/resnet_unit2_2/conv_bn1_1/batchnorm1_0/BatchNorm/beta:0' ] = d
    return mydict