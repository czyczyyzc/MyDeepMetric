import numpy as np
import tensorflow as tf

from Mybase import layers
from Mybase.layers import *
from Mybase.layers_utils import *
from Mybase.losses import *

from Mybase.deep_metric_utils.bbox import *
from Mybase.deep_metric_utils.embcls_layer import *
from Mybase.deep_metric_utils.embcls_target_layer import *


class Resnet101_Deep_Metric(object):
    
    def __init__(self, cls_num=81, reg=1e-4, drp=0.5, typ=tf.float32):
        
        self.cls_num = cls_num #class number
        self.reg     = reg     #regularization
        self.drp     = drp     #dropout
        self.typ     = typ     #dtype
        
        self.mod_tra = True    #mode training
        self.glb_pol = False   #global pooling
        self.inc_btm = True    #include bottom block
        
        #resnet block setting
        self.res_set = [( 256,  64, [1, 1], [1, 1],  3, True ),  #conv2x 256 /4
                        ( 512, 128, [1, 1], [2, 2],  4, True ),  #conv3x 128 /8  #use
                        (1024, 256, [1, 1], [2, 2], 23, True ),  #conv4x 64  /16 #23--->101 #6--->50
                        (2048, 512, [1, 1], [2, 2],  3, True )]  #conv5x 64  /16 #use
        self.out_srd = 4 #output stride

        
    def forward(self, imgs=None, lbls=None, gbxs=None, gmks=None, gbx_nums=None, mtra=None, scp=None): 
        #lbl=label #scp=scope #mod_tra=mode train
        
        img_shp = imgs.get_shape().as_list()
        img_num, img_hgt, img_wdh = img_shp[0], img_shp[1], img_shp[2]
        fet_hgt = img_hgt // (self.out_srd*2)
        fet_wdh = img_wdh // (self.out_srd*2)
        img_shp = np.stack([img_hgt, img_wdh], axis=0)
        fet_shp = np.stack([fet_hgt, fet_wdh], axis=0)
        #common parameters
        com_pams = {
            'com':    {'reg': self.reg, 'wscale': 0.01, 'dtype': self.typ, 'reuse': False, 'is_train': False, 'trainable': True},
            'bn':     {'eps': 1e-5, 'decay': 0.9997}, #0.9997
            'relu':   {'alpha': -0.1},
            'conv':   {'number': 64,'shape':[7, 7],'rate':1,'stride':[2, 2],'padding':'SAME','use_bias':True},
            'deconv': {'number':256,'shape':[2, 2],'rate':1,'stride':[2, 2],'padding':'SAME','out_shape':[128, 128],'use_bias':False},
            'max_pool':     {'shape': [3, 3], 'stride': [2, 2], 'padding': 'VALID'},
            'resnet_block': {'block_setting': self.res_set, 'output_stride': self.out_srd},
            'resnet_unit':  {'depth_output':1024, 'depth_bottle':256, 'use_branch':True, 'shape':[1, 1], 'stride':[1, 1], 'rate':1},
            'pyramid':   {'depth': 256},
            'glb_pool':  {'axis':  [1, 2]},
            'reshape':   {'shape': []},
            'squeeze':   {'axis':  [1, 2]},
            'transpose': {'perm':  [0, 3, 1, 2, 4]},
            'affine':    {'dim': 1024, 'use_bias': True},
            #'bilstm':   {'num_h': self.fet_dep//2, 'num_o': None, 'fbias': 1.0, 'tmajr': False},
            #'concat':   {'axis': 0},
            #'split':    {'axis': 0, 'number': img_num},
            #'dropout':  {'keep_p': self.dropout},
        }
        #####################Get the first feature map!####################
        if self.inc_btm:
            print('Get the first feature map!')
            opas = {'op': [{'op': 'conv_bn_relu1', 'loop': 1, 'params':{'com':{'trainable': True }}}, #(None, 512, 512, 64)
                           {'op': 'max_pool1',     'loop': 1, 'params':{}}, #(None, 256, 256, 64) #pool2
                          ], 'loop': 1}
            tsr_out = layers_module1(imgs, 0, com_pams, opas, mtra)
            print('')
        #####################Get the resnet blocks!#########################
        print('Get the resnet block!')
        fet_lst = []
        opas = {'op': [{'op': 'resnet_block2', 'loop': 1, 'params':{}}], 'loop': 1}
        fet_lst.extend(layers_module1(tsr_out, 1, com_pams, opas, mtra))
        assert len(fet_lst) == 4, 'The first resnet block is worng!'
        tsr_out = fet_lst[-1]
        print('')
        ################Get the image classification results!###############
        if self.glb_pol: # Global average pooling.
            print('Get the image classification results!')
            com_pams['conv'] = {'number':self.cls_num,'shape':[1, 1],'rate':1,'stride':[1, 1],'padding':'SAME','use_bias':True}
            opas = {'op': [{'op': 'glb_pool1', 'loop': 1, 'params': {}},
                           {'op': 'conv1',     'loop': 1, 'params': {}},
                           {'op': 'squeeze1',  'loop': 1, 'params': {}},
                          ], 'loop': 1}
            scrs     = layers_module1(tsr_out, 99, com_pams, opas, mtra) #class scores
            prbs     = tf.nn.softmax(scrs) #class probabilities
            los_dat  = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=lbls, logits=scrs))
            los_reg  = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            los      = los_dat + los_reg
            loss     = tf.stack([los, los_dat, los_reg], axis=0)
            return loss, scrs
            print('')
        ###############Get embedding vectors for each pixel!################
        print('Get embedding vectors for each pixel!')
        '''
        We start by learning an embedding space, so that pixels that correspond to the same object instance 
        are close, and pixels that correspond to different objects (including the background) are far. 
        That is to say, embedding vectors which are similar are more likely to belong to the same object instance.
        '''
        com_pams['com']['trainable'] = True
        com_pams['conv'] = {'number': 256, 'shape': [1, 1], 'rate': 1, 'stride': [1, 1], 'padding': 'SAME', 'use_bias': True}
        opas = {'op': [{'op': 'conv_bn_relu1',   'loop': 4, 'params':{'conv':{'shape': [3, 3]}}},
                       {"op": 'deconv_bn_relu1', 'loop': 1, 'params':{}},
                       {'op': 'conv1',           'loop': 1, 'params':{'conv':{'number': 64}}},
                      ], 'loop': 1}
        fet_emb = layers_module1(tsr_out, 2, com_pams, opas, mtra) #(N, H, W, 64)
        print('')
        ####Get Mask classification and seedness scores for each pixel!#####
        print('Get Mask classification and seedness scores for each pixel!')
        '''
        The model predicts a class label for the mask centered at each pixel, as well as a confidence score 
        that this pixel would make a good “seed” for creating a mask.
        That is to say, the model predicts the classification score of the mask each pixel will generate if 
        picked as a seed. We derive the seediness scores from the classification scores and use them to choose 
        which seed points in the image to sample. Each seed point generates a mask based on the embedding 
        vectors; each mask is then associated with a class label and confidence score.
        '''
        com_pams['com']['trainable'] = True
        com_pams['conv'] = {'number': 256, 'shape': [1, 1], 'rate': 1, 'stride': [1, 1], 'padding': 'SAME', 'use_bias': True}
        opas = {'op': [{'op': 'conv_bn_relu1',   'loop': 4, 'params':{'conv':{'shape': [3, 3]}}},
                       {"op": 'deconv_bn_relu1', 'loop': 1, 'params':{}},
                       {'op': 'conv1',           'loop': 1, 'params':{'conv':{'number': self.cls_num*4}}},
                       {'op': 'reshape1',        'loop': 1, 'params':{'reshape':{'shape': [img_num, fet_hgt, fet_wdh, 4, -1]}}},
                      ], 'loop': 1}
        fet_cls = layers_module1(tsr_out, 3, com_pams, opas, mtra) #(N, H, W, 4, C+1)
        print('')
        #####################Get the mask prediction!#######################
        print('Get the mask prediction!')
        EL = EmbclsLayer(self.cls_num, img_shp, fet_shp)
        boxs, msks, msk_clss, msk_prbs, msk_nums = EL.generate_msks(fet_emb, fet_cls)
        print('')
        
        if self.mod_tra: 
            #######################Get the mask losses!##########################
            print('Get the mask losses!')
            ET = EmbClsTargetLayer(self.cls_num, img_shp, fet_shp)
            sims_los, prbs_los = ET.generate_embcls_loss(fet_emb, fet_cls, gbxs, gmks, gbx_nums)
            los_dat  = sims_los * 10.0 + prbs_los * 2.0
            los_reg  = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            los      = los_dat  + los_reg
            loss     = tf.stack([los, sims_los, prbs_los, los_reg], axis=0)
            print('')
            return loss, boxs, msks, msk_clss, msk_prbs, msk_nums
        else:
            return boxs, msks, msk_clss, msk_prbs, msk_nums