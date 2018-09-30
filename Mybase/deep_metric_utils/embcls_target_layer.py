import cv2
import numpy as np
import tensorflow as tf
from Mybase.losses import *
from Mybase.comp_utils import *
from Mybase.deep_metric_utils.embcls_layer import *
from .bbox import *

class EmbClsTargetLayer(object):
    '''
    The similarity between pixels p and q is computed as follows:
        σ(p, q) = 2 / (1 + exp(||ep - eq||2^2))
                = 2 / (1 + exp(d))            d = ||ep - eq||2^2
                = 2 * (1 - 1/(1+exp(-d)))
                = 2 * (1 - sigmoid(d)) ∈ (0, 1]
                ??? sigmoid(ep * eq / sqrt(d)) ???
        ||tensor||2 = tf.norm(tensor, ord='euclidean', axis=-1, keepdims=False)
    We train the network by minimizing the following loss:
        Le = -1/|S| * ∑(p,q∈S) wpq*[1{yp=yq}*log(σ(p,q)) + 1{yp≠yq}*log(1-σ(p,q))]
        tf.nn.sigmoid_cross_entropy_with_logits(labels=z, logits=x)
        = z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        = z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
        = x - x * z + log(1+)
        sigmoid(x) = 1 / (1 + exp(-x))
        where S is the set of pixels that we choose, yp is the instance label of pixel p, 
        and wpq is the weight of the loss on the similarity between p and q. The weights 
        wpq are set to values inversely proportional to the size of the instances p and q 
        belong to, so the loss will not become biased towards the larger examples. We 
        normalize weights so that ∑(p,q)wpq = 1(p,q所处的两个实例的并集中的计算, wpq = 1 / (Sp * Sq)).
        |S|为图片中所选择的实例的个数。
    In contrast to semantic segmentation, where the pixels themselves are classified, here 
    we classify the mask that each pixel will generate if chosen as a seed. We see that most
    pixels inside the object make good seeds, but pixels near the boundaries are not so good.
    '''
    def __init__(self, cls_num=None, img_shp=None, fet_shp=None):
        
        self.cls_num     = cls_num
        self.img_shp     = img_shp
        self.fet_shp     = fet_shp
        self.fet_srd     = img_shp[0] / fet_shp[0]
        self.pix_smp_num = 16
        self.pix_sim_num = 4
        self.pix_msk_min = 0.5
        self.msk_ovp_min = 0.5
        self.EL          = EmbclsLayer(cls_num, img_shp, fet_shp)
        
    def generate_embcls_loss_img(self, elems=None):
        
        fet_emb, fet_cls, gbxs, gmks, gbx_num = elems
        #fet_emb-->(H, W, 64), fet_cls-->(H, W, 4, C+1)
        #若gbx_num==1，即无正样本，gbxs->(0,5)，gmks->(0, H, W)
        gbxs = gbxs[0:gbx_num]                                                             #(M+1, 5)    最后一个是无效边框
        gmks = gmks[0:gbx_num]                                                             #(M+1, H, W) 最后一个是无效Mask
        
        #把gmks根据gbxs融进fet_shp大小的特征图中
        def recover_mask(elems=None):
            gbx, gmk = elems                                                               #(5), (H‘, W’)
            crd = tf.cast(gbx[0:-1]/self.fet_srd, dtype=tf.int32)                          #(4)
            ymn = crd[0]
            xmn = crd[1]
            ymx = crd[2]
            xmx = crd[3]
            leh = tf.stack([ymx-ymn+1, xmx-xmn+1], axis=0)
            gmk = tf.expand_dims(gmk, axis=-1)                                             #(H“, W”, 1)
            gmk = tf.image.resize_images(gmk, leh, method=tf.image.ResizeMethod.BILINEAR, align_corners=False) #(H“, W”, 1)
            gmk = tf.squeeze(gmk, axis=[-1])                                               #(H“, W”)
            gmk = tf.cast(gmk>=self.pix_msk_min, dtype=tf.float32)                         #(H“, W”)
            paddings = [[ymn, self.fet_shp[0]-ymx-1], [xmn, self.fet_shp[1]-xmx-1]]        #(H, W)
            gmk = tf.pad(gmk, paddings, 'CONSTANT')
            return gmk
        #当没有box时，tf.map_fn不会拆开，所以tf.image.resize_images是安全的
        elems     = [gbxs[0:-1], gmks[0:-1]]
        gmks      = tf.map_fn(recover_mask, elems, dtype=tf.float32, \
                              parallel_iterations=10, back_prop=False, swap_memory=True, infer_shape=True) #(M, H, W)
        #得到纯净的背景mask(不包含任何形式的crowd)
        gmk_bgd   = tf.cast(tf.equal(tf.reduce_sum(gmks, axis=0, keepdims=True), 0), dtype=tf.float32)     #(1, H, W)
        gmks      = tf.concat([gmks, gmk_bgd], axis=0)                                     #(M+1, H, W)
        #统计各mask的pixels数量
        pix_nums  = tf.reduce_sum(gmks, axis=[1, 2])                                       #(M+1)
        #剔除不合理的mask(包含crowd)
        ncw_idxs0 = tf.where(pix_nums>=self.pix_smp_num)
        ncw_idxs1 = tf.where(gbxs[:, 4]>=0)                                                #最后一个是背景
        ncw_msks  = tf.zeros(shape=[gbx_num], dtype=tf.float32)
        ncw_msks0 = tensor_update(ncw_msks, ncw_idxs0, 1.0)
        ncw_msks1 = tensor_update(ncw_msks, ncw_idxs1, 1.0)
        ncw_msks  = ncw_msks0 * ncw_msks1
        ncw_idxs  = tf.where(tf.equal(ncw_msks, 1.0))
        gbxs      = tf.gather_nd(gbxs, ncw_idxs)                                           #(M, 5)
        gmks      = tf.gather_nd(gmks, ncw_idxs)                                           #(M, H, W)
        gbx_num   = tf.shape(gbxs)[0]                                                      #M
        pix_num   = gbx_num * self.pix_smp_num                                             #M*10
        
        ##########################Get the loss for embeddings!#########################
        #随机采样K个点
        def sample_pixels(elems=None):
            gmk      = elems                                                               #(H, W)
            pix_idxs = tf.where(tf.equal(gmk, 1.0))                                        #(N, 2)
            pix_idxs = tf.random_shuffle(pix_idxs)[:self.pix_smp_num]                      #(K, 2)
            return pix_idxs
        elems     = gmks
        pix_idxs  = tf.map_fn(sample_pixels, elems, dtype=tf.int64, \
                              parallel_iterations=10, back_prop=False, swap_memory=True, infer_shape=True) #(M, K, 2)
        pix_idxs  = tf.reshape(pix_idxs, [-1, 2])                                          #(M*K,  2)
        pix_embs  = tf.gather_nd(fet_emb, pix_idxs)                                        #(M*K, 64), (H, W, 64), (M*K, 2)
        #生成对应的prediciton
        '''
        sims_pst  = tf.matmul(pix_embs, pix_embs, transpose_b=True) / tf.sqrt(tf.cast(tf.shape(pix_embs)[-1], dtype=tf.float32))
        #sims_pst = tf.sigmoid(sims_pst)                                                   #(M*K, M*K)
        '''
        pix_embs0 = tf.reshape(pix_embs, [pix_num, 1, -1])                                 #(M*K,   1, 64)
        pix_embs1 = tf.reshape(pix_embs, [1, pix_num, -1])                                 #(  1, M*K, 64)
        #pix_embs = pix_embs0 - pix_embs1                                                  #(M*K, M*K, 64)
        pix_embs  = tf.abs(pix_embs0 - pix_embs1)                                          #(M*K, M*K, 64)
        pix_embs  = tf.clip_by_value(pix_embs, 1e-4, 5.0)                                  #(M*K, M*K, 64)
        pix_dsts  = tf.norm(pix_embs, ord='euclidean', axis=-1, keepdims=False) ** 2       #(M*K, M*K)
        #pix_dsts = tf.Print(pix_dsts, [pix_dsts], message=None, first_n=None, summarize=100)
        sims_pst  = 2.0 * (1.0 - tf.sigmoid(pix_dsts))                                     #(M*K, M*K)
        sims_pst  = tf.clip_by_value(sims_pst, 1e-4, 1-1e-4)
        #生成对应的target
        crd_offs  = tf.zeros(shape=[gbx_num, 1], dtype=tf.int32) + self.pix_smp_num        #(M,   1)
        crd_offs  = tf.cumsum(crd_offs, axis=0, exclusive=True, reverse=False)             #(M,   1)
        ycd_idxs  = tf.expand_dims(tf.range(self.pix_smp_num), axis=-1)                    #(K,   1)
        ycd_idxs  = tf.reshape(tf.tile(ycd_idxs, [1, self.pix_smp_num]), [1, -1])          #(1, K*K)
        ycd_idxs  = tf.add(ycd_idxs, crd_offs)                                             #(M, K*K)
        xcd_idxs  = tf.expand_dims(tf.range(self.pix_smp_num), axis= 0)                    #(1,   K)
        xcd_idxs  = tf.reshape(tf.tile(xcd_idxs, [self.pix_smp_num, 1]), [1, -1])          #(1, K*K)
        xcd_idxs  = tf.add(xcd_idxs, crd_offs)                                             #(M, K*K)
        crd_idxs  = tf.stack([ycd_idxs, xcd_idxs], axis=-1)                                #(M, K*K, 2)
        crd_idxs  = tf.reshape(crd_idxs, [-1, 2])                                          #(M*K*K,  2)
        sims_pre  = tf.zeros(shape=[pix_num, pix_num], dtype=tf.float32)                   #(M*K, M*K)
        sims_pre  = tensor_update(sims_pre, crd_idxs, 1.0)                                 #(M*K, M*K)
        #生成对应的loss
        '''
        sims_los  = tf.nn.sigmoid_cross_entropy_with_logits(labels=sims_pre, logits=sims_pst) #(M*K, M*K)
        sims_los  = tf.reduce_sum(sims_los)
        '''
        sims_los  = -(sims_pre * tf.log(sims_pst) + (1.0-sims_pre) * tf.log(1.0-sims_pst)) #(M*K*M*K)
        sims_los  = tf.reduce_sum(sims_los)
        ##########################Get the loss for classfication!#########################
        #再次随机采样K个点
        elems     = gmks
        pix_idxs  = tf.map_fn(sample_pixels, elems, dtype=tf.int64, \
                              parallel_iterations=10, back_prop=False, swap_memory=True, infer_shape=True) #(M, K, 2)
        #生成对应的prediciton
        prbs_pst  = tf.gather_nd(fet_cls, pix_idxs)                                        #(M, K, 4, C), (H, W, 4, C), (M, K, 2)
        #生成对应的target
        pix_idxs  = tf.reshape(pix_idxs, [-1, 2])                                          #(M*K, 2)
        msks      = self.EL.generate_msks_vld(pix_idxs, fet_emb)                           #(M*K, 4, H, W)
        msks      = tf.reshape(msks, [gbx_num, self.pix_smp_num, -1, self.fet_shp[0], self.fet_shp[1]]) #(M, K, 4, H, W)
        #计算每个点可能对应的类别标签
        prbs_pre0 = tf.cast(gbxs[:, -1], dtype=tf.int32)                                   #(M)
        prbs_pre0 = prbs_pre0[:, tf.newaxis, tf.newaxis]                                   #(M, 1, 1)
        prbs_pre0 = tf.tile(prbs_pre0, [1, self.pix_smp_num, self.pix_sim_num])            #(M, K, 4)
        prbs_pre1 = tf.zeros(shape=tf.shape(prbs_pre0), dtype=tf.int32)                    #(M, K, 4)
        #计算由K个点生成的mask和gmks之间的IOU
        gmks      = gmks[:, tf.newaxis, tf.newaxis, :, :]                                  #(M, 1, 1, H, W)
        gmks      = tf.cast(gmks, dtype=tf.bool)                                           #(M, 1, 1, H, W)
        msks      = tf.cast(msks, dtype=tf.bool)                                           #(M, K, 4, H, W)
        msk_iscs  = tf.cast(tf.logical_and(msks, gmks), dtype=tf.float32)                  #(M, K, 4, H, W)
        msk_uins  = tf.cast(tf.logical_or (msks, gmks), dtype=tf.float32)                  #(M, K, 4, H, W)
        msk_iscs  = tf.reduce_sum(msk_iscs, axis=[3, 4])                                   #(M, K, 4)
        msk_uins  = tf.reduce_sum(msk_uins, axis=[3, 4])                                   #(M, K, 4)
        msk_ovps  = tf.where(msk_uins>0, msk_iscs/msk_uins, tf.zeros(shape=tf.shape(msk_uins), dtype=tf.float32)) #(M, K, 4)
        prbs_pre  = tf.where(msk_ovps>=self.msk_ovp_min, prbs_pre0, prbs_pre1)             #(M, K, 4)
        #生成对应的loss
        prbs_los  = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=prbs_pre, logits=prbs_pst) #(M, K, 4)
        prbs_los  = tf.reduce_sum(prbs_los)
        #pix_num  = tf.Print(pix_num, [pix_num, sims_pst], message=None, first_n=None, summarize=100)
        pix_num0  = pix_num ** 2
        pix_num1  = pix_num *  4
        return sims_los, prbs_los, pix_num0, pix_num1

    def generate_embcls_loss(self, fet_emb=None, fet_cls=None, gbxs=None, gmks=None, gbx_nums=None):
        
        elems = [fet_emb, fet_cls, gbxs, gmks, gbx_nums]
        sims_loss, prbs_loss, pix_nums0, pix_nums1 = \
            tf.map_fn(self.generate_embcls_loss_img, elems, dtype=(tf.float32, tf.float32, tf.int32, tf.int32),
                      parallel_iterations=10, back_prop=True, swap_memory=True, infer_shape=True)
        sims_los = tf.reduce_sum(sims_loss)
        prbs_los = tf.reduce_sum(prbs_loss)
        pix_num0 = tf.cast(tf.reduce_sum(pix_nums0), dtype=tf.float32)
        pix_num1 = tf.cast(tf.reduce_sum(pix_nums1), dtype=tf.float32)
        sims_los = tf.cond(pix_num0>0, lambda: sims_los/pix_num0, lambda: tf.constant(0.0))
        prbs_los = tf.cond(pix_num1>0, lambda: prbs_los/pix_num1, lambda: tf.constant(0.0))
        return sims_los, prbs_los
