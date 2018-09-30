import numpy as np
import tensorflow as tf
from .bbox import *

class EmbclsLayer(object):
    '''
    The similarity between pixels p and q is computed as follows:
    σ(p, q) = 2 / (1 + exp(||ep - eq||2^2))
            = 2 / (1 + exp(||*||2^2))   --->   -x = ||*||2^2, x = -||*||2^2, x ∈ (-∞, 0], sigmoid(x) ∈ (0, 0.5]
            = 2 * sigmoid(x) ∈ (0, 1]
            ??? sigmoid(ep * eq / sqrt(d)) ???
    ||tensor||2 = tf.norm(tensor, ord='euclidean', axis=-1, keepdims=False)
    '''
    def __init__(self, cls_num=None, img_shp=None, fet_shp=None):
        
        self.cls_num     = cls_num
        self.img_shp     = img_shp
        self.fet_shp     = fet_shp
        self.fet_srd     = img_shp[0] / fet_shp[0]
        self.pix_smp_num = 100
        self.pix_sim_min = np.array([0.25, 0.5, 0.75, 0.9], dtype=np.float32)
        self.pix_sim_num = np.shape(self.pix_sim_min)[0]
        self.pix_msk_min = 0.2
        self.msk_siz_min = 16
        self.msk_prb_min = 0.8
        self.msk_nms_pre = None
        self.msk_nms_pst = 100
        self.msk_nms_max = 0.2
        self.msk_ovp_min = 0.4
        
    def generate_msks_vld(self, pix_idxs=None, fet_emb=None):
        #pix_embs-->(K, 64), fet_emb --> (H, W, 64), pix_idxs --> (K, 2)
        pix_embs = tf.gather_nd(fet_emb, pix_idxs)                                    #(K, 64)
        pix_embs = pix_embs[:, tf.newaxis, tf.newaxis, :]                             #(K, 1, 1, 64)
        fet_emb  = tf.expand_dims(fet_emb, axis=0)                                    #(1, H, W, 64)
        #pix_embs= pix_embs - fet_emb                                                 #(K, H, W, 64)
        pix_embs = tf.abs(pix_embs - fet_emb)                                         #(K, H, W, 64)
        pix_embs = tf.clip_by_value(pix_embs, 1e-4, 5.0)                              #(K, H, W, 64)
        pix_dsts = tf.norm(pix_embs, ord='euclidean', axis=-1, keepdims=False) ** 2   #(K, H, W)
        pix_sims = 2.0 * (1.0 - tf.sigmoid(pix_dsts))                                 #(K, H, W)
        '''
        #fet_emb-->(H, W, 64), pix_embs-->(K, 64)
        fet_emb  = tf.reshape(fet_emb, [self.fet_shp[0]*self.fet_shp[1], -1])         #(H*W, 64)
        pix_sims = tf.matmul(pix_embs, fet_emb, transpose_b=True) / tf.sqrt(tf.cast(tf.shape(pix_embs)[-1], dtype=tf.float32))
        pix_sims = tf.reshape(pix_sims, [-1, self.fet_shp[0], self.fet_shp[1]])       #(K, H, W)
        pix_sims = tf.sigmoid(pix_sims)                                               #(K, H, W)
        '''
        pix_sims = tf.expand_dims(pix_sims, axis=1)                                   #(K, 1, H, W)
        sim_min  = self.pix_sim_min[tf.newaxis, :, tf.newaxis, tf.newaxis]            #(1, 4, 1, 1)
        msks     = tf.cast(pix_sims>=sim_min, dtype=tf.float32)                       #(K, 4, H, W)
        return msks
    
    def generate_seds_vld(self, sed_msk=None, pix_msk=None, fet_emb=None):
        
        if pix_msk is None:
            pix_msk  = tf.zeros(shape=tf.shape(sed_msk), dtype=tf.float32)            #(H, W)
        fet_emb  = tf.reshape(fet_emb, [self.fet_shp[0]*self.fet_shp[1], -1])         #(H*W, 64)
        pix_msk  = tf.reshape(pix_msk, [-1])                                          #(H*W)
        sed_msk  = tf.reshape(sed_msk, [-1])                                          #(H*W)
        #sed_msk = tf.log(sed_msk)                                                    #(H*W)
        pix_idxs = tf.argmax(sed_msk+pix_msk, axis=0)                                 #(1) 此时pix_dsts为0
        pix_idxs = tf.expand_dims(pix_idxs, axis=0)                                   #(1)
        pix_msk  = tensor_update(pix_msk, pix_idxs, -np.inf)                          #(H*W)
        
        #选择剩下的点
        def cond(i, fet_emb, sed_msk, pix_msk, pix_idxs):
            c = tf.less(i, self.pix_smp_num)
            return c

        def body(i, fet_emb, sed_msk, pix_msk, pix_idxs):
            
            #和已有点pix_idxs算之间的距离
            pix_embs = tf.gather(fet_emb, pix_idxs)                                   #(K, 64)
            #pix_embs= pix_embs[:, tf.newaxis, :] - fet_emb[tf.newaxis, :, :]         #(K, H*W, 64) (K, 1, 64) (1, H*W, 64)
            pix_embs = tf.abs(pix_embs[:, tf.newaxis, :] - fet_emb[tf.newaxis, :, :]) #(K, H*W, 64) (K, 1, 64) (1, H*W, 64)
            pix_embs = tf.clip_by_value(pix_embs, 1e-4, 5.0)                          #(K, H*W, 64)
            pix_dsts = tf.norm(pix_embs, ord='euclidean', axis=-1, keepdims=False)**2 #(K, H*W)
            pix_sims = 2.0 * (1.0 - tf.sigmoid(pix_dsts))                             #(K, H*W)
            pix_sims = tf.reduce_max(pix_sims, axis=0)                                #(H*W)
            #pix_sims= tf.Print(pix_sims , [sed_msk, pix_sims], message=None, first_n=None, summarize=100)
            pix_idx  = tf.argmax(sed_msk - 0.7*pix_sims + pix_msk)                    #(1)
            '''
            pix_dsts = tf.reduce_min(pix_dsts, axis=0)                                #(H*W)
            pix_dsts = tf.log(pix_dsts)                                               #(H*W)
            pix_dsts = tf.Print(pix_dsts , [sed_msk, pix_dsts], message=None, first_n=None, summarize=100)
            pix_idx  = tf.argmax(sed_msk + 0.5*pix_dsts + pix_msk)                    #(1)
            '''
            '''
            pix_sims = tf.matmul(pix_embs, fet_emb, transpose_b=True) / tf.sqrt(tf.cast(tf.shape(pix_embs)[-1], dtype=tf.float32))
            pix_sims = tf.sigmoid(pix_sims)                                           #(K, H*W)
            pix_sims = tf.reduce_max(pix_sims, axis=0)                                #(H*W)
            #pix_sims= tf.log(pix_sims)                                               #(H*W)
            pix_idx  = tf.argmax(sed_msk - 2.0*pix_sims + pix_msk)                    #(1)
            '''
            pix_idx  = tf.expand_dims(pix_idx, axis=0)                                #(1)
            #pix_idx = tf.Print(pix_idx , [pix_idx], message=None, first_n=None, summarize=100)
            pix_msk  = tensor_update(pix_msk, pix_idx, -np.inf)                       #(H*W)
            pix_idxs = tf.concat([pix_idxs, pix_idx], axis=0)                         #(K+1)
            return [i+1, fet_emb, sed_msk, pix_msk, pix_idxs]
        
        i = tf.constant(1) #第一个点已经获取
        [i, fet_emb, sed_msk, pix_msk, pix_idxs] = \
            tf.while_loop(cond, body, loop_vars=[i, fet_emb, sed_msk, pix_msk, pix_idxs], \
                          shape_invariants=[i.get_shape(), fet_emb.get_shape(), sed_msk.get_shape(), \
                                            pix_msk.get_shape(), tf.TensorShape([None])], \
                          parallel_iterations=1, back_prop=False, swap_memory=True)
        
        #pix_idxs= tf.Print(pix_idxs, [sed_msk, pix_msk, pix_idxs], message=None, first_n=None, summarize=100)
        pix_ycds = pix_idxs // self.fet_shp[1]
        pix_xcds = pix_idxs %  self.fet_shp[1]
        pix_idxs = tf.stack([pix_ycds, pix_xcds], axis=-1) #(K, 2)
        #pix_idxs= tf.Print(pix_idxs, [pix_idxs], message=None, first_n=None, summarize=100)
        return pix_idxs
    
    def generate_msks_img(self, elems=None):
        
        #tf.gather是只在axis指定的维度上根据indcies进行gather操作，其余(axis-1)个维度所进行的操作相同
        #axis+1及以上各维度被看成同一个实体，indcies中的每个数值指明了对axis维的哪些实体进行操作
        #tf.gather    --> output.shape = params.shape[:axis] + indices.shape + params.shape[axis + 1:]
        #tf.gather_nd --> output.shape = indices.shape[:-1] + params.shape[indices.shape[-1]:]
        #fet_emb-->(H, W, 64), fet_cls-->(H, W, 4, C+1)
        fet_emb, fet_cls = elems
        #屏蔽背景类
        pix_prbs = tf.nn.softmax(fet_cls, axis=-1)                                    #(H, W, 4, C+1)
        pix_msk  = tf.zeros(shape=[self.cls_num], dtype=tf.float32)                   #(C+1)
        pix_msk  = tensor_update(pix_msk, [0], -np.inf)                               #(C+1)
        pix_prbs = pix_prbs + pix_msk                                                 #(H, W, 4, C+1)
        #pix_prbs= tf.Print(pix_prbs, [pix_prbs], message=None, first_n=None, summarize=100)
        #取出最佳类的预测值
        pix_clss = tf.argmax(pix_prbs, axis=-1)                                       #(H, W, 4)
        pix_clss = tf.cast(pix_clss, dtype=tf.int32)                                  #(H, W, 4)
        pix_prbs = tf.reduce_max(pix_prbs, axis=-1)                                   #(H, W, 4)
        sim_idxs = tf.argmax(pix_prbs, axis=-1)                                       #(H, W)
        ycd_idxs = tf.range(self.fet_shp[0])                                          #(H)
        ycd_idxs = tf.expand_dims(ycd_idxs, axis=-1)                                  #(H, 1)
        ycd_idxs = tf.tile(ycd_idxs, [1, self.fet_shp[1]])                            #(H, W)
        xcd_idxs = tf.range(self.fet_shp[1])                                          #(W)
        xcd_idxs = tf.expand_dims(xcd_idxs, axis= 0)                                  #(1, W)
        xcd_idxs = tf.tile(xcd_idxs, [self.fet_shp[0], 1])                            #(H, W)
        pix_idxs = tf.stack([ycd_idxs, xcd_idxs, sim_idxs],  axis=-1)                 #(H, W, 3)
        pix_prbs = tf.gather_nd(pix_prbs, pix_idxs)                                   #(H, W)
        pix_clss = tf.gather_nd(pix_clss, pix_idxs)                                   #(H, W)
        #pix_prbs= tf.Print(pix_prbs, [pix_prbs, pix_clss], message=None, first_n=None, summarize=256)
        #pix_prbs, pix_clss, sim_idxs --> (H, W)
        '''
        #屏蔽背景点
        pix_idxs = tf.where(tf.equal(pix_clss, 0))                                    #(K, 2)
        pix_msk  = tf.zeros(shape=tf.shape(pix_prbs), dtype=tf.float32)               #(H, W)
        pix_msk  = tensor_update(pix_msk, pix_idxs, -np.inf)                          #(H, W)
        #筛选种子点
        pix_idxs = self.generate_seds_vld(pix_prbs, pix_msk, fet_emb)                 #(K, 2)
        '''
        #筛选种子点
        pix_idxs = self.generate_seds_vld(pix_prbs, None, fet_emb)                    #(K, 2)
        '''
        ycd_idxs = tf.range(self.fet_shp[0])                                          #(H)
        ycd_idxs = tf.expand_dims(ycd_idxs, axis=-1)                                  #(H, 1)
        ycd_idxs = tf.tile(ycd_idxs, [1, self.fet_shp[1]])                            #(H, W)
        xcd_idxs = tf.range(self.fet_shp[1])                                          #(W)
        xcd_idxs = tf.expand_dims(xcd_idxs, axis= 0)                                  #(1, W)
        xcd_idxs = tf.tile(xcd_idxs, [self.fet_shp[0], 1])                            #(H, W)
        pix_idxs = tf.stack([ycd_idxs, xcd_idxs],  axis=-1)                           #(H, W, 2)
        pix_idxs = tf.reshape(pix_idxs, [-1, 2])                                      #(H*W, 2)
        '''
        #生成mask
        sim_idxs = tf.gather_nd(sim_idxs, pix_idxs)                                   #(K)
        tmp_idxs = tf.cast(tf.range(tf.shape(sim_idxs)[0]), dtype=tf.int64)           #(K)
        sim_idxs = tf.stack([tmp_idxs, sim_idxs], axis=-1)                            #(K, 2)
        msks     = self.generate_msks_vld(pix_idxs, fet_emb)                          #(K, 4, H, W)
        msks     = tf.gather_nd(msks,     sim_idxs)                                   #(K, H, W)
        msk_prbs = tf.gather_nd(pix_prbs, pix_idxs)                                   #(K)
        msk_clss = tf.gather_nd(pix_clss, pix_idxs)                                   #(K)
        #msk_prbs= tf.Print(msk_prbs, [msk_prbs, msk_clss], message=None, first_n=None, summarize=100)
        #msks --> (K, H, W), msk_prbs, msk_clss --> (K)
        msk_idxs = tf.range(tf.shape(msks)[0])                                        #mask索引
        #剔除背景mask
        idxs     = tf.where(msk_clss>0)
        msk_idxs = tf.gather_nd(msk_idxs, idxs)                                       #保留的masks是哪些
        #剔除得分较低的mask
        if self.msk_prb_min is not None:
            tmp_prbs       = tf.gather(msk_prbs, msk_idxs)                            #保留的masks的概率是什么
            idxs           = tf.where(tmp_prbs>=self.msk_prb_min)
            msk_idxs       = tf.gather_nd(msk_idxs, idxs)                             #保留的masks是哪些
        #进一步剔除过多的mask
        if self.msk_nms_pre is not None:
            tmp_prbs       = tf.gather(msk_prbs, msk_idxs)                            #保留的masks的概率是什么
            msk_nms_pre    = tf.minimum(self.msk_nms_pre, tf.shape(msk_idxs)[0])
            tmp_prbs, idxs = tf.nn.top_k(tmp_prbs, k=msk_nms_pre, sorted=True)
            msk_idxs       = tf.gather(msk_idxs, idxs)                                #保留的masks是哪些
        #剔除过小的mask
        if self.msk_siz_min is not None:
            tmp_msks       = tf.gather(msks,     msk_idxs)                            #保留的masks是什么 (K, H, W)
            tmp_nums       = tf.reduce_sum(tmp_msks, axis=[1, 2])                     #(K)
            idxs           = tf.where(tmp_nums>=self.msk_siz_min)
            msk_idxs       = tf.gather_nd(msk_idxs, idxs)                             #保留的masks是哪些
        #利用msk_idxs进行nms之前的一次汇总
        msks     = tf.gather(msks,     msk_idxs)                                      #(K, H, W)
        msk_clss = tf.gather(msk_clss, msk_idxs)                                      #(K)
        msk_prbs = tf.gather(msk_prbs, msk_idxs)                                      #(K)
        
        #做逐类的nms
        msk_idxs = tf.zeros(shape=[0, 1], dtype=tf.int64)
        unq_clss, idxs = tf.unique(msk_clss)
        
        def cond0(i, msks, msk_clss, msk_prbs, unq_clss, msk_idxs):
            cls_num = tf.shape(unq_clss)[0]
            c       = tf.less(i, cls_num)
            return c

        def body0(i, msks, msk_clss, msk_prbs, unq_clss, msk_idxs):
            #选出对应类的masks
            unq_cls      = unq_clss[i]
            msk_idxs_cls = tf.where(tf.equal(msk_clss, unq_cls))
            msks_cls     = tf.gather_nd(msks,     msk_idxs_cls)
            msk_prbs_cls = tf.gather_nd(msk_prbs, msk_idxs_cls)
            #进行非极大值抑制操作
            idxs         = mask_nms(msks_cls, msk_prbs_cls, self.msk_nms_pst, self.msk_nms_max)
            msk_idxs_cls = tf.gather(msk_idxs_cls, idxs)
            # 保存结果
            msk_idxs     = tf.concat([msk_idxs, msk_idxs_cls], axis=0)
            return [i+1, msks, msk_clss, msk_prbs, unq_clss, msk_idxs]
        
        i = tf.constant(0)
        [i, msks, msk_clss, msk_prbs, unq_clss, msk_idxs] = \
            tf.while_loop(cond0, body0, loop_vars=[i, msks, msk_clss, msk_prbs, unq_clss, msk_idxs], \
                          shape_invariants=[i.get_shape(), msks.get_shape(), msk_clss.get_shape(), \
                                            msk_prbs.get_shape(), unq_clss.get_shape(), tf.TensorShape([None, 1])], \
                          parallel_iterations=10, back_prop=False, swap_memory=True)
        
        msk_prbs_kep = tf.gather_nd(msk_prbs, msk_idxs)
        msk_num      = tf.minimum(self.msk_nms_pst, tf.shape(msk_idxs)[0])
        msk_prbs_kep, idxs = tf.nn.top_k(msk_prbs_kep, k=msk_num, sorted=True)        #最终的mask概率
        msk_idxs     = tf.gather   (msk_idxs, idxs    )
        msks_tmp     = tf.gather_nd(msks,     msk_idxs)                               #融合前的mask
        msk_clss_kep = tf.gather_nd(msk_clss, msk_idxs)                               #最终的mask类别
        
        #msk_clss, msk_prbs-->(K), msks_tmp-->(K, H, W)
        #融合mask，此时msks, msk_prbs, msk_clss的值并未被覆盖
        boxs_kep = tf.zeros(shape=[0, 4], dtype=tf.float32)
        msks_kep = tf.zeros(shape=[0, self.fet_shp[0], self.fet_shp[1]], dtype=tf.float32)
        idxs_kep = tf.zeros(shape=[0, 1], dtype=tf.int64)                             #用来保证msk_prbs和msk_clss对应关系的索引
        unq_clss, idxs = tf.unique(msk_clss_kep)
        
        def cond1(i, msks, msk_clss, msk_prbs, msks_tmp, msk_clss_kep, unq_clss, boxs_kep, msks_kep, idxs_kep):
            cls_num = tf.shape(unq_clss)[0]
            c       = tf.less(i, cls_num)
            return c

        def body1(i, msks, msk_clss, msk_prbs, msks_tmp, msk_clss_kep, unq_clss, boxs_kep, msks_kep, idxs_kep):
            #选出对应类的boxs
            unq_cls      = unq_clss[i]
            idxs         = tf.where(tf.equal(msk_clss,     unq_cls))
            msks_cls     = tf.gather_nd(msks,     idxs    )
            msk_prbs_cls = tf.gather_nd(msk_prbs, idxs    )
            
            idxs_run     = tf.where(tf.equal(msk_clss_kep, unq_cls))
            msks_run     = tf.gather_nd(msks_tmp, idxs_run)
            
            boxs_hld     = tf.zeros(shape=[0, 4], dtype=tf.float32)
            msks_hld     = tf.zeros(shape=[0, self.fet_shp[0], self.fet_shp[1]], dtype=tf.float32)
            idxs_hld     = tf.zeros(shape=[0],    dtype=tf.int32  )                   #并行计算要保证对应关系
            
            def cond(j, msks_run, msks_cls, msk_prbs_cls, boxs_hld, msks_hld, idxs_hld):
                c = tf.less(j, tf.shape(msks_run)[0])
                return c
            
            def body(j, msks_run, msks_cls, msk_prbs_cls, boxs_hld, msks_hld, idxs_hld):
                msk_run      = msks_run[j][tf.newaxis, :, :]                          #(1, H, W)
                msk_ovps     = mask_overlaps(msks_cls, msk_run)                       #(M0, 1)
                msk_ovps     = tf.squeeze(msk_ovps, axis=[-1])                        #(M0)
                idxs         = tf.where(msk_ovps>=self.msk_ovp_min)
                msks_meg     = tf.gather_nd(msks_cls,     idxs)                       #(M1, H, W)
                msk_prbs_meg = tf.gather_nd(msk_prbs_cls, idxs)                       #(M1)
                msk_wgts     = msk_prbs_meg / tf.reduce_sum(msk_prbs_meg)             #(M1)
                msk_wgts     = msk_wgts[:, tf.newaxis, tf.newaxis]                    #(M1, 1, 1)
                msks_meg     = msks_meg * msk_wgts                                    #(M1, H, W)
                msk_meg      = tf.reduce_sum(msks_meg, axis=0)                        #(H, W)
                pix_idxs     = tf.cast(tf.where(msk_meg>=self.pix_msk_min), dtype=tf.int32) #(K, 2)
                crd_min      = tf.maximum(tf.reduce_min(pix_idxs, axis=0), 0)         #(2) 防止没有msk_idxs出现-1
                crd_max      = tf.reduce_max(pix_idxs, axis=0)                        #(2)
                box_meg      = tf.concat([crd_min, crd_max], axis=0)                  #(4)
                box_meg      = tf.cast(box_meg, dtype=tf.float32) * self.fet_srd      #(4)
                box_meg      = tf.expand_dims(box_meg, axis=0)                        #(1, 4)
                msk_meg      = tf.expand_dims(msk_meg, axis=0)                        #(1, H, W)
                idx_meg      = tf.expand_dims(j,       axis=0)                        #(1)
                #保存结果
                boxs_hld     = tf.concat([boxs_hld, box_meg], axis=0)                 #(K, 4)
                msks_hld     = tf.concat([msks_hld, msk_meg], axis=0)                 #(K, H, W)
                idxs_hld     = tf.concat([idxs_hld, idx_meg], axis=0)                 #(K)
                '''
                assert_op = tf.Assert(tf.size(box_msks_meg)>0, [j, tf.shape(boxs_run)[0], box_ovps, box_run, boxs_cls, \
                                                                idxs, boxs_meg, box_prbs_meg, box_msks_meg], summarize=100)
                with tf.control_dependencies([assert_op]):
                    box_msks_meg = tf.identity(box_msks_meg)
                '''
                return [j+1, msks_run, msks_cls, msk_prbs_cls, boxs_hld, msks_hld, idxs_hld]
                
            j = tf.constant(0)
            [j, msks_run, msks_cls, msk_prbs_cls, boxs_hld, msks_hld, idxs_hld] = \
                tf.while_loop(cond, body, \
                              loop_vars=[j, msks_run, msks_cls, msk_prbs_cls, boxs_hld, msks_hld, idxs_hld], \
                              shape_invariants=[j.get_shape(), msks_run.get_shape(), msks_cls.get_shape(), \
                                                msk_prbs_cls.get_shape(), tf.TensorShape([None, 4]), \
                                                tf.TensorShape([None,self.fet_shp[0],self.fet_shp[1]]), tf.TensorShape([None])], \
                              parallel_iterations=10, back_prop=False, swap_memory=True)
            
            idxs_hld = tf.gather(idxs_run, idxs_hld)
            #保存结果
            boxs_kep = tf.concat([boxs_kep, boxs_hld], axis=0)
            msks_kep = tf.concat([msks_kep, msks_hld], axis=0)
            idxs_kep = tf.concat([idxs_kep, idxs_hld], axis=0)
            return [i+1, msks, msk_clss, msk_prbs, msks_tmp, msk_clss_kep, unq_clss, boxs_kep, msks_kep, idxs_kep]
        
        i = tf.constant(0)
        [i, msks, msk_clss, msk_prbs, msks_tmp, msk_clss_kep, unq_clss, boxs_kep, msks_kep, idxs_kep] = \
            tf.while_loop(cond1, body1, \
                          loop_vars=[i, msks, msk_clss, msk_prbs, msks_tmp, msk_clss_kep, unq_clss, boxs_kep, msks_kep, idxs_kep], \
                          shape_invariants=[i.get_shape(), msks.get_shape(), msk_clss.get_shape(), msk_prbs.get_shape(), \
                                            msks_tmp.get_shape(), msk_clss_kep.get_shape(), unq_clss.get_shape(), \
                                            tf.TensorShape([None, 4]), tf.TensorShape([None, self.fet_shp[0], self.fet_shp[1]]), \
                                            tf.TensorShape([None, 1])], \
                          parallel_iterations=10, back_prop=False, swap_memory=True)
            
        #匹配msk_clss, msk_prbs
        msk_clss_kep = tf.gather_nd(msk_clss_kep, idxs_kep)
        msk_prbs_kep = tf.gather_nd(msk_prbs_kep, idxs_kep)
        
        msk_num      = tf.shape(msks_kep)[0]
        paddings     = [[0, self.msk_nms_pst-msk_num], [0, 0]]
        boxs_kep     = tf.pad(boxs_kep,     paddings, 'CONSTANT')
        paddings     = [[0, self.msk_nms_pst-msk_num], [0, 0], [0, 0]]
        msks_kep     = tf.pad(msks_kep,     paddings, 'CONSTANT')
        paddings     = [[0, self.msk_nms_pst-msk_num]]
        msk_clss_kep = tf.pad(msk_clss_kep, paddings, 'CONSTANT')
        msk_prbs_kep = tf.pad(msk_prbs_kep, paddings, 'CONSTANT')
        return boxs_kep, msks_kep, msk_clss_kep, msk_prbs_kep, msk_num
    
    
    def generate_msks(self, fet_emb=None, fet_cls=None):
        #fet_emb --> (N, H, W, 64), fet_cls --> (N, H, W, C)
        elems = [fet_emb, fet_cls]
        boxs, msks, msk_clss, msk_prbs, msk_nums = \
            tf.map_fn(self.generate_msks_img, elems, dtype=(tf.float32, tf.float32, tf.int32, tf.float32, tf.int32),
                  parallel_iterations=10, back_prop=False, swap_memory=True, infer_shape=True)
        return boxs, msks, msk_clss, msk_prbs, msk_nums
    