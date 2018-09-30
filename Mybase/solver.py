import os
import cv2
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import defaultdict
from .load_weights import *
from .optim_utils import *
from .deep_metric_utils.make_image import *
from .deep_metric_utils.bboxes_target_layer import *

def get_data(fid):
    try:  
            a = pickle.load(fid)
            return 1, a
    except EOFError:  
            return 0, 0

def get_all_data(fid):
    data = []
    while(True):
        sig, dat = get_data(fid)
        if(sig == 0): break
        else:
            data.append(dat)
    return data

class Solver(object):
    
    def __init__(self, mdl, **kwargs):
        
        self.mdl         = mdl
        self.opm_cfg     = kwargs.pop('opm_cfg',       {})
        self.use_gpu     = kwargs.pop('use_gpu',     True)
        self.gpu_lst     = kwargs.pop('gpu_lst',      '0')
        self.gpu_num     = len(self.gpu_lst.split(',')) if self.use_gpu else 1
        self.mdl_dev     = '/gpu:%d'                    if self.use_gpu else '/cpu:%d'
        self.MDL_DEV     = 'GPU_%d'                     if self.use_gpu else 'CPU_%d'
        self.bat_siz     = kwargs.pop('bat_siz',        2)
        self.bat_siz_all = self.bat_siz                    * self.gpu_num
        self.tra_num     = kwargs.pop('tra_num',     8000) * self.gpu_num
        self.val_num     = kwargs.pop('val_num',       80) * self.gpu_num
        self.epc_num     = kwargs.pop('epc_num',       10)
        self.prt_ena     = kwargs.pop('prt_ena',     True)
        self.itr_per_prt = kwargs.pop('itr_per_prt',   20)
        self.tst_num     = kwargs.pop('tst_num',     None)
        self.tst_shw     = kwargs.pop('tst_shw',     True)
        self.tst_sav     = kwargs.pop('tst_sav',     True)
        self.mdl_nam     = kwargs.pop('mdl_nam',    'model.ckpt'     )
        self.mdl_dir     = kwargs.pop('mdl_dir',    'Mybase/Model'   )
        self.log_dir     = kwargs.pop('log_dir',    'Mybase/logdata' )
        self.dat_dir     = kwargs.pop('dat_dir',    'Mybase/datasets')
        self.mov_ave_dca = kwargs.pop('mov_ave_dca', 0.99)
        self.epc_per_dca = kwargs.pop('epc_per_dca',    1)
        self.dat_dir_tra = self.dat_dir + '/train'
        self.dat_dir_val = self.dat_dir + '/val'
        self.dat_dir_tst = self.dat_dir + '/test'
        self.dat_dir_rst = self.dat_dir + '/result'
        self.log_dir_tra = self.log_dir + '/train'
        self.log_dir_val = self.log_dir + '/val'
        self.log_dir_tst = self.log_dir + '/test'
        self.epc_cnt     = 0
        
        os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_lst
        
        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in kwargs.keys())
            raise ValueError('Unrecognized arguments %s' % extra)
            
    
    #################################For FCIS##################################
    def _train_step(self, mtra=None, mtst=None, itr_per_epc=None, glb_stp=None):
        #将简单的运算放在CPU上，只有神经网络的训练过程放在GPU上
        with tf.device("/cpu:0"):
            
            GC_tra = GeneratorForCOCO(True,  self.dat_dir_tra, self.dat_dir_tst, self.dat_dir_rst, self.tst_shw, self.tst_sav, \
                                      self.bat_siz_all, 200)
            GC_val = GeneratorForCOCO(False, self.dat_dir_val, self.dat_dir_tst, self.dat_dir_rst, self.tst_shw, self.tst_sav, \
                                      self.bat_siz_all,  50)
            
            imgs_tra, gbxs_tra, gmks_tra, gbx_nums_tra, img_wdws_tra, img_hgts_tra_, img_wdhs_tra_ = GC_tra.get_input()
            imgs_val, gbxs_val, gmks_val, gbx_nums_val, img_wdws_val, img_hgts_val_, img_wdhs_val_ = GC_val.get_input()
            
            imgs      = tf.cond(mtst, lambda: imgs_val,      lambda: imgs_tra,      strict=True)
            gbxs      = tf.cond(mtst, lambda: gbxs_val,      lambda: gbxs_tra,      strict=True)
            gmks      = tf.cond(mtst, lambda: gmks_val,      lambda: gmks_tra,      strict=True)
            gbx_nums  = tf.cond(mtst, lambda: gbx_nums_val,  lambda: gbx_nums_tra,  strict=True)
            img_wdws  = tf.cond(mtst, lambda: img_wdws_val,  lambda: img_wdws_tra,  strict=True)
            img_hgts_ = tf.cond(mtst, lambda: img_hgts_val_, lambda: img_hgts_tra_, strict=True)
            img_wdhs_ = tf.cond(mtst, lambda: img_wdhs_val_, lambda: img_wdhs_tra_, strict=True)
            
            #with tf.name_scope("input_image"):
            #    tf.summary.image("input", X, 10)
            
            self.opm_cfg["decay_step"] = itr_per_epc * self.epc_per_dca #decay
            tra_stp = update_rule(self.opm_cfg, glb_stp)
            
            self.mdl.mod_tra = True
            self.mdl.glb_pol = False
            self.mdl.inc_btm = True

            grds_lst     = []
            loss_lst     = []
            boxs_lst     = []
            msks_lst     = []
            msk_clss_lst = []
            msk_prbs_lst = []
            msk_nums_lst = []
            for i in range(self.gpu_num):
                with tf.device(self.mdl_dev % i):
                    with tf.name_scope(self.MDL_DEV % i) as scp:
                        sta = i     * self.bat_siz
                        end = (i+1) * self.bat_siz
                        loss, boxs, msks, msk_clss, msk_prbs, msk_nums = \
                            self.mdl.forward(imgs[sta:end], None, gbxs[sta:end], gmks[sta:end], gbx_nums[sta:end], mtra, scp)
                        #在第一次声明变量之后，将控制变量重用的参数设置为True。这样可以让不同的GPU更新同一组参数
                        #注意tf.name_scope函数并不会影响tf.get_variable的命名空间
                        tf.get_variable_scope().reuse_variables()
                        #使用当前GPU计算所有变量的梯度
                        grds = tra_stp.compute_gradients(loss[0])
                        #print(grds)
                grds_lst    .append(grds    )
                loss_lst    .append(loss    )
                boxs_lst    .append(boxs    )
                msks_lst    .append(msks    )
                msk_clss_lst.append(msk_clss)
                msk_prbs_lst.append(msk_prbs)
                msk_nums_lst.append(msk_nums)
            #print(grds_lst)
            '''
            with tf.variable_scope("average", reuse = tf.AUTO_REUSE):
                mov_ave    = tf.train.ExponentialMovingAverage(self.mov_ave_dca, glb_stp)
                mov_ave_op = mov_ave.apply(tf.trainable_variables())
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, mov_ave_op)
            '''
            with tf.variable_scope("optimize", reuse = tf.AUTO_REUSE):
                grds     = average_gradients(grds_lst, clip_norm=None)
                upd_opas = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(upd_opas):
                    tra_opa = tra_stp.apply_gradients(grds, global_step=glb_stp)
                '''
                tra_opa  = tra_stp.apply_gradients(grds, global_step=glb_stp)
                '''
            loss     = tf.stack(loss_lst,      axis=0) #一个向量而已
            loss     = tf.reduce_mean(loss,    axis=0) #以下皆有原来的维度
            boxs     = tf.concat(boxs_lst,     axis=0) #(N, M, 4)
            msks     = tf.concat(msks_lst,     axis=0) #(N, M, H, W)
            msk_clss = tf.concat(msk_clss_lst, axis=0) #(N, M)
            msk_prbs = tf.concat(msk_prbs_lst, axis=0) #(N, M)
            msk_nums = tf.concat(msk_nums_lst, axis=0) #(N)
            #tf.summary.scalar("loss", loss)
            #tf.summary.scalar("acc", acc)
            #for grad, var in grads:
            #    if grad is not None:
            #        tf.summary.histogram("gradients_on_average/%s" % var.op.name, grad)
            #for var in tf.trainable_variables():
            #    tf.summary.histogram(var.op.name, var)
        return tra_opa, loss, boxs, msks, msk_clss, msk_prbs, msk_nums, gbxs, gmks, gbx_nums
    
    """
    ###############################For CLASSIFY################################
    def _train_step(self, mtra=None, mtst=None, itr_per_epc=None, glb_stp=None):
        #将简单的运算放在CPU上，只有神经网络的训练过程放在GPU上
        with tf.device("/cpu:0"):
            
            GI_tra = GeneratorForImageNet(True,  self.dat_dir_tra, self.bat_siz_all, 250)
            GI_val = GeneratorForImageNet(False, self.dat_dir_val, self.bat_siz_all, 250)
            imgs_tra, lbls_tra = GI_tra.get_input()
            imgs_val, lbls_val = GI_val.get_input()
            imgs = tf.cond(mtst, lambda: imgs_val, lambda: imgs_tra, strict=True)
            lbls = tf.cond(mtst, lambda: lbls_val, lambda: lbls_tra, strict=True)
            #with tf.name_scope("input_image"):
            #    tf.summary.image("input", X, 10)

            self.opm_cfg["decay_step"] = itr_per_epc * self.epc_per_dca #decay
            tra_stp = update_rule(self.opm_cfg, glb_stp)

            self.mdl.mod_tra = True
            self.mdl.glb_pol = True

            grds_lst = []
            loss_lst = []
            scrs_lst = []
            for i in range(self.gpu_num):
                with tf.device(self.mdl_dev % i):
                    with tf.name_scope(self.MDL_DEV % i) as scp:
                        sta = i     * self.bat_siz
                        end = (i+1) * self.bat_siz
                        loss, scrs = \
                            self.mdl.forward(imgs=imgs[sta:end], lbls=lbls[sta:end], gbxs=None, gbx_nums=None, mtra=mtra, scp=scp)

                        #在第一次声明变量之后，将控制变量重用的参数设置为True。这样可以让不同的GPU更新同一组参数
                        #注意tf.name_scope函数并不会影响tf.get_variable的命名空间
                        tf.get_variable_scope().reuse_variables()
                        #使用当前GPU计算所有变量的梯度
                        grds = tra_stp.compute_gradients(loss[0])
                        #print(grds)
                grds_lst.append(grds)
                loss_lst.append(loss)
                scrs_lst.append(scrs)
            '''
            with tf.variable_scope("average", reuse = tf.AUTO_REUSE):
                mov_ave    = tf.train.ExponentialMovingAverage(self.mov_ave_dca, glb_stp)
                mov_ave_op = mov_ave.apply(tf.trainable_variables())
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, mov_ave_op)
            '''
            with tf.variable_scope("optimize", reuse = tf.AUTO_REUSE):
                grds = average_gradients(grds_lst)
                upd_opas = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(upd_opas):
                    tra_opa = tra_stp.apply_gradients(grds, global_step=glb_stp)

            loss = tf.stack(loss_lst,   axis=0)
            scrs = tf.concat(scrs_lst,  axis=0)
            loss = tf.reduce_mean(loss, axis=0)
            #tf.summary.scalar("loss", loss)
            #tf.summary.scalar("acc", acc)
            #for grad, var in grads:
            #    if grad is not None:
            #        tf.summary.histogram("gradients_on_average/%s" % var.op.name, grad)
            #for var in tf.trainable_variables():
            #    tf.summary.histogram(var.op.name, var)
        return tra_opa, loss, scrs, lbls
    """
    
    #################################For FCIS##################################
    def _test_step(self, imgs=None):
        
        with tf.device("/cpu:0"):
            
            mtra = tf.constant(False, dtype=tf.bool)
            self.mdl.mod_tra = False
            self.mdl.glb_pol = False
            self.mdl.inc_btm = True
            
            boxs_lst     = []
            msks_lst     = []
            msk_clss_lst = []
            msk_prbs_lst = []
            msk_nums_lst = []
            for i in range(self.gpu_num):
                with tf.device(self.mdl_dev % i):
                    with tf.name_scope(self.MDL_DEV % i) as scp:
                        sta = i     * self.bat_siz
                        end = (i+1) * self.bat_siz
                        boxs, msks, msk_clss, msk_prbs, msk_nums = \
                            self.mdl.forward(imgs[sta:end], None, None, None, None, mtra, scp)
                        #在第一次声明变量之后，将控制变量重用的参数设置为True。这样可以让不同的GPU更新同一组参数
                        #注意tf.name_scope函数并不会影响tf.get_variable的命名空间
                        tf.get_variable_scope().reuse_variables()
                boxs_lst    .append(boxs    )
                msks_lst    .append(msks    )
                msk_clss_lst.append(msk_clss)
                msk_prbs_lst.append(msk_prbs)
                msk_nums_lst.append(msk_nums)
            boxs     = tf.concat(boxs_lst,     axis=0) #(N, M, 4)
            msks     = tf.concat(msks_lst,     axis=0) #(N, M, H, W)
            msk_clss = tf.concat(msk_clss_lst, axis=0) #(N, M)
            msk_prbs = tf.concat(msk_prbs_lst, axis=0) #(N, M)
            msk_nums = tf.concat(msk_nums_lst, axis=0) #(N)
        return boxs, msks, msk_clss, msk_prbs, msk_nums
    
    
    def concat(self, sess=None, fetches=None, feed_dict=None, itr_num=None, gen=None, tsrs=None, keps=None):
        
        rsts_lst = [[] for _ in range(len(fetches))]
        if keps != None:
            rsts_kep = [[] for _ in range(len(keps))]
        for _ in range(itr_num):
            if gen != None:
                feds = next(gen)
                for i, tsr in enumerate(tsrs):
                    feed_dict[tsr] = feds[i]
                for i, kep in enumerate(keps):
                    rsts_kep[i].append(feds[kep])
            rsts = sess.run(fetches, feed_dict=feed_dict)
            for i, rst in enumerate(rsts):
                rsts_lst[i].append(rst)
        for i, rst in enumerate(rsts_lst):
            rsts_lst[i] = np.concatenate(rst, axis=0)
        if keps != None:
            for i, rst in enumerate(rsts_kep):
                rsts_kep[i] = np.concatenate(rst, axis=0)
            return rsts_lst, rsts_kep
        else:
            return rsts_lst
    
    
    def merge(self, rsts=None, rst_nums=None):
        
        rst_imxs = []
        rsts_lst = [[] for _ in range(len(rsts))]
        for i, rst_num in enumerate(rst_nums): #batch
            rst_imxs.extend([i]*rst_num)
            for j, rst in enumerate(rsts):     #tensors
                rsts_lst[j].append(rst[i][:rst_num])
        rst_imxs = np.asarray(rst_imxs, dtype=np.int32)
        for i, rst in enumerate(rsts_lst):
            rsts_lst[i] = np.concatenate(rst, axis=0)
        return rsts_lst, rst_imxs
    
    
    #################################For FCIS##################################
    def train(self):
        
        itr_per_epc = max(self.tra_num // self.bat_siz_all, 1)
        if self.tra_num % self.bat_siz_all != 0:
            itr_per_epc += 1
        tra_itr_num = self.epc_num * itr_per_epc
        
        val_itr_num = max(self.val_num // self.bat_siz_all, 1)
        if self.val_num % self.bat_siz_all != 0:
            val_itr_num += 1

        tf.reset_default_graph()
        mtra    = tf.placeholder(dtype=tf.bool, name="train")
        mtst    = tf.placeholder(dtype=tf.bool, name="test")
        glb_stp = tf.Variable(0, trainable=False, name="global_step", dtype=tf.int64)
        
        tra_opa, loss, boxs, msks, msk_clss, msk_prbs, msk_nums, gbxs, gmks, gbx_nums = \
            self._train_step(mtra, mtst, itr_per_epc, glb_stp)
        #tf.summary.scalar('loss', loss)
        #summary_op   = tf.summary.merge_all()
        #summary_loss = tf.summary.merge(loss)
        #writer       = tf.summary.FileWriter(LOG_PATH, sess.graph, flush_secs=5) #tf.get_default_graph()    
        #gpu_options  = tf.GPUOptions(per_process_gpu_memory_fraction = 0.8)
        #config       = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, gpu_options=gpu_options)
        #config       = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, device_count={"CPU": 2}, \
        #                              inter_op_parallelism_threads=16, intra_op_parallelism_threads=16)
        config        = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            
            init_op = (tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)
            coord   = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess = sess, coord = coord)
            saver   = tf.train.Saver()
            ckpt    = tf.train.get_checkpoint_state(self.mdl_dir)
            if ckpt and ckpt.model_checkpoint_path:
                '''
                var = tf.global_variables()
                mydict = load_weights()
                mykeys = mydict.keys()
                for i, v in enumerate(var):
                    if v.name in mykeys:
                        sess.run(tf.assign(v, mydict[v.name], validate_shape=True, use_locking=True))
                    else:
                        print(v.name)
                saver.save(sess, os.path.join(self.mdl_dir, self.mdl_nam), global_step=glb_stp)
                return
                '''
                #var = tf.global_variables()
                #var = [v for v in var if "layers_module1_0/" in v.name or "layers_module1_1/" in v.name]
                #var = [v for v in var if "average/" not in v.name and "optimize/" not in v.name]
                #var = [v for v in var if "layers_module1_19/" not in v.name and "layers_module1_20/" not in v.name
                #      and "layers_module1_21/" not in v.name and "layers_module1_22/" not in v.name]
                #var_ave = tf.train.ExponentialMovingAverage(self.mv_ave_decay, glb_stp)
                #var   = var_ave.variables_to_restore()
                #saver = tf.train.Saver(var)
                saver.restore(sess, ckpt.model_checkpoint_path)
                saver = tf.train.Saver()
            
            #初始化box评估类
            BT = BboxesTargetLayer()
            
            with open(os.path.join(self.log_dir_tra, "loss"), 'ab') as fid_tra_loss, \
                 open(os.path.join(self.log_dir_tra, "accs"), 'ab') as fid_tra_accs, \
                 open(os.path.join(self.log_dir_val, "accs"), 'ab') as fid_val_accs:
                
                for t in range(tra_itr_num):      
                    epc_end = (t + 1) % itr_per_epc == 0
                    itr_sta = (t == 0)
                    itr_end = (t == tra_itr_num - 1)
                    if epc_end:
                        self.epc_cnt += 1
                        
                    #_, summary, loss1, = sess.run([train_op, summary_op, loss], feed_dict = {mtrain: True})
                    #writer.add_summary(summary, global_step=glb_stp.eval())
                    _, loss_kep = sess.run([tra_opa, loss], feed_dict={mtra: True, mtst: False})
                    
                    if self.prt_ena and t % self.itr_per_prt == 0:
                        pickle.dump(loss_kep, fid_tra_loss, pickle.HIGHEST_PROTOCOL)
                        print('(Iteration %d / %d) losses: %s' % (t + 1, tra_itr_num, str(loss_kep)))

                    #if itr_sta or itr_end or epc_end:
                    if itr_end or epc_end: 
                        saver.save(sess, os.path.join(self.mdl_dir, self.mdl_nam), global_step=glb_stp)
                        ###Get the training accuracy###
                        print('Get the training accuracy!')
                        fetches   = [boxs, msks, msk_clss, msk_prbs, msk_nums, gbxs, gmks, gbx_nums]
                        feed_dict = {mtra: False, mtst: False}
                        boxs_kep, msks_kep, msk_clss_kep, msk_prbs_kep, msk_nums_kep, gbxs_kep, gmks_kep, gbx_nums_kep = \
                            self.concat(sess, fetches, feed_dict, val_itr_num)
                        [boxs_kep, msks_kep, msk_clss_kep, msk_prbs_kep], msk_imxs = \
                            self.merge([boxs_kep, msks_kep, msk_clss_kep, msk_prbs_kep], msk_nums_kep)
                        [gbxs_kep, gmks_kep], gbx_imxs = \
                            self.merge([gbxs_kep, gmks_kep], gbx_nums_kep)
                        boxs_kep = (boxs_kep, msk_clss_kep, msk_prbs_kep, msk_imxs)
                        gbxs_kep = (gbxs_kep[:, :-1], gbxs_kep[:, -1],    gbx_imxs)
                        '''
                        gbxs_kep = (gbxs_kep[:, :-1], gbxs_kep[:, -1]>0,  gbx_imxs)
                        '''
                        #print(boxs_kep)
                        #print(gbxs_kep)
                        rsts     = BT.generate_boxs_pre(boxs_kep, gbxs_kep)
                        pickle.dump(rsts, fid_tra_accs, pickle.HIGHEST_PROTOCOL)
                        print('')
                        ###Get the validaiton accuracy###
                        print('Get the validaiton accuracy!')
                        fetches   = [boxs, msks, msk_clss, msk_prbs, msk_nums, gbxs, gmks, gbx_nums]
                        feed_dict = {mtra: False, mtst: True}
                        boxs_kep, msks_kep, msk_clss_kep, msk_prbs_kep, msk_nums_kep, gbxs_kep, gmks_kep, gbx_nums_kep = \
                            self.concat(sess, fetches, feed_dict, val_itr_num)
                        [boxs_kep, msks_kep, msk_clss_kep, msk_prbs_kep], msk_imxs = \
                            self.merge([boxs_kep, msks_kep, msk_clss_kep, msk_prbs_kep], msk_nums_kep)
                        [gbxs_kep, gmks_kep], gbx_imxs = \
                            self.merge([gbxs_kep, gmks_kep], gbx_nums_kep)
                        boxs_kep = (boxs_kep, msk_clss_kep, msk_prbs_kep, msk_imxs)
                        gbxs_kep = (gbxs_kep[:, :-1], gbxs_kep[:, -1],    gbx_imxs)
                        '''
                        gbxs_kep = (gbxs_kep[:, :-1], gbxs_kep[:, -1]>0,  gbx_imxs)
                        '''
                        #print(boxs_kep)
                        #print(gbxs_kep)
                        rsts     = BT.generate_boxs_pre(boxs_kep, gbxs_kep)
                        pickle.dump(rsts, fid_val_accs, pickle.HIGHEST_PROTOCOL)
                        print('')
            coord.request_stop()
            coord.join(threads)
            
    """
    #####################################For CLASSIFY#####################################
    def train(self):
        
        itr_per_epc = max(self.tra_num // self.bat_siz_all, 1)
        if self.tra_num % self.bat_siz_all != 0:
            itr_per_epc += 1
        tra_itr_num = self.epc_num * itr_per_epc
        
        val_itr_num = max(self.val_num // self.bat_siz_all, 1)
        if self.val_num % self.bat_siz_all != 0:
            val_itr_num += 1

        tf.reset_default_graph()
        mtra = tf.placeholder(dtype=tf.bool, name="train")
        mtst = tf.placeholder(dtype=tf.bool, name="test")
        glb_stp = tf.Variable(0, trainable=False, name="global_step", dtype=tf.int64)
        
        tra_opa, loss, scrs, lbls = \
            self._train_step(mtra, mtst, itr_per_epc, glb_stp)
        
        scrs_tmp = tf.placeholder(dtype=tf.float32, name="scrs_tmp")
        lbls_tmp = tf.placeholder(dtype=tf.int32,   name="lbls_tmp")
        acc_top1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(scrs_tmp, lbls_tmp, k=1), tf.float32))
        acc_top5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(scrs_tmp, lbls_tmp, k=5), tf.float32))
        #tf.summary.scalar('loss', loss)
        #summary_op   = tf.summary.merge_all()
        #summary_loss = tf.summary.merge(loss)
        #writer       = tf.summary.FileWriter(LOG_PATH, sess.graph, flush_secs=5) #tf.get_default_graph()    
        #gpu_options  = tf.GPUOptions(per_process_gpu_memory_fraction = 0.8)
        #config       = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, gpu_options=gpu_options)
        #config       = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, device_count={"CPU": 2}, \
        #                              inter_op_parallelism_threads=16, intra_op_parallelism_threads=16)
        config        = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            
            init_op = (tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(self.mdl_dir)
            if ckpt and ckpt.model_checkpoint_path:
                
                var = tf.global_variables()
                mydict = load_weights()
                mykeys = mydict.keys()
                for i, v in enumerate(var):
                    if v.name in mykeys:
                        sess.run(tf.assign(v, mydict[v.name], validate_shape=True, use_locking=True))
                    else:
                        print(v.name)
                saver.save(sess, os.path.join(self.mdl_dir, self.mdl_nam), global_step=glb_stp)
                return
                
                #var = tf.global_variables()
                #var = [v for v in var if "layers_module1_0/" in v.name or "layers_module1_1/" in v.name]
                #var = [v for v in var if "average/" not in v.name and "optimize/" not in v.name]
                #var = [v for v in var if "_myprd2" not in v.name and "optimize/" not in v.name]
                #var_ave = tf.train.ExponentialMovingAverage(self.mv_ave_decay, glb_stp)
                #var = var_ave.variables_to_restore()
                #saver = tf.train.Saver(var)
                saver.restore(sess, ckpt.model_checkpoint_path)
                saver = tf.train.Saver()

            with open(os.path.join(self.log_dir_tra, "loss"), 'ab') as fid_tra_loss, \
                 open(os.path.join(self.log_dir_tra, "accs"), 'ab') as fid_tra_accs, \
                 open(os.path.join(self.log_dir_val, "accs"), 'ab') as fid_val_accs:
                
                for t in range(tra_itr_num):
                    epc_end = (t + 1) % itr_per_epc == 0
                    itr_sta = (t == 0)
                    itr_end = (t == tra_itr_num - 1)
                    if epc_end:
                        self.epc_cnt += 1
                        
                    #_, summary, loss1, = sess.run([train_op, summary_op, loss], feed_dict = {mtrain: True})
                    #writer.add_summary(summary, global_step=glb_stp.eval())
                    _, loss_kep = sess.run([tra_opa, loss], feed_dict = {mtra: True, mtst: False})
                    
                    if self.prt_ena and t % self.itr_per_prt == 0:
                        pickle.dump(loss_kep, fid_tra_loss, pickle.HIGHEST_PROTOCOL)
                        print('(Iteration %d / %d) losses: %s' % (t + 1, tra_itr_num, str(loss_kep)))

                    #if itr_sta or itr_end or epc_end:
                    if itr_end or epc_end: 
                        saver.save(sess, os.path.join(self.mdl_dir, self.mdl_nam), global_step=glb_stp)
                        
                        fetches   = [scrs, lbls]
                        feed_dict = {mtra: False, mtst: False}
                        tra_scrs, tra_lbls = self.concat(sess, fetches, feed_dict, val_itr_num)
                        fetches   = [scrs, lbls]
                        feed_dict = {mtra: False, mtst: True}
                        val_scrs, val_lbls = self.concat(sess, fetches, feed_dict, val_itr_num)
                        
                        tra_acc_top1, tra_acc_top5 = \
                            sess.run([acc_top1, acc_top5], feed_dict={scrs_tmp: tra_scrs, lbls_tmp:tra_lbls})
                        val_acc_top1, val_acc_top5 = \
                            sess.run([acc_top1, acc_top5], feed_dict={scrs_tmp: val_scrs, lbls_tmp:val_lbls})   
                        pickle.dump([tra_acc_top1, tra_acc_top5], fid_tra_accs, pickle.HIGHEST_PROTOCOL)
                        pickle.dump([val_acc_top1, val_acc_top5], fid_val_accs, pickle.HIGHEST_PROTOCOL)
                        if self.prt_ena:
                            print('(Epoch %d / %d) tra_acc1: %f, tra_acc5: %f, val_acc1: %f, val_acc5: %f'\
                                  % (self.epc_cnt, self.epc_num, tra_acc_top1, tra_acc_top5, val_acc_top1, val_acc_top5))
            coord.request_stop()
    """
    
    #################################For FCIS##################################
    def test(self):
        
        GC  = GeneratorForCOCO(False, self.dat_dir_tra, self.dat_dir_tst, self.dat_dir_rst, self.tst_shw, self.tst_sav, \
                               self.bat_siz_all, 2)
        #GV = GeneratorForVOC (False, self.dat_dir_tra, self.dat_dir_tst, self.dat_dir_rst, self.tst_shw, self.tst_sav, \
        #                      self.bat_siz_all, 2)
        cat_idx_to_cls_nam = GC.cls_idx_to_cls_nam
        img_siz_max        = GC.img_siz_max
        max_num            = GC.max_num
        if self.tst_num == None: self.tst_num = GC.img_num_tst
        print("There are {:d} pictures to test!".format(self.tst_num))
        tst_itr_num = max(self.tst_num // self.bat_siz_all, 1)
        if self.tst_num % self.bat_siz_all != 0:
            tst_itr_num += 1
        
        tf.reset_default_graph()
        imgs      = tf.placeholder(dtype=tf.float32, shape=[self.bat_siz_all, img_siz_max, img_siz_max, 3], name="images")
        img_hgts_ = tf.placeholder(dtype=tf.int32,   shape=[self.bat_siz_all], name="img_hgts_")
        img_wdhs_ = tf.placeholder(dtype=tf.int32,   shape=[self.bat_siz_all], name="img_wdhs_")
        
        glb_stp   = tf.Variable(0, trainable=False, name="global_step", dtype=tf.int64)
        
        boxs, msks, msk_clss, msk_prbs, msk_nums = self._test_step(imgs)

        #gpu_options  = tf.GPUOptions(per_process_gpu_memory_fraction = 0.8)
        #config       = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, gpu_options=gpu_options)
        #config       = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, device_count={"CPU": 2}, \
        #                              inter_op_parallelism_threads=16, intra_op_parallelism_threads=16)
        config        = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            
            init_op = (tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)
            
            saver = tf.train.Saver()
            ckpt  = tf.train.get_checkpoint_state(self.mdl_dir)
            if ckpt and ckpt.model_checkpoint_path:
                #var     = tf.global_variables()
                #var_ave = tf.train.ExponentialMovingAverage(self.mv_ave_decay, glb_stp)
                #var     = var_ave.variables_to_restore()
                #saver   = tf.train.Saver(var)
                saver.restore(sess, ckpt.model_checkpoint_path)
                saver    = tf.train.Saver()
            else:
                print("No checkpoint file found!")
                return
            
            with open(os.path.join(self.log_dir_tst, "imgs"), 'wb') as fid_tst_imgs, \
                 open(os.path.join(self.log_dir_tst, "boxs"), 'wb') as fid_tst_boxs, \
                 open(os.path.join(self.log_dir_tst, "msks"), 'wb') as fid_tst_msks:
                '''
                sri = ['ID', 'PATH', 'TYPE', 'SCORE', 'XMIN', 'YMIN', 'XMAX', 'YMAX']
                sri = ",".join(sri)
                sri = sri + '\n'
                fid_tst_rsts.writelines([sri])
                '''
                fetches   = [boxs, msks, msk_clss, msk_prbs, msk_nums]
                feed_dict = {}
                [boxs_kep, msks_kep, msk_clss_kep, msk_prbs_kep, msk_nums_kep],[img_nams, img_wdws, img_hgts_, img_wdhs_] = \
                    self.concat(sess, fetches, feed_dict, tst_itr_num, GC.get_input2(sess), [imgs], [1, 2, 3, 4])
                [boxs_kep, msks_kep, msk_clss_kep, msk_prbs_kep], msk_imxs = \
                    self.merge([boxs_kep, msks_kep, msk_clss_kep, msk_prbs_kep], msk_nums_kep)
                    
                for i, img_nam in enumerate(img_nams):
                    img_wdw  = img_wdws [i]
                    img_hgt_ = img_hgts_[i]
                    img_wdh_ = img_wdhs_[i]
                    idxs     = np.where(msk_imxs==i)[0]
                    if len(idxs) == 0:
                        print('There is no boxs for image %s' %(img_nam))
                        continue
                    boxs     = boxs_kep    [idxs]
                    msks     = msks_kep    [idxs]
                    msk_clss = msk_clss_kep[idxs]
                    msk_prbs = msk_prbs_kep[idxs]
                    _, boxs, msks = GC.recover_instances1(None, boxs, msks, img_wdw, img_hgt_, img_wdh_)
                    box_clss = msk_clss[:, np.newaxis].astype(dtype=np.float32, copy=False)
                    box_prbs = msk_prbs[:, np.newaxis]
                    boxs     = np.concatenate([boxs, box_clss, box_prbs], axis=-1)
                    pickle.dump(img_nam, fid_tst_imgs, pickle.HIGHEST_PROTOCOL)
                    pickle.dump(boxs,    fid_tst_boxs, pickle.HIGHEST_PROTOCOL)
                    pickle.dump(msks,    fid_tst_msks, pickle.HIGHEST_PROTOCOL)
                    '''
                    box_idxs = np.arange(len(boxs)) + 1
                    for i, box in enumerate(boxs):
                        box_ymn, box_xmn, box_ymx, box_xmx = box
                        box_idx = box_idxs[i]
                        box_cls = box_clss[i]
                        box_cls = cat_idx_to_cls_nam[box_cls]
                        box_prb = box_prbs[i]
                        sri = [str(box_idx), img_nam, box_cls, str(box_prb), \
                               str(box_xmn), str(box_ymn), str(box_xmx), str(box_ymx)]
                        sri = ",".join(sri)
                        sri = sri + '\n'
                        fid_tst_boxs.writelines([sri])
                    '''
    
    def display_detections(self):
        
        GC  = GeneratorForCOCO(False, self.dat_dir_tra, self.dat_dir_tst, self.dat_dir_rst, self.tst_shw, self.tst_sav, \
                               self.bat_siz_all, 2)
        #GV = GeneratorForVOC (False, self.dat_dir_tra, self.dat_dir_tst, self.dat_dir_rst, self.tst_shw, self.tst_sav, \
        #                      self.bat_siz_all, 2)
        
        with open(os.path.join(self.log_dir_tst, "imgs"), 'rb') as fid_tst_imgs, \
             open(os.path.join(self.log_dir_tst, "boxs"), 'rb') as fid_tst_boxs, \
             open(os.path.join(self.log_dir_tst, "msks"), 'rb') as fid_tst_msks:
            
            while True:
                try:  
                    img_nam = pickle.load(fid_tst_imgs)
                    boxs    = pickle.load(fid_tst_boxs)
                    msks    = pickle.load(fid_tst_msks)
                    #print(boxs.shape)
                    #print(msks.shape)
                    img_fil = os.path.join(self.dat_dir_tst, img_nam)
                    img     = cv2.imread(img_fil)
                    #print(img.shape)
                    if type(img) != np.ndarray:
                        print("Failed to find image %s" %(img_fil))
                        continue
                    img_hgt, img_wdh = img.shape[0], img.shape[1]
                    if img.size == img_hgt * img_wdh:
                        print ('Gray Image %s' %(img_fil))
                        img_zro = np.empty((img_hgt, img_wdh, 3), dtype=np.uint8)
                        img_zro[:, :, :] = img[:, :, np.newaxis]
                        img     = img_zro
                    assert img.size == img_wdh * img_hgt * 3, '%s' % img_nam
                    img      = img[:, :, ::-1]
                    box_prbs = boxs[:,  -1]
                    box_clss = boxs[:,  -2].astype(dtype=np.int32, copy=False)
                    boxs     = boxs[:, :-2]
                    GC.display_instances(img, boxs, box_clss, box_prbs, msks, img_nam)
                except EOFError:  
                    return
    
    
    def show_loss_acc(self):

        with open(os.path.join(LOG_PATH1, "loss"), 'rb') as fid_train_loss, \
             open(os.path.join(LOG_PATH1, "mAP"), 'rb') as fid_train_mAP, \
             open(os.path.join(LOG_PATH2, "mAP"), 'rb') as fid_val_mAP:
                    
            loss_history      = get_all_data(fid_train_loss)
            train_acc_history = get_all_data(fid_train_mAP)
            val_acc_history   = get_all_data(fid_val_mAP)

            plt.figure(1)

            plt.subplot(2, 1, 1)
            plt.title('Training loss')
            plt.xlabel('Iteration')

            plt.subplot(2, 1, 2)
            plt.title('accuracy')
            plt.xlabel('Epoch')
            
            #plt.subplot(3, 1, 3)
            #plt.title('Validation accuracy')
            #plt.xlabel('Epoch')
            
            plt.subplot(2, 1, 1)
            plt.plot(loss_history, 'o')

            plt.subplot(2, 1, 2)
            plt.plot(train_acc_history, '-o', label="train_acc")
            plt.plot(val_acc_history, '-o', label="val_acc")

            for i in [1, 2]:
                plt.subplot(2, 1, i)
                plt.legend(loc='upper center', ncol=4)

                plt.gcf().set_size_inches(15, 15)
            
            plt.show()