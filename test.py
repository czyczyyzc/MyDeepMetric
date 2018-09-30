import time
import numpy as np
import tensorflow as tf
from resnet101_deep_metric import Resnet101_Deep_Metric
from Mybase.solver import Solver


def test():
    
    mdl = Resnet101_Deep_Metric(cls_num=81, reg=1e-4, drp=0.5, typ=tf.float32)
    sov = Solver(mdl,
                 opm_cfg={
                     "decay_rule": "fixed",
                     #"optim_rule": "adam",
                     #"lr_base": 0.0002,
                     "optim_rule": "momentum",
                     "lr_base":  0.00005, #0.004
                     "momentum": 0.9,
                     #"optim_rule": "adadelta",
                     #"lr_base": 0.01,
                 },
                 use_gpu     = True, 
                 gpu_lst     = '0',
                 bat_siz     = 1,
                 tra_num     = 8000,
                 val_num     = 40,
                 epc_num     = 10000,
                 prt_ena     = True,
                 itr_per_prt = 20,
                 tst_num     = None,
                 tst_shw     = True,
                 tst_sav     = True,
                 mdl_nam     = 'model.ckpt',
                 mdl_dir     = 'Mybase/Model',
                 log_dir     = 'Mybase/logdata',
                 dat_dir     = 'Mybase/datasets',
                 mov_ave_dca = 0.99,
                 epc_per_dca = 1)
    #print("TRAINING...")
    #sov.train()
    print("TESTING...")
    sov.test()
    sov.display_detections()
    #sov.show_loss_acc()

test()