import numpy as np
import tensorflow as tf

def modified_smooth_l1(sigma, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights):
    """
        ResultLoss = outside_weights * SmoothL1(inside_weights * (bbox_pred - bbox_targets))
        SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                      |x| - 0.5 / sigma^2,    otherwise
    """
    #print("modified_smooth_l1 shape checking!")
    #print(bbox_pred.shape)
    #print(bbox_targets.shape)
    #print(bbox_inside_weights.shape)
    #print(bbox_outside_weights.shape)

    #assert bbox_pred.shape == bbox_targets.shape == bbox_inside_weights.shape == bbox_outside_weights.shape, \
    #    "The shapes of bbox_pred, bbox_targets, bbox_inside_weights and bbox_outside_weights are wrong!"
    
    sigma2 = sigma * sigma

    inside_mul = tf.multiply(bbox_inside_weights, tf.subtract(bbox_pred, bbox_targets))

    smooth_l1_sign = tf.cast(tf.less(tf.abs(inside_mul), 1.0 / sigma2), tf.float32)
    smooth_l1_option1 = tf.multiply(tf.multiply(inside_mul, inside_mul), 0.5 * sigma2)
    smooth_l1_option2 = tf.subtract(tf.abs(inside_mul), 0.5 / sigma2)
    smooth_l1_result = tf.add(tf.multiply(smooth_l1_option1, smooth_l1_sign),
                              tf.multiply(smooth_l1_option2, tf.abs(tf.subtract(smooth_l1_sign, 1.0))))

    outside_mul = tf.multiply(bbox_outside_weights, smooth_l1_result)

    return outside_mul

def smooth_l1(sigma, bbox_pred, bbox_targets):
    """
        ResultLoss = outside_weights * SmoothL1(inside_weights * (bbox_pred - bbox_targets))
        SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                      |x| - 0.5 / sigma^2,    otherwise
    """
    sigma2 = sigma ** 2
    diff = tf.abs(bbox_targets - bbox_pred)

    smooth_l1_sign = tf.cast(tf.less(diff, 1.0/sigma2), tf.float32)
    smooth_l1_option1 = 0.5 * sigma2 * diff**2
    smooth_l1_option2 = diff -  0.5 / sigma2
    smooth_l1_result = smooth_l1_sign * smooth_l1_option1 + (1.0 - smooth_l1_sign) * smooth_l1_option2
    
    return smooth_l1_result

        
def dice_loss1(prb_msks_pre, prb_msks_pst, vld_msks_pre):
    eps = 1e-5
    inter = tf.reduce_sum(prb_msks_pre*prb_msks_pst*vld_msks_pre)
    union = tf.reduce_sum(prb_msks_pre*vld_msks_pre) + tf.reduce_sum(prb_msks_pst*vld_msks_pre) + eps
    loss = 1.0 - (2 * inter / union)
    return loss
    
