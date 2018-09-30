import numpy as np
import tensorflow as tf
from Mybase.comp_utils import *
#from shapely.geometry import Polygon


def bbox_transform(rpns, gbxs, std_dev=[0.1, 0.1, 0.2, 0.2]):
    
    rpn_ymn, rpn_xmn, rpn_ymx, rpn_xmx = tf.split(rpns, 4, axis=-1)
    gbx_ymn, gbx_xmn, gbx_ymx, gbx_xmx = tf.split(gbxs, 4, axis=-1)

    rpn_yct = (rpn_ymn + rpn_ymx) / 2.    
    rpn_xct = (rpn_xmn + rpn_xmx) / 2.
    rpn_hgt =  rpn_ymx - rpn_ymn + 1.0
    rpn_wdh =  rpn_xmx - rpn_xmn + 1.0
    
    gbx_yct = (gbx_ymn + gbx_ymx) / 2.
    gbx_xct = (gbx_xmn + gbx_xmx) / 2.
    gbx_hgt =  gbx_ymx - gbx_ymn + 1.0
    gbx_wdh =  gbx_xmx - gbx_xmn + 1.0
    
    rpn_ypd = (gbx_yct - rpn_yct) / rpn_hgt
    rpn_xpd = (gbx_xct - rpn_xct) / rpn_wdh
    rpn_hpd = tf.log(gbx_hgt / rpn_hgt)
    rpn_wpd = tf.log(gbx_wdh / rpn_wdh)

    prds = tf.concat([rpn_ypd, rpn_xpd, rpn_hpd, rpn_wpd], axis=-1)
    prds = prds / std_dev
    return prds


def bbox_transform_inv(rpns, prds, std_dev=[0.1, 0.1, 0.2, 0.2]):
    
    prds = prds * std_dev
    rpn_ymn, rpn_xmn, rpn_ymx, rpn_xmx = tf.split(rpns, 4, axis=-1)
    rpn_ypd, rpn_xpd, rpn_hpd, rpn_wpd = tf.split(prds, 4, axis=-1)
    
    rpn_yct = (rpn_ymn + rpn_ymx) / 2.    
    rpn_xct = (rpn_xmn + rpn_xmx) / 2.
    rpn_hgt =  rpn_ymx - rpn_ymn + 1.0
    rpn_wdh =  rpn_xmx - rpn_xmn + 1.0
    
    box_yct = rpn_ypd * rpn_hgt + rpn_yct
    box_xct = rpn_xpd * rpn_wdh + rpn_xct
    box_hgt = rpn_hgt * tf.exp(rpn_hpd)
    box_wdh = rpn_wdh * tf.exp(rpn_wpd)

    box_ymn = box_yct - (box_hgt - 1.0) / 2.
    box_xmn = box_xct - (box_wdh - 1.0) / 2.
    box_ymx = box_yct + (box_hgt - 1.0) / 2.
    box_xmx = box_xct + (box_wdh - 1.0) / 2.

    boxs = tf.concat([box_ymn, box_xmn, box_ymx, box_xmx], axis=-1)
    return boxs


def bbox_filter(boxs, siz_min):
    
    box_ymn, box_xmn, box_ymx, box_xmx = tf.split(boxs, 4, axis=-1)
    box_hgt = box_ymx - box_ymn + 1.0
    box_wdh = box_xmx - box_xmn + 1.0
    idxs = tf.where(tf.logical_and(box_hgt>=siz_min, box_wdh>=siz_min)[:, 0])
    return idxs


def bbox_clip(boxs, box_ref):

    box_ymn, box_xmn, box_ymx, box_xmx = tf.split(boxs, 4, axis=-1)
    
    box_ymn = tf.minimum(tf.maximum(box_ymn, box_ref[0]), box_ref[2])
    box_xmn = tf.minimum(tf.maximum(box_xmn, box_ref[1]), box_ref[3])
    box_ymx = tf.maximum(tf.minimum(box_ymx, box_ref[2]), box_ref[0])
    box_xmx = tf.maximum(tf.minimum(box_xmx, box_ref[3]), box_ref[1])
    
    # Double check! Empty boxes when no-intersection.
    box_ymn = tf.minimum(box_ymn, box_ymx)
    box_xmn = tf.minimum(box_xmn, box_xmx)
    boxs = tf.concat([box_ymn, box_xmn, box_ymx, box_xmx], axis=-1)
    return boxs


def bbox_clip_py(boxs, ref):

    box_ymn, box_xmn, box_ymx, box_xmx = np.split(boxs, 4, axis=-1)
    
    box_ymn = np.minimum(np.maximum(box_ymn, ref[0]), ref[2])
    box_xmn = np.minimum(np.maximum(box_xmn, ref[1]), ref[3])
    box_ymx = np.maximum(np.minimum(box_ymx, ref[2]), ref[0])
    box_xmx = np.maximum(np.minimum(box_xmx, ref[3]), ref[1])
    
    # Double check! Empty boxes when no-intersection.
    box_ymn = np.minimum(box_ymn, box_ymx)
    box_xmn = np.minimum(box_xmn, box_xmx)
    boxs = np.concatenate([box_ymn, box_xmn, box_ymx, box_xmx], axis=-1)
    return boxs


def bbox_clip2(boxs, ref):
    #注意p0, p1, p2, p3对应top_lft, top_rgt, btm_rgt, btm_lft
    #调整y时，根据p12和p03调整x
    def clip0(y0, x0, y1, x1, y2, x2, y3, x3):
        p0 = tf.concat([y0, x0], axis=-1)
        p1 = tf.concat([y1, x1], axis=-1)
        p2 = tf.concat([y2, x2], axis=-1)
        p3 = tf.concat([y3, x3], axis=-1)
        p12 = lin_fit_pin(p1, p2)
        p30 = lin_fit_pin(p3, p0)
        y0 = tf.minimum(tf.maximum(y0, ref[0]), ref[2])
        y1 = tf.minimum(tf.maximum(y1, ref[0]), ref[2])
        y2 = tf.minimum(tf.maximum(y2, ref[0]), ref[2])
        y3 = tf.minimum(tf.maximum(y3, ref[0]), ref[2])
        x0 = lin_y(p30, y0)
        x1 = lin_y(p12, y1)
        x2 = lin_y(p12, y2)
        x3 = lin_y(p30, y3)
        return y0, x0, y1, x1, y2, x2, y3, x3
    
    def clip1(y0, x0, y1, x1, y2, x2, y3, x3):
        p0 = tf.concat([y0, x0], axis=-1)
        p1 = tf.concat([y1, x1], axis=-1)
        p2 = tf.concat([y2, x2], axis=-1)
        p3 = tf.concat([y3, x3], axis=-1)
        p01 = lin_fit_pin(p0, p1)
        p23 = lin_fit_pin(p2, p3)
        x0 = tf.minimum(tf.maximum(x0, ref[1]), ref[3])
        x1 = tf.minimum(tf.maximum(x1, ref[1]), ref[3])
        x2 = tf.minimum(tf.maximum(x2, ref[1]), ref[3])
        x3 = tf.minimum(tf.maximum(x3, ref[1]), ref[3])
        y0 = lin_x(p01, x0)
        y1 = lin_x(p01, x1)
        y2 = lin_x(p23, x2)
        y3 = lin_x(p23, x3)
        return y0, x0, y1, x1, y2, x2, y3, x3
    
    def clip2(y0, x0, y1, x1, y2, x2, y3, x3):
        y0 = tf.minimum(tf.maximum(y0, ref[0]), ref[2])
        y1 = tf.minimum(tf.maximum(y1, ref[0]), ref[2])
        y2 = tf.minimum(tf.maximum(y2, ref[0]), ref[2])
        y3 = tf.minimum(tf.maximum(y3, ref[0]), ref[2])
        x0 = tf.minimum(tf.maximum(x0, ref[1]), ref[3])
        x1 = tf.minimum(tf.maximum(x1, ref[1]), ref[3])
        x2 = tf.minimum(tf.maximum(x2, ref[1]), ref[3])
        x3 = tf.minimum(tf.maximum(x3, ref[1]), ref[3])
        boxs = tf.concat([y0, x0, y1, x1, y2, x2, y3, x3], axis=-1)
        return boxs
    
    def clip_0(boxs):
        y0, x0, y1, x1, y2, x2, y3, x3 = tf.split(boxs, 8, axis=-1)
        y0, x0, y1, x1, y2, x2, y3, x3 = clip0(y0, x0, y1, x1, y2, x2, y3, x3)
        y0, x0, y1, x1, y2, x2, y3, x3 = clip1(y0, x0, y1, x1, y2, x2, y3, x3)
        boxs = clip2(y0, x0, y1, x1, y2, x2, y3, x3)
        return boxs
    
    def clip_1(boxs):
        y0, x0, y1, x1, y2, x2, y3, x3 = tf.split(boxs, 8, axis=-1)
        y0, x0, y1, x1, y2, x2, y3, x3 = clip1(y0, x0, y1, x1, y2, x2, y3, x3)
        y0, x0, y1, x1, y2, x2, y3, x3 = clip0(y0, x0, y1, x1, y2, x2, y3, x3)
        boxs = clip2(y0, x0, y1, x1, y2, x2, y3, x3)
        return boxs
    
    box_edgs = bbox_edges2(boxs)
    leh_amx  = tf.argmax(box_edgs, axis=1)
    leh_amx  = tf.equal(leh_amx%2, 0) #看最长边是否为p01或p23
    leh_amx  = tf.tile(leh_amx[:, tf.newaxis], [1, 8])
    boxs = tf.where(leh_amx, clip_1(boxs), clip_0(boxs))
    return boxs


def bbox_clip_py2(boxs, ref):
    #注意p0, p1, p2, p3对应top_lft, top_rgt, btm_rgt, btm_lft
    #调整y时，根据p12和p03调整x
    def clip0(y0, x0, y1, x1, y2, x2, y3, x3):
        p0 = np.concatenate([y0, x0], axis=-1)
        p1 = np.concatenate([y1, x1], axis=-1)
        p2 = np.concatenate([y2, x2], axis=-1)
        p3 = np.concatenate([y3, x3], axis=-1)
        p12 = lin_fit_pin_py(p1, p2)
        p30 = lin_fit_pin_py(p3, p0)
        y0 = np.minimum(np.maximum(y0, ref[0]), ref[2])
        y1 = np.minimum(np.maximum(y1, ref[0]), ref[2])
        y2 = np.minimum(np.maximum(y2, ref[0]), ref[2])
        y3 = np.minimum(np.maximum(y3, ref[0]), ref[2])
        x0 = lin_y_py(p30, y0)
        x1 = lin_y_py(p12, y1)
        x2 = lin_y_py(p12, y2)
        x3 = lin_y_py(p30, y3)
        return y0, x0, y1, x1, y2, x2, y3, x3
    
    def clip1(y0, x0, y1, x1, y2, x2, y3, x3):
        p0 = np.concatenate([y0, x0], axis=-1)
        p1 = np.concatenate([y1, x1], axis=-1)
        p2 = np.concatenate([y2, x2], axis=-1)
        p3 = np.concatenate([y3, x3], axis=-1)
        p01 = lin_fit_pin_py(p0, p1)
        p23 = lin_fit_pin_py(p2, p3)
        x0 = np.minimum(np.maximum(x0, ref[1]), ref[3])
        x1 = np.minimum(np.maximum(x1, ref[1]), ref[3])
        x2 = np.minimum(np.maximum(x2, ref[1]), ref[3])
        x3 = np.minimum(np.maximum(x3, ref[1]), ref[3])
        y0 = lin_x_py(p01, x0)
        y1 = lin_x_py(p01, x1)
        y2 = lin_x_py(p23, x2)
        y3 = lin_x_py(p23, x3)
        return y0, x0, y1, x1, y2, x2, y3, x3
    
    def clip2(y0, x0, y1, x1, y2, x2, y3, x3):
        y0 = np.minimum(np.maximum(y0, ref[0]), ref[2])
        y1 = np.minimum(np.maximum(y1, ref[0]), ref[2])
        y2 = np.minimum(np.maximum(y2, ref[0]), ref[2])
        y3 = np.minimum(np.maximum(y3, ref[0]), ref[2])
        x0 = np.minimum(np.maximum(x0, ref[1]), ref[3])
        x1 = np.minimum(np.maximum(x1, ref[1]), ref[3])
        x2 = np.minimum(np.maximum(x2, ref[1]), ref[3])
        x3 = np.minimum(np.maximum(x3, ref[1]), ref[3])
        boxs = np.concatenate([y0, x0, y1, x1, y2, x2, y3, x3], axis=-1)
        return boxs
    
    def clip_0(boxs):
        y0, x0, y1, x1, y2, x2, y3, x3 = np.split(boxs, 8, axis=-1)
        y0, x0, y1, x1, y2, x2, y3, x3 = clip0(y0, x0, y1, x1, y2, x2, y3, x3)
        y0, x0, y1, x1, y2, x2, y3, x3 = clip1(y0, x0, y1, x1, y2, x2, y3, x3)
        boxs = clip2(y0, x0, y1, x1, y2, x2, y3, x3)
        return boxs
    
    def clip_1(boxs):
        y0, x0, y1, x1, y2, x2, y3, x3 = np.split(boxs, 8, axis=-1)
        y0, x0, y1, x1, y2, x2, y3, x3 = clip1(y0, x0, y1, x1, y2, x2, y3, x3)
        y0, x0, y1, x1, y2, x2, y3, x3 = clip0(y0, x0, y1, x1, y2, x2, y3, x3)
        boxs = clip2(y0, x0, y1, x1, y2, x2, y3, x3)
        return boxs
    
    box_edgs = bbox_edges_py2(boxs)
    leh_amx  = np.argmax(box_edgs, axis=1)
    leh_amx  = np.equal(leh_amx%2, 0) #看最长边是否为p01或p23
    leh_amx  = np.tile(leh_amx[:, np.newaxis], [1, 8])
    boxs = np.where(leh_amx, clip_1(boxs), clip_0(boxs))
    return boxs

def bbox_area(boxs):
    box_ymn, box_xmn, box_ymx, box_xmx = tf.split(boxs, 4, axis=-1)
    box_aras = (box_ymx - box_ymn) * (box_xmx - box_xmn)
    return box_aras


def bbox_area_py(boxs):
    box_ymn, box_xmn, box_ymx, box_xmx = np.split(boxs, 4, axis=-1)
    box_aras = (box_ymx - box_ymn) * (box_xmx - box_xmn)
    return box_aras


def bbox_area2(boxs):
    y0, x0, y1, x1, y2, x2, y3, x3 = tf.split(boxs, 8, axis=-1) #8个N*1
    box_aras = tf.abs(((x0-x2)*(y1-y3)-(y0-y2)*(x1-x3))/2.0)
    return box_aras


def bbox_area_py2(boxs):
    y0, x0, y1, x1, y2, x2, y3, x3 = np.split(boxs, 8, axis=-1) #8个N*1
    box_aras = np.absolute(((x0-x2)*(y1-y3)-(y0-y2)*(x1-x3))/2.0)
    return box_aras


def lin_fit_pin(p0, p1):
    # fit a line ax+by+c = 0
    y0, x0 = tf.split(p0, 2, axis=-1)
    y1, x1 = tf.split(p1, 2, axis=-1)
    a = y0 - y1
    b = x1 - x0
    c = y1*x0 - y0*x1
    lin = tf.concat([a, b, c], axis=-1)
    return lin


def lin_fit_pin_py(p0, p1):
    # fit a line ax+by+c = 0
    y0, x0 = np.split(p0, 2, axis=-1)
    y1, x1 = np.split(p1, 2, axis=-1)
    a = y0 - y1
    b = x1 - x0
    c = y1*x0 - y0*x1
    lin = np.concatenate([a, b, c], axis=-1)
    return lin


def lin_y(lin, y):
    a, b, c = tf.split(lin, 3, axis=-1)
    x = -(b*y+c) / (a+1e-8)
    return x


def lin_y_py(lin, y):
    a, b, c = np.split(lin, 3, axis=-1)
    x = -(b*y+c) / (a+1e-8)
    return x


def lin_x(lin, x):
    a, b, c = tf.split(lin, 3, axis=-1)
    y = -(a*x+c) / (b+1e-8)
    return y


def lin_x_py(lin, x):
    a, b, c = np.split(lin, 3, axis=-1)
    y = -(a*x+c) / (b+1e-8)
    return y


def pin_lin_vrt(lin, p0):
    #line_verticle
    a0, b0, c0 = tf.split(lin, 3, axis=-1)
    y0, x0 = tf.split(p0, 2, axis=-1)
    a1  =  b0
    b1  = -a0
    c1  =  a0*y0 - b0*x0
    lin = tf.concat([a1, b1, c1], axis=-1)
    return lin


def pin_lin_dst(lin, p0):
    #point to line distance
    a, b, c = tf.split(lin, 3, axis=-1)
    y0, x0 = tf.split(p0, 2, axis=-1)
    dst = tf.abs(a*x0+b*y0+c) / tf.sqrt(a**2+b**2)
    return dst


def pin_lin_pra(lin, p0):
    #line through the point parallel to another line
    a, b, c = tf.split(lin, 3, axis=-1)
    y0, x0 = tf.split(p0, 2, axis=-1)
    c0 = -(a*x0 + b*y0)
    lin = tf.concat([a, b, c0], axis=-1)
    return lin

'''
def pin_lin_dst(p0, p1, p2):
    #point to line distance
    return np.linalg.norm(np.cross(p1-p0, p2-p0)) / np.linalg.norm(p1-p0) # (x, y)颠倒不影响
'''

def lin_lin_crs(lin0, lin1):
    #cross point of two lines
    a0, b0, c0 = tf.split(lin0, 3, axis=-1)
    a1, b1, c1 = tf.split(lin1, 3, axis=-1)
    x = (b0*c1-b1*c0) / (a0*b1 - a1*b0)
    y = (a1*c0-a0*c1) / (a0*b1 - a1*b0)
    p = tf.concat([y, x], axis=-1)
    return p


def lin_lin_pra(lin, d, y_first=True, btm_rgt=True):
    a, b, c = tf.split(lin, 3, axis=-1)
    if y_first:
        k = -a / b #斜率
        m = -c / b #偏移
        if btm_rgt: #compute the line under the line given
            m = m + d*tf.sqrt(k**2+1.0)
        else:
            m = m - d*tf.sqrt(k**2+1.0)
        b = tf.zeros(shape=tf.shape(k), dtype=tf.float32) - 1.0
        lin = tf.concat([k, b, m], axis=-1)
        return lin
    else:
        k = -b / a
        m = -c / a
        if btm_rgt: #compute the line right to the line given
            m = m + d*tf.sqrt(1.0+k**2)
        else:
            m = m - d*tf.sqrt(1.0+k**2)
        a = tf.zeros(shape=tf.shape(k), dtype=tf.float32) - 1.0
        lin = tf.concat([a, k, m], axis=-1)
        return lin
    

def pin_pin_agl(p0, p1, p2):
    p1_0 = p1 - p0
    p2_0 = p2 - p0
    y0, x0 = tf.split(p1_0, 2, axis=-1)
    y1, x1 = tf.split(p2_0, 2, axis=-1)
    agl = tf.acos((x0*x1+y0*y1)/tf.sqrt((x0**2+y0**2)*(x1**2+y1**2)))
    return agl


def pin_pin_dst(p0, p1):
    y0, x0 = tf.split(p0, 2, axis=-1)
    y1, x1 = tf.split(p1, 2, axis=-1)
    dst = tf.sqrt((x0-x1)**2+(y0-y1)**2)
    return dst


def pin_pin_dst_py(p0, p1):
    y0, x0 = np.split(p0, 2, axis=-1)
    y1, x1 = np.split(p1, 2, axis=-1)
    dst = np.sqrt((x0-x1)**2+(y0-y1)**2)
    return dst


def pin_lin_crs(lin, p0, p1):
    a, b, c = tf.split(lin, 3, axis=-1)
    y0, x0  = tf.split(p0,  2, axis=-1)
    y1, x1  = tf.split(p1,  2, axis=-1)
    z0 = a*x0 + b*y0 + c
    z1 = a*x1 + b*y1 + c
    z  = (z0*z1) < 0
    z  = tf.concat([z, z], axis=-1)
    return z


def crs_mul(p0, p1): #向量叉积
    y0, x0 = tf.split(p0, 2, axis=-1)
    y1, x1 = tf.split(p1, 2, axis=-1)
    z = x0*y1 - x1*y0
    return z


def pin_pin_crs(p0, p1, p2, p3):
    #注意p0、p1为一组构成线段, p2、p3为一组构成线段
    v0 = p1 - p0
    v1 = p3 - p2
    s0 = p0 - p2
    s1 = p1 - p2
    t0 = p2 - p0
    t1 = p3 - p0
    z0 = crs_mul(s0, v1)*crs_mul(s1, v1) < 0
    z1 = crs_mul(t0, v0)*crs_mul(t1, v0) < 0
    z  = tf.logical_and(z0, z1)
    z  = tf.concat([z, z], axis=-1)
    return z


def pin_qud_inn(p0, p1, p2, p3, p):
    y0, x0 = tf.split(p0, 2, axis=-1)
    y1, x1 = tf.split(p1, 2, axis=-1)
    y2, x2 = tf.split(p2, 2, axis=-1)
    y3, x3 = tf.split(p3, 2, axis=-1)
    y,  x  = tf.split(p,  2, axis=-1)
    z0 = (x1-x0)*(y-y0) - (y1-y0)*(x-x0)
    z1 = (x2-x1)*(y-y1) - (y2-y1)*(x-x1)
    z2 = (x3-x2)*(y-y2) - (y3-y2)*(x-x2)
    z3 = (x0-x3)*(y-y3) - (y0-y3)*(x-x3)
    z4 = tf.cast(z0>=0, tf.int8) + tf.cast(z1>=0, tf.int8) + tf.cast(z2>=0, tf.int8) + tf.cast(z3>=0, tf.int8)
    z5 = tf.cast(z0< 0, tf.int8) + tf.cast(z1< 0, tf.int8) + tf.cast(z2< 0, tf.int8) + tf.cast(z3< 0, tf.int8)
    z  = tf.logical_or(tf.equal(z4, 4), tf.equal(z5, 4))
    z  = tf.concat([z, z], axis=-1)
    return z


def pin_qud_inn2(p0, p1, p2, p3, p):
    y0, x0 = tf.split(p0, 2, axis=-1)
    y1, x1 = tf.split(p1, 2, axis=-1)
    y2, x2 = tf.split(p2, 2, axis=-1)
    y3, x3 = tf.split(p3, 2, axis=-1)
    y,  x  = tf.split(p,  2, axis=-1)
    z0 = (x1-x0)*(y-y0) - (y1-y0)*(x-x0)
    z1 = (x2-x1)*(y-y1) - (y2-y1)*(x-x1)
    z2 = (x3-x2)*(y-y2) - (y3-y2)*(x-x2)
    z3 = (x0-x3)*(y-y3) - (y0-y3)*(x-x3)
    z4 = tf.cast(z0>=0, tf.int8) + tf.cast(z1>=0, tf.int8) + tf.cast(z2>=0, tf.int8) + tf.cast(z3>=0, tf.int8)
    z5 = tf.cast(z0< 0, tf.int8) + tf.cast(z1< 0, tf.int8) + tf.cast(z2< 0, tf.int8) + tf.cast(z3< 0, tf.int8)
    z  = tf.where(tf.logical_or(tf.equal(z4, 4), tf.equal(z5, 4)))[:, 0]
    return z


def bbox_overlaps2(boxs, gbxs):
    #交点、在内部的点
    #boxs = tf.reshape(boxs, [-1, 8])
    #gbxs = tf.reshape(gbxs, [-1, 8])
    box_num = tf.shape(boxs)[0]
    gbx_num = tf.shape(gbxs)[0]
    boxs = tf.reshape(tf.tile(boxs, [1, gbx_num]), [-1, 8])
    gbxs = tf.reshape(tf.tile(gbxs, [box_num, 1]), [-1, 8])
    box_aras = bbox_area2(boxs)
    gbx_aras = bbox_area2(gbxs)
    
    p0, p1, p2, p3 = tf.split(boxs, 4, axis=-1)
    q0, q1, q2, q3 = tf.split(gbxs, 4, axis=-1)
    p01 = lin_fit_pin(p0, p1)
    p12 = lin_fit_pin(p1, p2)
    p23 = lin_fit_pin(p2, p3)
    p30 = lin_fit_pin(p3, p0)
    q01 = lin_fit_pin(q0, q1)
    q12 = lin_fit_pin(q1, q2)
    q23 = lin_fit_pin(q2, q3)
    q30 = lin_fit_pin(q3, q0)
    tmp = tf.zeros(shape=tf.shape(p0), dtype=tf.float32) + np.inf
    p01_q01 = tf.where(pin_pin_crs(p0, p1, q0, q1), lin_lin_crs(p01, q01), tmp)
    p12_q01 = tf.where(pin_pin_crs(p1, p2, q0, q1), lin_lin_crs(p12, q01), tmp)
    p23_q01 = tf.where(pin_pin_crs(p2, p3, q0, q1), lin_lin_crs(p23, q01), tmp)
    p30_q01 = tf.where(pin_pin_crs(p3, p0, q0, q1), lin_lin_crs(p30, q01), tmp)
    p01_q12 = tf.where(pin_pin_crs(p0, p1, q1, q2), lin_lin_crs(p01, q12), tmp)
    p12_q12 = tf.where(pin_pin_crs(p1, p2, q1, q2), lin_lin_crs(p12, q12), tmp)
    p23_q12 = tf.where(pin_pin_crs(p2, p3, q1, q2), lin_lin_crs(p23, q12), tmp)
    p30_q12 = tf.where(pin_pin_crs(p3, p0, q1, q2), lin_lin_crs(p30, q12), tmp)
    p01_q23 = tf.where(pin_pin_crs(p0, p1, q2, q3), lin_lin_crs(p01, q23), tmp)
    p12_q23 = tf.where(pin_pin_crs(p1, p2, q2, q3), lin_lin_crs(p12, q23), tmp)
    p23_q23 = tf.where(pin_pin_crs(p2, p3, q2, q3), lin_lin_crs(p23, q23), tmp)
    p30_q23 = tf.where(pin_pin_crs(p3, p0, q2, q3), lin_lin_crs(p30, q23), tmp)
    p01_q30 = tf.where(pin_pin_crs(p0, p1, q3, q0), lin_lin_crs(p01, q30), tmp)
    p12_q30 = tf.where(pin_pin_crs(p1, p2, q3, q0), lin_lin_crs(p12, q30), tmp)
    p23_q30 = tf.where(pin_pin_crs(p2, p3, q3, q0), lin_lin_crs(p23, q30), tmp)
    p30_q30 = tf.where(pin_pin_crs(p3, p0, q3, q0), lin_lin_crs(p30, q30), tmp)
    p_q0 = tf.where(pin_qud_inn(p0, p1, p2, p3, q0), q0, tmp)
    p_q1 = tf.where(pin_qud_inn(p0, p1, p2, p3, q1), q1, tmp)
    p_q2 = tf.where(pin_qud_inn(p0, p1, p2, p3, q2), q2, tmp)
    p_q3 = tf.where(pin_qud_inn(p0, p1, p2, p3, q3), q3, tmp)
    q_p0 = tf.where(pin_qud_inn(q0, q1, q2, q3, p0), p0, tmp)
    q_p1 = tf.where(pin_qud_inn(q0, q1, q2, q3, p1), p1, tmp)
    q_p2 = tf.where(pin_qud_inn(q0, q1, q2, q3, p2), p2, tmp)
    q_p3 = tf.where(pin_qud_inn(q0, q1, q2, q3, p3), p3, tmp)
    ply = tf.stack([p01_q01, p12_q01, p23_q01, p30_q01, 
                    p01_q12, p12_q12, p23_q12, p30_q12, 
                    p01_q23, p12_q23, p23_q23, p30_q23,
                    p01_q30, p12_q30, p23_q30, p30_q30,
                    p_q0, p_q1, p_q2, p_q3, q_p0, q_p1, q_p2, q_p3], axis=1)
    
    def ply_ara_sig(ply):
        idx = tf.where(tf.not_equal(ply[:, 0], np.inf))
        ply = tf.gather_nd(ply, idx)
        ply_num = tf.shape(ply)[0]
        ctr = tf.reduce_mean(ply, axis=0)
        ply = ply - ctr
        agl = tf.atan2(ply[:, 0], ply[:, 1])
        agl, idx = tf.nn.top_k(-agl, k=ply_num, sorted=True) #逆时针排序，否则面积计算为负
        ply = tf.gather(ply, idx)
        ply = tf.pad(ply, [[0, 8-ply_num], [0, 0]])
        y8, x8 = tf.split(ply[ply_num-1], 2, axis=-1)
        ply = tf.reshape(ply, [16])
        y0, x0, y1, x1, y2, x2, y3, x3, y4, x4, y5, x5, y6, x6, y7, x7 = tf.split(ply, 16, axis=-1)
        ara = ((x0*y1-x1*y0)+(x1*y2-x2*y1)+(x2*y3-x3*y2)+(x3*y4-x4*y3)+
               (x4*y5-x5*y4)+(x5*y6-x6*y5)+(x6*y7-x7*y6)+(x8*y0-x0*y8)) / 2.0
        ara = tf.where(ply_num>=3, ara, tf.constant([0.0], dtype=tf.float32))
        return ara
    ply_int = tf.map_fn(ply_ara_sig, ply, dtype=None,
                        parallel_iterations=10, back_prop=False, 
                        swap_memory=False, infer_shape=True)
    ply_uin = box_aras + gbx_aras - ply_int
    ply_ovp = ply_int / ply_uin
    ply_ovp = tf.reshape(ply_ovp, [box_num, gbx_num])
    return ply_ovp
    
"""
def bbox_overlaps_py2(boxs, gbxs):
    
    boxs = boxs.reshape([-1, 4, 2])
    gbxs = gbxs.reshape([-1, 4, 2])
    ovps = np.zeros(shape=[len(boxs), len(gbxs)], dtype=np.float32)
    for i in range(len(boxs)):
        for j in range(len(gbxs)):
            g = Polygon(boxs[i])
            p = Polygon(gbxs[j])
            inter = Polygon(g).intersection(Polygon(p)).area
            union = g.area + p.area - inter
            if union == 0:
                ovps[i, j] = 0.0
            else:
                ovps[i, j] = inter / union
    return ovps
"""

def bbox_intersects(boxs, gbxs):
    
    box_num = tf.shape(boxs)[0]
    gbx_num = tf.shape(gbxs)[0]

    boxs = tf.reshape(tf.tile(boxs, [1, gbx_num]), [-1, 4])
    gbxs = tf.reshape(tf.tile(gbxs, [box_num, 1]), [-1, 4])
    
    box_ymn, box_xmn, box_ymx, box_xmx = tf.split(boxs, 4, axis=-1)
    gbx_ymn, gbx_xmn, gbx_ymx, gbx_xmx = tf.split(gbxs, 4, axis=-1)
    
    isc_ymn = tf.maximum(box_ymn, gbx_ymn)
    isc_xmn = tf.maximum(box_xmn, gbx_xmn)
    isc_ymx = tf.minimum(box_ymx, gbx_ymx)
    isc_xmx = tf.minimum(box_xmx, gbx_xmx)
    
    isc_hgt = tf.maximum(isc_ymx-isc_ymn+1, 0)
    isc_wdh = tf.maximum(isc_xmx-isc_xmn+1, 0)
    isc_ara = isc_hgt * isc_wdh
    
    box_ara = (box_ymx - box_ymn + 1) * (box_xmx - box_xmn + 1)
    iscs    = tf.where(box_ara>0, isc_ara/box_ara, tf.zeros(shape=tf.shape(box_ara), dtype=tf.float32))
    iscs    = tf.reshape(iscs, [box_num, gbx_num])
    return iscs


def bbox_intersects1(boxs, gbxs):
    
    box_ymn, box_xmn, box_ymx, box_xmx = tf.split(boxs, 4, axis=-1)
    gbx_ymn, gbx_xmn, gbx_ymx, gbx_xmx = tf.split(gbxs, 4, axis=-1)
    
    isc_ymn = tf.maximum(box_ymn, gbx_ymn)
    isc_xmn = tf.maximum(box_xmn, gbx_xmn)
    isc_ymx = tf.minimum(box_ymx, gbx_ymx)
    isc_xmx = tf.minimum(box_xmx, gbx_xmx)
    
    isc_hgt = tf.maximum(isc_ymx-isc_ymn+1, 0)
    isc_wdh = tf.maximum(isc_xmx-isc_xmn+1, 0)
    isc_ara = isc_hgt * isc_wdh
    
    box_ara = (box_ymx - box_ymn + 1) * (box_xmx - box_xmn + 1)
    iscs    = tf.where(box_ara>0, isc_ara/box_ara, tf.zeros(shape=tf.shape(box_ara), dtype=tf.float32))
    iscs    = iscs[:, 0]
    return iscs


def bbox_intersects_py(boxs, gbxs):
    
    box_num = np.shape(boxs)[0]
    gbx_num = np.shape(gbxs)[0]

    boxs = np.reshape(np.tile(boxs, [1, gbx_num]), [-1, 4])
    gbxs = np.reshape(np.tile(gbxs, [box_num, 1]), [-1, 4])
    
    box_ymn, box_xmn, box_ymx, box_xmx = np.split(boxs, 4, axis=-1)
    gbx_ymn, gbx_xmn, gbx_ymx, gbx_xmx = np.split(gbxs, 4, axis=-1)
    
    isc_ymn = np.maximum(box_ymn, gbx_ymn)
    isc_xmn = np.maximum(box_xmn, gbx_xmn)
    isc_ymx = np.minimum(box_ymx, gbx_ymx)
    isc_xmx = np.minimum(box_xmx, gbx_xmx)
    
    isc_hgt = np.maximum(isc_ymx-isc_ymn+1, 0)
    isc_wdh = np.maximum(isc_xmx-isc_xmn+1, 0)
    isc_ara = isc_hgt * isc_wdh
    
    box_ara = (box_ymx - box_ymn) * (box_xmx - box_xmn)
    iscs    = np.where(box_ara>0, isc_ara/box_ara, np.zeros(shape=np.shape(box_ara), dtype=np.float32))
    iscs    = np.reshape(iscs, [box_num, gbx_num])
    return iscs


def bbox_overlaps(boxs, gbxs):
    
    box_num = tf.shape(boxs)[0]
    gbx_num = tf.shape(gbxs)[0]

    boxs = tf.reshape(tf.tile(boxs, [1, gbx_num]), [-1, 4])
    gbxs = tf.reshape(tf.tile(gbxs, [box_num, 1]), [-1, 4])
    
    box_ymn, box_xmn, box_ymx, box_xmx = tf.split(boxs, 4, axis=-1)
    gbx_ymn, gbx_xmn, gbx_ymx, gbx_xmx = tf.split(gbxs, 4, axis=-1)
    
    isc_ymn = tf.maximum(box_ymn, gbx_ymn)
    isc_xmn = tf.maximum(box_xmn, gbx_xmn)
    isc_ymx = tf.minimum(box_ymx, gbx_ymx)
    isc_xmx = tf.minimum(box_xmx, gbx_xmx)
    
    isc_hgt = tf.maximum(isc_ymx-isc_ymn+1, 0)
    isc_wdh = tf.maximum(isc_xmx-isc_xmn+1, 0)
    isc_ara = isc_hgt * isc_wdh
    
    box_ara = (box_ymx - box_ymn + 1) * (box_xmx - box_xmn + 1)
    gbx_ara = (gbx_ymx - gbx_ymn + 1) * (gbx_xmx - gbx_xmn + 1)
    uin_ara = box_ara + gbx_ara - isc_ara #union_area
    
    ovps    = tf.where(uin_ara>0, isc_ara/uin_ara, tf.zeros(shape=tf.shape(uin_ara), dtype=tf.float32))
    ovps    = tf.reshape(ovps, [box_num, gbx_num])
    return ovps


def bbox_overlaps_py(boxs, gbxs):

    box_num = np.shape(boxs)[0]
    gbx_num = np.shape(gbxs)[0]

    boxs = np.reshape(np.tile(boxs, [1, gbx_num]), [-1, 4])
    gbxs = np.reshape(np.tile(gbxs, [box_num, 1]), [-1, 4])

    box_ymn, box_xmn, box_ymx, box_xmx = np.split(boxs, 4, axis=-1)
    gbx_ymn, gbx_xmn, gbx_ymx, gbx_xmx = np.split(gbxs, 4, axis=-1)

    isc_ymn = np.maximum(box_ymn, gbx_ymn)
    isc_xmn = np.maximum(box_xmn, gbx_xmn)
    isc_ymx = np.minimum(box_ymx, gbx_ymx)
    isc_xmx = np.minimum(box_xmx, gbx_xmx)

    isc_hgt = np.maximum(isc_ymx-isc_ymn+1, 0)
    isc_wdh = np.maximum(isc_xmx-isc_xmn+1, 0)
    isc_ara = isc_hgt * isc_wdh

    box_ara = (box_ymx - box_ymn + 1) * (box_xmx - box_xmn + 1)
    gbx_ara = (gbx_ymx - gbx_ymn + 1) * (gbx_xmx - gbx_xmn + 1)
    uin_ara = box_ara + gbx_ara - isc_ara #union_area

    ovps    = np.where(uin_ara>0, isc_ara/uin_ara, np.zeros(shape=np.shape(uin_ara), dtype=np.float32))
    ovps    = np.reshape(ovps, [box_num, gbx_num])
    return ovps


def mask_overlaps(msks, gmks):
    
    #msks-->(M, H, W), gmks-->(K, H, W)
    msks    = tf.cast(tf.expand_dims(msks, axis=1), dtype=tf.bool)    #(M, 1, H, W)
    gmks    = tf.cast(tf.expand_dims(gmks, axis=0), dtype=tf.bool)    #(1, K, H, W)
    msk_isc = tf.cast(tf.logical_and(msks, gmks), dtype=tf.float32)   #(M, K, H, W)
    msk_uin = tf.cast(tf.logical_or (msks, gmks), dtype=tf.float32)   #(M, K, H, W)
    msk_isc = tf.reduce_sum(msk_isc, axis=[-1, -2])                     #(M, K) 
    msk_uin = tf.reduce_sum(msk_uin, axis=[-1, -2])                     #(M, K) 
    msk_ovp = tf.where(msk_uin>0, msk_isc/msk_uin, tf.zeros(shape=tf.shape(msk_uin), dtype=tf.float32)) #(M, K)
    return msk_ovp


def mask_nms(msks, prbs, nms_pst, nms_max):
    #按得分对msks进行排序
    _, idxs = tf.nn.top_k(prbs, k=tf.shape(msks)[0], sorted=True)
    msks    = tf.gather(msks, idxs)                                   #(M, H, W)
    #计算overlaps
    ovps    = mask_overlaps(msks, msks)                               #(M, M)
    keps    = tf.zeros(shape=[tf.shape(msks)[0]], dtype=tf.int32)     #(M)
    
    def cond(i, ovps, keps):
        c = tf.less(i, tf.shape(ovps)[0])
        return c

    def body(i, ovps, keps):                              #我不知道哪些会keep下来，但我知道哪些需要剔除掉
        
        ovp  = ovps[i]
        idxs = tf.where(ovp>nms_max)                      #超过阈值的msks可能需要剔除
        idxs = tf.cast(idxs, dtype=tf.int32)
        ixxs = tf.where(idxs>i)                           #只能剔除得分比自己低的那些boxs
        idxs = tf.gather_nd(idxs, ixxs)
        ivd  = tf.cast(tf.gather(keps, i), dtype=tf.bool) #已经被剔除掉的box没有资格剔除其余的boxs(invalid==ivd)
        tmp  = tf.zeros(shape=[1, 0], dtype=tf.int32)
        idxs = tf.cond(ivd, lambda: tmp, lambda: idxs)
        keps = tensor_update(keps, idxs, 1)
        return [i+1, ovps, keps]
    #为了保证剔除box的先后关系，令parallel_iterations=1，防止乱序操作
    i = tf.constant(0)
    [i, ovps, keps] = tf.while_loop(cond, body, loop_vars=[i, ovps, keps], \
                                    shape_invariants=[i.get_shape(), ovps.get_shape(), keps.get_shape()], \
                                    parallel_iterations=1, back_prop=False, swap_memory=False)
    ixxs = tf.where(tf.equal(keps, 0))
    idxs = tf.gather_nd(idxs, ixxs)
    idxs = idxs[:nms_pst]
    return idxs
    

def bbox_nms(boxs, prbs, nms_pst, nms_max):
    #按得分对boxs进行排序
    _, idxs = tf.nn.top_k(prbs, k=tf.shape(boxs)[0], sorted=True)
    boxs    = tf.gather(boxs, idxs)
    #计算overlaps
    ovps    = bbox_overlaps(boxs, boxs)
    keps    = tf.zeros(shape=[tf.shape(boxs)[0]], dtype=tf.int32)
    
    def cond(i, ovps, keps):
        c = tf.less(i, tf.shape(ovps)[0])
        return c

    def body(i, ovps, keps):                              #我不知道哪些会keep下来，但我知道哪些需要剔除掉
        
        ovp  = ovps[i]
        idxs = tf.where(ovp>nms_max)                      #超过阈值的boxs可能需要剔除
        idxs = tf.cast(idxs, dtype=tf.int32)
        ixxs = tf.where(idxs>i)                           #只能剔除得分比自己低的那些boxs
        idxs = tf.gather_nd(idxs, ixxs)
        ivd  = tf.cast(tf.gather(keps, i), dtype=tf.bool) #已经被剔除掉的box没有资格剔除其余的boxs(invalid==ivd)
        tmp  = tf.zeros(shape=[1, 0], dtype=tf.int32)
        idxs = tf.cond(ivd, lambda: tmp, lambda: idxs)
        keps = tensor_update(keps, idxs, 1)
        return [i+1, ovps, keps]
    #为了保证剔除box的先后关系，令parallel_iterations=1，防止乱序操作
    i = tf.constant(0)
    [i, ovps, keps] = tf.while_loop(cond, body, loop_vars=[i, ovps, keps], \
                                    shape_invariants=[i.get_shape(), ovps.get_shape(), keps.get_shape()], \
                                    parallel_iterations=1, back_prop=False, swap_memory=False)
    ixxs = tf.where(tf.equal(keps, 0))
    idxs = tf.gather_nd(idxs, ixxs)
    idxs = idxs[:nms_pst]
    return idxs


def bbox_nms2(boxs, prbs, nms_pst, nms_max):
    #按得分对boxs进行排序
    _, idxs = tf.nn.top_k(prbs, k=tf.shape(boxs)[0], sorted=True)
    boxs = tf.gather(boxs, idxs)
    #计算overlaps
    ovps = bbox_overlaps2(boxs, boxs)
    keps = tf.zeros(shape=[tf.shape(boxs)[0]], dtype=tf.int32)
    
    def cond(i, ovps, keps):
        c = tf.less(i, tf.shape(ovps)[0])
        return c

    def body(i, ovps, keps): #我不知道哪些会keep下来，但我知道哪些需要剔除掉
        
        ovp  = ovps[i]
        idxs = tf.where(ovp>nms_max)    #超过阈值的boxs可能需要剔除
        idxs = tf.cast(idxs, dtype=tf.int32)
        ixxs = tf.where(idxs>i)         #只能剔除得分比自己低的那些boxs
        idxs = tf.gather_nd(idxs, ixxs)
        ivd  = tf.cast(tf.gather(keps, i), dtype=tf.bool) #已经被剔除掉的box没有资格剔除其余的boxs(invalid==ivd)
        tmp  = tf.zeros(shape=[1, 0], dtype=tf.int32)
        idxs = tf.cond(ivd, lambda: tmp, lambda: idxs)
        keps = tensor_update(keps, idxs, 1)
        return [i+1, ovps, keps]
    #为了保证剔除box的先后关系，令parallel_iterations=1，防止乱序操作
    i = tf.constant(0)
    [i, ovps, keps] = tf.while_loop(cond, body, loop_vars=[i, ovps, keps], \
                                    shape_invariants=[i.get_shape(), ovps.get_shape(), keps.get_shape()], \
                                    parallel_iterations=1, back_prop=False, swap_memory=False)
    ixxs = tf.where(tf.equal(keps, 0))
    idxs = tf.gather_nd(idxs, ixxs)
    idxs = idxs[:nms_pst]
    return idxs


def bbox_edges(boxs):
    
    box_ymn, box_xmn, box_ymx, box_xmx = tf.split(boxs, 4, axis=-1)
    box_hgt = box_ymx - box_ymn
    box_wdh = box_xmx - box_xmn
    box_edg = tf.concat([box_wdh, box_hgt, box_wdh, box_hgt], axis=-1)
    return box_edg


def bbox_edges2(boxs):
    
    p0, p1, p2, p3 = tf.split(boxs, 4, axis=-1)
    p01 = pin_pin_dst(p0, p1)
    p12 = pin_pin_dst(p1, p2)
    p23 = pin_pin_dst(p2, p3)
    p30 = pin_pin_dst(p3, p0)
    box_edg = tf.concat([p01, p12, p23, p30], axis=-1)    
    return box_edg


def bbox_edges_py2(boxs):
    
    p0, p1, p2, p3 = np.split(boxs, 4, axis=-1)
    p01 = pin_pin_dst_py(p0, p1)
    p12 = pin_pin_dst_py(p1, p2)
    p23 = pin_pin_dst_py(p2, p3)
    p30 = pin_pin_dst_py(p3, p0)
    box_edg = np.concatenate([p01, p12, p23, p30], axis=-1)    
    return box_edg


def bbox_bound2(boxs):
    
    box_ys  = boxs[:, 0::2]
    box_xs  = boxs[:, 1::2]
    box_ymn = tf.reduce_min(box_ys, axis=-1)
    box_xmn = tf.reduce_min(box_xs, axis=-1)
    box_ymx = tf.reduce_max(box_ys, axis=-1)
    box_xmx = tf.reduce_max(box_xs, axis=-1)
    box_bnd = tf.stack([box_ymn, box_xmn, box_ymx, box_xmx], axis=-1)
    return box_bnd


def bbox_bound_py2(boxs):
    
    box_ys  = boxs[:, 0::2]
    box_xs  = boxs[:, 1::2]
    box_ymn = np.amin(box_ys, axis=-1)
    box_xmn = np.amin(box_xs, axis=-1)
    box_ymx = np.amax(box_ys, axis=-1)
    box_xmx = np.amax(box_xs, axis=-1)
    box_bnd = np.stack([box_ymn, box_xmn, box_ymx, box_xmx], axis=-1)
    return box_bnd
