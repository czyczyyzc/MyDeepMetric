import pickle as pickle
import numpy as np
import os
from scipy.misc import imread

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='iso-8859-1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y).astype(np.int64)
        return X, Y

def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)    
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=10000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for classifiers. These are the same steps as we used for the SVM, but
    condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'Mybase/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
        
    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]
    
    #Normalize the data: subtract the mean image
    """
    X_train -= np.mean(X_train, axis = (1,2,3), keepdims = True)
    X_test  -= np.mean(X_test,  axis = (1,2,3), keepdims = True)
    X_val   -= np.mean(X_val,   axis = (1,2,3), keepdims = True)
    """
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val   -= mean_image
    X_test  -= mean_image

    X_train /= 255.0
    X_val   /= 255.0
    X_test  /= 255.0
    """
    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val   = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()
    """
    # Package data into a dictionary
    return {
      'X_train': X_train, 'y_train': y_train,
      'X_val': X_val, 'y_val': y_val,
      'X_test': X_test, 'y_test': y_test,
    }

'''
        imgs_dir_lst = ["/home/ziyechen/VOCdevkit/VOC2007/JPEGImages", 
                        "/home/ziyechen/VOCdevkit/VOC2012/JPEGImages"]
        if train:
            sets_dir_lst = ["/home/ziyechen/VOCdevkit/VOC2007/ImageSets/Main/train.txt",
                            "/home/ziyechen/VOCdevkit/VOC2007/ImageSets/Main/test.txt",
                            "/home/ziyechen/VOCdevkit/VOC2012/ImageSets/Main/trainval.txt"]
        else:
            sets_dir_lst = ["/home/ziyechen/VOCdevkit/VOC2007/ImageSets/Main/val.txt"]
        #anns_dir_lst = ["/home/ziyechen/VOCdevkit/VOC2007/Annotations", 
        #                "/home/ziyechen/VOCdevkit/VOC2012/Annotations"]
        
        imgs_lst = []
        for img_dir in imgs_dir_lst:
            for ext in ['jpg', 'png', 'jpeg', 'JPG']:
                imgs_lst.extend(glob.glob(os.path.join(img_dir, '*.{}'.format(ext))))
        np.random.shuffle(imgs_lst)
        img_num = len(imgs_lst)
        print("The datasets have a total of {:d} images!".format(img_num))
        anns_lst = []
        imgs_kep = []
        for img in imgs_lst:
            img_ext = img.split('.')[-1]
            img_bas = img.split('.')[:-1]
            img_bas = img_bas[0].split('/')
            img_bas[-2] = "Annotations"
            img_bas = ["/".join(img_bas)]
            ann = ".".join(img_bas+["xml"])
            if os.path.exists(ann) == True:
                anns_lst.append(ann)
                imgs_kep.append(img)
        imgs_lst = imgs_kep
        img_num = len(imgs_lst)
        print("The datasets have {:d} valid images!".format(img_num))
        
        gbxs_lst = []
        for i, ann in enumerate(anns_lst):
            if i % 500 == 0:
                print("Converting %d annotations" % (i + 1))
            tree = ET.parse(ann)
            objs = tree.findall('object') #list
            img_siz = tree.find('size')
            img_hgt = float(img_siz.find('height').text)
            img_wdh = float(img_siz.find('width' ).text)
            boxs = []
            clss = []
            for idx, obj in enumerate(objs):
                box     = obj.find('bndbox')
                box_ymn = float(box.find('ymin').text)
                box_xmn = float(box.find('xmin').text)
                box_ymx = float(box.find('ymax').text)
                box_xmx = float(box.find('xmax').text)
                cls = self.cat_nam_to_cls_idx[obj.find('name').text.lower().strip()]
                dif = obj.find('difficult')
                dif = 0 if dif == None else int(dif.text)
                if dif: cls *= -1
                boxs.append([box_ymn, box_xmn, box_ymx, box_xmx])
                clss.append([cls])
            boxs = np.asarray(boxs, dtype=np.float32)
            clss = np.asarray(clss, dtype=np.float32)
            boxs = bbox_clip_py(boxs, [0.0, 0.0, img_hgt-1.0, img_wdh-1.0])
            gbxs = np.concatenate([boxs, clss], axis=-1)
            gbxs_lst.append(gbxs)

        with tf.Graph().as_default(), tf.device('/cpu:0'):
            sha_num = int(img_num/num_per_sha)
            if sha_num == 0:
                sha_num = 1
                num_per_sha = img_num
            else:
                num_per_sha = int(math.ceil(img_num/sha_num))

            for sha_idx in range(sha_num):
                out_nam = 'dianwang_%05d-of-%05d.tfrecord' % (sha_idx, sha_num)
                rcd_nam = os.path.join(rcds_dir, out_nam)

                options = tf.python_io.TFRecordOptions(TFRecordCompressionType.ZLIB)
                with tf.python_io.TFRecordWriter(rcd_nam, options=options) as writer:

                    sta_idx = sha_idx * num_per_sha
                    end_idx = min((sha_idx + 1) * num_per_sha, img_num)
                    for i in range(sta_idx, end_idx):
                        if i % 50 == 0:
                            print("Converting image %d/%d shard %d" % (i + 1, img_num, sha_idx))
                        #读取图像
                        img = imgs_lst[i]
                        img = cv2.imread(img)
                        if type(img) != np.ndarray:
                            print("Failed to find image %s" %(imgs_lst[i]))
                            continue
                        img_hgt, img_wdh = img.shape[0], img.shape[1]
                        if img.size == img_hgt * img_wdh:
                            print ('Gray Image %s' %(imgs_lst[i]))
                            img_tmp = np.empty((img_hgt, img_wdh, 3), dtype=np.uint8)
                            img_tmp[:, :, :] = img[:, :, np.newaxis]
                            img = img_tmp
                        img = img.astype(np.uint8)
                        assert img.size == img_wdh * img_hgt * 3, '%s' % str(i)
                        img = img[:, :, ::-1]
                        #读取标签
                        gbxs = gbxs_lst[i]
                        if len(gbxs) == 0:
                            print("No gt_boxes in this image!")
                            continue
'''