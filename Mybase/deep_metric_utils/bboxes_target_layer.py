import time
import datetime
import numpy as np
import tensorflow as tf
from .bbox import *
from collections import defaultdict
#from shapely.geometry import Polygon

class BboxesTargetLayer(object):
    
    def __init__(self):
        
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, np.round((1.00 - .0) / .01) + 1, endpoint=True)
        self.maxDets = [1, 10, 100]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'small', 'medium', 'large']
        self.useCats = 1
        self.iouType = 'bbox'
        
    def generate_boxs_pre(self, boxs, gbxs):
        
        boxs, box_clss, box_prbs, box_imxs = boxs
        gbxs, gbx_clss, gbx_imxs = gbxs
        gbx_clss_tmp = np.absolute(gbx_clss).astype(np.int32, copy=False)
        
        self.imgIds = list(set(box_imxs))
        self.catIds = list(set(list(box_clss) + list(gbx_clss_tmp)))
        
        box_aras = bbox_area_py(boxs)[:, 0]
        gbx_aras = bbox_area_py(gbxs)[:, 0]

        gts = []
        for i in range(len(gbx_imxs)):
            gt                = {}
            gt['id']          = i + 1
            gt['bbox']        = gbxs[i]
            gt['area']        = gbx_aras[i]
            gt['iscrowd']     = 1 if gbx_clss[i] <= 0 else 0
            gt['ignore']      = gt['iscrowd']
            gt['image_id']    = gbx_imxs[i]
            gt['category_id'] = gbx_clss_tmp[i]
            gts.append(gt)
        
        dts = []    
        for i in range(len(box_imxs)):
            dt                = {}
            dt['id']          = i + 1
            dt['bbox']        = boxs[i]
            dt['area']        = box_aras[i]
            dt['score']       = box_prbs[i]
            dt['image_id']    = box_imxs[i]
            dt['category_id'] = box_clss[i]
            dts.append(dt)

        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        for gt in gts:
            self._gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results
        self.eval     = {}                  # accumulated evaluation results
        
        self.evaluate()
        self.accumulate()
        self.summarize()
        return self.stats
        
        
    def evaluate(self):
        
        catIds = self.catIds if self.useCats else [-1]
        self.ious = {(imgId, catId): self.computeIoU(imgId, catId) \
                     for imgId in self.imgIds
                     for catId in catIds}

        maxDet = self.maxDets[-1]
        self.evalImgs = [self.evaluateImg(imgId, catId, areaRng, maxDet)
                         for catId in catIds
                         for areaRng in self.areaRng
                         for imgId in self.imgIds
                        ]
    
    
    def computeIoU(self, imgId, catId):

        if self.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in self.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in self.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > self.maxDets[-1]:
            dt=dt[0:self.maxDets[-1]]

        if self.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif self.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        #iscrowd = [int(o['iscrowd']) for o in gt]
        #ious = maskUtils.iou(d,g,iscrowd)
        g = np.asarray(g, dtype=np.float32)
        d = np.asarray(d, dtype=np.float32)
        ious = bbox_overlaps_py(d, g)
        return ious
    
    
    def evaluateImg(self, imgId, catId, aRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        if self.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in self.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in self.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return None

        for g in gt:
            if g['ignore'] or (g['area']<aRng[0] or g['area']>aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score']  for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious(重新把iou按照gt的顺序排序)
        ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

        T = len(self.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm  = np.zeros((T,G)) #在不同的IOU阈值下，有哪些gt匹配上了dt
        dtm  = np.zeros((T,D)) #在不同的IOU阈值下，有哪些dt匹配上了gt
        gtIg = np.array([g['_ignore'] for g in gt]) #在所选面积范围下，有哪些gt是不用考虑的
        dtIg = np.zeros((T,D)) #在不同的IOU阈值下，有哪些dt被忽略掉了
        if not len(ious)==0:
            for tind, t in enumerate(self.iouThrs): #选一个IOU阈值进行评估
                for dind, d in enumerate(dt): #选一个dt，用它去匹配所有的gt
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t,1-1e-10])
                    m   = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind,gind]>0 and not iscrowd[gind]: #如果某gt已经匹配上一个dt并且该gt有效，则忽略该gt
                            continue                               #无效的gt可以由多个dt与之匹配，并且这些dt的忽略与否都由该gt决定
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m>-1 and gtIg[m]==0 and gtIg[gind]==1:  #如果dt已经匹配到一个有效的gt，且碰到了一个无效的gt，则停止匹配
                            break                                  #就算某dt匹配到了一个gt，该gt不一定有效，需要继续匹配
                        # continue to next gt unless better match made
                        if ious[dind,gind] < iou: #如果dt与某gt之间的IOU小于预设值或之前值，则忽略该gt
                            continue
                        # if match successful and best so far, store appropriately
                        iou=ious[dind,gind] #dt与某gt匹配成功(该gt可能是有效的也可能是无效的)
                        m=gind
                    # if match made store id of match for both dt and gt
                    if m ==-1: #如果dt没有匹配上任何一个gt，则该dt什么记号都没留下
                        continue
                    dtIg[tind,dind] = gtIg[m] #dt是否被忽略，由它所匹配的gt决定
                    dtm[tind,dind]  = gt[m]['id']
                    gtm[tind,m]     = d['id']
        # set unmatched detections outside of area range to ignore
        a = np.array([d['area']<aRng[0] or d['area']>aRng[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm==0, np.repeat(a,T,0))) #用dt的面积是否在范围内确定未成功匹配的dt(dtm==0)是否需要忽略
        # store results for given image and category            #这对precision的计算是很重要的，如果无效的dt被剔除了，则precision会显著提高
        return {
                'image_id':     imgId,
                'category_id':  catId,
                'aRng':         aRng,
                'maxDet':       maxDet,
                'dtIds':        [d['id'] for d in dt],
                'gtIds':        [g['id'] for g in gt],
                'dtMatches':    dtm,
                'gtMatches':    gtm,
                'dtScores':     [d['score'] for d in dt],
                'gtIgnore':     gtIg,
                'dtIgnore':     dtIg,
            }
    
    
    def accumulate(self):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param : None
        :return: None
        '''
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')

        self.catIds = self.catIds if self.useCats == 1 else [-1]
        T           = len(self.iouThrs)
        R           = len(self.recThrs)
        K           = len(self.catIds) if self.useCats else 1
        A           = len(self.areaRng)
        M           = len(self.maxDets)
        precision   = -np.ones((T,R,K,A,M)) # -1 for the precision of absent categories
        recall      = -np.ones((T,K,A,M))

        catIds = self.catIds if self.useCats else [-1]
        I0 = len(self.imgIds)
        A0 = len(self.areaRng)
        # retrieve E at each category, area range, and max number of detections
        for k, cat in enumerate(catIds): #对每一类的结果进行统计
            Nk = k*A0*I0
            for a, ara in enumerate(self.areaRng):
                Na = a*I0
                for m, maxDet in enumerate(self.maxDets):
                    E = [self.evalImgs[Nk + Na + i] for i, imx in enumerate(self.imgIds)]
                    E = [e for e in E if not e is None] #只有当既没有gt也没有dt时，结果才会为None
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E]) #(maxDet*img_num)

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')

                    dtm  = np.concatenate([e['dtMatches'][:,0:maxDet] for e in E], axis=1)[:,inds] #(T, maxDet*img_num)
                    dtIg = np.concatenate([e['dtIgnore' ][:,0:maxDet] for e in E], axis=1)[:,inds] #(T, maxDet*img_num)
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg==0) #有效的gt有多少个
                    if npig == 0:
                        continue
                    tps = np.logical_and(               dtm,  np.logical_not(dtIg)) #对有效的dt中，匹配成功的dt进行标记, true  positive
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg)) #对有效的dt中，匹配失败的dt进行标记, fasle positive

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)): #对每一个IOU阈值进行登记
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig                   #预测成功的数目 / 真实有效的数目
                        pr = tp / (fp+tp+np.spacing(1))  #预测成功的数目 / 总共预测的数目
                        q  = np.zeros((R,))

                        if nd:
                            recall[t,k,a,m] = rc[-1]
                        else:
                            recall[t,k,a,m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist(); q = q.tolist()

                        for i in range(nd-1, 0, -1):
                            if pr[i] > pr[i-1]:
                                pr[i-1] = pr[i]

                        inds = np.searchsorted(rc, self.recThrs, side='left')
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                        except:
                            pass
                        precision[t,:,k,a,m] = np.array(q)
        self.eval = {
            #'params': p,
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall':    recall,
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))
        
    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(self.iouThrs[0], self.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(self.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(self.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == self.iouThrs)[0]
                    s = s[t]
                s = s[:,:,:,aind,mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == self.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind,mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s
        
        def _summarizeDets():
            stats = np.zeros((12,))
            stats[0]  = _summarize(1)
            stats[1]  = _summarize(1, iouThr=.5,        maxDets=self.maxDets[2])
            stats[2]  = _summarize(1, iouThr=.75,       maxDets=self.maxDets[2])
            stats[3]  = _summarize(1, areaRng='small',  maxDets=self.maxDets[2])
            stats[4]  = _summarize(1, areaRng='medium', maxDets=self.maxDets[2])
            stats[5]  = _summarize(1, areaRng='large',  maxDets=self.maxDets[2])
            stats[6]  = _summarize(0,                   maxDets=self.maxDets[0])
            stats[7]  = _summarize(0,                   maxDets=self.maxDets[1])
            stats[8]  = _summarize(0,                   maxDets=self.maxDets[2])
            stats[9]  = _summarize(0, areaRng='small',  maxDets=self.maxDets[2])
            stats[10] = _summarize(0, areaRng='medium', maxDets=self.maxDets[2])
            stats[11] = _summarize(0, areaRng='large',  maxDets=self.maxDets[2])
            return stats
        
        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        #elif iouType == 'keypoints':
        #    summarize = _summarizeKps
        self.stats = summarize()

        
    def __str__(self):
        self.summarize()
        