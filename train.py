# Suofei ZHANG, 2017.

import numpy as np
from numpy.matlib import repmat
import tensorflow as tf
# import matplotlib.image as mpimg
import matplotlib.pyplot as plt
# from PIL import Image
import os
import time
import cv2
import scipy.io as sio
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

from siamese import *
from parameters import configParams
import utils

def getOpts(opts):
    print("config opts...")

    opts['validation'] = 0.1
    opts['exemplarSize'] = 127
    opts['instanceSize'] = 255-2*8
    opts['lossRPos'] = 16
    opts['lossRNeg'] = 0
    opts['labelWeight'] = 'balanced'
    opts['numPairs'] = 53200
    opts['frameRange'] = 100
    opts['trainNumEpochs'] = 50
    opts['trainLr1'] = np.logspace(-2, -5, opts['trainNumEpochs'])
    opts['trainWeightDecay'] = 5e-04
    opts['randomSeed'] = 1
    opts['momentum'] = 0.9
    opts['stddev'] = 0.01

    opts['start'] = 0
    opts['expName'] = 'base_s'
    opts['summaryFile'] = './summaryFile_r/'+opts['expName']
    opts['ckptPath'] = './ckpt_r/'+opts['expName']

    opts['modelPath'] = './models/'
    opts['modelName'] = opts['modelPath'] + "model_epoch49.ckpt"
    return opts

def getEig(mat):
    d, v = np.linalg.eig(mat)
    idx = np.argsort(d)
    d.sort()
    d = np.diag(d)
    v = -v;
    v = v[:, idx]

    return d, v

def loadStats(path):
    imgStats = utils.loadImageStats(path)

    if 'z' not in imgStats:
        print("to implement...")
        return
    else:
        rgbMeanZ = np.reshape(imgStats['z']['rgbMean'][0][0], [1, 1, 3])
        rgbMeanX = np.reshape(imgStats['x']['rgbMean'][0][0], [1, 1, 3])
        d, v = getEig(imgStats['z']['rgbCovariance'][0][0])
        rgbVarZ = 0.1*np.dot(np.sqrt(d), v.T)
        d, v = getEig(imgStats['x']['rgbCovariance'][0][0])
        rgbVarX = 0.1*np.dot(np.sqrt(d), v.T)
        return rgbMeanZ, rgbVarZ, rgbMeanX, rgbVarX

def chooseValSet(imdb, opts):
    TRAIN_SET = 1
    VAL_SET = 2

    sizeDataset = len(imdb.id)
    sizeVal = round(opts['validation']*sizeDataset)
    sizeTrain = sizeDataset-sizeVal
    imdb.set = np.zeros([sizeDataset], dtype='uint8')
    imdb.set[:sizeTrain] = TRAIN_SET
    imdb.set[sizeTrain:] = VAL_SET

    imdbInd = {}
    imdbInd['id'] = [i for i in range(0, opts['numPairs'])]
    imdbInd['imageSet'] = np.zeros([opts['numPairs']], dtype='uint8')
    nPairsTrain = round(opts['numPairs']*(1-opts['validation']))
    imdbInd['imageSet'][:nPairsTrain] = TRAIN_SET
    imdbInd['imageSet'][nPairsTrain:] = VAL_SET

    return imdb, imdbInd

def createLogLossLabel(labelSize, rPos, rNeg):
    labelSide = labelSize[0]

    logLossLabel = np.zeros(labelSize, dtype=np.float32,)
    labelOrigin = np.array([np.floor(labelSide/2), np.floor(labelSide/2)])

    for i in range(0, labelSide):
        for j in range(0, labelSide):
            distFromOrigin = np.linalg.norm(np.array([i, j])-labelOrigin)
            if distFromOrigin <= rPos:
                logLossLabel[i, j] = 1
            else:
                if distFromOrigin <= rNeg:
                    logLossLabel[i, j] = 0
                else:
                    logLossLabel[i, j] = -1

    return logLossLabel

def createLabels(labelSize, rPos, rNeg, batchSize):
    half = np.floor(labelSize[0]/2)

    fixedLabel = createLogLossLabel(labelSize, rPos, rNeg)
    instanceWeight = np.ones(fixedLabel.shape)
    idxP = np.where(fixedLabel == 1)
    idxN = np.where(fixedLabel == -1)

    sumP = len(idxP[0])
    sumN = len(idxN[0])

    # instanceWeight = instanceWeight/225.
    instanceWeight[idxP[0], idxP[1]] = 0.5*instanceWeight[idxP[0], idxP[1]]/sumP
    instanceWeight[idxN[0], idxN[1]] = 0.5*instanceWeight[idxN[0], idxN[1]]/sumN

    fixedLabels = np.zeros([batchSize, labelSize[0], labelSize[1], 1], dtype=np.float32)
    instanceWeights = np.zeros([batchSize, labelSize[0], labelSize[1], 1], dtype=np.float32)

    for i in range(batchSize):
        fixedLabels[i, :, :, 0] = fixedLabel
        instanceWeights[i, :, :, 0] = instanceWeight

    return fixedLabels, instanceWeights

def createLabels_G(labelSize, batchSize):
    sz = labelSize[0]
    fixedLabel = gaussian_shaped_labels(sz)
    instanceWeight = np.ones(fixedLabel.shape)
    idxP = np.where(fixedLabel == 1)
    idxN = np.where(fixedLabel == 0)

    sumP = len(idxP[0])
    sumN = len(idxN[0])

    # instanceWeight = instanceWeight/225.
    instanceWeight[idxP[0], idxP[1]] = 5
    instanceWeight[idxN[0], idxN[1]] = 1

    fixedLabels = np.zeros([batchSize, labelSize[0], labelSize[1], 1], dtype=np.float32)
    instanceWeights = np.zeros([batchSize, labelSize[0], labelSize[1], 1], dtype=np.float32)

    for i in range(batchSize):
        fixedLabels[i, :, :, 0] = fixedLabel
        instanceWeights[i, :, :, 0] = instanceWeight

    return fixedLabels, instanceWeights

def gaussian_shaped_labels(sz):
    sigma = [0.7,0.7]
    v = np.linspace(1, sz, sz) - np.ceil(sz / 2)
    xs, ys = np.meshgrid(v, v)
    alpha=0.25
    labels = np.exp(-alpha * ((xs ** 2) / (sigma[0] ** 2) + (ys ** 2) / (sigma[1] ** 2)))
    return labels

def precisionAuc(positions, groundTruth, radius, nStep):
    thres = np.linspace(0, radius, nStep)

    errs = np.zeros([nStep], dtype=np.float32)

    distances = np.sqrt(np.power(positions[:, 0]-groundTruth[:, 0], 2)+np.power(positions[:, 1]-groundTruth[:, 1], 2))
    distances[np.where(np.isnan(distances))] = []

    for p in range(0, nStep):
        errs[p] = np.shape(np.where(distances > thres[p]))[-1]

    score = np.trapz(errs)

    return score

def centerThrErr(score, labels, oldRes, m):
    radiusInpix = 50
    totalStride = 8
    nStep = 100
    batchSize = score.shape[0]
    posMask = np.where(labels > 0)
    numPos = posMask[0].shape[-1]

    responses = np.squeeze(score[posMask, :, :, :], axis=(0,))
    half = np.floor(score.shape[1]/2)
    centerLabel = repmat([half, half], numPos, 1)
    positions = np.zeros([numPos, 2], dtype=np.float32)

    for b in range(0, numPos):
        sc = np.squeeze(responses[b, :, :, 0])
        r = np.where(sc == np.max(sc))
        positions[b, :] = [r[0][0], r[1][0]]

    res = precisionAuc(positions, centerLabel, radiusInpix/totalStride, nStep)
    res = (oldRes*m+res)/(m+batchSize)
    return res

def centerScore(x):
    m1, m2 = x.shape
    c1 = (m1+1)/2-1
    c2 = (m2+1)/2-1
    v = x[int(c1), int(c2)]

    return v

def maxScoreErr(x, yGt, oldRes, m):
    b, m1, m2, k = x.shape

    errs = np.zeros([b], dtype=np.float32)

    for i in range(0, b):
        score = np.squeeze(x[i, :, :, 0])

        if yGt[i] > 0:
            errs[i] = centerScore(score)
        else:
            errs[i] = -np.max(score)

    res = len(np.where(errs <= 0)[0])
    res = (oldRes*m+res)/(m+b)

    return res

def choosePosPair(imdb, idx, frameRange):
    validTrackIds = np.where(imdb.valid_trackids[:, idx] > 1)[0]
    randTrackidZ = np.random.permutation(validTrackIds)[0]

    frames = imdb.valid_per_trackid[idx][randTrackidZ]
    randZ = np.random.permutation(frames)[0]
    randZPos = frames.index(randZ)

    possibleX = frames
    possibleX = possibleX[:min(len(frames), randZPos+frameRange)]
    possibleX = possibleX[max(randZPos-frameRange, 0):]
    possibleX.remove(randZ)

    randX = np.random.permutation(possibleX)[0]

    z = imdb.objects[idx]
    x = imdb.objects[idx]

    return z, randZ, x, randX

def acquireAugment(im, imageSize, rgbVar, augOpts):
    if not isinstance(imageSize, list): #len(imageSize) == 1:
        imageSize = [imageSize, imageSize]

    if not isinstance(augOpts['maxTranslate'], list): #len(augOpts['maxTranslate']) == 1:
        augOpts['maxTranslate'] = [augOpts['maxTranslate'], augOpts['maxTranslate']]

    if im.shape[-1] == 1:
        imt = np.zeros([im.shape[0], im.shape[0], 3])
        imt[:, :, 0] = imt[:, :, 1] = imt[:, :, 2] = im
    else:
        imt = im

    h, w, _ = imt.shape
    cx = (w+1)/2-1
    cy = (h+1)/2-1

    if augOpts['stretch']:
        scale = np.squeeze((1+augOpts['maxStretch']*(-1+2*np.random.rand(2, 1))))
        test = np.multiply(imageSize, scale)
        sz = np.around(np.min([test, [h, w]], 0))
    else:
        sz = imageSize

    if augOpts['translate']:
        if not isinstance(augOpts['maxTranslate'], list):
            dx = np.random.randint(1, w-sz[1]+1, 1)
            dy = np.random.randint(1, h-sz(0)+1, 1)
        else:
            mx = min(augOpts['maxTranslate'][1], np.floor((w-sz[1])/2))
            my = min(augOpts['maxTranslate'][0], np.floor((h-sz[0])/2))
            dx = cx-(sz[1]-1)/2+np.random.randint(-mx, mx+1, 1)
            dy = cy-(sz[0]-1)/2+np.random.randint(-my, my+1, 1)
    else:
        dx = cx-(sz[1]-1)/2
        dy = cy-(sz[0]-1)/2

    sx = np.around(np.linspace(dx, dx+sz[1]-1, imageSize[1]))
    sy = np.around(np.linspace(dy, dy+sz[0]-1, imageSize[0]))
    sx = sx.astype(int).tolist()
    sy = sy.astype(int).tolist()

    imo = imt[sy, :, :]
    imo = imo[:, sy, :]
    if augOpts['color']:
        offset = np.dot(rgbVar, np.random.randn(3, 1))
        imo[:, :, 0] = imo[:, :, 0]-offset[0]
        imo[:, :, 1] = imo[:, :, 1]-offset[1]
        imo[:, :, 2] = imo[:, :, 2]-offset[2]

    return imo

def vidGetRandBatch(imdbInd, imdb, batch, opts):
    TRAIN_SET = 1
    VAL_SET = 2

    batchSet = imdbInd['imageSet'][batch[0]]
    assert all(batchSet == imdbInd['imageSet'][batch])
    batchSize = len(batch)
    pairTypesRgb = 1
    dataDir = opts['crops_train']
    imdb.set = imdb.set[0:4366]
    idsSet = np.where(imdb.set == batchSet)[0]
    rndVideos = np.random.permutation(idsSet)[:batchSize]
    idsPairs = rndVideos

    imoutZ1 = np.zeros([batchSize, opts['exemplarSize'], opts['exemplarSize'], 3], dtype=np.float32)
    imoutX = np.zeros([batchSize, opts['instanceSize'], opts['instanceSize'], 3], dtype=np.float32)
    imoutX1 = np.zeros([batchSize, opts['instanceSize'], opts['instanceSize'], 3], dtype=np.float32)

    objectsZ = []
    objectsX = []
    idxZ = []
    idxX = []
    cropsZ1Str = []
    cropsXStr = []
    cropsX1Str = []
    for i in range(0, batchSize):
        z, randZ, x, randX = choosePosPair(imdb, idsPairs[i], opts['frameRange'])
        objectsZ.append(z)
        idxZ.append(randZ)
        objectsX.append(x)
        idxX.append(randX)

        z1Str = dataDir + z.frame_path[randZ]
        z1Str = z1Str.replace(".JPEG", "")+".%02d.crop.z.jpg" % z.track_id[randZ]
        xStr = dataDir+x.frame_path[randX]
        xStr = xStr.replace(".JPEG", "")+".%02d.crop.x.jpg" % x.track_id[randX]
        if randZ<randX:
            randZt = np.random.randint(randZ, randX, 1)[0]
        else:
            randZt = np.random.randint(randX, randZ, 1)[0]
        x1Str = dataDir + z.frame_path[randZt]
        x1Str = x1Str.replace(".JPEG", "") + ".%02d.crop.x.jpg" % z.track_id[randZt]
        cropsZ1Str.append(z1Str)
        cropsXStr.append(xStr)
        cropsX1Str.append(x1Str)

    augOpts = {}
    if batchSet == TRAIN_SET:
        augOpts['translate'] = True
        augOpts['maxTranslate'] = 4
        augOpts['stretch'] = True
        augOpts['maxStretch'] = 0.05
        augOpts['color'] = True
        augOpts['grayscale'] = 0
    else:
        augOpts['translate'] = False
        augOpts['maxTranslate'] = 0
        augOpts['stretch'] = False
        augOpts['maxStretch'] = 0
        augOpts['color'] = False

    for i in range(batchSize):
        imz1 = cv2.imread(cropsZ1Str[i])
        imx = cv2.imread(cropsXStr[i])
        imx1 = cv2.imread(cropsX1Str[i])

        augZ1 = acquireAugment(imz1, opts['exemplarSize'], opts['rgbVarZ'], augOpts)
        augX = acquireAugment(imx, opts['instanceSize'], opts['rgbVarX'], augOpts)
        augX1 = acquireAugment(imx1, opts['instanceSize'], opts['rgbVarX'], augOpts)

        imoutZ1[i, :, :, :] = augZ1
        imoutX[i, :, :, :] = augX
        imoutX1[i, :, :, :] = augX1

    return imoutZ1, imoutX, imoutX1

def main(_):
    opts = configParams()
    opts = getOpts(opts)
    # opts['trainBatchSize'] = 4
    # curation.py should be executed once before
    imdb = utils.loadImdbFromPkl(opts['curation_path'], opts['crops_train'])
    rgbMeanZ, rgbVarZ, rgbMeanX, rgbVarX = loadStats(opts['curation_path'])
    imdb, imdbInd = chooseValSet(imdb, opts)

    # random seed should be fixed here
    np.random.seed(opts['randomSeed'])
    exemplarOp1 = tf.placeholder(tf.float32, [opts['trainBatchSize'], opts['exemplarSize'], opts['exemplarSize'], 3])
    instanceOpt = tf.placeholder(tf.float32, [opts['trainBatchSize'], opts['instanceSize'], opts['instanceSize'], 3])
    instanceOp = tf.placeholder(tf.float32, [opts['trainBatchSize'], opts['instanceSize'], opts['instanceSize'], 3])
    lr = tf.placeholder(tf.float32, shape=())

    sn = SiameseNet()

    # with tf.variable_scope('model') as scope:
    scoreOp1, l2Op = sn.buildTrainNetwork(exemplarOp1, instanceOp, opts)

    labels = np.ones([8], dtype=np.float32)
    respSz = int(scoreOp1.get_shape()[1])
    respSz = [respSz, respSz]
    respStride = 8  # calculated from stride of convolutional layers and pooling layers
    fixedLabel, instanceWeight = createLabels(respSz, opts['lossRPos']/respStride, opts['lossRNeg']/respStride, opts['trainBatchSize'])
    # fixedLabel, instanceWeight = createLabels_G(respSz, opts['trainBatchSize'])
    # sio.savemat('labels.mat', {'fixedLabel': fixedLabel, 'instanceWeight': instanceWeight})
    opts['rgbMeanZ'] = rgbMeanZ
    opts['rgbVarZ'] = rgbVarZ
    opts['rgbMeanX'] = rgbMeanX
    opts['rgbVarX'] = rgbVarX

    instanceWeightOp = tf.constant(instanceWeight, dtype=tf.float32)
    yOp = tf.placeholder(tf.float32, fixedLabel.shape)
    with tf.name_scope("logistic_loss"):
        lossOp1 = sn.loss(scoreOp1, yOp, instanceWeightOp)

    grad1 = tf.gradients(lossOp1, l2Op)

    scoreOp2 = sn.buildTrainNetwork1(l2Op, grad1[0], instanceOpt, opts)
    lossOp2 = sn.loss(scoreOp2, yOp, instanceWeightOp)
    tf.summary.scalar('loss', lossOp2)
    tf.summary.image('res1', scoreOp1)
    tf.summary.image('res2', scoreOp2)
    errDispVar = tf.Variable(0, 'tbVarErrDisp', dtype=tf.float32)
    errDispPH = tf.placeholder(tf.float32, shape=())
    errDispSummary = errDispVar.assign(errDispPH)
    tf.summary.scalar("errDisp", errDispSummary)
    errMaxVar = tf.Variable(0, 'tbVarErrMax', dtype=tf.float32)
    errMaxPH = tf.placeholder(tf.float32, shape=())
    errMaxSummary = errMaxVar.assign(errMaxPH)
    tf.summary.scalar("errMax", errMaxSummary)

    '''train and save part network'''
    scala1_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='siamese/scala1/')
    scala2_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='siamese/scala2/')
    scala3_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='siamese/scala3/')
    scala4_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='siamese/scala4/')
    scala5_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='siamese/scala5/')
    siamese_original_list = scala1_list + scala2_list + scala3_list + scala4_list + \
                            scala5_list 
    saver_restore = tf.train.Saver(var_list=siamese_original_list)

    all_vars = tf.trainable_variables()
    optimizer4 = tf.train.MomentumOptimizer(learning_rate=lr, momentum=opts['momentum'])
    var_list4 = [var for var in all_vars if ('_z' in var.name)]
    grads4 = optimizer4.compute_gradients(lossOp2, var_list=var_list4)
    gradsOp4 = optimizer4.apply_gradients(grads_and_vars=grads4)

    batchNormUpdates = tf.get_collection(UPDATE_OPS_COLLECTION)
    batchNormUpdates_res = []
    for var in batchNormUpdates:
        if '_z' in var.name:
            batchNormUpdates_res.append(var)
    batchNormUpdatesOp_res = tf.group(*batchNormUpdates_res)

    trainOp = tf.group(gradsOp4, batchNormUpdatesOp_res)

    summaryOp = tf.summary.merge_all()
    writer = tf.summary.FileWriter(opts['summaryFile'])
    saver = tf.train.Saver(max_to_keep=40)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    saver_restore.restore(sess, './ckpt/base_256/model_epoch45.ckpt')
    writer.add_graph(sess.graph)

    step = 0
    epochStep = opts['numPairs']/opts['trainBatchSize']

    for i in range(opts['start'], opts['trainNumEpochs']):
        trainSamples = opts['numPairs'] * (1 - opts['validation'])
        sampleNum = 0
        errDisp = 0
        errMax = 0
        sampleIdx = np.random.permutation(int(trainSamples))
        j=0
        while sampleNum < trainSamples-9:
            j+=1
            t0 = time.clock()
            batch = sampleIdx[sampleNum:sampleNum+opts['trainBatchSize']]

            imoutZ1, imoutX, imoutX1 = vidGetRandBatch(imdbInd, imdb, batch, opts)
            score = sess.run(scoreOp2, feed_dict={exemplarOp1: imoutZ1,
                                                   instanceOp: imoutX,
                                                  instanceOpt: imoutX,
                                                   yOp: fixedLabel,
                                                   lr: opts['trainLr1'][i]})#opts['trainLr1'][i]

            errDisp = centerThrErr(score, labels, errDisp, sampleNum)
            errMax = maxScoreErr(score, labels, errMax, sampleNum)

            sess.run(trainOp, feed_dict={exemplarOp1: imoutZ1,
                                           instanceOp: imoutX,
                                           instanceOpt: imoutX,
                                           yOp: fixedLabel,
                                           lr: opts['trainLr1'][i]})


            _, _, s, l2, l1, g1 = sess.run([errDispSummary, errMaxSummary, summaryOp, lossOp2, lossOp1, grad1], feed_dict={errDispPH: errDisp,
                                                                                      errMaxPH: errMax,
                                                                                      exemplarOp1: imoutZ1,
                                                                                      instanceOp: imoutX,
                                                                                      instanceOpt: imoutX,
                                                                                      yOp: fixedLabel,
                                                                                      lr: opts['trainLr1'][i]})
            writer.add_summary(s, step)

            sampleNum = sampleNum + opts['trainBatchSize']
            step = step+1
            print('the %d epoch %d round training is finished in %f, errDisp: %f, loss1: %f, loss2: %f' % (
                i, np.mod(step, epochStep), time.clock() - t0, errDisp, l1, l2))
        if not os.path.exists(opts['ckptPath']):
            os.mkdir(opts['ckptPath'])

        ckptName = os.path.join(opts['ckptPath'], 'model_epoch'+str(i)+'.ckpt')
        saveRes = saver.save(sess, ckptName)

        valSamples = opts['numPairs']*opts['validation']
        sampleNum = 0
        errDisp = 0
        errMax = 0
        sampleIdx = np.random.permutation(int(valSamples))+int(trainSamples)
        while sampleNum < valSamples-9:
            t0 = time.clock()
            batch = sampleIdx[sampleNum:sampleNum + opts['trainBatchSize']]
            imoutZ1, imoutX, imoutX1 = vidGetRandBatch(imdbInd, imdb, batch, opts)
            score = sess.run(scoreOp2, feed_dict={exemplarOp1: imoutZ1,
                                                   instanceOp: imoutX,
                                                  instanceOpt: imoutX,
                                                   yOp: fixedLabel,
                                                   lr: opts['trainLr1'][i]})

            errDisp = centerThrErr(score, labels, errDisp, sampleNum)
            errMax = maxScoreErr(score, labels, errMax, sampleNum)

            _, _, s, l2, l1 = sess.run([errDispSummary, errMaxSummary, summaryOp, lossOp2, lossOp1], feed_dict={errDispPH: errDisp,
                                                                                      errMaxPH: errMax,
                                                                                      exemplarOp1: imoutZ1,
                                                                                      instanceOp: imoutX,
                                                                                      instanceOpt: imoutX,
                                                                                      yOp: fixedLabel,
                                                                                      lr: opts['trainLr1'][i]})
            writer.add_summary(s, step)
            sampleNum = sampleNum + opts['trainBatchSize']
            step = step + 1
            print('the %d epoch %d round training is finished in %f, errDisp: %f, loss1: %f, loss2: %f' % (
                i, np.mod(step, epochStep), time.clock() - t0, errDisp, l1, l2))
    return

if __name__ == '__main__':
    tf.app.run()


