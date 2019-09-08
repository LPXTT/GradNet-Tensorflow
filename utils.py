# Suofei ZHANG, 2017.

import numpy as np
import pickle
import scipy.io as sio

# there maybe more than one target in the same video, in this case, the field 'objects' has an ImdbObjects entry for all targets in the video
# the field 'valid_trackids' and 'valid_per_trackid' also have more than one entry at current column
# the value of Imdb.id starts from 1, other indices start from 0
class Imdb:
    id = None
    nframes = None
    path = None

    n_valid_objects = None
    valid_trackids = None

    total_valid_objects = np.uint32(0)

    objects = []
    valid_per_trackid = []

    def __init__(self, numVideos=None, maxTrackIds=None, _id=None, _nframes=None, _path=None):
        if (not numVideos is None) and (not _id is None):
            self.id = _id
            self.nframes = _nframes
            self.path = _path

            self.n_valid_objects = np.zeros(numVideos, dtype=np.uint32)
            self.valid_trackids = np.zeros([maxTrackIds, numVideos], dtype=np.uint32)

# an ImdbObjects corresponds to a video containing frames
# there maybe more than one target in the same video, in this case, the field of 'track_id' should be like [0, 1, 2, 0, 1, 2,...], each entry corresponds to a row in other fields
class ImdbObjects:
    track_id = None
    oClass = None
    frames_sz = None
    extent = None
    valid = None
    frame_path = None

    def __init__(self, _track_id = None, _oclass = None, _frame_sz = None, _extent = None, _valid = None, _frame_path = None):
        self.track_id = np.array(_track_id)
        self.oClass = np.array(_oclass)
        self.frames_sz = np.array(_frame_sz)
        self.extent = np.array(_extent)
        self.valid = np.array(_valid)
        self.frame_path = _frame_path

def deleteFromImdb(imdb, idx):
    imdb.id = np.delete(imdb.id, idx)
    numDelete = np.sum(imdb.n_valid_objects[idx])
    imdb.n_valid_objects = np.delete(imdb.n_valid_objects, idx)
    imdb.nframes = np.delete(imdb.nframes, idx)
    imdb.total_valid_objects -= numDelete
    imdb.valid_trackids = np.delete(imdb.valid_trackids, idx, 1)

    upper = len(imdb.objects)

    for i in range(len(idx)-1, -1, -1):
        if idx[i] < upper:
            del imdb.objects[idx[i]]
            del imdb.path[idx[i]]
            del imdb.valid_per_trackid[idx[i]]

    return imdb

# the functions is a python implementation of the function imdb_video = vid_setup_data(root) in original siamese-fc
# it collects all information from imagenet vid, cooking the imdb data
def vidSetupData(curation_path, root, crops_train):
    rootPath = root+"Data/VID/train/"
    MAX_TRACKIDS = 50;
    framesIdPath = curation_path+"vid_id_frames.txt"

    videoPaths = []
    videoIds = []
    videoNFrames = []

    with open(framesIdPath, 'r') as vidFiles:
        while True:
            line = vidFiles.readline()
            if not line:
                break

            videoPath, videoId, videoNFrame = [str for str in line.split(' ')]
            videoPaths.append(videoPath)
            videoIds.append(np.uint32(videoId))
            videoNFrames.append(np.uint32(videoNFrame))

        vidFiles.close()
        videoIds = np.array(videoIds)
        videoNFrames = np.array(videoNFrames)

    nVideos = videoIds.shape[0]
    # nVideos = 4367
    imdb = Imdb(nVideos, MAX_TRACKIDS, videoIds, videoNFrames, videoPaths)

    for i in range(0, nVideos):      #
        print("Objects from video %d" % i + "/%d" % nVideos)

        with open(rootPath+imdb.path[i]+".txt", 'r') as vidFile:
            trackIds = []
            oClasses = []
            framesSize = []
            extents = []
            valids = []
            framePathes = []
            validPerTrackids = []
            targetIdx = 0       #targetIdx here corresponds to l in the Matlab version, however targetIdx starts from 0 rather than 1
            validPerTrackidPath = ""

            while True:
                line = vidFile.readline()
                if (not line) or (len(line) < 1):
                    break

                trackId, oClass, frameW, frameH, oXMins, oYMinx, oWs, ohS, framePath = [str for str in line.split(',')]

                trackId = np.uint8(trackId)
                trackIds.append(trackId)
                oClasses.append(np.uint8(oClass))
                frameW = np.uint16(frameW)
                frameH = np.uint16(frameH)
                framesSize.append([frameW, frameH])
                oXMins = np.int16(oXMins)
                oYMinx = np.int16(oYMinx)
                oWs = np.int16(oWs)
                ohS = np.int16(ohS)
                extents.append([oXMins, oYMinx, oWs, ohS])
                valids.append(np.bool(1))
                _, framePath = [str for str in framePath.split("train/")]
                framePath, _ = [str for str in framePath.split("\n")]
                framePathes.append(framePath)

                if True:        #if valids[length(valids)-1] == True
                    imdb.n_valid_objects[i] += 1
                    imdb.valid_trackids[trackId, i] += 1
                    while trackId+1 > len(validPerTrackids):
                        tmp = []
                        validPerTrackids.append(tmp)

                    validPerTrackids[trackId].append(np.uint16(targetIdx))

                targetIdx += 1

            imdbObjects = ImdbObjects(trackIds, oClasses, framesSize, extents, valids, framePathes)
            imdb.objects.append(imdbObjects)
            imdb.valid_per_trackid.append(validPerTrackids)
            imdb.total_valid_objects += imdb.n_valid_objects[i]
            print(imdb.valid_trackids[:, i])

            vidFile.close()
            print("Found %d" % imdb.n_valid_objects[i] + " valid objects in %d" % imdb.nframes[i] + " frames")

    toDelete = np.where(imdb.n_valid_objects < 2)[0]
    imdb = deleteFromImdb(imdb, toDelete)
    toDelete = np.unique(np.where(imdb.valid_trackids == 1)[1])
    imdb = deleteFromImdb(imdb, toDelete)
    saveImdbToPkl(imdb, curation_path, crops_train)
    return imdb

def saveImdbToPkl(imdb, curation_path, crops_train):
    with open(curation_path+"imdb.pkl", 'wb') as imdbFile:
        pickle.dump(imdb, imdbFile)
        imdbFile.close()

    for i in range(0, imdb.id.shape[0]):
        with open(crops_train+imdb.path[i]+"/object.pkl", 'wb') as objFile:
            pickle.dump(imdb.objects[i], objFile)
            objFile.close()

        for j in range(0, len(imdb.valid_per_trackid[i])):
            with open(crops_train+imdb.path[i]+"/trackid_%d" % j+".pkl", 'wb') as idFile:
                pickle.dump(imdb.valid_per_trackid[i][j], idFile)
                idFile.close()


def loadImdbFromPkl(curation_path, crops_train):
    imdb = Imdb()
    with open(curation_path+"imdb.pkl", 'rb') as imdbFile:
        imdb = pickle.load(imdbFile)

    imdb.objects = []
    for i in range(0, imdb.id.shape[0]):
        imdbObject = ImdbObjects()
        with open(crops_train+imdb.path[i]+"/object.pkl", 'rb') as objFile:
            imdbObject = pickle.load(objFile)
            imdb.objects.append(imdbObject)

        trackIdNum = np.where(imdb.valid_trackids[:, i] > 0)[0][-1]
        validPerTrackids = []

        for j in range(0, trackIdNum+1):
            with open(crops_train+imdb.path[i]+"/trackid_%d" % j+".pkl", 'rb') as idFile:
                validPerTrackid = pickle.load(idFile)
                validPerTrackids.append(validPerTrackid)
                idFile.close()

        imdb.valid_per_trackid.append(validPerTrackids)

    return imdb

def loadImageStatsFromMat(path):
    imgStats = {}# ImgStats()

    imgStatsMat = sio.loadmat(path+"x.mat")
    imgStatsMat = imgStatsMat['x']
    imgStats['x'] = {}
    imgStats['x']['averageImage'] = imgStatsMat['averageImage']
    imgStats['x']['rgbm1'] = imgStatsMat['rgbm1']
    imgStats['x']['rgbMean'] = imgStatsMat['rgbMean']
    imgStats['x']['rgbCovariance'] = imgStatsMat['rgbCovariance']

    imgStatsMat = sio.loadmat(path + "z.mat")
    imgStatsMat = imgStatsMat['z']
    imgStats['z'] = {}
    imgStats['z']['averageImage'] = imgStatsMat['averageImage']
    imgStats['z']['rgbm1'] = imgStatsMat['rgbm1']
    imgStats['z']['rgbm2'] = imgStatsMat['rgbm2']
    imgStats['z']['rgbMean'] = imgStatsMat['rgbMean']
    imgStats['z']['rgbCovariance'] = imgStatsMat['rgbCovariance']

    with open(path + "imageStats.pkl", 'wb') as imgStatsFile:
        pickle.dump(imgStats, imgStatsFile)
        imgStatsFile.close()

    return imgStatsMat

def loadImageStats(path):
    with open(path+"imageStats.pkl", 'rb') as imgStatsFile:
        imgStats = pickle.load(imgStatsFile)

    return imgStats



