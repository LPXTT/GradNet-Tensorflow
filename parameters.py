def configParams():
    params = {}

    # print("config parameters...")

    params['gpuId'] = 0
    params['data_path'] = "./.."
    params['ilsvrc2015'] = params['data_path']+"ILSVRC2015/"
    params['crops_path'] = params['data_path']+"ILSVRC2015_C/"
    params['crops_train'] = params['crops_path']+"Data/VID/train/"
    params['curation_path'] = "/your_path/ILSVRC15-curation/"
    params['seq_base_path'] = "./dataset/"#OTB-100
    params['model_path'] = './ckpt/base_l5_1t_49/model_epoch49.ckpt'
    params['trainBatchSize'] = 8
    params['numScale'] = 3

    return params

