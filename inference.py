import copy
import os.path as osp
import mmcv
import numpy as np
import pandas
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
from mmcv import Config
from mmdet.apis import set_random_seed

from mmcv import Config

from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.apis import init_detector, inference_detector

@DATASETS.register_module()

class CREATEDataset(CustomDataset):

    CLASSES = ('ultrasound', 'guidewire', 'guidewire_casing', 'syringe', 'scalpel', 'dilator', 'catheter', 'anesthetic')
    def load_annotations(self, ann_file):
        #print(self.ann_file)
        cat2label = {k: i for i, k in enumerate(self.CLASSES)}
        dataSet = pandas.read_csv(self.ann_file)
        imageList = dataSet.FileName
        annoList = dataSet.ToolBX

        # load image list from file
        data_infos = []
        COUNT = len(imageList)
        # convert annotations to middle format
        for i in range(COUNT):
            imgName = imageList[i]

            #img_prefix = '/home/ROBARTS/sxing/myModels/FasterRCNN/Training_Data_Part1/AN01-20210104-160254/'
            imgfilename = osp.join(self.img_prefix, imgName)
            #imgfilename = f'{img_prefix}/{imgName}'
            #imgfilename = f'{self.img_prefix}/{imgName}'
            #print(imgfilename)

            image = mmcv.imread(imgfilename)
            height, width = image.shape[:2]
            data_info = dict(filename=imgName, width = width, height = height)
            stringBX = annoList[i]
            if stringBX == '[]':
                gt_bboxes = []
                gt_labels = []
                gt_bboxes_ignore = []
                gt_labels_ignore = []
                gt_labels_ignore.append(-1)
                gt_bboxes_ignore.append([0,0,0,0])
            else: 
                boudingBox = stringBX.replace(" ","")
                boudingBox = boudingBox.replace("'", "")
                boudingBox = boudingBox.replace("[", "")
                boudingBox = boudingBox.replace("]", "")
                listBoundingBox = boudingBox.split("},{")
        
                gt_bboxes = []
                gt_labels = []
                gt_bboxes_ignore = []
                gt_labels_ignore = []
       
                for boundingBox in listBoundingBox:
                    boundingBox = boundingBox.replace("{", "")
                    boundingBox = boundingBox.replace("}", "")
                    keyEntryPairs = boundingBox.split(",")
                    boundingBoxDict = {}
                    for pair in keyEntryPairs:
                        key, entry = pair.split(":")
                        if entry.isnumeric():
                            boundingBoxDict[key] = int(entry)
                        else:
                            boundingBoxDict[key] = entry
                    x1 = boundingBoxDict["xmin"]
                    x2 = boundingBoxDict["xmax"]
                    y1 = boundingBoxDict["ymin"]
                    y2 = boundingBoxDict["ymax"]

                    xmin_ = min(x1,x2)
                    xmax_ = max(x1,x2)
                    ymin_ = min(y1,y2)
                    ymax_ = max(y1,y2)
                    bbox = [xmin_, ymin_, xmax_, ymax_]
                    bbox_name = boundingBoxDict["class"]
                    #print(boundingBoxDict)
                    #print(bbox_name)
                    if bbox_name in cat2label:
                        gt_labels.append(cat2label[bbox_name])
                        gt_bboxes.append(bbox)
                    else:
                        gt_labels_ignore.append(-1)
                        gt_bboxes_ignore.append(bbox)
            data_anno = dict(
                bboxes=np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
                labels=np.array(gt_labels, dtype=np.int64),
                bboxes_ignore=np.array(gt_bboxes_ignore, dtype=np.float32).reshape(-1,4),
                labels_ignore=np.array(gt_labels_ignore, dtype=np.int64))
            data_info.update(ann=data_anno)
            data_infos.append(data_info)  

        return data_infos
    



if __name__ == '__main__':

    device = 'cuda:0'
    cfg = Config.fromfile('/home/ROBARTS/sxing/mmdetection/configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py')
    # Modify dataset type and path
    cfg.dataset_type = 'CREATEDataset'
    cfg.data_root = '/home/ROBARTS/sxing/myModels/FasterRCNN/'
    cfg.device='cuda'
    cfg.data.samples_per_gpu = 12
    cfg.runner.max_epochs = 25
    cfg.data.test.type = 'CREATEDataset'
    cfg.data.test.data_root = '/home/ROBARTS/sxing/myModels/FasterRCNN/'
    cfg.data.test.ann_file = '/home/ROBARTS/sxing/myModels/FasterRCNN/Validation_data.csv'
    cfg.data.test.img_prefix = '/home/ROBARTS/sxing/myModels/FasterRCNN/Validation_data/'

    cfg.data.train.type = 'CREATEDataset'
    cfg.data.train.data_root = '/home/ROBARTS/sxing/myModels/FasterRCNN/'
    cfg.data.train.ann_file = '/home/ROBARTS/sxing/myModels/FasterRCNN/Training_data.csv'
    cfg.data.train.img_prefix = '/home/ROBARTS/sxing/myModels/FasterRCNN/Training_data/'

    cfg.data.val.type = 'CREATEDataset'
    cfg.data.val.data_root = '/home/ROBARTS/sxing/myModels/FasterRCNN/'
    cfg.data.val.ann_file = '/home/ROBARTS/sxing/myModels/FasterRCNN/Validation_data.csv'
    cfg.data.val.img_prefix = '/home/ROBARTS/sxing/myModels/FasterRCNN/Validation_data/'
    # modify num classes of the model in box head
    cfg.model.roi_head.bbox_head.num_classes = 8
    
    # If we need to finetune a model based on a pre-trained detector, we need to
    # use load_from to set the path of checkpoints.
    cfg.load_from = 'checkpoints/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth'
    # Set up working dir to save files and logs.
    cfg.work_dir = './tutorial_exps'
    # The original learning rate (LR) is set for 8-GPU training.
    # We divide it by 8 since we only use one GPU.
    cfg.optimizer.lr = 0.02 / 8
    cfg.lr_config.warmup = None
    cfg.log_config.interval = 10

    # Change the evaluation metric since we use customized dataset.
    cfg.evaluation.metric = 'mAP'
    # We can set the evaluation interval to reduce the evaluation times
    cfg.evaluation.interval = 1
    # We can set the checkpoint saving interval to reduce the storage cost
    cfg.checkpoint_config.interval = 2
    # Set seed thus the results are more reproducible
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)
    # We can also use tensorboard to log the training process
    cfg.log_config.hooks = [
    dict(type='TextLoggerHook'),
    dict(type='TensorboardLoggerHook')]


    # We can initialize the logger for training and have a look
    # at the final config used for training
    #print(f'Config:\n{cfg.pretty_text}')
    
    # dataset = CREATEDataset(ann_file, test_pipeline)
    # data_infos = dataset.load_annotations(ann_file)
    # print(data_infos)
    
    # # Build dataset
    #datasets = [build_dataset(cfg.data.train)]
    # # Build the detector

    #model = build_detector(cfg.model)
    # Add an attribute for visualization convenience
    
    checkpoint_file = '/home/ROBARTS/sxing/myModels/FasterRCNN/tutorial_exps/epoch_10.pth'
    
    # build the model from a config file and a checkpoint file
    model = init_detector(cfg, checkpoint_file, device='cuda:0')
    #model.CLASSES = datasets[0].CLASSES
    model.to(device)


    # # test a single image and show the results
    img = '/home/ROBARTS/sxing/myModels/FasterRCNN/Test_data/AN05-20210216-205144/AN05-20210216-205144_0464.jpg'  # or img = mmcv.imread(img), which will only load it once
    fileName = '/home/ROBARTS/sxing/myModels/FasterRCNN/Test_data/AN05-20210216-205144_Labels.csv'
    dataSetTest = pandas.read_csv(fileName)
    testImageList = dataSetTest.FileName
    NUM = len(testImageList)
    # NUM = 1
    for i in range(NUM):
        
        imgName = testImageList[i]
        imgFile = osp.join('/home/ROBARTS/sxing/myModels/FasterRCNN/Test_data/AN05-20210216-205144/', imgName)
        
        result = inference_detector(model, imgFile)
    # visualize the results in a new window
    # model.show_result(img, result)
    # or save the visualization results to image files
        saveOutName = osp.join('/home/ROBARTS/sxing/myModels/FasterRCNN/Test_data/', imgName)
        model.show_result(imgFile, result, out_file=saveOutName)
        if (i%10 == 0):
            print(i)
    # # test a video and show the results
    # video = mmcv.VideoReader('/home/ROBARTS/sxing/myModels/FasterRCNN/Test_data/AN05-20210216-205144/AN05-20210216-205144_0000.mp4')
    # for frame in video:
    #     result = inference_detector(model, frame)
    #     model.show_result(frame, result, wait_time=1, out_file='/home/ROBARTS/sxing/myModels/FasterRCNN/Test_data/AN05-20210216-205144_00000_results.mp4')
        