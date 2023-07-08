#Adapt the Feature Extarctor from ChartQA to VLT5
import json
import cv2
import time
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
import detectron2.data.transforms as T
from detectron2.structures import ImageList


import os
import torch
from tqdm import tqdm
from pathlib import Path
import h5py

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trained_model_path = "/dvmm-filer2/projects/mingyang/semafor/chart_qa/VisualFeature_Extractor/model_final_all.pth"
# images_path = "/content/data/images_validation/"
# save_path = "/content/output_feats/"

NUM_OBJECTS = 36
DIM = 2048
CLASSES_NUM = 15

cfg = get_cfg()
aug = T.ResizeShortestEdge(
    [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
)
input_format = cfg.INPUT.FORMAT

def build_detector():
	cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml"))
	cfg.DATALOADER.NUM_WORKERS = 0
	cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml")  # Let training initialize from model zoo
	cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 
	cfg.MODEL.ROI_HEADS.NUM_CLASSES = CLASSES_NUM  # number of classes

	trained_model_path = "/dvmm-filer2/projects/mingyang/semafor/chart_qa/VisualFeature_Extractor/model_final_all.pth"
	cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, trained_model_path)  # path to the model we just trained
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
	model = build_model(cfg)
	DetectionCheckpointer(model).load(trained_model_path)
	model.eval()
	return model

def collate_fn(batch):
    img_ids = []
    imgs = []

    for i, entry in enumerate(batch):
        img_ids.append(entry['img_id'])
        imgs.append(entry['img'])

    batch_out = {}
    batch_out['img_ids'] = img_ids
    batch_out['imgs'] = imgs

    return batch_out

def extract(output_fname, dataloader, desc):
	detector = build_detector()
	with h5py.File(output_fname, 'w') as f:
		with torch.no_grad():
			for i, batch in tqdm(enumerate(dataloader),
                                 desc=desc,
                                 ncols=150,
                                 total=len(dataloader)):

				img_ids = batch['img_ids']
				# feat_list, info_list = feature_extractor.get_detectron_features(batch)

				imgs = batch['imgs']

				assert len(imgs) == 1


				img_id = img_ids[0]
				orig_image = imgs[0] #cv read img


				image = aug.get_transform(orig_image).apply_image(orig_image)
				image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
				c, height, width = image.shape
				inputs = [{"image": image, "height": height, "width": width}]
				#run model
				image_data_dict = {}

				images = detector.preprocess_image(inputs) 
				features = detector.backbone(images.tensor)
				proposals, _ = detector.proposal_generator(images, features) 
				#### orig
				try:
				    instances, _ = detector.roi_heads(images, features, proposals)
				    mask_features = [features[f] for f in detector.roi_heads.in_features]
				    mask_features = detector.roi_heads.pooler(mask_features, [x.pred_boxes for x in instances])
				    mask_features = detector.roi_heads.res5(mask_features)
				    average_pooling = nn.AvgPool2d(7)
				    
				    mask_features = average_pooling(mask_features)
				    orig_size = mask_features.size()
				    mask_features = mask_features.resize_(orig_size[0], orig_size[1])
				    ###

				    boxes = instances[0].pred_boxes.tensor.detach().cpu().numpy()
				    
				    grp = f.create_group(img_id)
				    grp['features'] = mask_features.detach().cpu().numpy()
				    grp['boxes'] = boxes 
				    grp['img_w'] = width
				    grp['img_h'] = height
				except Exception as e:
				    print(batch)
				    print(e)
				    continue
		        	
