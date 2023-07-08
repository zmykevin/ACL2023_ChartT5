from torch.utils.data import DataLoader, Dataset, Sampler
from pathlib import Path
from collections import defaultdict
import json
import random
from multiprocessing import Pool
import h5py
import pickle
import math
from tqdm import tqdm
import torch
import numpy as np
from copy import deepcopy
import os
from torch.utils.data.distributed import DistributedSampler

from transformers import T5TokenizerFast, BartTokenizer
from tokenization import VLT5TokenizerFast
from PIL import Image

project_dir = Path(__file__).resolve().parent.parent  # VLT5
#workspace_dir = project_dir.parent
#dataset_dir = workspace_dir.joinpath('datasets/').resolve()
dataset_dir = Path("/dvmm-filer2/projects/mingyang/semafor/chart_summary")
coco_dir = dataset_dir.joinpath('Chart-to-text')
vg_dir = dataset_dir.joinpath('VG')
#dataset_img_dir = dataset_dir.joinpath('images/')
coco_feature_dir = coco_dir.joinpath('features')



class COCOCaptionFineTuneDataset(Dataset):
    def __init__(self, split='karpathy_train', raw_dataset=None, rank=-1, topk=-1, verbose=True, args=None, mode='train'):
        super().__init__()

        self.raw_dataset = raw_dataset
        self.topk = topk
        self.verbose = verbose
        self.args = args

        self.mode = mode

        # Loading datasets to data
        self.source = split
        if self.verbose:
            print('Data source: ', self.source)


        if self.args.tokenizer is None:
            self.args.tokenizer = self.args.backbone

        if 't5' in self.args.tokenizer:
            if self.args.use_vision:
                self.tokenizer = VLT5TokenizerFast.from_pretrained(
                    args.backbone,
                    # max_length=self.args.max_text_length,
                    do_lower_case=self.args.do_lower_case)
            else:
                self.tokenizer = T5TokenizerFast.from_pretrained(
                    args.backbone,
                    # max_length=self.args.max_text_length,
                    do_lower_case=self.args.do_lower_case)
        elif 'bart' in self.args.tokenizer:
            self.tokenizer = BartTokenizer.from_pretrained(
                args.backbone,
                # max_length=self.args.max_text_length,
                do_lower_case=self.args.do_lower_case)

            additional_special_tokens = [f'<extra_id_{i}>' for i in range(100-1, -1, -1)] + \
                    [f'<vis_extra_id_{i}>' for i in range(100-1, -1, -1)]
            special_tokens_dict = {'additional_special_tokens': additional_special_tokens}
            num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)


        data_info_path = dataset_dir.joinpath('dataset_fix.json')
        self.chartsum_dir = dataset_dir

        with open(data_info_path) as f:
            karpathy_data = json.load(f)

        split_rename = {
            'train': 'train',
            'restval': 'train',
            'val': 'val',
            'test': 'test'
        }

        n_images = 0
        
        self.f1 = h5py.File(self.chartsum_dir.joinpath('features').joinpath('train_boxes36.h5'), 'r')
        self.f2 = h5py.File(self.chartsum_dir.joinpath('features').joinpath('val_boxes36.h5'), 'r')
        self.f3 = h5py.File(self.chartsum_dir.joinpath('features').joinpath('val_boxes36.h5'), 'r') #don't have text features. 
        if args.visfeat_type == 'chart_element':
            self.f1 = h5py.File(self.chartsum_dir.joinpath('features').joinpath('train_chart_elements.h5'), 'r')
            self.f2 = h5py.File(self.chartsum_dir.joinpath('features').joinpath('val_chart_elements.h5'), 'r')
            self.f3 = h5py.File(self.chartsum_dir.joinpath('features').joinpath('val_chart_elements.h5'), 'r') #don't have test features
        feat_keys_train = list(self.f1.keys())
        feat_keys_val = list(self.f2.keys())
        feat_keys_test = list(self.f3.keys())
        
        data = []
        
        for datum in karpathy_data['images']:
            if datum['split'] !=mode or datum["dataset"]!=args.dataset_name:
                continue
            
            re_split = split_rename[datum['split']]
            #print(datum['split'] , re_split != self.source.split('_')[-1])
            #if re_split != self.source.split('_')[-1]:
            #    continue
            img_id = datum['filename'].split('.')[0]
            new_datum = {
                'img_id': img_id,
                'targets': [d['raw'].strip() for d in datum['sentences']],
                'is_train': re_split != 'test'
            }
            #if re_split == 'train':
            if self.args.ocr_copy:
                new_datum["sent"] = datum['ocr_sentences'][0]['raw']
            else:
                new_datum["sent"] = datum['sentences'][0]['raw']

            if mode not in ["train", "val", "test"]:
                print(mode)
            if (mode=="train" and new_datum["img_id"] in feat_keys_train) or \
                (mode=="val" and new_datum["img_id"] in feat_keys_val) or \
                (mode=="test" and new_datum["img_id"] in feat_keys_test):
                data.append(new_datum)

            n_images += 1
        
        if self.verbose:
            print(f"{self.source} has {n_images} images")
            print(f"Loaded {len(data)} data from", split)

        self.n_gpus = torch.cuda.device_count()

        self.rank = rank
        if self.topk > 0:
            data = data[:self.topk]
            if self.verbose:
                print(f"Use only {self.topk} data")

        self.data = data

        if self.verbose:
            print("# all sentences:", len(self.data))


        self.source_to_h5 = {}
        # if self.args.max_n_boxes == 36:
        #     self.source_to_h5.update({
        #         'train2014': dataset_dir.joinpath('features').joinpath('train_boxes36.h5'),
        #         'val2014': dataset_dir.joinpath('features').joinpath('val_boxes36.h5'),
        #     })
        if self.args.max_n_boxes == "vlt5":
            self.source_to_h5.update({
                'train2014': dataset_dir.joinpath('features').joinpath('train_boxes36.h5'),
                'val2014': dataset_dir.joinpath('features').joinpath('val_boxes36.h5'),
            })
        else:
            self.source_to_h5.update({
                'train2014': dataset_dir.joinpath('features').joinpath('train_chart_elements.h5'),
                'val2014': dataset_dir.joinpath('features').joinpath('val_chart_elements.h5'),
            })

        self.f1 = h5py.File(self.source_to_h5['train2014'], 'r')
        self.f2 = h5py.File(self.source_to_h5['val2014'], 'r')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        out_dict = {}
        out_dict['args'] = self.args

        datum = self.data[idx]
        #print(datum.keys())
        ###### Image ######
        if self.args.use_vision:
            img_id = datum['img_id']
            out_dict['img_id'] = img_id
            
            # Normalize the boxes (to 0 ~ 1)
            f = self.f1 if img_id in list(self.f1.keys()) else self.f2 \
                    if img_id in list(self.f2.keys()) else self.f3

            #get the img_root
            img_root = self.chartsum_dir.joinpath('train/images') if img_id in list(self.f1.keys()) else self.chartsum_dir.joinpath('val/images')
            current_split = "train" if img_id in list(self.f1.keys()) else "val"
            img_path = f"{str(img_root)}/{img_id}.png"
            im = Image.open(img_path)
            imw,imh = im.size
            #del the image
            del im

            img_h = f[f'{img_id}/img_h'][()]
            img_w = f[f'{img_id}/img_w'][()]
            boxes = f[f'{img_id}/boxes'][()]  # (x1, y1, x2, y2)
            boxes[:, (0, 2)] /= img_w
            boxes[:, (1, 3)] /= img_h
            np.testing.assert_array_less(boxes, 1+1e-5)
            np.testing.assert_array_less(-boxes, 0+1e-5)
            boxes = torch.from_numpy(boxes)

            boxes.clamp_(min=0.0, max=1.0)

            n_boxes = len(boxes)

            feats = np.zeros(shape=(n_boxes, 2048), dtype=np.float32)
            f[f'{img_id}/features'].read_direct(feats)
            feats = torch.from_numpy(feats)

            n_boxes = min(n_boxes, self.args.max_n_boxes)
            out_dict['n_boxes'] = n_boxes
            boxes = boxes[:n_boxes]
            feats = feats[:n_boxes]
            out_dict['boxes'] = boxes
            out_dict['vis_feats'] = feats

            # #check if we want ocr bbox
            # if self.args.ocr_tags:
            #     if self.args.ocr_position_encoding == "ocr_bbox":



        ###### Text #####
        if self.args.no_prefix:
            input_text = ''
            input_ids = []

        else:
            prefix = ""
            if self.args.prefix is None:
                prefix = 'caption:'
            elif self.args.prefix == 'span':
                prefix = "span prediction:"
            elif self.args.prefix == 'denoise':
                prefix = "denoise text: <mask>"
            elif self.args.prefix == 'mask':
                if 'bart' in self.args.tokenizer:
                    prefix = "<mask>"

            input_tokens = [prefix]

            if self.args.ocr_tags:
                if os.path.exists(self.chartsum_dir.joinpath(f"{current_split}/ocr_results/"+img_id+".json")):
                    prefix = 'describe image with ocr:'
                    input_tokens = [prefix]
                    with open(self.chartsum_dir.joinpath(f"{current_split}/ocr_results/"+img_id+".json"), "r") as f:
                        obj_ocrs = json.load(f)
                    # for bbox, ocr in obj_ocrs:
                    #     input_tokens.append(ocr[0])
                    
                    if self.args.ocr_position_encoding == "oscar_style":
                        if self.args.ocr_copy:
                            for i, (bbox, ocr) in enumerate(obj_ocrs):
                                input_tokens.append(f"<ocr_{i}> {ocr[0]}")
                        else:
                            for bbox, ocr in obj_ocrs:
                                input_tokens.append(ocr[0])


                    elif self.args.ocr_position_encoding == "ocr_bbox":
                        ocr_ids = []
                        ocr_bboxes = []
                        for i, (bbox, ocr) in enumerate(obj_ocrs):
                            if self.args.ocr_copy:
                                current_ocr_ids = self.tokenizer.encode(f"<ocr_{i}> {ocr[0]}")
                            else:
                                current_ocr_ids = self.tokenizer.encode(ocr[0])
                            ocr_ids += current_ocr_ids
                            #rescale bbox
                            if imw == img_w and imh == img_h:
                                new_bbox = [bbox[0][0], bbox[0][1], bbox[2][0], bbox[2][1]]
                            else:
                                scale_ratio_1 = img_w/imw
                                scale_ratio_2 = img_h/imh
                                
                                new_bbox = [scale_ratio_1*bbox[0][0], scale_ratio_2*bbox[0][1], scale_ratio_1*bbox[2][0], scale_ratio_2*bbox[2][1]]
                            ocr_bboxes += [new_bbox] * len(current_ocr_ids)

                        
                        #recompute the bbox
                        out_dict['visocrs'] = torch.LongTensor(ocr_ids)
                        ocr_bboxes = np.array(ocr_bboxes)
                        #print(ocr_bboxes.shape)
                        ocr_bboxes[:, (0, 2)] /= img_w
                        ocr_bboxes[:, (1, 3)] /= img_h
                        np.testing.assert_array_less(ocr_bboxes, 1+1e-5)
                        # np.testing.assert_array_less(boxes, 1+5e-2)
                        np.testing.assert_array_less(-ocr_bboxes, 0+1e-5)
                        
                        ocr_bboxes = torch.from_numpy(ocr_bboxes)
                        out_dict['visocrs_bboxes'] = ocr_bboxes
                        #get the length
                        out_dict['visocrs_length'] = len(ocr_ids)
            
            input_text = ' '.join(input_tokens)

            if 't5' in self.args.tokenizer:
                input_ids = self.tokenizer.encode(
                    input_text,
                    max_length=self.args.max_text_length, truncation=True)
            elif 'bart' in self.args.tokenizer:
                input_ids = self.tokenizer.encode(
                    input_text,
                    max_length=self.args.max_text_length, truncation=True)
            else:
                input_ids = self.tokenizer.convert_tokens_to_ids(
                    self.tokenizer.tokenize(input_text)[:self.args.max_text_length - 1] + ['[SEP]'])
        
        out_dict['input_text'] = input_text
        out_dict['input_ids'] = torch.LongTensor(input_ids)
        out_dict['input_length'] = len(input_ids)
        if True: #'sent' in datum:
            sent = datum['sent'].strip()
            if 't5' in self.args.tokenizer:
                target_ids = self.tokenizer.encode(sent, max_length=self.args.gen_max_length, truncation=True)
            elif 'bart' in self.args.tokenizer:
                target_ids = self.tokenizer.encode(sent, max_length=self.args.gen_max_length, truncation=True)

            assert len(target_ids) <= self.args.gen_max_length, len(target_ids)
            out_dict['target_ids'] = torch.LongTensor(target_ids)
            out_dict['target_length'] = len(target_ids)
        #if datum['is_train'] and 'sent' in datum:
            out_dict['sent'] = sent
        if 'targets' in datum:
            out_dict['targets'] = datum['targets']


        return out_dict

    def collate_fn(self, batch):
        batch_entry = {}

        B = len(batch)

        S_W_L = max(entry['input_length'] for entry in batch)
        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        if self.args.no_prefix:
            assert input_ids.size() == (B, 0)

        if self.args.use_vision:
            V_L = max(entry['n_boxes'] for entry in batch)
            # V_L = len(batch[0]['boxes'])
            feat_dim = batch[0]['vis_feats'].shape[-1]

            boxes = torch.zeros(B, V_L, 4, dtype=torch.float)
            vis_feats = torch.zeros(B, V_L, feat_dim, dtype=torch.float)
            vis_attention_mask = torch.zeros(B, V_L, dtype=torch.float)
        
        #Added by Mingyang Zhou
        if self.args.ocr_position_encoding == "ocr_bbox" and self.args.ocr_tags:
            V_W_L = max(entry['visocrs_length'] for entry in batch)
            ocr_ids = torch.ones(B, V_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

            #get ocr_bbox
            ocr_bboxes = torch.zeros(B, V_W_L, 4, dtype=torch.float)

        if 'target_ids' in batch[0]:
            T_W_L = max(entry['target_length'] for entry in batch)
            target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        # sentences = []

        targets = []
        img_ids = []
        img_paths = []
        input_text = []

        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']

            if self.args.use_vision:
                n_boxes = entry['n_boxes']
                boxes[i, :n_boxes] = entry['boxes']
                vis_feats[i, :n_boxes] = entry['vis_feats']
                vis_attention_mask[i, :n_boxes] = 1
                img_ids.append(entry['img_id'])
                # img_paths.append(entry['img_path'])

            if 'target_ids' in entry:
                target_ids[i, :entry['target_length']] = entry['target_ids']

            if 'input_text' in entry:
                input_text.append(entry['input_text'])

            # sentences.append(entry['sent'])

            if 'targets' in entry:
                targets.append(entry['targets'])


        batch_entry['input_ids'] = input_ids
        if 'target_ids' in batch[0]:
            word_mask = target_ids != self.tokenizer.pad_token_id
            target_ids[~word_mask] = -100
            batch_entry['target_ids'] = target_ids

        if self.args.use_vision:
            batch_entry['boxes'] = boxes
            batch_entry['vis_feats'] = vis_feats
            batch_entry['vis_attention_mask'] = vis_attention_mask
            batch_entry['img_id'] = img_ids
            batch_entry['img_paths'] = img_paths
        
        if self.args.ocr_position_encoding == "ocr_bbox" and self.args.ocr_tags:
            batch_entry['visocrs'] = ocr_ids
            batch_entry['visocrs_bboxes'] = ocr_bboxes
        # batch_entry['sent'] = sentences

        batch_entry['input_text'] = input_text

        batch_entry['targets'] = targets

        batch_entry['task'] = 'caption'

        return batch_entry


def get_loader(args, split='karpathy_train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0,
               topk=-1):

    verbose = (gpu == 0)

    dataset = COCOCaptionFineTuneDataset(
        "_"+mode,
        rank=gpu,
        topk=topk,
        verbose=verbose,
        args=args,
        mode=mode)

    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None

    if mode == 'train':
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(sampler is None),
            num_workers=workers, pin_memory=True, sampler=sampler,
            collate_fn=dataset.collate_fn)
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers, pin_memory=True,
            sampler=sampler,
            shuffle=None if (sampler is not None) else False,
            collate_fn=dataset.collate_fn,
            drop_last=False)

    if verbose:
        loader.evaluator = COCOCaptionEvaluator()

    loader.task = 'caption'

    return loader



class COCOCaptionEvaluator:
    def __init__(self):
        import language_evaluation
        self.evaluator = language_evaluation.CocoEvaluator(verbose=False)


    def evaluate(self, predicts, answers):

        results = self.evaluator.run_evaluation(predicts, answers)

        return results