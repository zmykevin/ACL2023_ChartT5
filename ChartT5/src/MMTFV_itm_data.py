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
from torch.utils.data.distributed import DistributedSampler
from copy import deepcopy


from transformers import T5Tokenizer, T5TokenizerFast
from tokenization import VLT5Tokenizer, VLT5TokenizerFast

import preprocess
from qa_answer_table import AnswerTable
import pandas as pd

project_dir = Path(__file__).resolve().parent.parent # VLT5
workspace_dir = project_dir.parent
dataset_dir = workspace_dir.joinpath('datasets/').resolve()
coco_dir = dataset_dir.joinpath('COCO')
vg_dir = dataset_dir.joinpath('VG')
coco_img_dir = coco_dir.joinpath('images/')

chartitm_dir = Path("/dvmm-filer2/projects/mingyang/semafor/chart_itm")
infographicitm_dir = Path("/dvmm-filer2/projects/mingyang/semafor/infographic_itm")
#chartitm_feature_dir = chartsum_dir.joinpath("features")



###################Create Own Dataset#############
class ChartITMFinetuneDataset(Dataset):
    def __init__(self, split='train', raw_dataset=None, rank=-1, topk=-1, verbose=True, args=None, mode='train'):
        super().__init__()

        self.raw_dataset = raw_dataset
        self.topk = topk
        self.verbose = verbose
        self.args = args

        self.mode = mode

        # Loading datasets to data
        #data_info_path = chartitm_dir.joinpath(f'ChartITM_annotation.json')
        data_info_path = infographicitm_dir.joinpath(f'MMTFV_annotation.json')
        self.instances = json.load(open(data_info_path, "r"))
        self.split = split
        if self.verbose:
            print('Data source: ', self.split)

        data = [datum for datum in self.instances if datum['split'] == split]

        # if topk > 0:
        #     data = data[:topk]
        #     if self.verbose:
        #         print(f"Use only {topk} data")

        self.n_gpus = torch.cuda.device_count()

        self.rank = rank

        self.data = data

        if self.verbose:
            # if 'sent' not in self.data_out:
            #     print("# all images:", len(self.data))
            # else:
            print("# all sentences:", len(self.data))

        self.n_boxes = args.n_boxes

        if 't5' in self.args.backbone:
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

        
        #Assume using just the chart element features
        self.source_to_h5 = {
            'infovqa': infographicitm_dir.joinpath(f'features/infovqa_boxes36.h5'),
            'chart2text': infographicitm_dir.joinpath(f'features/chart2text_boxes36.h5'),
            'test_hard_infovqa': infographicitm_dir.joinpath(f'features/testH_boxes36.h5'),
            'test_hard_chart2text':infographicitm_dir.joinpath(f'features/testH_boxes36.h5'),
            'test_easy_infovqa': infographicitm_dir.joinpath(f'features/testE_boxes36.h5'),
        }


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        out_dict = {}
        out_dict['args'] = self.args

        datum = self.data[idx]
        uid = datum['uid']
        out_dict['uid'] = uid
        img_id = datum['img_id']
        ###### Image ######
        source = datum['text_source']

        f = self.source_to_h5[source]

        if isinstance(f, Path):
            # path = self.data_source_to_h5_path[source]
            f = h5py.File(f, 'r')
            # self.split_to_h5_features[split_i] = f
            self.source_to_h5[source] = f
        
        try:
            feats = np.array(f[f'{img_id}/features'])
        except KeyError:
            print('img_id', img_id)
            # print(datum)
            exit()
        # feats2 = []
        # for key in ['img0', 'img1']:
        #     img_id = datum[key]
        #     feats = np.zeros(
        #         shape=(self.n_boxes, 2048), dtype=np.float32)
        #     f[f'{img_id}/features'].read_direct(feats)
        #     feats2.append(feats)
        # feats = np.stack(feats2)  # [2, n_boxes, feat_dim]
        feats = torch.from_numpy(feats)
        max_tensor_size = min(36, feats.size()[0])
        out_dict['vis_feats'] = feats[:max_tensor_size, :]

        # Normalize the boxes (to 0 ~ 1)
        img_h = f[f'{img_id}/img_h'][()]
        img_w = f[f'{img_id}/img_w'][()]
        boxes = f[f'{img_id}/boxes'][()]  # (x1, y1, x2, y2)
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        # np.testing.assert_array_less(boxes, 1+5e-2)
        np.testing.assert_array_less(-boxes, 0+1e-5)
        boxes = torch.from_numpy(boxes)
        boxes.clamp_(min=0.0, max=1.0)
        
        max_tensor_size = min(36, boxes.size()[0])
        out_dict['boxes'] = boxes[:max_tensor_size,:]

        ###### Text #####
        # caption = datum['caption']
        sent = datum['sent']
        if datum.get('article', None) is not None:
            if self.args.use_article:
                article = datum['article']
                source_text = f'image text match: <article> {article} <claim> {sent}'
            else:
                source_text = f'image text match: <claim> {sent}'
        #input_ids = self.tokenizer.encode(f'image text match: {sent}')
        else:
            source_text = f'image text match: {sent}'
        
        if self.args.ocr_tags:
            #Add OCRs to the data_csv
            ocr_tags = datum['ocr']
            #Add an argument to define the threshold
            if len(ocr_tags) > 60:
                ocr_tags = ocr_tags[:60]
            #input_tokens = [source_text]
            input_tokens = []
            input_tokens.append('ocrs:')
            for ocr_tag in ocr_tags:
                input_tokens.append(ocr_tag)
            #Append the source_text to the back
            input_tokens.append(source_text)
            source_text = ' '.join(input_tokens)
        input_ids = self.tokenizer.encode(source_text, max_length=400, truncation=True)


        question_id = uid
        out_dict['question_id'] = question_id


        out_dict['sent'] = sent
        out_dict['input_ids'] = torch.LongTensor(input_ids)
        out_dict['input_length'] = len(input_ids)
  
        if 'label' in datum:
            label = datum['label']

            # if label == 1:
            #     answer = 'true'
            # elif label == 0:
            #     answer = 'false'

            if label == 0:
                answer = 'support'
            elif label == 1:
                answer = 'refute'
            else:
                answer = 'nei'

            out_dict['answer'] = answer
            
            #print("classifier: {}".format(self.args.classifier))
            if self.args.classifier:
                target = torch.zeros(self.raw_dataset.num_answers)
                # for ans, score in label.items():
                #     target[self.raw_dataset.ans2label[ans]] = score
                target[self.raw_dataset.ans2label[answer]] = 1.0
                out_dict['target'] = target

            target_ids = self.tokenizer.encode(answer)

            out_dict['target_ids'] = torch.LongTensor(target_ids)
            out_dict['target_length'] = len(target_ids)
        else:
            label = None
        out_dict['label'] = label

        return out_dict


    def collate_fn(self, batch):
        batch_entry = {}

        args = batch[0]['args']

        B = len(batch)
        #V_L = batch[0]['boxes'].size(1)
        V_L = 36
        S_W_L = max(entry['input_length'] for entry in batch)
        if 'target_ids' in batch[0]:
            T_W_L = max(entry['target_length'] for entry in batch)

        feat_dim = batch[0]['vis_feats'].size(-1)

        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        # boxes = torch.zeros(B, 2, V_L, 4, dtype=torch.float)
        # vis_feats = torch.zeros(B, 2, V_L, feat_dim, dtype=torch.float)
        boxes = torch.zeros(B, V_L, 4, dtype=torch.float)
        vis_feats = torch.zeros(B, V_L, feat_dim, dtype=torch.float)

        if 'target' in batch[0]:
            # targets = []
            #targets = torch.zeros(B, dtype=torch.long)
            targets = torch.zeros(B, len(batch[0]['target']), dtype=torch.float)
        if 'target_ids' in batch[0]:
            target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        if batch[0]['label'] is not None:
            labels = torch.zeros(B, dtype=torch.long)


        sentences = []
        question_ids = []
        answers = []
        labels = []
        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']

            # boxes[i] += entry['boxes']
            # vis_feats[i] += entry['vis_feats']
            boxes[i, :len(entry['boxes']), :] += entry['boxes']
            vis_feats[i, :len(entry['vis_feats']), :] += entry['vis_feats']

            if 'target_ids' in entry:
                target_ids[i, :entry['target_length']] = entry['target_ids']

            if 'target' in entry:
                targets[i] += entry['target']
                # targets.append(entry['target'])

            sentences.append(entry['sent'])
            question_ids.append(entry['question_id'])
            if 'answer' in entry:
                answers.append(entry['answer'])

            if entry['label'] is not None:
                #labels[i] += entry['label']
                labels.append(entry['label'])

        batch_entry['input_ids'] = input_ids
        if 'target_ids' in batch[0]:
            word_mask = target_ids != self.tokenizer.pad_token_id
            target_ids[~word_mask] = -100
            batch_entry['target_ids'] = target_ids
        if 'target' in batch[0]:
            # targets = torch.stack(targets, dim=0)
            batch_entry['targets'] = targets

        batch_entry['boxes'] = boxes
        batch_entry['vis_feats'] = vis_feats

        batch_entry['sent'] = sentences
        batch_entry['question_ids'] = question_ids
        batch_entry['answers'] = answers

        if batch[0]['label'] is not None:
            batch_entry['labels'] = labels

        batch_entry['task'] = 'itm'

        return batch_entry

def get_loader(args, split='train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0,
               topk=-1):

    verbose = (gpu == 0)

    _dset = ChartITMDataset(split, verbose)

    dataset = ChartITMFinetuneDataset(
        split,
        raw_dataset=_dset,
        rank=gpu,
        topk=topk,
        verbose=verbose,
        args=args,
        mode=mode)
    
    if distributed and mode == 'train':
        train_sampler = DistributedSampler(dataset)
    else:
        train_sampler = None

    # if split == "val":
    #     print(len(dataset))

    if mode == 'train':
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(train_sampler is None),
            num_workers=workers, pin_memory=True, sampler=train_sampler,
            collate_fn=dataset.collate_fn)
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True,
            sampler=None,
            collate_fn=dataset.collate_fn,
            drop_last=False)

    if verbose:
        loader.evaluator = ChartITMEvaluator(_dset)

    loader.task = 'itm'

    return loader

# def get_loader(args, split='chartitm_train', mode='train',
#                batch_size=32, workers=4, distributed=False, gpu=0,
#                topk=-1):


#     verbose = (gpu == 0)
#     # dataset = PretrainDataset(
#     #     split,
#     #     rank=gpu,
#     #     topk=topk,
#     #     verbose=verbose,
#     #     args=args,
#     #     is_train=(mode == 'train'),
#     #     )
#     #Using the New Pre-training Dataset
#     _dset = ChartITMDataset(split, verbose)
#     dataset = ChartITMFinetuneDataset(
#         split,
#         rank=gpu,
#         topk=topk,
#         verbose=verbose,
#         args=args,
#         is_train=(mode == 'train'),
#         )

#     if distributed and mode=='train':
#         sampler = DistributedSampler(dataset)
#     else:
#         sampler = None

#     if mode == 'train':
#         loader = DataLoader(
#             dataset, batch_size=batch_size, shuffle=(sampler is None),
#             num_workers=workers, pin_memory=True, sampler=sampler,
#             collate_fn=dataset.collate_fn)
#     else:
#         loader = DataLoader(
#             dataset,
#             batch_size=batch_size,
#             num_workers=workers, pin_memory=True,
#             sampler=sampler,
#             shuffle=None if (sampler is not None) else False,
#             collate_fn=dataset.collate_fn,
#             drop_last=False)

#     if verbose:
#         loader.evaluator = ITMEvaluator(_dset)

#     loader.task = 'itm'

#     return loader
class ChartITMDataset:
    """
    An ChartITM data example in json file:
    {
        "uid": "train-10171-0-img0",
        "label": 0,
        "sent": "An image shows one leather pencil case, displayed open with writing implements tucked inside.",
    }
    """

    def __init__(self, splits: str, verbose=True):
        self.name = splits
        self.splits = splits.split(',')

        # Loading datasets to data
        self.data = []
        for split in self.splits:
            # self.data.extend(
            #     json.load(open(nlvr_dir.joinpath(f'{split}.json'))))
            #data_info_path = chartitm_dir.joinpath(f'ChartITM_annotation.json')
            data_info_path = infographicitm_dir.joinpath(f'MMTFV_annotation.json')
            with open(data_info_path) as f:
                _data = json.load(f)
            for datum in _data:
                if datum['split'] == split:
                    self.data.append(datum)
        if verbose:
            print("Load %d data from split(s) %s." %
                  (len(self.data), self.name))

        # List to dict (for evaluation and others)
        self.id2datum = {}
        # self.identifier2uid = {}
        for datum in self.data:
            self.id2datum[datum['uid']] = datum
        self.label2ans = {0: "support", 1: "refute", 2: "nei"}
        self.ans2label = {"support": 0, "refute": 1, "nei": 2}
            #self.identifier2uid[datum['identifier']] = datum['uid']
        if verbose:
            print('# Answers:', len(self.ans2label))

    @property
    def num_answers(self):
        return len(self.ans2label)
    def __len__(self):
        return len(self.data)

class ChartITMEvaluator:
    def __init__(self, dataset: ChartITMDataset):
        self.dataset = dataset

    def evaluate_train(self, quesid2ans: dict):
        score = 0.
        score_dict = {}
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['label']
            if int(ans) == int(label):
                score += 1
        accuracy = score / len(quesid2ans)
        score_dict['accuracy'] = accuracy
        return score_dict

    def dump_result(self, quesid2ans: dict, path):
        """
        Dump result to a CSV file, which is compatible with NLVR2 evaluation system.
        NLVR2 CSV file requirement:
            Each line contains: identifier, answer
        :param quesid2ans: nlvr2 uid to ans (either "True" or "False")
        :param path: The desired path of saved file.
        :return:
        """
        with open(path, 'w') as f:
            for uid, ans in quesid2ans.items():
                # idt = self.dataset.id2datum[uid]["identifier"]
                ans = 'True' if ans == 1 else 'False'
                f.write("%s,%s\n" % (uid, ans))

    def evaluate(self, quesid2ans: dict):
        # https://github.com/lil-lab/nlvr/blob/master/nlvr2/eval/metrics.py
        score = 0.
        score_dict = {}
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['label']
            if int(ans) == int(label):
                score += 1
        accuracy = score / len(quesid2ans)
        score_dict['accuracy'] = accuracy
        return score_dict
        # labeled_examples = self.dataset.data
        # predictions = quesid2ans

        # total_num = len(labeled_examples)
        # if len(predictions) < total_num:
        #     print("Some predictions are missing!")
        #     print("Got " + str(len(predictions)) + " predictions but expected " + str(total_num))

        #     # for example in labeled_examples:
        #     #     identifier = example["identifier"]
        #     #     uid = self.dataset.identifier2uid[identifier]
        #     #     if not uid in predictions:
        #     #         print("Missing prediction for item " + str(identifier))
        #     exit()

        # num_correct = 0.
        # consistency_dict = {}

        # for example in labeled_examples:
        #     # anon_label = example["identifier"].split("-")
        #     # anon_label[2] = ''
        #     # anon_label = '-'.join(anon_label)
        #     # if not anon_label in consistency_dict:
        #     #     consistency_dict[anon_label] = True
        #     # identifier = example["identifier"]
        #     # uid = self.dataset.identifier2uid[identifier]
        #     uid = example['uid']
        #     prediction = quesid2ans[uid]
        #     # if prediction.lower() == example["label"].lower():
        #     if int(prediction) == int(example["label"]):
        #         num_correct += 1.
        #     # else:
        #     #     consistency_dict[anon_label] = False

        # # Calculate consistency.
        # # num_consistent = 0.
        # # unique_sentence = len(consistency_dict)
        # # for identifier, consistent in consistency_dict.items():
        # #     if consistent:
        # #         num_consistent += 1

        # score_dict = {}
        # accuracy = num_correct / total_num
        # #consistency = num_consistent / unique_sentence
        # score_dict['accuracy'] = accuracy
        # #score_dict['consistency'] = consistency

        # return score_dict


