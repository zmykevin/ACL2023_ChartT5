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
import re
import pandas as pd
from torch.utils.data.distributed import DistributedSampler

from transformers import T5TokenizerFast, BartTokenizer
from tokenization import VLT5TokenizerFast
import ast


project_dir = Path(__file__).resolve().parent.parent  # VLT5
workspace_dir = project_dir.parent
dataset_dir = workspace_dir.joinpath('datasets/').resolve()
coco_dir = dataset_dir.joinpath('COCO')
vg_dir = dataset_dir.joinpath('VG')
coco_img_dir = coco_dir.joinpath('images/')
coco_feature_dir = coco_dir.joinpath('features')
vqa_dir = dataset_dir.joinpath('vqa')
#chartqa_dir = Path("/dvmm-filer2/projects/mingyang/semafor/chart_qa/ChartQAv1")
#chartqa_feature_dir = chartqa_dir.joinpath("features")
chartqa_root = Path("/dvmm-filer2/projects/mingyang/semafor/chart_qa")

exp_symbol = "[EXP]"
num_symbol = "<num_extra_id_0>"
MANTISSA_NORM = 10
EXPONENT_NORM = 20

def scifnum_sent(caption):
    caption_words = caption.split(" ")
    new_caption_words = []
    for w in caption_words:
        try:
            float_w = float(w)
        except:
            new_caption_words.append(w)
            continue
        scif_w = "{:2e}".format(float_w)
        scif_w = scif_w.replace('e', exp_symbol)
        new_caption_words.append(scif_w)
    new_caption = " ".join(new_caption_words)
    return new_caption

def scifnum_ocr(ocrs): 
    new_ocrs = []
    for w in ocrs:
        try:
            float_w = float(w)
        except:
            new_ocrs.append(w)
            continue
        scif_w = "{:2e}".format(float_w)
        scif_w = scif_w.replace('e', exp_symbol)
        new_ocrs.append(scif_w)

    return new_ocrs

def numerify_ocrs(ocrs):
    # caption_words = caption.split(" ")
    num_values = []
    new_ocrs = []
    for w in ocrs:
        try:
            float_w = float(w)
        except:
            new_ocrs.append(w)
            num_values.append((0.0,0.0))
            continue
        scif_w = "{:2e}".format(float_w)
        try:
            mantissa = float(scif_w.split('e')[0])/(2*MANTISSA_NORM) + 0.5
            exponent = float(scif_w.split('e')[1])/(2*EXPONENT_NORM) + 0.5
        except:
            new_ocrs.append(w)
            num_values.append((0.0, 0.0))
            continue

        num_values.append((mantissa, exponent))
        new_ocrs.append(num_symbol)
    #new_caption = ' '.join(new_caption_words)
    return new_ocrs, num_values

def numerify_sent(caption, input_value=True):
    caption_words = caption.split(" ")
    num_values = []
    new_caption_words = []
    for w in caption_words:
        try:
            float_w = float(w)
        except:
            new_caption_words.append(w)
            num_values.append((0.0,0.0))
            continue
        scif_w = "{:2e}".format(float_w)
        # mantissa = float(scif_w.split('e')[0])/(2*MANTISSA_NORM) + 0.5
        # exponent = float(scif_w.split('e')[1])/(2*EXPONENT_NORM) + 0.5
        try:
            mantissa = float(scif_w.split('e')[0])/(2*MANTISSA_NORM) + 0.5
            exponent = float(scif_w.split('e')[1])/(2*EXPONENT_NORM) + 0.5
        except:
            new_caption_words.append(w)
            num_values.append((0.0, 0.0))
            continue

        num_values.append((mantissa, exponent))
        # if input_value:
        #     new_caption_words.append(w)
        # else:
        new_caption_words.append(num_symbol)
    new_caption = ' '.join(new_caption_words)
    assert len(new_caption.split(" ")) == len(num_values), "original_caption is: {}, new_caption is: {}".format(caption, new_caption)
    return new_caption, num_values


class ChartQAFineTuneDataset(Dataset):
    def __init__(self, src_folder="/path/2/data/", split='train', raw_dataset=None, rank=-1, topk=-1, verbose=True, args=None, mode='train'):
        super().__init__()

        #self.raw_dataset = raw_dataset
        self.topk = topk
        self.verbose = verbose
        self.args = args

        self.mode = mode
        
        # Loading datasets to data, this part might need to be changed
        #instances = pd.read_csv(src_folder + "data_fix_origin.csv")
        instances = pd.read_csv(src_folder + "data_fix_origin.csv")
        self.instances = instances
        self.inputs = instances["Input"].values
        self.ocrs = instances["OCR"].values
        

        #Added to encode the ocr bbox values
        self.ocr_bboxes = instances["OCR_bbox"].values
        self.outputs = None
        if "Output" in instances:
            self.outputs = instances["Output"].values
        #Added by Mingyang for OCR Copy
        if "OCR_Input" in instances:
            self.ocr_inputs = instances["OCR_Input"].values
        if "OCR_Output" in instances:
            self.ocr_outputs = instances["OCR_Output"].values

        if "OCR_Symbols" in instances:
            self.ocr_symbols = instances["OCR_Symbols"].values

        self.images_indices = instances['Image Index'].values
        self.questions_ids = instances['Question ID'].values
        self.src_folder = src_folder

        self.sources = split.split(',')
        # if self.verbose:
        #     print('Data sources: ', self.sources)
        if self.args.table_flat:
            self.full_tables = instances['Full_Table'].values

        if 't5' in self.args.backbone:
            if self.args.use_vision:
                self.tokenizer = VLT5TokenizerFast.from_pretrained(
                    args.backbone,
                    max_length=self.args.max_text_length,
                    do_lower_case=self.args.do_lower_case
                    )
            else:
                self.tokenizer = T5TokenizerFast.from_pretrained(
                    args.backbone,
                    max_length=self.args.max_text_length,
                    do_lower_case=self.args.do_lower_case)

        elif 'bart' in self.args.backbone:
            self.tokenizer = BartTokenizer.from_pretrained(
                args.backbone,
                # max_length=self.args.max_text_length,
                do_lower_case=self.args.do_lower_case)

            if args.use_vis_order_embedding:
                additional_special_tokens = [f'<extra_id_{i}>' for i in range(100-1, -1, -1)] + \
                        [f'<vis_extra_id_{i}>' for i in range(100-1, -1, -1)]
                special_tokens_dict = {'additional_special_tokens': additional_special_tokens}
                num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)

        # self.answer_normalizer = ChartQAEvaluator()
        
        self.img_ids_to_source = {}

        for source in self.sources:
            for img_id in self.images_indices:
                self.img_ids_to_source[img_id] = source

            #data_info_dicts.extend(_data_info_dicts)
        # if self.verbose:
        #     print(f"Loaded {len(_data_info_dicts)} data from", source)

        # data = data_info_dicts

        self.n_gpus = torch.cuda.device_count()

        self.rank = rank

        # if self.topk > 0:
        #     data = data[:self.topk]
        #     if self.verbose:
        #         print(f"Use only {self.topk} data")

        # self.data = data

        # if self.verbose:
        #     print("# all sentences:", len(self.data))

        self.n_boxes = args.n_boxes
        
        if args.dataset_name == "chartqa":
            self.chartqa_dir = chartqa_root.joinpath('ChartQAv1')
        elif args.dataset_name == "plotqa":
            self.chartqa_dir = chartqa_root.joinpath('PlotQA')


        #Get the default VLT5 Features
        chartqa_feature_dir = self.chartqa_dir.joinpath("features")
        if self.args.visfeat_type == "vlt5":
            self.source_to_h5 = {
                'train': chartqa_feature_dir.joinpath(f'train_boxes36.h5'),
                'val': chartqa_feature_dir.joinpath(f'val_boxes36.h5'),
                'test': chartqa_feature_dir.joinpath(f'test_boxes36.h5'),
                'testv1': chartqa_feature_dir.joinpath(f'test_boxes36.h5'),
                'testv2': chartqa_feature_dir.joinpath(f'test_boxes36.h5'),
            }
        else:
            self.source_to_h5 = {
                'train': chartqa_feature_dir.joinpath(f'train_chart_elements.h5'),
                'val': chartqa_feature_dir.joinpath(f'val_chart_elements.h5'),
                'test': chartqa_feature_dir.joinpath(f'test_chart_elements.h5'),
                'testv1': chartqa_feature_dir.joinpath(f'test_chart_elements.h5'),
                'testv2': chartqa_feature_dir.joinpath(f'test_chart_elements.h5'),
            }
            
        # else:
        if self.args.ocr_tags:
            self.ocr_threds = 30
        

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):

        out_dict = {}
        out_dict['args'] = self.args

        # datum = self.data[idx]

        ###### Image ######
        if self.args.use_vision:
            # img_id = datum['img_id']
            img_id = self.images_indices[idx]
            out_dict['img_id'] = img_id

            source = self.img_ids_to_source[img_id]

            f = self.source_to_h5[source]

            if isinstance(f, Path):
                # path = self.data_source_to_h5_path[source]
                f = h5py.File(f, 'r')
                # self.split_to_h5_features[split_i] = f
                self.source_to_h5[source] = f

            if self.args.visfeat_type == "vlt5":
                feats = np.zeros(shape=(self.n_boxes, 2048), dtype=np.float32)
                try:
                    f[f'{img_id}/features'].read_direct(feats)
                except KeyError:
                    print('img_id', img_id)
                    # print(datum)
                    exit()

                feats = torch.from_numpy(feats)
            
                out_dict['vis_feats'] = feats
            else:
                try:
                    feats = np.array(f[f'{img_id}/features'])
                except KeyError:
                    print('img_id', img_id)
                    # print(datum)
                    exit()

                feats = torch.from_numpy(feats)
                max_tensor_size = min(36, feats.size()[0])
                out_dict['vis_feats'] = feats[:max_tensor_size, :]

            # Normalize the boxes (to 0 ~ 1)
            img_h = f[f'{img_id}/img_h'][()]
            img_w = f[f'{img_id}/img_w'][()]
            boxes = f[f'{img_id}/boxes'][()]  # (x1, y1, x2, y2)
            # print(boxes)
            # print(img_w)
            # print(img_h)
            # print(img_id)
            # print(type(boxes))
            # print(boxes)
            boxes[:, (0, 2)] /= img_w
            boxes[:, (1, 3)] /= img_h
            np.testing.assert_array_less(boxes, 1+1e-5)
            # np.testing.assert_array_less(boxes, 1+5e-2)
            np.testing.assert_array_less(-boxes, 0+1e-5)
            boxes = torch.from_numpy(boxes)

            boxes.clamp_(min=0.0, max=1.0)
            if self.args.visfeat_type == "vlt5":
                out_dict['boxes'] = boxes
            else:
                max_tensor_size = min(36, boxes.size()[0])
                out_dict['boxes'] = boxes[:max_tensor_size,:]

        ###### Text #####
        # caption = datum['caption']
        # if 'sent' in datum:
        #     sent = datum['sent']
        # elif 'question' in datum:
        #     sent = datum['question']
        
        # if self.args.ocr_copy:
        #     sent = self.ocr_inputs[idx]
        # else:
        #     sent = self.inputs[idx]
        sent = self.inputs[idx]
        
        #Fixed Scifnum Sentence
        if self.args.scif_num:
            sent = scifnum_sent(sent)
        
        #obtain the ocrs
        c_ocrs = ast.literal_eval(self.ocrs[idx])
        
        if self.args.num_modeling:
            #convert the sent and c_ocrs's numerical value to ['NUM']
            sent, sent_nums = numerify_sent(sent)
            c_ocrs, ocr_nums = numerify_ocrs(c_ocrs)

        if self.args.ocr_tags:
            if self.args.ocr_copy:
                #replace ocr with symbolic format
                ocr_s = ast.literal_eval(self.ocr_symbols[idx])
                ocr_t = c_ocrs
                #Keep both the symbolic representation and original text

                assert len(ocr_s) == len(ocr_t)
                ocr_tags = [f"{x} {y}" for x,y in zip(ocr_s,ocr_t)]
            else:
                ocr_tags = c_ocrs

            if self.args.ocr_position_encoding == "oscar_style":
                #Add OCRs to the data_csv
                #ocr_tags = json.loads(self.ocrs[idx])
                #ocr_tags = self.ocrs[idx].strip('][').split(',')
                
                if self.args.scif_num:
                    ocr_tags = scifnum_ocr(ocr_tags)

                #Add an argument to define the threshold
                if not self.args.num_modeling:
                    if len(ocr_tags) > self.ocr_threds:
                        ocr_tags = ocr_tags[:self.ocr_threds]
                    ocr_tag_string = ' '.join(ocr_tags)
                    #input_ids = self.tokenizer.encode(f'ocrs: {ocr_tag_string} vqa: {sent}', max_length=400, truncation=True)
                    input_ids = self.tokenizer.encode(f'vqa: {sent} ocrs: {ocr_tag_string}', max_length=400, truncation=True)
                else:
                    #Get the numerical modeling
                    if len(ocr_tags) > self.ocr_threds:
                        ocr_tags = ocr_tags[:self.ocr_threds]

                    input_tokens = ['vqa: '] + sent.split()  + ['ocrs: '] + ocr_tags
                    input_mantiss_tokens = [0.0] + [x[0] for x in sent_nums] + [0.0] + [x[0] for x in ocr_nums]
                    input_exponent_tokens = [0.0] + [x[1] for x in sent_nums] + [0.0] + [x[1] for x in ocr_nums]
                    input_ids = []
                    input_mantissa = []
                    input_exponent = []
                    for t,m,e in zip(input_tokens, input_mantissa_tokens, input_exponent_tokens):
                        token_ids = self.tokenizer.encode(t)[:-1]
                        input_ids += token_ids
                        input_mantissa += [m]*len(token_ids)
                        input_exponent += [e]*len(token_ids)
                    #Add an end token
                    input_ids.append(1)
                    input_mantissa.append(0.0)
                    input_exponent.append(0.0)

            #adding the ocr_bbox position embedding
            elif self.args.ocr_position_encoding == "ocr_bbox":
                
                ocr_locations = json.loads(self.ocr_bboxes[idx])
                assert len(ocr_tags) == len(ocr_locations), "length does not match: ocr: {} vs bbox: {}".format(len(ocr_tags), len(ocr_locations))
                if len(ocr_tags) > self.ocr_threds:
                    ocr_tags = ocr_tags[:self.ocr_threds]
                    ocr_locations = ocr_locations[:self.ocr_threds]
                # print(ocr_tags)
                # print(ocr_bboxes)
                if not self.args.num_modeling:
                    ocr_ids = []
                    ocr_bboxes = []
                    #How will the OCR Tokenization Differ from directly tokenize the string
                    for current_ocr, current_ocr_bbox in zip(ocr_tags, ocr_locations):
                        current_ocr_ids = self.tokenizer.encode(current_ocr+" ")
                        ocr_ids += current_ocr_ids
                        ocr_bboxes += [current_ocr_bbox]*len(current_ocr_ids)
                    
                    # print(ocr_tags)
                    # print(ocr_ids)
                    assert len(ocr_ids) == len(ocr_bboxes), "length does not match: ocr: {} vs bbox: {}".format(len(ocr_tags), len(ocr_bboxes))
                    #load the ocr ids and ocr_bboxes into a separate feature
                    input_ids = self.tokenizer.encode(f'vqa: {sent}', max_length=30, truncation=True) #Add a short input_ids
                else:
                    ocr_ids = []
                    ocr_bboxes = []

                    ocr_mantissa = []
                    ocr_exponent = []
                    #How will the OCR Tokenization Differ from directly tokenize the string
                    for current_ocr, current_ocr_bbox, current_ocr_nums in zip(ocr_tags, ocr_locations, ocr_nums):
                        current_ocr_ids = self.tokenizer.encode(current_ocr+" ")[:-1]
                        ocr_ids += current_ocr_ids
                        ocr_bboxes += [current_ocr_bbox]*len(current_ocr_ids)
                        #update the ocr_mantissa information
                        ocr_mantissa += [current_ocr_nums[0]]*len(current_ocr_ids)
                        ocr_exponent += [current_ocr_nums[1]]*len(current_ocr_ids)
                    ocr_ids.append(1)
                    ocr_mantissa.append(0.0)
                    ocr_exponent.append(0.0)
                    ocr_bboxes.append([0.0,0.0,0.0,0.0])

                    assert len(ocr_ids) == len(ocr_exponent), "length does not match: ocr_ids: {} vs ocr_exponent: {}".format(len(ocr_ids), len(ocr_exponent))
                    
                    
                    # print(ocr_ids)
                    # if len(ocr_ids) != len(ocr_bboxes):
                    #     print(ocr_ids)
                    #     print(ocr_bboxes)
                    #     print(len(ocr_ids))
                    #     print(len(ocr_bboxes))
                    assert len(ocr_ids) == len(ocr_bboxes), "length does not match: ocr: {} vs bbox: {}".format(len(ocr_tags), len(ocr_bboxes))
                    
                    #load the ocr ids and ocr_bboxes into a separate feature
                    #input_ids = self.tokenizer.encode(f'vqa: {sent}', max_length=30, truncation=True) #Add a short input_ids
                    input_tokens = ['vqa: '] + sent.split(" ")
                    input_mantissa_tokens = [0.0] + [x[0] for x in sent_nums]
                    input_exponent_tokens = [0.0] + [x[1] for x in sent_nums]
                    input_ids = []
                    input_mantissa = []
                    input_exponent = []
                    for t,m,e in zip(input_tokens, input_mantissa_tokens, input_exponent_tokens):
                        token_ids = self.tokenizer.encode(t)[:-1]
                        input_ids += token_ids
                        input_mantissa += [m]*len(token_ids)
                        input_exponent += [e]*len(token_ids)
                    #Add an end token
                    input_ids.append(1)
                    input_mantissa.append(0.0)
                    input_exponent.append(0.0)
                    
                    # print(sent)
                    # print(input_ids)
                    # print(input_mantissa)
                    # print(input_exponent)
                #load the vis_tags
                out_dict['visocrs'] = torch.LongTensor(ocr_ids)
                #normalize the ocr_bboxes
                ocr_bboxes = np.array(ocr_bboxes)
                ocr_bboxes[:, (0, 2)] /= img_w
                ocr_bboxes[:, (1, 3)] /= img_h
                np.testing.assert_array_less(ocr_bboxes, 1+1e-5)
                # np.testing.assert_array_less(boxes, 1+5e-2)
                np.testing.assert_array_less(-ocr_bboxes, 0+1e-5)
                
                ocr_bboxes = torch.from_numpy(ocr_bboxes)
                out_dict['visocrs_bboxes'] = ocr_bboxes

                #get the length
                out_dict['visocrs_length'] = len(ocr_ids)
            else:
                ("invalid style is captured")
                exit()
                #print(ocr_bboxes[0])
                
                #tag_ids = []
                # tag_bboxes = []
                # for tag,tag_bbox in zip(ocr_tags, ocr_bboxes):
                #     tag_id = self.tokenizer.encode(tag)
                #     tag_bbox = 
                #     tag_ids += tag_id
                #input_ids = self.tokenizer.encode(f'vqa: {sent}', max_length=400, truncation=True)
                #exit()
        elif self.args.table_flat:
            #load the table
            table_path = str(self.chartqa_dir.joinpath(f"{source}/tables/{img_id}.csv"))
            table_data = pd.read_csv(table_path)
            table_column_head = [col for col in table_data.columns]
            if table_column_head[0] == "<random_legend>":
                table_flat = table_data.to_numpy().flatten().tolist()
            else:
                table_flat = table_column_head + table_data.to_numpy().flatten().tolist()
            table_flat_string = ' '.join([str(x) for x in table_flat])
            # table_flat_string = self.full_tables[idx]
            input_ids = self.tokenizer.encode(f'vqa: {sent} {table_flat_string}', max_length=400, truncation=True)
        else:
            input_ids = self.tokenizer.encode(f'vqa: {sent}', max_length=400, truncation=True) #changed from 20 to 400


        # question_id = datum['question_id']
        question_id = self.questions_ids[idx]
        out_dict['question_id'] = question_id


        out_dict['sent'] = sent
        # if self.args.scif_num:
        #     out_dict['sent'] = scifnum_sent(sent)

        out_dict['input_ids'] = torch.LongTensor(input_ids)
        out_dict['input_length'] = len(input_ids)
        # out_dict['target_ids'] = torch.LongTensor(target_ids)
        # out_dict['target_length'] = len(target_ids)

        # if 'is_topk_optimal' in datum:
        #     out_dict['is_topk_optimal'] = datum['is_topk_optimal']
        if self.args.num_modeling:
            out_dict['input_mantissa'] = torch.FloatTensor(input_mantissa)
            out_dict['input_exponent'] = torch.FloatTensor(input_exponent)
            if self.args.ocr_position_encoding == 'ocr_bbox':
                out_dict['ocr_mantissa'] = torch.FloatTensor(ocr_mantissa)
                out_dict['ocr_exponent'] = torch.FloatTensor(ocr_exponent)

        if self.outputs is not None:

            if self.args.ocr_copy:
                label = self.ocr_outputs[idx]
            else:
                label = self.outputs[idx]
            

            out_dict['label'] = label

            # 3129 topk answers
            if self.args.classifier:
                target = torch.zeros(self.raw_dataset.num_answers)
                for ans, score in label.items():
                    target[self.raw_dataset.ans2label[ans]] = score
                out_dict['target'] = target

            elif self.args.raw_label:

                # 10 raw answers
                # ex) 'answers': [{'answer': 'net', 'answer_confidence': 'maybe', 'answer_id': 1},
                #     {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 2},
                #     {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 3},
                #     {'answer': 'netting', 'answer_confidence': 'yes', 'answer_id': 4},
                #     {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 5},
                #     {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 6},
                #     {'answer': 'mesh', 'answer_confidence': 'maybe', 'answer_id': 7},
                #     {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 8},
                #     {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 9},
                #     {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 10}],

                # answers = datum['answers']
                # answer = random.choice(answers)['answer']

                # if self.args.answer_normalize:
                #     answer = self.answer_normalizer.normalize_answer(answer)

                # score = int(len(answers) > 0)
                if self.args.ocr_copy:
                    answer = str(self.ocr_outputs[idx])
                else:
                    answer = str(self.outputs[idx])
                #print(answer)
                if self.args.scif_num:
                    answer = scifnum_sent(answer)

                if self.args.num_modeling:
                    #convert the sent and c_ocrs's numerical value to ['NUM']
                    answer, answer_nums = numerify_sent(answer, input_value=False)


                
                #print(answer)
                score = 1
                #print(answer)
                out_dict['answer'] = answer
                out_dict['score'] = score
                out_dict['all_answers'] = [answer]
                
                if not self.args.num_modeling:
                    target_ids = self.tokenizer.encode(answer, max_length=100, truncation=True) #changed 10 to 100
                else:
                    target_ids = []
                    target_mantissa = []
                    target_exponent = []
                    assert len(answer.split(" ")) == len(answer_nums), "answer is {} and answer_numis {}".format(answer, answer_nums)
                    for token, target_num in zip(answer.split(), answer_nums):
                        token_ids = self.tokenizer.encode(answer, max_length=100, truncation=True)[:-1]
                        target_ids += token_ids
                        target_mantissa += [target_num[0]]*len(token_ids)
                        target_exponent += [target_num[1]]*len(token_ids)
                    target_ids.append(1)
                    target_mantissa.append(0.0)
                    target_exponent.append(0.0)
                    assert len(target_ids) == len(target_mantissa)
                #print(target_ids)
                out_dict['target_ids'] = torch.LongTensor(target_ids)
                out_dict['target_length'] = len(target_ids)
                if self.args.num_modeling:
                    out_dict['target_mantissa'] = torch.FloatTensor(target_mantissa)
                    out_dict['target_exponent'] = torch.FloatTensor(target_exponent)
            else:
                # https://github.com/airsplay/lxmert/blob/master/src/pretrain/lxmert_pretrain.py#L191

                answers = []
                scores = []
                for a, s in label.items():
                    answers.append(a)
                    scores.append(s)

                score_sum = sum(scores)

                if score_sum == 0:
                    answer = ''
                    score = 0.
                else:
                    prob = [score / score_sum for score in scores]
                    choice = np.random.multinomial(1, prob).argmax()
                    answer = answers[choice]
                    score = scores[choice]
                    assert len(answer) > 0, (sent, label, choice, answer)

                out_dict['answer'] = answer
                out_dict['score'] = score
                out_dict['all_answers'] = answers


                target_ids = self.tokenizer.encode(answer, max_length=10, truncation=True)

                out_dict['target_ids'] = torch.LongTensor(target_ids)
                out_dict['target_length'] = len(target_ids)

        return out_dict


    def collate_fn(self, batch):
        batch_entry = {}

        args = batch[0]['args']

        B = len(batch)

        S_W_L = max(entry['input_length'] for entry in batch)
        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        
        if self.args.num_modeling:
            input_mantissa = torch.zeros(B, S_W_L, dtype=torch.float)
            input_exponent = torch.zeros(B, S_W_L, dtype=torch.float)

        if args.use_vision:
            if self.args.visfeat_type == "vlt5":
                V_L = len(batch[0]['boxes'])
            else:
                V_L = 36 #For padding purpose
            feat_dim = batch[0]['vis_feats'].shape[-1]

            boxes = torch.zeros(B, V_L, 4, dtype=torch.float)
            vis_feats = torch.zeros(B, V_L, feat_dim, dtype=torch.float)
            
        #include the visual_ocrs, visual_ocr_bbox
        if self.args.ocr_position_encoding == "ocr_bbox" and self.args.ocr_tags:
            V_W_L = max(entry['visocrs_length'] for entry in batch)
            ocr_ids = torch.ones(B, V_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

            
            ocr_bboxes = torch.zeros(B, V_W_L, 4, dtype=torch.float)

            if self.args.num_modeling:
                ocr_mantissa = torch.zeros(B, V_W_L, dtype=torch.float)
                ocr_exponent = torch.zeros(B, V_W_L, dtype=torch.float)

            #Get rid of the extra padding
            # S_W_L = max(entry['input_length'] + entry['visocrs_length'] for entry in batch)
            # input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

            #get ocr_bbox
            # ocr_bboxes = torch.zeros(B, S_W_L, 4, dtype=torch.float)
            # ocr_bboxes = 
        # else:
        #     S_W_L = max(entry['input_length'] for entry in batch)
        #     input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        #     visocrs = 
        if 'target' in batch[0]:
            # targets = []
            targets = torch.zeros(B, len(batch[0]['target']), dtype=torch.float)
        if 'target_ids' in batch[0]:
            T_W_L = max(entry['target_length'] for entry in batch)
            target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
            if self.args.num_modeling:
                target_mantissa = torch.zeros(B, T_W_L, dtype=torch.float)
                target_exponent = torch.zeros(B, T_W_L, dtype=torch.float)


        sentences = []
        question_ids = []
        answers = []
        all_answers = []
        img_ids = []
        img_paths = []
        labels = []
        scores = []
        is_topk_optimal = []

        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']
            
            if self.args.num_modeling:
                input_mantissa[i, :entry['input_length']] = entry['input_mantissa']
                input_exponent[i, :entry['input_length']] = entry['input_exponent']

            if args.use_vision:
                if self.args.visfeat_type == "vlt5":
                    boxes[i] += entry['boxes']
                    vis_feats[i] += entry['vis_feats']
                else:
                    boxes[i, :len(entry['boxes']), :] += entry['boxes']
                    vis_feats[i, :len(entry['vis_feats']), :] += entry['vis_feats']
                # img_ids.append(entry['img_id'])
                # img_paths.append(entry['img_path'])
            if self.args.ocr_position_encoding == "ocr_bbox" and self.args.ocr_tags:
                #fill in the ocr_ids
                ocr_ids[i, :entry['visocrs_length']] = entry['visocrs']
                ocr_bboxes[i, :len(entry['visocrs_bboxes']), :] += entry['visocrs_bboxes']

                # input_ids[i, entry['input_length']:entry['input_length']+entry['visocrs_length']] = entry['visocrs']
                if self.args.num_modeling:
                    ocr_mantissa[i, :entry['visocrs_length']] = entry['ocr_mantissa']
                    ocr_exponent[i, :entry['visocrs_length']] = entry['ocr_exponent']

            if 'target_ids' in entry:
                target_ids[i, :entry['target_length']] = entry['target_ids']
                if self.args.num_modeling:
                    target_mantissa[i, :entry['target_length']] = entry['target_mantissa']
                    target_exponent[i, :entry['target_length']] = entry['target_exponent']


            if 'target' in entry:
                targets[i] += entry['target']
                # targets.append(entry['target'])

            sentences.append(entry['sent'])
            question_ids.append(entry['question_id'])
            if 'answer' in entry:
                answers.append(entry['answer'])
            if 'all_answers' in entry:
                all_answers.append(entry['all_answers'])
            if 'score' in entry:
                scores.append(entry['score'])

            if 'label' in entry:
                labels.append(entry['label'])

            if 'is_topk_optimal' in entry:
                is_topk_optimal.append(entry['is_topk_optimal'])

        batch_entry['input_ids'] = input_ids
        if self.args.num_modeling:
            batch_entry['input_mantissa'] = input_mantissa.unsqueeze(-1)
            batch_entry['input_exponent'] = input_exponent.unsqueeze(-1)

        if 'target_ids' in batch[0]:
            word_mask = target_ids != self.tokenizer.pad_token_id
            target_ids[~word_mask] = -100
            batch_entry['target_ids'] = target_ids

            if self.args.num_modeling:
                target_numerical_mask = (target_mantissa > 0.0).float()

                batch_entry['target_mantissa'] = target_mantissa.unsqueeze(-1)
                batch_entry['target_exponent'] = target_exponent.unsqueeze(-1)
                batch_entry['target_numericals_mask'] = target_numerical_mask.unsqueeze(-1)

        if 'target' in batch[0]:
            # targets = torch.stack(targets, dim=0)
            batch_entry['targets'] = targets

        if args.use_vision:
            batch_entry['boxes'] = boxes
            batch_entry['vis_feats'] = vis_feats
            # batch_entry['img_id'] = img_ids
            # batch_entry['img_paths'] = img_paths

        if self.args.ocr_position_encoding == "ocr_bbox" and self.args.ocr_tags:
            batch_entry['visocrs'] = ocr_ids
            batch_entry['visocrs_bboxes'] = ocr_bboxes
            if self.args.num_modeling:
                batch_entry['visocr_mantissa'] = ocr_mantissa.unsqueeze(-1)
                batch_entry['visocr_exponent'] = ocr_exponent.unsqueeze(-1)

        batch_entry['sent'] = sentences
        batch_entry['question_ids'] = question_ids
        batch_entry['answers'] = answers
        batch_entry['all_answers'] = all_answers
        batch_entry['scores'] = torch.FloatTensor(scores)
        batch_entry['labels'] = labels

        batch_entry['args'] = args
        batch_entry['task'] = 'vqa'

        return batch_entry


def get_loader(args, split='train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0, topk=-1):

    verbose = (gpu == 0)

    # _dset = VQADataset(split, verbose)

    dataset = ChartQAFineTuneDataset(
        src_folder=args.src_folder+split+"/",
        split=split,
        # raw_dataset=_dset,
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
        loader.evaluator = ChartQAEvaluator(args.src_folder+split+"/")

    loader.task = 'vqa'

    return loader


class ChartQADataset:
    """
    Need to change this to chartQA style. 
    A VQA data example in json file:
        {
            "answer_type": "other",
            "img_id": "COCO_train2014_000000458752",
            "label": {
                "net": 1
            },
            "question_id": 458752000,
            "question_type": "what is this",
            "sent": "What is this photo taken looking through?"
        }
    """

    def __init__(self, splits: str, verbose=True):
        self.name = splits
        self.splits = splits.split(',')

        with open(dataset_dir.joinpath(f'vqa/v2_mscoco_train2014_annotations.json')) as f:
            train2014_data = json.load(f)
        with open(dataset_dir.joinpath(f'vqa/v2_mscoco_val2014_annotations.json')) as f:
            val2014_data = json.load(f)
        train2014_id2datum = {}
        for datum in train2014_data['annotations']:
            qid = datum['question_id']
            train2014_id2datum[qid] = datum
        val2014_id2datum = {}
        for datum in val2014_data['annotations']:
            qid = datum['question_id']
            val2014_id2datum[qid] = datum
        self.id2datum_gt = {**train2014_id2datum, **val2014_id2datum}

        # Loading datasets
        self.data = []
        for split in self.splits:
            self.data.extend(
                json.load(open(vqa_dir.joinpath("%s.json" % split))))

        if verbose:
            print("Load %d data from split(s) %s." %
                  (len(self.data), self.name))

        # Convert list to dict (for evaluation)
        self.id2datum = {
            datum['question_id']: datum
            for datum in self.data
        }

        # Topk Answers
        self.ans2label = json.load(
            open(vqa_dir.joinpath("trainval_ans2label.json")))
        self.label2ans = json.load(
            open(vqa_dir.joinpath("trainval_label2ans.json")))
        assert len(self.ans2label) == len(self.label2ans)

        if verbose:
            print('# Answers:', len(self.ans2label))

    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)


class ChartQAEvaluator:
    def __init__(self, src_folder):
        #self.dataset = dataset
        # Loading datasets to data
        instances = pd.read_csv(src_folder + "data.csv")
        self.instances = instances
        self.inputs = instances["Input"].values
        self.ocrs = instances["OCR"].values
        self.outputs = None
        if "Output" in instances:
            self.outputs = instances["Output"].values
        self.images_indices = instances['Image Index'].values
        self.questions_ids = instances['Question ID'].values
        self.src_folder = src_folder

        self.qidtoans = {}
        self.qidtoocrs = {}
        for qid, ans in zip(self.questions_ids, self.outputs):
            self.qidtoans[qid] = ans

        self.qidtoocrs = {}
        for qid, ocr in zip(self.questions_ids, self.ocrs):
            self.qidtoocrs[qid] = ast.literal_eval(ocr)

        """https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py"""

        self.contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't", \
							 "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't", \
							 "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've", \
							 "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", \
							 "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's", \
							 "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", \
							 "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't", \
							 "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've", \
							 "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", \
							 "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", \
							 "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've", \
							 "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've", \
							 "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd", "thered've": "there'd've", \
							 "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", \
							 "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", \
							 "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're", \
							 "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've", \
							 "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", \
							 "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", \
							 "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've", \
							 "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", \
							 "youll": "you'll", "youre": "you're", "youve": "you've"}

        self.manualMap    = { 'none': '0',
							  'zero': '0',
							  'one': '1',
							  'two': '2',
							  'three': '3',
							  'four': '4',
							  'five': '5',
							  'six': '6',
							  'seven': '7',
							  'eight': '8',
							  'nine': '9',
							  'ten': '10'
							}

        self.articles     = ['a',
							 'an',
							 'the'
							]

        self.periodStrip  = re.compile("(?!<=\d)(\.)(?!\d)")
        self.commaStrip   = re.compile("(\d)(\,)(\d)")
        self.punct        = [';', r"/", '[', ']', '"', '{', '}',
							 '(', ')', '=', '+', '\\', '_', '-',
							 '>', '<', '@', '`', ',', '?', '!']

        self.n = 2

    def evaluate(self, quesid2ans: dict):
        score = 0.
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['label']
            if ans in label:
                score += label[ans]
        return score / len(quesid2ans)

    def dump_result(self, quesid2ans: dict, path):
        """
        Dump results to a json file, which could be submitted to the VQA online evaluation.
        VQA json file submission requirement:
            results = [result]
            result = {
                "question_id": int,
                "answer": str
            }
        :param quesid2ans: dict of quesid --> ans
        :param path: The desired path of saved file.
        """
        with open(path, 'w') as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                result.append({
                    'question_id': ques_id,
                    'answer': ans
                })
            json.dump(result, f, indent=4, sort_keys=True)

    def relaxed_measure_fix(self, gt_ans, resAns):
        try:
            gt_ans_num = float(gt_ans)
            resAns_num = float(resAns)
        except:
            return str(gt_ans).strip() == str(resAns).strip()
        if "." in resAns:
            return (abs(resAns_num-gt_ans_num)+1e-12)/(gt_ans_num+1e-12) <= 0.05
        else:
            return str(gt_ans).strip() == str(resAns).strip()

    # def relaxed_measure(self, gt_ans, resAns):
    #     try:
    #         gt_ans_num = float(gt_ans)
    #         resAns_num = float(resAns)
    #     except:
    #         return str(gt_ans).strip() == str(resAns).strip()
    #     return (abs(resAns_num-gt_ans_num)+1e-12)/(gt_ans_num+1e-12) <= 0.05
    
    def relaxed_measure_scifnum_ocrcopy(self, gt_ans, resAns, gt_ocrs):
        #convert scifnum into the gt
        if exp_symbol in resAns:
            resAns = resAns.replace(exp_symbol, 'e')
            #print(v)
            #convert v into a string
            new_resAns_list = []
            for w in resAns.split(' '):
                try:
                    new_w = str(int(float(w)))
                except:
                    try:
                        new_w = str(float(w))
                    except:
                        new_w = w
                new_resAns_list.append(new_w)
            resAns = ' '.join(new_resAns_list)
                
        #convert the ocr_id into its original text
        if re.search(u"ocr_\d+>", resAns):
            match = re.findall(u"ocr_\d+>", resAns)
            for m in match:
                matched_index = int(re.findall(u'\d+',m)[0])
                try:
                    matched_ocr = gt_ocrs[matched_index]
                    resAns = resAns.replace(m, matched_ocr)
                except:
                    #list index out of range
                    continue

        #At this point resAns should be returned to actual predicted answer
        # print(gt_ans)
        # print(resAns)
        # print("\n")
        try:
            resAns_num = float(resAns)
            gt_ans_num = float(gt_ans)
        except:
            return str(resAns).strip() == str(gt_ans).strip()
        
        return (abs(resAns_num-gt_ans_num)+1e-12)/(gt_ans_num +1e-12) <= 0.05

    def relaxed_measure_scifnum(self, gt_ans, resAns, gt_ocrs):
        #convert scifnum into the gt
        if exp_symbol in resAns:
            resAns = resAns.replace(exp_symbol, 'e')
            #print(v)
            #convert v into a string
            new_resAns_list = []
            for w in resAns.split(' '):
                try:
                    new_w = str(int(float(w)))
                except:
                    try:
                        new_w = str(float(w))
                    except:
                        new_w = w
                new_resAns_list.append(new_w)
            resAns = ' '.join(new_resAns_list)
                
        #convert the ocr_id into its original text
        if re.search(u"<ocr_extra_id_\d+>", resAns):
            match = re.findall(u"<ocr_extra_id_\d+>", resAns)
            for m in match:
                matched_index = int(re.findall(u'\d+',m)[0])
                try:
                    matched_ocr = gt_ocrs[matched_index]
                    resAns = resAns.replace(m, matched_ocr)
                except:
                    #list index out of range
                    continue

        #At this point resAns should be returned to actual predicted answer
        # print(gt_ans)
        # print(resAns)
        # print("\n")
        try:
            resAns_num = float(resAns)
            gt_ans_num = float(gt_ans)
        except:
            return str(resAns).strip() == str(gt_ans).strip()
        
        return (abs(resAns_num-gt_ans_num)+1e-12)/(gt_ans_num +1e-12) <= 0.05
                
    def evaluate_raw(self, quesid2ans: dict, is_topk_optimal=None):
        """https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py"""

        # gts = self.dataset.id2datum_gt

        self.accuracy     = {}
        self.evalQA       = {}
        self.evalQuesType = {}
        self.evalAnsType  = {}

        accQA = []
        accQuesType = {}
        accAnsType = {}

        # print("Computing accuracy")
        for quesId, resAns in tqdm(quesid2ans.items(), total=len(quesid2ans), ncols=80):

            #quesId = int(quesId)
            quesId = quesId

            gt_ans = self.qidtoans[quesId]
            
            gt_ocrs = self.qidtoocrs[quesId]
            # if str(gt_ans).strip() == str(resAns).strip():
            #     accQA.append(1)
            # else:
            #     accQA.append(0)

            #Process the resAns
            #Added by Mingyang
            #print(resAns)
            removed_special_tokens = ["<pad>", "</s>", "<unk>"]
            for t in removed_special_tokens:
                resAns = resAns.replace(t, "")
            resAns=resAns.strip()

            # print(resAns)
            # print(gt_ans)
            #if self.relaxed_measure_scifnum_ocrcopy(gt_ans, resAns, gt_ocrs):

            if self.relaxed_measure_scifnum_ocrcopy(gt_ans, resAns, gt_ocrs):
                accQA.append(1)
            else:
                accQA.append(0)
                # print("prediction: {}".format(resAns))
                # print("ground_truth: {}".format(gt_ans))

            

            # if is_topk_optimal is None:
            #     pass
            # elif 'is_topk_optimal' in datum:
            #     if datum['is_topk_optimal'] != is_topk_optimal:
            #         continue

            # resAns      = resAns.replace('\n', ' ')
            # resAns      = resAns.replace('\t', ' ')
            # resAns      = resAns.strip()
            # resAns      = self.processPunctuation(resAns)
            # resAns      = self.processDigitArticle(resAns)

            # gtAcc  = []
            # gtAnswers = [ans['answer'] for ans in gts[quesId]['answers']]
            # if len(set(gtAnswers)) > 1:
            #     for ansDic in gts[quesId]['answers']:
            #         ansDic['answer'] = self.processPunctuation(ansDic['answer'])
            # for gtAnsDatum in gts[quesId]['answers']:
            #     otherGTAns = [item for item in gts[quesId]['answers'] if item!=gtAnsDatum]
            #     matchingAns = [item for item in otherGTAns if item['answer']==resAns]
            #     acc = min(1, float(len(matchingAns))/3)
            #     gtAcc.append(acc)
            # quesType    = gts[quesId]['question_type']
            # ansType     = gts[quesId]['answer_type']
            # avgGTAcc = float(sum(gtAcc))/len(gtAcc)
            # accQA.append(avgGTAcc)
            # if quesType not in accQuesType:
            #     accQuesType[quesType] = []
            # accQuesType[quesType].append(avgGTAcc)
            # if ansType not in accAnsType:
            #     accAnsType[ansType] = []
            # accAnsType[ansType].append(avgGTAcc)

            # self.setEvalQA(quesId, avgGTAcc)
            # self.setEvalQuesType(quesId, quesType, avgGTAcc)
            # self.setEvalAnsType(quesId, ansType, avgGTAcc)


        if len(accQA) == 0:
            return {
                'overall': 0,
                'perQuestionType': {},
                'perAnswerType': {}
            }
        else:
            self.setAccuracy(accQA, accQuesType, accAnsType)

        return self.accuracy

    def normalize_answer(self, resAns):
        resAns      = resAns.replace('\n', ' ')
        resAns      = resAns.replace('\t', ' ')
        resAns      = resAns.strip()
        resAns      = self.processPunctuation(resAns)
        resAns      = self.processDigitArticle(resAns)
        resAns = resAns.replace(',', '')
        return resAns

    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + ' ' in inText or ' ' + p in inText) or (re.search(self.commaStrip, inText) != None):
                outText = outText.replace(p, '')
            else:
                outText = outText.replace(p, ' ')
        outText = self.periodStrip.sub("",
                                        outText,
                                        re.UNICODE)
        return outText

    def processDigitArticle(self, inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = self.manualMap.setdefault(word, word)
            if word not in self.articles:
                outText.append(word)
            else:
                pass
        for wordId, word in enumerate(outText):
            if word in self.contractions:
                outText[wordId] = self.contractions[word]
        outText = ' '.join(outText)
        return outText

    def setEvalQA(self, quesId, acc):
        self.evalQA[quesId] = round(100*acc, self.n)

    def setEvalQuesType(self, quesId, quesType, acc):
        if quesType not in self.evalQuesType:
            self.evalQuesType[quesType] = {}
        self.evalQuesType[quesType][quesId] = round(100*acc, self.n)

    def setEvalAnsType(self, quesId, ansType, acc):
        if ansType not in self.evalAnsType:
            self.evalAnsType[ansType] = {}
        self.evalAnsType[ansType][quesId] = round(100*acc, self.n)

    def setAccuracy(self, accQA, accQuesType, accAnsType):
        self.accuracy['overall'] = round(100*float(sum(accQA))/len(accQA), self.n)
        # self.accuracy['perQuestionType'] = {quesType: round(100*float(sum(accQuesType[quesType]))/len(accQuesType[quesType]), self.n) for quesType in accQuesType}
        # self.accuracy['perAnswerType']   = {ansType:  round(100*float(sum(accAnsType[ansType]))/len(accAnsType[ansType]), self.n) for ansType in accAnsType}

