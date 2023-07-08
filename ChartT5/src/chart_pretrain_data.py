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


from transformers import T5Tokenizer, BartTokenizer, T5TokenizerFast, BartTokenizerFast
from tokenization import VLT5Tokenizer, VLT5TokenizerFast

import preprocess
from qa_answer_table import AnswerTable
import pandas as pd
import re

project_dir = Path(__file__).resolve().parent.parent # VLT5
workspace_dir = project_dir.parent
dataset_dir = workspace_dir.joinpath('datasets/').resolve()
coco_dir = dataset_dir.joinpath('COCO')
vg_dir = dataset_dir.joinpath('VG')
coco_img_dir = coco_dir.joinpath('images/')

chartsum_dir = Path("/dvmm-filer2/projects/mingyang/semafor/chart_pretrain_compress")
#chartsum_feature_dir = chartsum_dir.joinpath("features")
exp_symbol = "[EXP]"

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

def numerify_flattable(caption, input_value=True):
    caption_words = caption.split("|")
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
    assert len(new_caption.split("|")) == len(num_values), "original_caption is: {}, new_caption is: {}".format(caption, new_caption)
    return new_caption, num_values
# Load VG Classes
# vg_classes = []
# with open(vg_dir.joinpath('objects_vocab.txt')) as f:
#     for obj in f.readlines():
#         vg_classes.append(obj.split(',')[0].lower().strip())

# vg_attrs = []
# with open(vg_dir.joinpath('attributes_vocab.txt')) as f:
#     for attr in f.readlines():
#         vg_attrs.append(attr.split(',')[0].lower().strip())

def make_uid(img_id, dset, sent_idx):
    if type(sent_idx) is not int:
        sent_idx = int(sent_idx)
    return "%s_%s_%03d" % (img_id, dset, sent_idx)

def replace_ocr_symbols(origin_string):
    match = re.findall(u"<ocr_\d+>", origin_string)
    new_string = origin_string
    for m in match:
        matched_index = int(re.findall(u'\d+',m)[0])
        new_string = new_string.replace(m, "<ocr_extra_id_{}>".format(matched_index))
    return new_string

def get_datum(datum):
    data = []
    # _sents = []

    args = datum['args']

    if datum['is_train']:
        # if 'COCO_train2014' in datum['img_id']:
        #     img_source = 'mscoco_resplit_train_train2014'
        # elif 'COCO_val2014' in datum['img_id']:
        #     img_source = 'mscoco_resplit_train_val2014'
        # else:
        #     img_source = 'vgnococo'
        img_source = 'chartsum_train'
    else:
        img_source = 'chartsum_val'

    # for text_source, sents in datum['sentf'].items():
    #     if datum['caption_only']:
    #         if text_source not in ['mscoco', 'vg']:
    #             continue

    #     if args.coco_only:
    #         if text_source != 'mscoco':
    #             continue

    # if args.table_mlm and not datum['sent']:
    #     #load the table as flat string to the sentence:
    #     img_id = datum['img_id']
    #     datum_split = datum['split']

    #     #dataset_name = datum['text_source'].split('_')[0]
    #     if "chartsum" in datum['text_source']:
    #         data_name = "statista"
    #     elif "dvqa" in datum['text_source']:
    #         data_name = "dvqa"
    #     elif "figqa" in datum['text_source']:
    #         data_name = "figureqa"
    #     elif "plotqa" in datum['text_source']:
    #         data_name = "plotqa"
    #     #Hacky Way
    #     if "figqa" in datum['text_source']:
    #         table_path = str(chartsum_dir.joinpath(f"{data_name}/train/tables/{img_id}.csv"))
    #     else:
    #         table_path = str(chartsum_dir.joinpath(f"{data_name}/{datum['split']}/tables/{img_id}.csv"))

    #     #table_path = str(chartsum_dir.joinpath(f'{dataset_name}/{datum_split}/tables/{img_id}.csv'))
    #     table_data = pd.read_csv(table_path)
    #     table_flat = table_data.to_numpy().flatten().tolist()
    #     table_string = ' '.join([str(x) for x in table_flat])
    #     datum['sent'] = table_string

    labels = None
    # if datum['qa'] and text_source in datum['labelf']:
    #     labels = datum['labelf'][text_source]

    img_id = datum['img_id']

    # for sent_idx, sent in enumerate(sents):
    sent = datum['sent']
    sent_idx = datum['sent_id']
    text_source = datum['text_source']
    #print(datum)
    ocrs = datum['OCR']
    
    #added by MIngyang
    new_datum = {
        'uid': make_uid(img_id, text_source, sent_idx),
        'img_id': img_id,
        'img_source': img_source,
        'sent': sent,
        'text_source': text_source,
        'ocr': ocrs, #Added by Mingyang
        'split': datum['split']
    }

    if args.scif_num:
        #convert numerical value to scif representation:
        new_sent = scifnum_sent(new_datum['sent'])
        new_ocr = scifnum_ocr(new_datum['ocr'])
        #update the sent and ocrs
        new_datum['sent'] = new_sent
        new_datum['ocr'] = new_ocr

    if args.ocr_position_encoding == "ocr_bbox":
        #print(datum.keys())
        assert datum.get('OCR_bbox', None) is not None
        new_datum ['ocr_bbox'] = datum['OCR_bbox']

    if args.ocr_copy:
        new_datum['ocr_sent'] = datum['OCR_sent'] #Added by Mingyang
        new_datum['ocr_symbols'] = datum['OCR_Symbols'] #Added by Mingyang

    if datum.get('col_head', None) is not None:
        col_heads = datum['col_head']
        new_datum['col_head'] = col_heads
        
    if datum.get('row_head', None) is not None:
        row_heads = datum['row_head']
        new_datum['row_head'] = row_heads

    # Task: Language modeling
    if datum['lm'] and labels is None and datum['sent']:
        new_datum = deepcopy(new_datum)
        new_datum['task'] = 'lm'
        new_datum['label'] = None
        data.append(new_datum)
    
    if datum['table_lm'] and datum['row_head']:
        new_datum = deepcopy(new_datum)
        new_datum['task'] = 'table_lm'
        new_datum['label'] = None
        data.append(new_datum)

    if datum['table_lmh'] and datum['row_head']:
        new_datum = deepcopy(new_datum)
        new_datum['task'] = 'table_lmh'
        new_datum['label'] = None
        data.append(new_datum)


    if datum['table_lmd'] and datum['row_head']:
        new_datum = deepcopy(new_datum)
        new_datum['task'] = 'table_lmd'
        new_datum['label'] = None
        data.append(new_datum)

    # Task: Image captioning
    if datum['caption'] and datum['sent']:
        new_datum = deepcopy(new_datum)
        new_datum['task'] = 'caption'
        new_datum['label'] = None
        data.append(new_datum)

    #     if args.caption_cocoonly:
    #         if text_source == 'mscoco':
    #             new_datum = deepcopy(new_datum)
    #             new_datum['task'] = 'caption'
    #             new_datum['label'] = None
    #             data.append(new_datum)
    #     else:
    #         if text_source in ['mscoco', 'vg']:
    #             new_datum = deepcopy(new_datum)
    #             new_datum['task'] = 'caption'
    #             new_datum['label'] = None
    #             data.append(new_datum)

    # Task: Image-text matching
    if datum['itm']:
        new_datum = deepcopy(new_datum)
        new_datum['task'] = 'itm'
        new_datum['label'] = None
        data.append(new_datum)

        # _sents.append(sent)
    #Task: tdp
    if datum['tdp'] and datum['row_head']:
        new_datum = deepcopy(new_datum)
        new_datum['task'] = 'tdp'
        new_datum['label'] = None
        data.append(new_datum)

    if datum['tsp'] and datum['row_head']:
        new_datum = deepcopy(new_datum)
        new_datum['task'] = 'tsp'
        new_datum['label'] = None
        data.append(new_datum)

    for d in data:
        assert 'task' in d

    return data
###################Create Own Table Dataset#############
class ChartTablePretrainDataset(Dataset):
    def __init__(self, split='chartsum_train', rank=-1, topk=-1, verbose=True, args=None, is_train=True):

        self.topk = topk
        self.verbose = verbose
        self.args = args


        # Loading datasets to data
        self.sources = split.split(',')
        if self.verbose:
            print('Data sources: ', self.sources)

        # Answer Table from LXMERT (Could be removed)
        # self.answer_table = AnswerTable()
        # if self.verbose:
        #     print("Load an answer table of size %d." % (len(self.answer_table.ans2id_map())))

        self.img_ids_to_source = {}

        losses = args.losses.split(',')

        #No need to consider different sources, we only have one unique source. 
        #Need to modify to adapt to this change. Prepare a list of dictionaries.
        data = []
        for img_source in self.sources:
            # data_info_path = chartsum_dir.joinpath(f'chart_pretrain_annotation_tables_gtocrs_dfp_ocrcopy_special.json')
            data_info_path = chartsum_dir.joinpath(f'chart_pretrain_annotation_tables_gtocrs_dfp_ocrcopy.json')
            #data_info_path = chartsum_dir.joinpath(f'chartsum_table_annotation.json')
            with open(data_info_path) as f:
                _data = json.load(f)
                if self.verbose:
                    print(f"Loaded {len(_data)} data from", img_source)
                # source_img_ids.append([d['img_id'] for d in _data])
                for datum in _data:
                    if datum['split'] == split:
                        self.img_ids_to_source[datum['img_id']] = img_source
                        # datum['img_source'] = img_source
                        datum['args'] = args
                        datum['is_train'] = is_train
                        # datum['caption_only'] = args.caption_only

                        datum['lm'] = 'lm' in losses
                        datum['itm'] = 'itm' in losses
                        datum['tdp'] = 'tdp' in losses #Added by Mingyang for table data prediction
                        # datum['caption'] = 'caption' in losses
                        datum['tsp'] = 'tsp' in losses
                        #add the table language model
                        datum ['table_lm'] = 'table_lm' in losses
                        datum ['table_lmh'] = 'table_lmh' in losses
                        datum ['table_lmd'] = 'table_lmd' in losses
                        datum ['caption'] = 'caption' in losses
                        datum['backbone'] = self.args.backbone

                        data.append(datum)
        del _data

        if self.verbose:
            print("# images:", len(data))

        # if self.topk > 0:
        #     data = data[:self.topk]
        #     if self.verbose:
        #         print(f"Use only {self.topk} data")

        # if 'qa' in args.losses:
        #     self.evaluator = QAEvaluator(data)

        with Pool(8) as pool:
            if self.verbose:
                data = [datum for _data in tqdm(
                    pool.imap(get_datum, data), total=len(data), ncols=100, desc="Creating pretrainig data examples") for datum in _data]
            else:
                data = [datum for _data in pool.imap(
                    get_datum, data) for datum in _data]

        # if self.args.itm_cocoonly:
        #     caption_sources = ['mscoco']
        # else:
        #     caption_sources = ['mscoco', 'vg']
        self.data_captions = [datum for datum in data if datum['split'] == split] #Load thee data that contains cerrtain captions
        self.n_data_captions = len(self.data_captions)

        if self.verbose:
            print('# itm data:', self.n_data_captions)

        self.data = data
        # print(self.data[0].keys())
        self.n_data = len(self.data)

        if self.verbose and is_train:
            from collections import Counter
            task_counter = Counter()
            for datum in data:
                try:
                    task_counter.update([datum['task']])
                except KeyError:
                    print(datum)
                    exit()

            print(task_counter)
            for k, v in task_counter.items():
                print(k, f'{v/len(data)*100:.1f}%')

        if self.verbose:
            print("# examples:", len(data))
        
        if self.args.visfeat_type == "vlt5":
            self.source_to_h5 = {
                'chartsum_train': chartsum_feature_dir.joinpath('train_boxes36.h5'),
                'chartsum_val': chartsum_feature_dir.joinpath('val_boxes36.h5'),

            }
        else:
            self.source_to_h5 = {
                'chartsum_train': chartsum_dir.joinpath('statista/features/train_chart_elements.h5'),
                'chartsum_val': chartsum_dir.joinpath('statista/features/val_chart_elements.h5'),
                'dvqa_train': chartsum_dir.joinpath('dvqa/features/train_chart_elements.h5'),
                'dvqa_val': chartsum_dir.joinpath('dvqa/features/train_val_chart_elements.h5'),
                'dvqa_test': chartsum_dir.joinpath('dvqa/features/val_chart_elements.h5'),
                'figqa_train': chartsum_dir.joinpath('figureqa/features/train_chart_elements.h5'),
                'plotqa_train': chartsum_dir.joinpath('plotqa/features/train_chart_elements.h5'),
                'plotqa_val': chartsum_dir.joinpath('plotqa/features/val_chart_elements.h5'),
                'scicap_train': chartsum_dir.joinpath('scicap/features/train_chart_elements.h5'),
                'scicap_val': chartsum_dir.joinpath('scicap/features/val_chart_elements.h5'),
            }

        self.n_boxes = args.n_boxes

        if 't5' in self.args.backbone:
            if self.args.use_vision:
                # self.tokenizer = VLT5Tokenizer.from_pretrained(
                #     args.backbone, do_lower_case=args.do_lower_case)
                self.tokenizer = VLT5TokenizerFast.from_pretrained(
                    args.backbone, do_lower_case=args.do_lower_case)
            else:
                # self.tokenizer = T5Tokenizer.from_pretrained(
                #     args.backbone, do_lower_case=args.do_lower_case)
                self.tokenizer = T5TokenizerFast.from_pretrained(
                    args.backbone, do_lower_case=args.do_lower_case)
        elif 'bart' in self.args.backbone:
            self.tokenizer = BartTokenizer.from_pretrained(args.backbone)
            additional_special_tokens = [f'<extra_id_{i}>' for i in range(100-1, -1, -1)] + \
                    [f'<vis_extra_id_{i}>' for i in range(100-1, -1, -1)]
            special_tokens_dict = {'additional_special_tokens': additional_special_tokens}
            self.tokenizer.add_special_tokens(special_tokens_dict)
        if self.args.ocr_tags:
            self.ocr_threds = 30


    def __len__(self):
        # return len(self.data)
        return self.n_data

    def __getitem__(self, idx):
        
        out_dict = {}
        out_dict['args'] = self.args

        datum = self.data[idx]

        uid = datum['uid'] #Created by get_datum function. 
        out_dict['uid'] = uid

        ###### Image ######
        img_id = datum['img_id']
        #source = datum['img_source'] #Add to the get_datum function
        source = datum['text_source']

        f = self.source_to_h5[source]
        if isinstance(f, Path):
            path = self.source_to_h5[source]
            f = h5py.File(path, 'r')
            self.source_to_h5[source] = f

        if 't5' in self.args.backbone:

            #text_source = datum['text_source']
            task = datum['task']

            loss_weight = 1

            # T5 Corrupt span
            if task == 'lm':
                #assert text_source in ["mscoco", 'vg']

                prefix = "span prediction:"
                if self.args.ocr_copy:
                    sent = datum['ocr_sent'] #get the ocr replaced caption
                else:
                    sent = datum['sent'] #get the caption
                source_text, target_text = preprocess.corrupt_spans(
                    sent, mask_ratio=self.args.word_mask_rate, prefix=prefix)
                
                input_tokens = [source_text]
                if self.args.ocr_tags:
                    if self.args.ocr_copy:
                        #replace ocr with symbolic format
                        ocr_s = datum['ocr_symbols']
                        ocr_t = datum['ocr']
                        #Keep both the symbolic representation and original text

                        assert len(ocr_s) == len(ocr_t)
                        ocr_tags = [f"{x}_{y}" for x,y in zip(ocr_s,ocr_t)]
                    else:
                        ocr_tags = datum['ocr']

                    if self.args.ocr_position_encoding == "oscar_style":
                        #Add OCRs to the data_csv
                        # ocr_tags = datum['ocr'] #Need to be Modified
                        #Add an argument to define the threshold
                        if len(ocr_tags) > 60:
                            ocr_tags = ocr_tags[:60]
                        #input_tokens = [source_text]
                        #input_tokens = []
                        input_tokens.append('ocrs:')
                        for ocr_tag in ocr_tags:
                            input_tokens.append(ocr_tag)
                        #Append the source_text to the back
                        #input_tokens.append(source_text)
                        source_text = ' '.join(input_tokens)

            #if task == "table_lmh":
            if "table_lm" in task:
                #prefix = "table span prediction:"
                #sent = datum['sent'] #get the table
                if "chartsum" in datum['text_source']:
                    data_name = "statista"
                elif "dvqa" in datum['text_source']:
                    data_name = "dvqa"
                elif "figqa" in datum['text_source']:
                    data_name = "figureqa"
                elif "plotqa" in datum['text_source']:
                    data_name = "plotqa"
                
                #Hacky Way
                if self.args.ocr_copy:
                    table_folder_name = "ocrcopy_tables"
                else:
                    table_folder_name = "tables"

                if "figqa" in datum['text_source']:
                    table_path = str(chartsum_dir.joinpath(f"{data_name}/train/{table_folder_name}/{img_id}.csv"))
                else:
                    table_path = str(chartsum_dir.joinpath(f"{data_name}/{datum['split']}/{table_folder_name}/{img_id}.csv"))

                #table_path = str(chartsum_dir.joinpath(f'{dataset_name}/{datum_split}/tables/{img_id}.csv'))
                table_data = pd.read_csv(table_path)
                table_column_head = [col for col in table_data.columns]
                if table_column_head[0] == "<random_legend>":
                    table_flat = table_data.to_numpy().flatten().tolist()
                else:
                    table_flat = table_column_head + table_data.to_numpy().flatten().tolist()
                
                # if self.args.ocr_copy:
                #     table_flat = [replace_ocr_symbols(str(x)) for x in table_flat]

                #print(table_flat)
                if self.args.scif_num:
                    new_table_flat = []
                    for x in table_flat:
                        #determine the float or integer value
                        if type(x) is float or type(x) is int:
                            #convert to scientific notification:
                            scif_x = "{:2e}".format(x)
                            scif_x = scif_x.replace('e', exp_symbol)
                            new_table_flat.append(scif_x)
                        else:
                            new_table_flat.append(x)
                    sent = '|'.join([str(x) for x in new_table_flat])
                else:
                    sent = '|'.join([str(x) for x in table_flat])
                
                c_ocrs = datum['ocr']

                #Currently Only Implement in Table Prediction
                # if self.args.num_modeling:
                #     #convert the sent and c_ocrs's numerical value to ['NUM']
                #     sent, sent_nums = numerify_sent(sent)
                #     c_ocrs, ocr_nums = numerify_ocrs(c_ocrs)

                if task == "table_lmh":
                    prefix = "table header prediction:"
                    source_text, target_text = preprocess.table_corrupt_spans(
                        sent, mask_ratio=self.args.table_mask_rate, prefix=prefix, mask_type="word")
                elif task == "table_lmd":
                    prefix = "table value prediction:"
                    source_text, target_text = preprocess.table_corrupt_spans(
                        sent, mask_ratio=self.args.table_mask_rate, prefix=prefix, mask_type="numeric")
                elif task == "table_lm":
                    prefix = random.choice(["table header prediction", "table value prediction"])
                    mask_type = "numeric" if prefix == "table value prediction" else "word"
                    source_text, target_text = preprocess.table_corrupt_spans(
                        sent, mask_ratio=self.args.table_mask_rate, prefix=prefix, mask_type=mask_type)

                    # prefix = "table_prediction"
                    # source_text, target_text = preprocess.table_corrupt_spans(
                    #     sent, mask_ratio=self.args.table_mask_rate, prefix=prefix, mask_strategy="random_masking")

                #print(source_text)
                input_tokens = [source_text]
                if self.args.ocr_tags:
                    #Add the ocr_copy mechanism
                    if self.args.ocr_copy:
                        #replace ocr with symbolic format
                        ocr_s = datum['ocr_symbols']
                        ocr_t = c_ocrs
                        #Keep both the symbolic representation and original text

                        assert len(ocr_s) == len(ocr_t)
                        ocr_tags = [f"{x} {y}" for x,y in zip(ocr_s,ocr_t)]
                    else:
                        ocr_tags = c_ocrs

                    if self.args.ocr_position_encoding == "oscar_style":
                        #Add OCRs to the data_csv
                        #ocr_tags = datum['ocr'] #Need to be Modified
                        #Add an argument to define the threshold
                        # if len(ocr_tags) > 60:
                        #     ocr_tags = ocr_tags[:60]
                        #input_tokens = [source_text]
                        #input_tokens = []
                        input_tokens.append('ocrs:')
                        for ocr_tag in ocr_tags:
                            input_tokens.append(ocr_tag)
                        #Append the source_text to the back
                        #input_tokens.append(source_text)
                        source_text = ' '.join(input_tokens)


                    # input_ids = self.tokenizer.encode(f'ocrs: {ocr_tags} chartqa: {sent}', max_length=400, truncation=True)
    
            elif task == "caption":
                prefix = 'describe image with ocr:'
                #source_text = prefix
                sent = datum['sent']
                target_text = sent
                input_tokens = [prefix]
                if self.args.ocr_tags:
                    if self.args.ocr_position_encoding == "oscar_style":
                        #Add OCRs to the data_csv
                        ocr_tags = datum['ocr'] #Need to be Modified
                        #Add an argument to define the threshold
                        ocr_tokens = []
                        if len(ocr_tags) > 60:
                            ocr_tags = ocr_tags[:60]
                        #input_tokens = [source_text]
                        #ocr_tokens.append('ocrs:')
                        for ocr_tag in ocr_tags:
                            ocr_tokens.append(ocr_tag)
                        #Append the source_text to the back
                        input_tokens = ocr_tokens + input_tokens
                source_text = ' '.join(input_tokens)


            elif task == 'itm':

                # assert text_source in ["mscoco", 'vg']
                is_matched = 1
                sent = datum['sent']
                if random.random() < 0.5:
                    is_matched = 0

                    rand_idx = random.randint(0, self.n_data_captions-1)
                    # rand_idx = int(self.n_data_captions * random.random())

                    other_datum = self.data_captions[rand_idx]
                    # other_datum = self.data[random.randint(0, len(self.data)-1)]
                    while other_datum['img_id'] == img_id:

                        rand_idx = random.randint(0, self.n_data_captions-1)
                        # rand_idx = int(self.n_data_captions * random.random())

                        other_datum = self.data_captions[rand_idx]
                        # other_datum = self.data[random.randint(0, len(self.data)-1)]
                    sent = other_datum['sent']
                
                #Append the Prefix
                prefix = "image text match:"
                source_text = f"{prefix} {sent}"
                
                #Append the tags as Oscar. 
                # if self.args.oscar_tags:
                #     input_tokens = [source_text]
                #     input_tokens.append('tags:')
                #     obj_ids = f[f'{img_id}/obj_id'][()]
                #     for obj_id in obj_ids:
                #         obj = vg_classes[obj_id]
                #         if obj not in input_tokens:
                #             input_tokens.append(obj)
                #     source_text = ' '.join(input_tokens)

                if self.args.ocr_tags:
                    if self.args.ocr_position_encoding == "oscar_style":
                        #Add OCRs to the data_csv
                        ocr_tags = datum['ocr']
                        #Add an argument to define the threshold
                        # if len(ocr_tags) > 60:
                        #     ocr_tags = ocr_tags[:60]
                        #input_tokens = [source_text]
                        input_tokens = []
                        input_tokens.append('ocrs:')
                        for ocr_tag in ocr_tags:
                            input_tokens.append(ocr_tag)
                        #Append the source_text to the back
                        input_tokens.append(source_text)
                        source_text = ' '.join(input_tokens)
                    
                if is_matched:
                    target_text = 'true'
                    target_text = 'false'
            # elif task == 'tdp':
            #     sent = datum['sent']
            #     #Get the Column and Target Values
            #     col_heads = datum['col_head']
            #     #print(col_heads)
            #     if col_heads[0] == "<legend>" or col_heads[0] == "<random_legend>":
            #         col_heads = col_heads[1:]
            #     row_heads = datum['row_head']
            #     # print(col_heads)
            #     col_index = random.randint(0, len(col_heads)-1)
            #     row_index = random.randint(0, len(row_heads)-1)
                
            #     source_text = f"table data prediction column: {col_heads[col_index]} row: {row_heads[row_index]}"
                
            #     input_tokens = [source_text]
            #     if self.args.ocr_tags:
            #         if self.args.ocr_position_encoding == "oscar_style":
            #             ocr_tags = datum['ocr']
            #             #Add an argument to define the threshold
            #             if len(ocr_tags) > 60:
            #                 ocr_tags = ocr_tags[:60]
            #             #input_tokens = [source_text]
            #             input_tokens = []
            #             input_tokens.append('ocrs:')
            #             for ocr_tag in ocr_tags:
            #                 input_tokens.append(ocr_tag)
            #             #input_tokens.append(source_text)
            #             source_text = ' '.join(input_tokens)

            #     #Define the target text
            #     if "chartsum" in datum['text_source']:
            #         data_name = "statista"
            #     elif "dvqa" in datum['text_source']:
            #         data_name = "dvqa"
            #     elif "figqa" in datum['text_source']:
            #         data_name = "figureqa"
            #     elif "plotqa" in datum['text_source']:
            #         data_name = "plotqa"
            #     #Hacky Way
            #     if "figqa" in datum['text_source']:
            #         target_table_path = str(chartsum_dir.joinpath(f"{data_name}/train/tables/{img_id}.csv"))
            #     else:
            #         target_table_path = str(chartsum_dir.joinpath(f"{data_name}/{datum['split']}/tables/{img_id}.csv"))
            #     target_table = pd.read_csv(target_table_path)
            #     target_data = target_table.to_numpy()
            #     target_text =  str(target_data[row_index, col_index+1]) 

            # elif task == 'tsp':
            #     sent = datum['sent']
            #     #Get the Column and Target Values
            #     row_heads = datum['row_head']
            #     col_heads = datum['col_head']
            #     if col_heads[0] == "<legend>" or col_heads[0] == "<random_legend>":
            #         if col_heads[0] == "<random_legend>":
            #             source_text = f"row_head_prediction: "
            #             target_text = str(" ".join(row_heads))
            #         else:
            #             if random.randint(0,1) == 0:
            #                 source_text = f"row_head_prediction: "
            #                 target_text = str(" ".join(row_heads))
            #             else:
            #                 source_text = f"col_head_prediction: "
            #                 try:
            #                     target_text = str(" ".join(col_heads[1:]))
            #                 except:
            #                     target_text = str(" ".join([str(x) for x in col_heads[1:]]))

            #         col_heads = col_heads[1:]
            #     else:
            #         if random.randint(0,1) == 0:
            #             source_text = f"row_head_prediction: "
            #             target_text = str(" ".join(row_heads))
            #         else:
            #             source_text = f"col_head_prediction: "
            #             try:
            #                 target_text = str(" ".join(col_heads))
            #             except:
            #                 target_text = str(" ".join([str(x) for x in col_heads]))

                
                

                
                # if self.args.ocr_tags:
                #     if self.args.ocr_position_encoding == "oscar_style":
                #         ocr_tags = datum['ocr']
                #         #Add an argument to define the threshold
                #         if len(ocr_tags) > 60:
                #             ocr_tags = ocr_tags[:60]
                #         #input_tokens = [source_text]
                        
                #         input_tokens = [source_text]
                #         input_tokens.append('ocrs:')
                #         for ocr_tag in ocr_tags:
                #             input_tokens.append(ocr_tag)
                #         #input_tokens.append(source_text)
                #         source_text = ' '.join(input_tokens)
                    

           


            input_ids = self.tokenizer.encode(
                source_text, padding=True, truncation=True, max_length=self.args.max_text_length)
            target_ids = self.tokenizer.encode(
                target_text, padding=True, truncation=True, max_length=self.args.gen_max_length)

            # if task in ['refer', 'itm']:
            #     target_ids = target_ids[:-1]

            out_dict['input_ids'] = torch.LongTensor(input_ids)
            out_dict['input_length'] = len(input_ids)
            out_dict['target_ids'] = torch.LongTensor(target_ids)
            out_dict['target_length'] = len(target_ids)

            out_dict['source_text'] = source_text
            out_dict['target_text'] = target_text

            out_dict['task'] = task
            out_dict['sent'] = sent

            out_dict['loss_weight'] = loss_weight

            

            if self.args.visfeat_type == "vlt5":
                feats = np.zeros(shape=(self.n_boxes, 2048), dtype=np.float32)
                try:
                    f[f'{img_id}/features'].read_direct(feats)
                except KeyError:
                    print(uid)
                    print(source)
                    print(img_id)
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
            
            if self.args.ocr_tags:
                if self.args.ocr_position_encoding == "ocr_bbox":
                    ocr_locations = datum['ocr_bbox']
                    assert len(ocr_tags) == len(ocr_locations), "length does not match: ocr: {} vs bbox: {}".format(len(ocr_tags), len(ocr_locations))
                    if len(ocr_tags) > self.ocr_threds:
                        ocr_tags = ocr_tags[:self.ocr_threds]
                        ocr_locations = ocr_locations[:self.ocr_threds]
                    #if not self.args.num_modeling:
                    #print(ocr_tags)
                    ocr_ids = []
                    ocr_bboxes = []
                    for current_ocr, current_ocr_bbox in zip(ocr_tags, ocr_locations):
                        current_ocr_ids = self.tokenizer.encode(current_ocr+" ")
                        ocr_ids += current_ocr_ids
                        ocr_bboxes += [current_ocr_bbox]*len(current_ocr_ids)
       
                    assert len(ocr_ids) == len(ocr_bboxes), "length does not match: ocr: {} vs bbox: {}".format(len(ocr_tags), len(ocr_bboxes))
                    


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

            return out_dict

        elif 'bart' in self.args.backbone:

            text_source = datum['text_source']
            task = datum['task']

            loss_weight = 1

            # T5 Corrupt span
            if task == 'lm':
                assert text_source in ["mscoco", 'vg'], (datum, text_source)

                # LM only
                if self.args.losses == 'lm':
                    prefix = None
                else:
                    prefix = "denoise text:"
                sent = datum['sent']
                source_text, target_text = preprocess.corrupt_bart(
                    sent, mask_ratio=self.args.word_mask_rate, prefix=prefix)

                if self.args.oscar_tags:
                    input_tokens = [source_text]
                    input_tokens.append('tags:')
                    obj_ids = f[f'{img_id}/obj_id'][()]
                    for obj_id in obj_ids:
                        obj = vg_classes[obj_id]
                        if obj not in input_tokens:
                            input_tokens.append(obj)
                    source_text = ' '.join(input_tokens)

            elif task == 'qa':
                assert text_source in ['vqa', 'gqa',
                    'visual7w'], (text_source, uid)

                label = datum['label']
                assert len(label) > 0
                # for ans in list(label.keys()):
                #     label[self.answer_table.ans2id(ans)] = label.pop(ans)
                keys, values = zip(*label.items())
                # single answer
                if len(keys) == 1:
                    ans = keys[0]
                # multiple answers -> sample one answer
                else:
                    value_sum = sum(values)
                    prob = [value / value_sum for value in values]
                    choice = np.random.multinomial(1, prob).argmax()
                    ans = keys[choice]

                sent = datum['sent']

                if self.args.single_vqa_prefix:
                    source_text = f"vqa: {sent}"
                else:
                    source_text = f"{text_source}: {sent}"
                if self.args.oscar_tags:
                    input_tokens = [source_text]
                    input_tokens.append('tags:')
                    obj_ids = f[f'{img_id}/obj_id'][()]
                    for obj_id in obj_ids:
                        obj = vg_classes[obj_id]
                        if obj not in input_tokens:
                            input_tokens.append(obj)
                    source_text = ' '.join(input_tokens)
                target_text = ans

            elif task == 'itm':

                assert text_source in ["mscoco", 'vg']
                is_matched = 1
                sent = datum['sent']
                if random.random() < 0.5:
                    is_matched = 0

                    rand_idx = random.randint(0, self.n_data_captions-1)
                    # rand_idx = int(self.n_data_captions * random.random())

                    other_datum = self.data_captions[rand_idx]
                    # other_datum = self.data[random.randint(0, len(self.data)-1)]
                    while other_datum['img_id'] == img_id:

                        rand_idx = random.randint(
                            0, self.n_data_captions-1)
                        # rand_idx = int(self.n_data_captions * random.random())

                        other_datum = self.data_captions[rand_idx]
                        # other_datum = self.data[random.randint(0, len(self.data)-1)]
                    sent = other_datum['sent']

                prefix = "image text match:"
                source_text = f"{prefix} {sent}"

                if self.args.oscar_tags:
                    input_tokens = [source_text]
                    input_tokens.append('tags:')
                    obj_ids = f[f'{img_id}/obj_id'][()]
                    for obj_id in obj_ids:
                        obj = vg_classes[obj_id]
                        if obj not in input_tokens:
                            input_tokens.append(obj)
                    source_text = ' '.join(input_tokens)
                if is_matched:
                    target_text = 'true'
                else:
                    target_text = 'false'

            if task == 'ground_caption':
                obj_ids = f[f'{img_id}/obj_id'][()]
                attr_ids = f[f'{img_id}/attr_id'][()]

                captions = []
                for obj_id, attr_id in zip(obj_ids, attr_ids):
                    obj = vg_classes[obj_id]
                    attr = vg_attrs[attr_id]

                    caption = f'{attr} {obj}'
                    captions.append(caption)

                # prefix = "describe visual inputs:"
                prefix = "caption region:"
                source_text, target_text = preprocess.ground_caption(
                    captions, self.args.n_ground, prefix=prefix, sort=False)

                sent = source_text

                loss_weight = self.args.ground_weight

            if task == 'refer':
                obj_ids = f[f'{img_id}/obj_id'][()]
                attr_ids = f[f'{img_id}/attr_id'][()]

                captions = []
                for obj_id, attr_id in zip(obj_ids, attr_ids):
                    obj = vg_classes[obj_id]
                    attr = vg_attrs[attr_id]

                    caption = f'{attr} {obj}'
                    captions.append(caption)

                # prefix = "refer expressions:"
                prefix = "visual grounding:"
                source_text, target_text = preprocess.refer_expression(
                    captions, self.args.n_ground, prefix=prefix, sort=False)

                sent = source_text

                loss_weight = self.args.ground_weight

            input_ids = self.tokenizer.encode(
                source_text, padding=True, truncation=True, max_length=self.args.max_text_length)
            target_ids = self.tokenizer.encode(
                target_text, padding=True, truncation=True, max_length=self.args.gen_max_length)

            # if task in ['refer', 'itm']:
            #     target_ids = target_ids[:-1]

            out_dict['input_ids'] = torch.LongTensor(input_ids)
            out_dict['input_length'] = len(input_ids)
            out_dict['target_ids'] = torch.LongTensor(target_ids)
            out_dict['target_length'] = len(target_ids)

            out_dict['source_text'] = source_text
            out_dict['target_text'] = target_text

            out_dict['task'] = task
            out_dict['sent'] = sent

            out_dict['loss_weight'] = loss_weight

            feats = np.zeros(shape=(self.n_boxes, 2048), dtype=np.float32)
            try:
                f[f'{img_id}/features'].read_direct(feats)
            except KeyError:
                print(uid)
                print(source)
                print(img_id)
                exit()

            feats = torch.from_numpy(feats)
            out_dict['vis_feats'] = feats

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
            out_dict['boxes'] = boxes

            return out_dict


    def collate_fn(self, batch):
        batch_entry = {}

        B = len(batch)

        args = self.args

        # V_L = len(batch[0]['boxes'])
        if self.args.visfeat_type == "vlt5":
            V_L = len(batch[0]['boxes'])
        else:
            V_L = 36 #For padding purpose

        S_W_L = max(entry['input_length'] for entry in batch)
        T_W_L = max(entry['target_length'] for entry in batch)

        feat_dim = batch[0]['vis_feats'].shape[-1]

        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        boxes = torch.zeros(B, V_L, 4, dtype=torch.float)
        vis_feats = torch.zeros(B, V_L, feat_dim, dtype=torch.float)

        loss_weights = torch.ones(B, dtype=torch.float)
        
        if self.args.ocr_position_encoding == "ocr_bbox" and self.args.ocr_tags:
            V_W_L = max(entry['visocrs_length'] for entry in batch)
            ocr_ids = torch.ones(B, V_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
            ocr_bboxes = torch.zeros(B, V_W_L, 4, dtype=torch.float)


        sentences = []
        ans = []
        uids = []
        tasks = []

        source_text = []
        target_text = []

        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']
            target_ids[i, :entry['target_length']] = entry['target_ids']
            if self.args.visfeat_type == "vlt5":
                boxes[i] += entry['boxes']
                vis_feats[i] += entry['vis_feats']
            else:
                boxes[i, :len(entry['boxes']), :] += entry['boxes']
                vis_feats[i, :len(entry['vis_feats']), :] += entry['vis_feats']

            if self.args.ocr_position_encoding == "ocr_bbox" and self.args.ocr_tags:
                #fill in the ocr_ids
                ocr_ids[i, :entry['visocrs_length']] = entry['visocrs']
                ocr_bboxes[i, :len(entry['visocrs_bboxes']), :] += entry['visocrs_bboxes']

            if 'ans' in entry:
                ans.append(entry['ans'])

            if 'task' in entry:
                tasks.append(entry['task'])

            sentences.append(entry['sent'])
            uids.append(entry['uid'])
            
            if 'source_text' in entry:
                source_text.append(entry['source_text'])
            if 'target_text' in entry:
                target_text.append(entry['target_text'])

            if 'loss_weight' in entry:
                loss_weights[i] = entry['loss_weight']

        assert 't5' in args.backbone or 'bart' in args.backbone
        word_mask = target_ids != self.tokenizer.pad_token_id
        target_ids[~word_mask] = -100
        batch_entry['task'] = tasks

        batch_entry['source_text'] = source_text
        batch_entry['target_text'] = target_text

        batch_entry['input_ids'] = input_ids
        batch_entry['target_ids'] = target_ids

        batch_entry['boxes'] = boxes
        batch_entry['vis_feats'] = vis_feats

        if self.args.ocr_position_encoding == "ocr_bbox" and self.args.ocr_tags:
            batch_entry['visocrs'] = ocr_ids
            batch_entry['visocrs_bboxes'] = ocr_bboxes

        batch_entry['loss_weights'] = loss_weights

        batch_entry['uid'] = uids
        batch_entry['sent'] = sentences

        return batch_entry

########################################################


class PretrainDataset(Dataset):
    def __init__(self, split='chartsum_train', rank=-1, topk=-1, verbose=True, args=None, is_train=True):
        self.topk = topk
        self.verbose = verbose
        self.args = args


        # Loading datasets to data
        self.sources = split.split(',')
        if self.verbose:
            print('Data sources: ', self.sources)

        # Answer Table from LXMERT (Could be removed)
        # self.answer_table = AnswerTable()
        # if self.verbose:
        #     print("Load an answer table of size %d." % (len(self.answer_table.ans2id_map())))

        self.img_ids_to_source = {}

        losses = args.losses.split(',')

        #No need to consider different sources, we only have one unique source. 
        #Need to modify to adapt to this change. Prepare a list of dictionaries.
        data = []
        for img_source in self.sources:
            data_info_path = chartsum_dir.joinpath(f'chartsum_annotation.json')
            with open(data_info_path) as f:
                _data = json.load(f)
                if self.verbose:
                    print(f"Loaded {len(_data)} data from", img_source)
                # source_img_ids.append([d['img_id'] for d in _data])
                for datum in _data:
                    if datum['text_source'] == split:
                        self.img_ids_to_source[datum['img_id']] = img_source
                        # datum['img_source'] = img_source
                        datum['args'] = args
                        datum['is_train'] = is_train
                        datum['caption_only'] = args.caption_only

                        datum['lm'] = 'lm' in losses
                        datum['qa'] = 'qa' in losses
                        datum['ground_caption'] = 'ground_caption' in losses
                        datum['refer'] = 'refer' in losses
                        datum['itm'] = 'itm' in losses
                        datum['caption'] = 'caption' in losses

                        datum['backbone'] = self.args.backbone

                        data.append(datum)
        del _data

        if self.verbose:
            print("# images:", len(data))

        # if self.topk > 0:
        #     data = data[:self.topk]
        #     if self.verbose:
        #         print(f"Use only {self.topk} data")

        # if 'qa' in args.losses:
        #     self.evaluator = QAEvaluator(data)

        with Pool(8) as pool:
            if self.verbose:
                data = [datum for _data in tqdm(
                    pool.imap(get_datum, data), total=len(data), ncols=100, desc="Creating pretrainig data examples") for datum in _data]
            else:
                data = [datum for _data in pool.imap(
                    get_datum, data) for datum in _data]

        # if self.args.itm_cocoonly:
        #     caption_sources = ['mscoco']
        # else:
        #     caption_sources = ['mscoco', 'vg']
        self.data_captions = [datum for datum in data if datum['text_source'] == split] #Load thee data that contains cerrtain captions
        self.n_data_captions = len(self.data_captions)

        if self.verbose:
            print('# itm data:', self.n_data_captions)

        self.data = data
        # print(self.data[0].keys())
        self.n_data = len(self.data)

        if self.verbose and is_train:
            from collections import Counter
            task_counter = Counter()
            for datum in data:
                try:
                    task_counter.update([datum['task']])
                except KeyError:
                    print(datum)
                    exit()

            print(task_counter)
            for k, v in task_counter.items():
                print(k, f'{v/len(data)*100:.1f}%')

        if self.verbose:
            print("# examples:", len(data))
        
        if self.args.visfeat_type == "vlt5":
            self.source_to_h5 = {
                'chartsum_train': chartsum_feature_dir.joinpath('train_boxes36.h5'),
                'chartsum_val': chartsum_feature_dir.joinpath('val_boxes36.h5'),

            }
        else:
            self.source_to_h5 = {
                'chartsum_train': chartsum_feature_dir.joinpath('train_chart_elements.h5'),
                'chartsum_val': chartsum_feature_dir.joinpath('val_chart_elements.h5'),

            }

        self.n_boxes = args.n_boxes

        if 't5' in self.args.backbone:
            if self.args.use_vision:
                # self.tokenizer = VLT5Tokenizer.from_pretrained(
                #     args.backbone, do_lower_case=args.do_lower_case)
                self.tokenizer = VLT5TokenizerFast.from_pretrained(
                    args.backbone, do_lower_case=args.do_lower_case)
            else:
                # self.tokenizer = T5Tokenizer.from_pretrained(
                #     args.backbone, do_lower_case=args.do_lower_case)
                self.tokenizer = T5TokenizerFast.from_pretrained(
                    args.backbone, do_lower_case=args.do_lower_case)
        elif 'bart' in self.args.backbone:
            self.tokenizer = BartTokenizer.from_pretrained(args.backbone)
            additional_special_tokens = [f'<extra_id_{i}>' for i in range(100-1, -1, -1)] + \
                    [f'<vis_extra_id_{i}>' for i in range(100-1, -1, -1)]
            special_tokens_dict = {'additional_special_tokens': additional_special_tokens}
            self.tokenizer.add_special_tokens(special_tokens_dict)


    def __len__(self):
        # return len(self.data)
        return self.n_data

    def __getitem__(self, idx):

        out_dict = {}
        out_dict['args'] = self.args

        datum = self.data[idx]

        uid = datum['uid'] #Created by get_datum function. 
        out_dict['uid'] = uid

        ###### Image ######
        img_id = datum['img_id']
        source = datum['img_source'] #Add to the get_datum function

        f = self.source_to_h5[source]
        if isinstance(f, Path):
            path = self.source_to_h5[source]
            f = h5py.File(path, 'r')
            self.source_to_h5[source] = f

        if 't5' in self.args.backbone:

            #text_source = datum['text_source']
            task = datum['task']

            loss_weight = 1

            # T5 Corrupt span
            if task == 'lm':
                #assert text_source in ["mscoco", 'vg']

                prefix = "span prediction:"
                sent = datum['sent'] #get the caption
                source_text, target_text = preprocess.corrupt_spans(
                    sent, mask_ratio=self.args.word_mask_rate, prefix=prefix)
                
                #Appeend the oscar tags, if required
                if self.args.oscar_tags:
                    input_tokens = [source_text]
                    input_tokens.append('tags:')
                    obj_ids = f[f'{img_id}/obj_id'][()]
                    for obj_id in obj_ids:
                        obj = vg_classes[obj_id]
                        if obj not in input_tokens:
                            input_tokens.append(obj)
                    source_text = ' '.join(input_tokens)
                if self.args.ocr_tags:
                    #Add OCRs to the data_csv
                    ocr_tags = datum['ocr'] #Need to be Modified
                    #Add an argument to define the threshold
                    if len(ocr_tags) > 60:
                        ocr_tags = ocr_tags[:60]
                    input_tokens = [source_text]
                    input_tokens.append('ocrs:')
                    for ocr_tag in ocr_tags:
                        input_tokens.append(ocr_tag)
                    source_text = ' '.join(input_tokens)
                    # input_ids = self.tokenizer.encode(f'ocrs: {ocr_tags} chartqa: {sent}', max_length=400, truncation=True)

            # elif task == 'qa':
            #     assert text_source in ['vqa', 'gqa', 'visual7w'], (text_source, uid)

            #     label = datum['label']
            #     assert len(label) > 0

            #     keys, values = zip(*label.items())
            #     # single answer
            #     if len(keys) == 1:
            #         ans = keys[0]
            #     # multiple answers -> sample one answer
            #     else:
            #         value_sum = sum(values)
            #         prob = [value / value_sum for value in values]
            #         choice = np.random.multinomial(1, prob).argmax()
            #         ans = keys[choice]

            #     sent = datum['sent']

            #     if self.args.single_vqa_prefix:
            #         source_text = f"vqa: {sent}"
            #     else:
            #         source_text = f"{text_source}: {sent}"
            #     if self.args.oscar_tags:
            #         input_tokens = [source_text]
            #         input_tokens.append('tags:')
            #         obj_ids = f[f'{img_id}/obj_id'][()]
            #         for obj_id in obj_ids:
            #             obj = vg_classes[obj_id]
            #             if obj not in input_tokens:
            #                 input_tokens.append(obj)
            #         source_text = ' '.join(input_tokens)
            #     target_text = ans

            elif task == 'itm':

                # assert text_source in ["mscoco", 'vg']
                is_matched = 1
                sent = datum['sent']
                if random.random() < 0.5:
                    is_matched = 0

                    rand_idx = random.randint(0, self.n_data_captions-1)
                    # rand_idx = int(self.n_data_captions * random.random())

                    other_datum = self.data_captions[rand_idx]
                    # other_datum = self.data[random.randint(0, len(self.data)-1)]
                    while other_datum['img_id'] == img_id:

                        rand_idx = random.randint(0, self.n_data_captions-1)
                        # rand_idx = int(self.n_data_captions * random.random())

                        other_datum = self.data_captions[rand_idx]
                        # other_datum = self.data[random.randint(0, len(self.data)-1)]
                    sent = other_datum['sent']
                
                #Append the Prefix
                prefix = "image text match:"
                source_text = f"{prefix} {sent}"
                
                #Append the tags as Oscar. 
                if self.args.oscar_tags:
                    input_tokens = [source_text]
                    input_tokens.append('tags:')
                    obj_ids = f[f'{img_id}/obj_id'][()]
                    for obj_id in obj_ids:
                        obj = vg_classes[obj_id]
                        if obj not in input_tokens:
                            input_tokens.append(obj)
                    source_text = ' '.join(input_tokens)

                if self.args.ocr_tags:
                    #Add OCRs to the data_csv
                    ocr_tags = datum['ocr']
                    #Add an argument to define the threshold
                    if len(ocr_tags) > 60:
                        ocr_tags = ocr_tags[:60]
                    input_tokens = [source_text]
                    input_tokens.append('ocrs:')
                    for ocr_tag in ocr_tags:
                        input_tokens.append(ocr_tag)
                    source_text = ' '.join(input_tokens)
                    
                if is_matched:
                    target_text = 'true'
                else:
                    target_text = 'false'

            # if task == 'ground_caption':
            #     obj_ids = f[f'{img_id}/obj_id'][()]
            #     attr_ids = f[f'{img_id}/attr_id'][()]

            #     captions = []
            #     for obj_id, attr_id in zip(obj_ids, attr_ids):
            #         obj = vg_classes[obj_id]
            #         attr = vg_attrs[attr_id]

            #         caption = f'{attr} {obj}'
            #         captions.append(caption)

            #     # prefix = "describe visual inputs:"
            #     prefix = "caption region:"
            #     source_text, target_text = preprocess.ground_caption(
            #         captions, self.args.n_ground, prefix=prefix, sort=False)

            #     sent = source_text

            #     loss_weight = self.args.ground_weight

            # if task == 'refer':
            #     obj_ids = f[f'{img_id}/obj_id'][()]
            #     attr_ids = f[f'{img_id}/attr_id'][()]

            #     captions = []
            #     for obj_id, attr_id in zip(obj_ids, attr_ids):
            #         obj = vg_classes[obj_id]
            #         attr = vg_attrs[attr_id]

            #         caption = f'{attr} {obj}'
            #         captions.append(caption)

            #     # prefix = "refer expressions:"
            #     prefix = "visual grounding:"
            #     source_text, target_text = preprocess.refer_expression(
            #         captions, self.args.n_ground, prefix=prefix, sort=False)

            #     sent = source_text

            #     loss_weight = self.args.ground_weight

            input_ids = self.tokenizer.encode(
                source_text, padding=True, truncation=True, max_length=self.args.max_text_length)
            target_ids = self.tokenizer.encode(
                target_text, padding=True, truncation=True, max_length=self.args.gen_max_length)

            # if task in ['refer', 'itm']:
            #     target_ids = target_ids[:-1]

            out_dict['input_ids'] = torch.LongTensor(input_ids)
            out_dict['input_length'] = len(input_ids)
            out_dict['target_ids'] = torch.LongTensor(target_ids)
            out_dict['target_length'] = len(target_ids)

            out_dict['source_text'] = source_text
            out_dict['target_text'] = target_text

            out_dict['task'] = task
            out_dict['sent'] = sent

            out_dict['loss_weight'] = loss_weight
            if self.args.visfeat_type == "vlt5":
                feats = np.zeros(shape=(self.n_boxes, 2048), dtype=np.float32)
                try:
                    f[f'{img_id}/features'].read_direct(feats)
                except KeyError:
                    print(uid)
                    print(source)
                    print(img_id)
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

            return out_dict

        elif 'bart' in self.args.backbone:

            text_source = datum['text_source']
            task = datum['task']

            loss_weight = 1

            # T5 Corrupt span
            if task == 'lm':
                assert text_source in ["mscoco", 'vg'], (datum, text_source)

                # LM only
                if self.args.losses == 'lm':
                    prefix = None
                else:
                    prefix = "denoise text:"
                sent = datum['sent']
                source_text, target_text = preprocess.corrupt_bart(
                    sent, mask_ratio=self.args.word_mask_rate, prefix=prefix)

                if self.args.oscar_tags:
                    input_tokens = [source_text]
                    input_tokens.append('tags:')
                    obj_ids = f[f'{img_id}/obj_id'][()]
                    for obj_id in obj_ids:
                        obj = vg_classes[obj_id]
                        if obj not in input_tokens:
                            input_tokens.append(obj)
                    source_text = ' '.join(input_tokens)

            elif task == 'qa':
                assert text_source in ['vqa', 'gqa',
                    'visual7w'], (text_source, uid)

                label = datum['label']
                assert len(label) > 0
                # for ans in list(label.keys()):
                #     label[self.answer_table.ans2id(ans)] = label.pop(ans)
                keys, values = zip(*label.items())
                # single answer
                if len(keys) == 1:
                    ans = keys[0]
                # multiple answers -> sample one answer
                else:
                    value_sum = sum(values)
                    prob = [value / value_sum for value in values]
                    choice = np.random.multinomial(1, prob).argmax()
                    ans = keys[choice]

                sent = datum['sent']

                if self.args.single_vqa_prefix:
                    source_text = f"vqa: {sent}"
                else:
                    source_text = f"{text_source}: {sent}"
                if self.args.oscar_tags:
                    input_tokens = [source_text]
                    input_tokens.append('tags:')
                    obj_ids = f[f'{img_id}/obj_id'][()]
                    for obj_id in obj_ids:
                        obj = vg_classes[obj_id]
                        if obj not in input_tokens:
                            input_tokens.append(obj)
                    source_text = ' '.join(input_tokens)
                target_text = ans

            elif task == 'itm':

                assert text_source in ["mscoco", 'vg']
                is_matched = 1
                sent = datum['sent']
                if random.random() < 0.5:
                    is_matched = 0

                    rand_idx = random.randint(0, self.n_data_captions-1)
                    # rand_idx = int(self.n_data_captions * random.random())

                    other_datum = self.data_captions[rand_idx]
                    # other_datum = self.data[random.randint(0, len(self.data)-1)]
                    while other_datum['img_id'] == img_id:

                        rand_idx = random.randint(
                            0, self.n_data_captions-1)
                        # rand_idx = int(self.n_data_captions * random.random())

                        other_datum = self.data_captions[rand_idx]
                        # other_datum = self.data[random.randint(0, len(self.data)-1)]
                    sent = other_datum['sent']

                prefix = "image text match:"
                source_text = f"{prefix} {sent}"

                if self.args.oscar_tags:
                    input_tokens = [source_text]
                    input_tokens.append('tags:')
                    obj_ids = f[f'{img_id}/obj_id'][()]
                    for obj_id in obj_ids:
                        obj = vg_classes[obj_id]
                        if obj not in input_tokens:
                            input_tokens.append(obj)
                    source_text = ' '.join(input_tokens)
                if is_matched:
                    target_text = 'true'
                else:
                    target_text = 'false'

            if task == 'ground_caption':
                obj_ids = f[f'{img_id}/obj_id'][()]
                attr_ids = f[f'{img_id}/attr_id'][()]

                captions = []
                for obj_id, attr_id in zip(obj_ids, attr_ids):
                    obj = vg_classes[obj_id]
                    attr = vg_attrs[attr_id]

                    caption = f'{attr} {obj}'
                    captions.append(caption)

                # prefix = "describe visual inputs:"
                prefix = "caption region:"
                source_text, target_text = preprocess.ground_caption(
                    captions, self.args.n_ground, prefix=prefix, sort=False)

                sent = source_text

                loss_weight = self.args.ground_weight

            if task == 'refer':
                obj_ids = f[f'{img_id}/obj_id'][()]
                attr_ids = f[f'{img_id}/attr_id'][()]

                captions = []
                for obj_id, attr_id in zip(obj_ids, attr_ids):
                    obj = vg_classes[obj_id]
                    attr = vg_attrs[attr_id]

                    caption = f'{attr} {obj}'
                    captions.append(caption)

                # prefix = "refer expressions:"
                prefix = "visual grounding:"
                source_text, target_text = preprocess.refer_expression(
                    captions, self.args.n_ground, prefix=prefix, sort=False)

                sent = source_text

                loss_weight = self.args.ground_weight

            input_ids = self.tokenizer.encode(
                source_text, padding=True, truncation=True, max_length=self.args.max_text_length)
            target_ids = self.tokenizer.encode(
                target_text, padding=True, truncation=True, max_length=self.args.gen_max_length)

            # if task in ['refer', 'itm']:
            #     target_ids = target_ids[:-1]

            out_dict['input_ids'] = torch.LongTensor(input_ids)
            out_dict['input_length'] = len(input_ids)
            out_dict['target_ids'] = torch.LongTensor(target_ids)
            out_dict['target_length'] = len(target_ids)

            out_dict['source_text'] = source_text
            out_dict['target_text'] = target_text

            out_dict['task'] = task
            out_dict['sent'] = sent

            out_dict['loss_weight'] = loss_weight

            feats = np.zeros(shape=(self.n_boxes, 2048), dtype=np.float32)
            try:
                f[f'{img_id}/features'].read_direct(feats)
            except KeyError:
                print(uid)
                print(source)
                print(img_id)
                exit()

            feats = torch.from_numpy(feats)
            out_dict['vis_feats'] = feats

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
            out_dict['boxes'] = boxes

            return out_dict


    def collate_fn(self, batch):
        batch_entry = {}

        B = len(batch)

        args = self.args

        # V_L = len(batch[0]['boxes'])
        if self.args.visfeat_type == "vlt5":
            V_L = len(batch[0]['boxes'])
        else:
            V_L = 36 #For padding purpose

        S_W_L = max(entry['input_length'] for entry in batch)
        T_W_L = max(entry['target_length'] for entry in batch)

        feat_dim = batch[0]['vis_feats'].shape[-1]

        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        boxes = torch.zeros(B, V_L, 4, dtype=torch.float)
        vis_feats = torch.zeros(B, V_L, feat_dim, dtype=torch.float)

        loss_weights = torch.ones(B, dtype=torch.float)

        sentences = []
        ans = []
        uids = []
        tasks = []

        source_text = []
        target_text = []

        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']
            target_ids[i, :entry['target_length']] = entry['target_ids']
            if self.args.visfeat_type == "vlt5":
                boxes[i] += entry['boxes']
                vis_feats[i] += entry['vis_feats']
            else:
                boxes[i, :len(entry['boxes']), :] += entry['boxes']
                vis_feats[i, :len(entry['vis_feats']), :] += entry['vis_feats']

            if 'ans' in entry:
                ans.append(entry['ans'])

            if 'task' in entry:
                tasks.append(entry['task'])

            sentences.append(entry['sent'])
            uids.append(entry['uid'])

            if 'source_text' in entry:
                source_text.append(entry['source_text'])
            if 'target_text' in entry:
                target_text.append(entry['target_text'])

            if 'loss_weight' in entry:
                loss_weights[i] = entry['loss_weight']

        assert 't5' in args.backbone or 'bart' in args.backbone
        word_mask = target_ids != self.tokenizer.pad_token_id
        target_ids[~word_mask] = -100
        batch_entry['task'] = tasks

        batch_entry['source_text'] = source_text
        batch_entry['target_text'] = target_text

        batch_entry['input_ids'] = input_ids
        batch_entry['target_ids'] = target_ids

        batch_entry['boxes'] = boxes
        batch_entry['vis_feats'] = vis_feats

        batch_entry['loss_weights'] = loss_weights

        batch_entry['uid'] = uids
        batch_entry['sent'] = sentences

        return batch_entry


def get_loader(args, split='chartsum_train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0,
               topk=-1):


    verbose = (gpu == 0)
    # dataset = PretrainDataset(
    #     split,
    #     rank=gpu,
    #     topk=topk,
    #     verbose=verbose,
    #     args=args,
    #     is_train=(mode == 'train'),
    #     )
    #Using the New Pre-training Dataset
    dataset = ChartTablePretrainDataset(
        split,
        rank=gpu,
        topk=topk,
        verbose=verbose,
        args=args,
        is_train=(mode == 'train'),
        )

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

    return loader


class QAEvaluator:
    def __init__(self, data):

        # Create QA Eval Data
        self.data = []
        for datum in data:
            sentf = datum['sentf']
            for sents_cat, sents in sentf.items():
                if sents_cat in datum['labelf']:    # A labeled dataset
                    labels = datum['labelf'][sents_cat]
                    for sent_idx, sent in enumerate(sents):
                        new_datum = {
                            'uid': make_uid(datum['img_id'], sents_cat, sent_idx),
                            'img_id': datum['img_id'],
                            'sent': sent,
                            'dset': sents_cat,
                            'label': labels[sent_idx]
                        }
                        self.data.append(new_datum)

        # uid2datum
        self.uid2datum = {}
        for datum in self.data:
            self.uid2datum[datum['uid']] = datum

    def evaluate(self, uid2ans: dict):
        score = 0.
        cnt = 0
        dset2score = defaultdict(lambda: 0.)
        dset2cnt = defaultdict(lambda: 0)
        for uid, ans in uid2ans.items():
            if uid not in self.uid2datum:   # Not a labeled data
                continue
            datum = self.uid2datum[uid]
            label = datum['label']
            dset = datum['dset']
            if ans in label:
                score += label[ans]
                dset2score[dset] += label[ans]
            cnt += 1
            dset2cnt[dset] += 1
        return dset2score, dset2cnt, score, cnt

    def _evaluate(self, uid2ans: dict, pprint=False):
        score = 0.
        cnt = 0
        dset2score = defaultdict(lambda: 0.)
        dset2cnt = defaultdict(lambda: 0)
        for uid, ans in uid2ans.items():
            if uid not in self.uid2datum:   # Not a labeled data
                continue
            datum = self.uid2datum[uid]
            label = datum['label']
            dset = datum['dset']
            if ans in label:
                score += label[ans]
                dset2score[dset] += label[ans]
            cnt += 1
            dset2cnt[dset] += 1
        accu = score / cnt
        dset2accu = {}
        for dset in dset2cnt:
            dset2accu[dset] = dset2score[dset] / dset2cnt[dset]

        if pprint:
            accu_str = "Overall Accu %0.4f, " % (accu)
            sorted_keys = sorted(dset2accu.keys())
            for key in sorted_keys:
                accu_str += "%s Accu %0.4f, " % (key, dset2accu[key])
            print(accu_str)

        return accu, dset2accu

    def dump_result(self, uid2ans: dict, path):
        raise NotImplementedError
