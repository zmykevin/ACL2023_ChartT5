import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import ast
import random
import csv

from tokenization import VLT5TokenizerFast
from torch.nn import MSELoss
import torch
from torch import nn

num_symbol = "<num_extra_id_0>"
MANTISSA_NORM = 10
EXPONENT_NORM = 20

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
        mantissa = float(scif_w.split('e')[0])/(2*MANTISSA_NORM) + 0.5
        exponent = float(scif_w.split('e')[1])/(2*EXPONENT_NORM) + 0.5
        num_values.append((mantissa, exponent))
        new_ocrs.append(num_symbol)
    #new_caption = ' '.join(new_caption_words)
    return new_ocrs, num_values

def numerify_sent(caption):
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
        mantissa = float(scif_w.split('e')[0])/(2*MANTISSA_NORM) + 0.5
        exponent = float(scif_w.split('e')[1])/(2*EXPONENT_NORM) + 0.5
        num_values.append((mantissa, exponent))
        new_caption_words.append(num_symbol)
    new_caption = ' '.join(new_caption_words)
    return new_caption, num_values

def recover_mantissa(norm_man):
	orig_mantissa = (norm_man - 0.5) * (2*MANTISSA_NORM)
	return orig_mantissa

def recover_exponent(norm_exp):
	orig_exponent = int((norm_exp - 0.5) * (2*EXPONENT_NORM))
	return orig_exponent

def recover_numeric(norm_man, norm_exp):
	orig_man = recover_mantissa(norm_man)
	orig_exp = recover_exponent(norm_exp)

	orig_num = orig_man*10**orig_exp
	return orig_num
#Read the data_fix.csv
data_root = "/dvmm-filer2/projects/mingyang/semafor/chart_qa/ChartQAv1/val/data_fix.csv"

data = pd.read_csv(data_root)
inputs = data['Input'].values
outputs = data['Output'].values
ocrs = data['OCR'].values

# my_tokenizer = VLT5TokenizerFast.from_pretrained('t5-base', max_length=20, do_lower_case=False)
# #print(my_tokenizer.extra_ids)
# print(my_tokenizer.vocab_size)

# my_loss = nn.MESLoss()

# input_tensor = torch.randn((3,5), requires_grad=True)
# sigmoid_activate = nn.Sigmoid()
# input_tensor = sigmoid_activate(input_tensor)
# output_array = np.zeros((3,5))
# output_array[0,0] = 0.5
# output_array[1,1] = 0.7
# output_array[2,3] = 0.8
# output_tensor = torch.from_numpy(output_array)

#output_tensor = torch.from_numpy()


# output_mask = np.zeros((3,5))
# output_mask[0,0] = 1
# output_mask[1,1] = 1
# output_mask[2,3] = 1

#output_mask_tensor = torch.from_numpy(output_mask)
#target_mask = output_tensor > 0.0
#print(target_mask)
# input_tensor = target_mask * input_tensor

# print(input_tensor)
# print(output_tensor)

# my_loss = MSELoss()
# loss_value = my_loss(input_tensor, output_tensor)
# print(loss_value)
numerical_answer = 0
for answer in tqdm(outputs):
	new_question, new_values = numerify_sent(answer)
	if num_symbol in new_question:
		numerical_answer += 1

print(len(outputs))
print(numerical_answer)
print((len(outputs)-numerical_answer)/len(outputs))
# 		break
# for ocr in ocrs:
# 	ocr = ast.literal_eval(ocr)
# 	new_ocr, num_values = numerify_ocrs(ocr)
# 	print(ocr)
# 	print(new_ocr)
# 	print(num_values)
# 	break
# for question in inputs:
# 	new_question, num_values = numerify_sent(question)
# 	if '[NUM]' in new_question:
# 		print(question)
# 		print(new_question)
# 		print(num_values)

# 		for num in num_values:
# 			if num[0] != 0.0:
# 				print(num)
# 				print(recover_numeric(num[0], num[1]))
# 		break

# def token_classify(tokens):
# 	word_index = []
# 	num_index = []
# 	for i, w in enumerate(tokens):
# 		try:
# 			float_w = float(w)
# 		except:
# 			word_index.append(i)
# 			continue
# 		num_index.append(i)
# 	return word_index, num_index
 
# sample_number = 1500
# scif_num = "{:2e}".format(sample_number)
# print(scif_num)

# mantissa = float(scif_num.split('e')[0])
# exponent = float(scif_num.split('e')[1])
# print(mantissa)
# print(exponent)
# mask_ratio = 0.40
# #Load the annotation:
# data_root = "/dvmm-filer2/projects/mingyang/semafor/chart_pretrain_compress/"
# with open(f"{data_root}/chart_pretrain_annotation_tables_gtocrs_dfp_scicap_ocrcopy.json", "r") as f:
# 	data_annotation = json.load(f)

# for d in data_annotation:
#     text_source = d['text_source']
# 	break
#ocr_table_path = "/dvmm-filer2/projects/mingyang/semafor/chart_pretrain_compress/figureqa/train/tables/train1_51543.csv"
# table_data = pd.read_csv(table_path)
# table_column_head = [col for col in table_data.columns]
# # print(table_column_head)
# # print(table_data)
# if table_column_head[0] == "<random_legend>":
# 	table_flat = table_data.to_numpy().flatten().tolist()
# else:
# 	table_flat = table_column_head + table_data.to_numpy().flatten().tolist()
# table_string = ' '.join([str(x) for x in table_flat])


# tokens = table_string.split()
# #print(tokens)
# word_index, num_index = token_classify(tokens)

# if random.uniform(0,1) > 0.5:
# 	#mask head tokens
# 	n_mask = int(max(mask_ratio * len(word_index), 1))
# 	mask_indices = random.sample(word_index, n_mask)
# 	mask_indices.sort()
# else:
# 	#mask value tokens
#     n_mask = int(max(mask_ratio * len(num_index), 1))
#     mask_indices = random.sample(num_index, n_mask)
#     mask_indices.sort()
# print(n_mask)
# print(mask_indices)
#Convert the Numeric Value to Scientific Statement
# print(table_flat)
# exp_symbol = "[EXP]"
# for x in table_flat:
# 	#determine the float or integer value
# 	if type(x) is float or type(x) is int:
# 		#convert to scientific notification:
# 		scif_x = "{:2e}".format(x)
# 		scif_x = scif_x.replace('e', exp_symbol)

# for x in data_annotation:
# 	img_id = x['img_id']
# 	ocr = x['OCR']
# 	caption = x['sent']
# 	print(caption)
# 	print(ocr)
# 	break

		


# csv_path = "/dvmm-filer2/projects/mingyang/semafor/chart_qa/ChartQAv1/train/data.csv"
# instances = pd.read_csv(csv_path)
# ocrs = instances["OCR"].values
# ocr_locations= instances["OCR_bbox"].values

# for ocr_tag, ocr_location in tqdm(zip(ocrs, ocr_locations)):
# 	ocr_tag_processed = ast.literal_eval(ocr_tag)
# 	ocr_location = json.loads(ocr_location)
# 	assert len(ocr_tag_processed) == len(ocr_location), "length does not match: ocr: {} vs bbox: {}".format(len(ocr_tag), len(ocr_location))
	