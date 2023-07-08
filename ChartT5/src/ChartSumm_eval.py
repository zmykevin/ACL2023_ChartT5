import os, re, csv, sys, json, math, torch
from bleurt import score
from statistics import mean, stdev
from torchmetrics.functional import bleu_score, rouge_score
from pycocoevalcap.cider.cider import Cider 
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from transformers import Trainer, AutoModelForCausalLM, AutoTokenizer
from datetime import datetime


model = AutoModelForCausalLM.from_pretrained("gpt2-medium").eval()
tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")

ptb_tokenizer = PTBTokenizer()
cider = Cider()
checkpoint = "bleurt-base-128"
if not os.path.exists(checkpoint):
    os.system("wget https://storage.googleapis.com/bleurt-oss/bleurt-base-128.zip && \
            unzip bleurt-base-128.zip")
bleurt_scorer = score.BleurtScorer(checkpoint)

fillers = ['in', 'the', 'and', 'or', 'an', 'as', 'can', 'be', 'a', ':', '-',
           'to', 'but', 'is', 'of', 'it', 'on', '.', 'at', '(', ')', ',', ';']

def content_selection(pred_list, tgt_list):
    count = 0
    generatedScores = []
    untemplatedScores = [1,1]
    with open("datasets/Chart-to-text/"+dataset_name+"_c2t_data/testTitle.txt", 'r', encoding='utf-8') as f1, \
            open("datasets/Chart-to-text/"+dataset_name+"_c2t_data/testOriginalSummary.txt", 'r', encoding='utf-8') as f2:
        titleFile, goldFile = f1.readlines(), f2.readlines()
    with open("datasets/Chart-to-text/"+dataset_name+"_c2t_data/testData.txt", 'r', encoding='utf-8') as f:
        dataFile = f.readlines()
    for datas, titles, gold in zip(dataFile, titleFile, goldFile):
        titleArr = titles.split()
        goldArr = gold.split()
        recordList = []
        for gld in goldArr:
            data_string = datas.replace("_", " ")
            if gld.lower() in " ".join([data_string,titles]).lower()  and gld.lower() not in fillers and gld.lower() not in recordList:
                recordList.append(gld.lower())
        list1 = recordList
        list2 = recordList
        list3 = recordList
        recordLength = len(recordList)
        generatedList = []
        summary1 = [gen_summ for gen_summ,gt_summ in zip(pred_list, tgt_list) if "".join(gt_summ[0].split())=="".join(gold.split())]
        if len(summary1)==0:
            continue
        summary1 = summary1[0] #.split()
        for token in summary1.split():
            if token.lower() in list1:
                list1.remove(token.lower())
                generatedList.append(token.lower())
        count += 1
        if recordLength==0:
            generatedRatio=0
        else:
            generatedRatio = len(generatedList) / recordLength
        generatedScores.append(generatedRatio)
    if len(generatedScores)==0:
        return 0.0
    return round(mean(generatedScores)*100,2)

def eval_ChartSumm(pred_list, tgt_list, eval_mode):
    predictions, references = {}, {}
    bleu_scores, bleurt_scores, ppl_scores = [], [], []
    cs_score = 0.0
    if eval_mode=="test":
        cs_score = content_selection(pred_list, tgt_list)
    for i, (x,y) in enumerate(zip(pred_list, tgt_list)):
        y = y[0]
        bleu_scores.append(bleu_score([x], [y], 1))
        bleurt_scores.append(bleurt_scorer.score(references=[x], candidates=[y])[0])
        tok = tokenizer.encode(x,return_tensors="pt")
        #tok = tok.to(device)
        out = model(tok, labels=tok).loss.item()
        ppl = math.exp(out)
        ppl_scores.append(ppl)
        predictions[str(i)] = [{"caption":x}]
        references[str(i)] = [{"caption":y}]
    print(100*(sum(bleu_scores)/len(bleu_scores)).item(), cs_score, sum(bleurt_scores)/len(bleurt_scores), sum(ppl_scores)/len(ppl_scores))
    predictions = ptb_tokenizer.tokenize(predictions)
    references = ptb_tokenizer.tokenize(references) 
    uh = cider.compute_score(gts=predictions, res=references)
    print("BLeU,CS,BLEURT,CIDER,PPL",100*(sum(bleu_scores)/len(bleu_scores)).item(), \
            cs_score, sum(bleurt_scores)/len(bleurt_scores), \
            uh[0], sum(ppl_scores)/len(ppl_scores))

dataset_name, epoch, train_eval_mode = sys.argv[1], sys.argv[2], sys.argv[3]
if train_eval_mode=="test":
    epoch="test"
with open("yi_results/"+dataset_name+"_ep_"+epoch+"_.json", "r") as f:
    results = json.load(f)

eval_ChartSumm(results["predictions"],results["targets"], train_eval_mode)
