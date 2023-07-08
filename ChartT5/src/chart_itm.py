import collections
import os
import random
from pathlib import Path
import logging
import shutil
from packaging import version


from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from param import parse_args
from chart_itm_data import get_loader
from utils import LossMeter
from dist_utils import reduce_dict

import wandb

#set_global_logging_level(logging.ERROR, ["transformers"])
os.environ["TOKENIZERS_PARALLELISM"] = "false"
proj_dir = Path(__file__).resolve().parent.parent


_use_native_amp = False
_use_apex = False

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transormers.file_utils import is_apex_available
    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast

from trainer_base import TrainerBase

class Trainer(TrainerBase):
    def __init__(self, args, train_loader=None, val_loader=None, test_loader=None, train=True):
        super().__init__(
            args,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            train=train)

        from chart_itm_model import VLT5ChartITM
        
        assert 't5' in args.backbone
        #if 't5' in args.backbone:
        model_class = VLT5ChartITM

        config = self.create_config()
        self.tokenizer = self.create_tokenizer()

        self.model = self.create_model(model_class, config)

        #if 't5' in self.args.tokenizer:
        self.model.resize_token_embeddings(self.tokenizer.vocab_size)
        

        self.model.tokenizer = self.tokenizer
        if 't5' in self.args.tokenizer or 'bart' in self.args.tokenizer:
            self.model.true_id = self.tokenizer('true', add_special_tokens=False).input_ids[0]
            self.model.false_id = self.tokenizer('false', add_special_tokens=False).input_ids[0]

        # Load Checkpoint
        self.start_epoch = None
        if args.load is not None:
            ckpt_path = args.load + '.pth'
            self.load_checkpoint(ckpt_path)

        if self.args.from_scratch:
            self.init_weights()

        # GPU Options
        print(f'Model Launching at GPU {self.args.gpu}')
        if self.verbose:
            from time import time
            start = time()
        self.model = self.model.to(args.gpu)

        # Optimizer
        if train:
            self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler()

            if self.args.fp16 and _use_native_amp:
                self.scaler = torch.cuda.amp.GradScaler()
            elif _use_apex:
                self.model, self.optim = amp.initialize(
                    self.model, self.optim, opt_level='O1', verbosity=self.verbose)

        if args.multiGPU:
            if args.distributed:
                self.model = DDP(self.model, device_ids=[args.gpu],
                                find_unused_parameters=True
                                )
        if self.verbose:
            print(f'It took {time() - start:.1f}s')

    def train(self):
        if self.verbose:
            loss_meter = LossMeter()


            quesid2ans = {}
            best_valid = 0.
            best_epoch = 0

            #if 't5' in self.args.backbone:
            if self.args.use_vision:
                project_name = "VLT5_Chart_ITM"
            else:
                project_name = "T5_Chart_ITM"
            

            wandb.init(project=project_name)
            wandb.run.name = self.args.run_name
            wandb.config.update(self.args)
            wandb.watch(self.model)

            src_dir = Path(__file__).resolve().parent
            base_path = str(src_dir.parent)
            src_dir = str(src_dir)
            wandb.save(os.path.join(src_dir + "/*.py"), base_path=base_path)

        if self.args.distributed:
            dist.barrier()

        global_step = 0
        for epoch in range(self.args.epochs):
            if self.start_epoch is not None:
                epoch += self.start_epoch
            self.model.train()
            if self.args.distributed:
                self.train_loader.sampler.set_epoch(epoch)
            if self.verbose:
                pbar = tqdm(total=len(self.train_loader), ncols=200)
                itm_results = np.zeros(4, dtype=int)

            epoch_results = {
                'loss': 0,

            }
            quesid2ans = {}
            train_acc = 0.
            train_acc_steps = int(len(self.train_loader) * 0.05)
            last_acc_step = 0
            
            #Let's try this.
            self.evaluate(self.val_loader)

            for step_i, batch in enumerate(self.train_loader):

                self.optim.zero_grad()
                if self.args.fp16 and _use_native_amp:
                    with autocast():
                        if self.args.distributed:
                            results = self.model.module.train_step(batch)
                        else:
                            results = self.model.train_step(batch)
                else:
                    if self.args.distributed:
                        results = self.model.module.train_step(batch)
                    else:
                        results = self.model.train_step(batch)

                loss = results['loss']

                # if self.args.gradient_accumulation_steps > 1:
                #     loss /= self.args.gradient_accumulation_steps

                if self.args.fp16 and _use_native_amp:
                    self.scaler.scale(loss).backward()
                elif self.args.fp16 and _use_apex:
                    with amp.scale_loss(loss, self.optim) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                
                loss = loss.detach()

                # Update Parameters
                if self.args.clip_grad_norm > 0:
                    if self.args.fp16 and _use_native_amp:
                        self.scaler.unscale_(self.optim)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.args.clip_grad_norm)
                    elif self.args.fp16 and _use_apex:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(
                            self.optim), self.args.clip_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.args.clip_grad_norm)

                if self.args.fp16 and _use_native_amp:
                    self.scaler.step(self.optim)
                    self.scaler.update()
                else:
                    self.optim.step()

                if self.lr_scheduler:
                    self.lr_scheduler.step()
                # self.model.zero_grad()
                for param in self.model.parameters():
                    param.grad = None

                global_step += 1

                for k, v in results.items():
                    if k in epoch_results:
                        epoch_results[k] += v.item()

                if self.lr_scheduler:
                    if version.parse(torch.__version__) >= version.parse("1.4"):
                        lr = self.lr_scheduler.get_last_lr()[0]
                    else:
                        lr = self.lr_scheduler.get_lr()[0]
                else:
                    try:
                        lr = self.optim.get_lr()[0]
                    except AttributeError:
                        lr = self.args.lr

                if self.verbose:
                    loss_meter.update(loss.item())
                    desc_str = f'Epoch {epoch} | LR {lr:.6f} | '
                    desc_str += f'Loss {loss_meter.val:4f} |'

                    pred_ans = results['pred_ans_id']
                    ques_ids = batch['question_ids']

                    for qid, ans in zip(ques_ids, pred_ans):
                        quesid2ans[qid] = ans

                    label = batch['labels'].cpu().numpy()
                    predict = results['pred_ans_id']
                    #itm_results[0] += sum((label == 1) & (predict == 1)) + sum((label == 0) & (predict == 0)) + sum((label == 2) & (predict == 2))

                    # itm_results[0] += sum((label == 1) & (predict == 1))
                    # itm_results[1] += sum((label == 1) & (predict == 0))
                    # itm_results[2] += sum((label == 0) & (predict == 1))
                    # itm_results[3] += sum((label == 0) & (predict == 0))
                    # n_total = sum(itm_results)

                    # desc_str += f' TP {itm_results[0]} ({itm_results[0]/n_total*100:.1f}%)'
                    # desc_str += f' FN {itm_results[1]} ({itm_results[1]/n_total*100:.1f}%)'
                    # desc_str += f' FP {itm_results[2]} ({itm_results[2]/n_total*100:.1f}%)'
                    # desc_str += f' TN {itm_results[3]} ({itm_results[3]/n_total*100:.1f}%)'

                    pbar.set_description(desc_str)
                    pbar.update(1)

            if self.verbose:
                pbar.close()

                log_str = ''

                # train_score_dict = self.train_loader.evaluator.evaluate(quesid2ans)
                # train_acc = train_score_dict['accuracy']  * 100.
                # train_cons = train_score_dict['consistency'] * 100.

                train_acc = self.train_loader.evaluator.evaluate_train(quesid2ans) * 100.

                train_score = train_acc

                log_str += "\nEpoch %d: Train %0.2f" % (epoch, train_score)

                # Validation
                valid_score_dict = self.evaluate(self.val_loader)
                valid_acc = valid_score_dict['accuracy'] * 100.
                # valid_cons = valid_score_dict['consistency'] * 100.

                valid_score = valid_acc

                if valid_score > best_valid:
                    best_valid = valid_score
                    best_epoch = epoch
                    self.save("BEST")

                log_str += "\nEpoch %d: Valid %0.2f" % (epoch, valid_score)
                log_str += "\nEpoch %d: Best %0.2f\n" % (best_epoch, best_valid)

                wandb_log_dict = {}
                # wandb_log_dict['Train/Loss'] = loss_meter.val
                # wandb_log_dict['Train/score'] = score
                # wandb_log_dict['Valid/score'] = valid_score

                # for score_name, score in train_score_dict.items():
                    # wandb_log_dict[f'Train/{score_name}'] = score * 100.
                wandb_log_dict['Train/accuracy'] = train_acc

                for score_name, score in valid_score_dict.items():
                    wandb_log_dict[f'Valid/{score_name}'] = score * 100.

                wandb.log(wandb_log_dict, step=epoch)

                print(log_str)

            if self.args.distributed:
                dist.barrier()

        if self.verbose:
            self.save("LAST")

            # Test Set
            best_path = os.path.join(self.args.output, 'BEST')
            self.load(best_path)

            log_str = 'Test set results\n'

            dump_path = os.path.join(self.args.output, 'submit.csv')
            test_score_dict = self.evaluate(self.test_loader, dump_path=dump_path)
            wandb.save(dump_path, base_path=self.args.output)

            wandb_log_dict = {}
            for score_name, score in test_score_dict.items():
                wandb_log_dict[f'Test/{score_name}'] = score * 100.
            wandb.log(wandb_log_dict, step=epoch)

            from pprint import pformat

            log_str += pformat(test_score_dict)

            print(log_str)


            wandb.log({'finished': True})

    def predict(self, loader, dump_path=None):
        """
        Predict the answers to questions in a data split.
        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        with torch.no_grad():
            quesid2ans = {}
            for i, batch in enumerate(tqdm(loader, ncols=150, desc="Prediction")):

                if self.args.distributed:
                    results = self.model.module.test_step(batch)
                else:
                    results = self.model.test_step(batch)

                pred_ans = results['pred_ans_id']
                ques_ids = batch['question_ids']

                for qid, ans in zip(ques_ids, pred_ans):
                    quesid2ans[qid] = ans

            if dump_path is not None:
                loader.evaluator.dump_result(quesid2ans, dump_path)
            return quesid2ans

    def evaluate(self, loader, dump_path=None):
        #print(len(loader))
        evaluator = loader.evaluator
        quesid2ans = self.predict(loader, dump_path)
        return evaluator.evaluate(quesid2ans)

def main_worker(gpu, args):
    # GPU is assigned
    args.gpu = gpu
    args.rank = gpu
    print(f'Process Launching at GPU {gpu}')

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl')

    print(f'Building train loader at GPU {gpu}')
    train_loader = get_loader(
        args,
        split=args.train, mode='train', batch_size=args.batch_size,
        distributed=args.distributed, gpu=args.gpu,
        workers=args.num_workers,
        topk=args.train_topk,
    )

    print(f'Building val loader at GPU {gpu}')
    val_loader = get_loader(
        args,
        split=args.valid, mode='val', batch_size=args.batch_size,
        distributed=args.distributed, gpu=args.gpu,
        workers=4,
        topk=args.valid_topk,
    )
    print(f'Building test loader at GPU {gpu}')
    test_loader = get_loader(
        args,
        split=args.test, mode='val', batch_size=args.batch_size,
        distributed=args.distributed, gpu=args.gpu,
        workers=4,
        topk=args.valid_topk,
    )

    trainer = Trainer(args, train_loader, val_loader, test_loader, train=True)
    trainer.train()

if __name__ == "__main__":
    cudnn.benchmark = True
    args = parse_args()
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node
    if args.local_rank in [0, -1]:
        print(args)

        comments = []
        if args.load is not None:
            ckpt_str = "_".join(args.load.split('/')[-3:])
            comments.append(ckpt_str)
        if args.comment != '':
            comments.append(args.comment)
        comment = '_'.join(comments)

        # if not args.test:
        from datetime import datetime
        current_time = datetime.now().strftime('%b%d_%H-%M')

        run_name = f'{current_time}_GPU{args.world_size}'
        if len(comments) > 0:
            run_name += f'_{comment}'

        args.run_name = run_name

    if args.distributed:
        main_worker(args.local_rank, args)
