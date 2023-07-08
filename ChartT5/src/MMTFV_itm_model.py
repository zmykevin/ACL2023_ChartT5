import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modeling_t5 import VLT5

class VLT5MMTFVITM(VLT5):
    def __init__(self, config, num_answers=3, label2ans={0:"support", 1:"refute", 2:"nei"}):
        super().__init__(config)
        if config.classifier:
            self.answer_head = nn.Sequential(
                nn.Linear(config.d_model, config.d_model * 2),
                nn.GELU(),
                nn.LayerNorm(config.d_model * 2),
                nn.Linear(config.d_model * 2, num_answers)
            )

        self.num_answers = num_answers
        self.label2ans = label2ans
        self.bce_loss = nn.BCEWithLogitsLoss()

    def train_step(self, batch):

        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)
        # input_ids = batch['input_ids'].to(device)
        B = len(input_ids)
        # V_L = batch['vis_feats'].size(2)
        # vis_feats = batch['vis_feats'].to(device).view(B, 2*V_L, 2048)
        # vis_pos = batch['boxes'].to(device).view(B, 2*V_L, 4)
        result = {}
        if self.config.classifier:
            B = len(input_ids)

            decoder_input_ids = torch.ones(
                B, 1, dtype=torch.long, device=device) * self.config.decoder_start_token_id

            output = self(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos),
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=True,
                return_dict=True
            )
            target = batch['targets'].to(device)

            last_layer_hidden_state = output.decoder_hidden_states[-1]
            last_hidden_state = last_layer_hidden_state.view(B, -1, self.config.d_model)[:, -1]

            # [B, num_answers]
            logit = self.answer_head(last_hidden_state)

            loss = self.bce_loss(logit, target)

            score, pred_ans_id = logit.max(1)
            pred_ans_id = pred_ans_id.cpu().numpy()
            pred_ans = [self.label2ans[ans_id] for ans_id in pred_ans_id]

            result['pred_ans_ids'] = pred_ans_id
            result['pred_ans'] = pred_ans
        else:
            #TODO
            lm_labels = batch["target_ids"].to(device)

            # img_order_ids = [0] * V_L + [1] * V_L
            # img_order_ids = torch.tensor(img_order_ids, dtype=torch.long, device=device)
            # img_order_ids = img_order_ids.view(1, 2*V_L).expand(B, -1)

            # obj_order_ids = torch.arange(V_L, dtype=torch.long, device=device)
            # obj_order_ids = obj_order_ids.view(1, 1, V_L).expand(B, 2, -1).contiguous().view(B, 2*V_L)

            output = self(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos),
                labels=lm_labels,
                return_dict=True
            )
            assert 'loss' in output

            lm_mask = (lm_labels != -100).float()
            B, L = lm_labels.size()

            loss = output['loss']

            loss = loss.view(B, L) * lm_mask

            loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)  # B

            loss = loss.mean()

            

            # logits = output['logits'].detach()[:, 0]
            # logits = logits.view(B, self.lm_head.out_features)
            # true_logit = logits[:, self.true_id]
            # false_logit = logits[:, self.false_id]

            # pred_true = true_logit > false_logit
            # pred_true = pred_true.long().cpu().numpy()
            # result['pred_ans_id'] = pred_true
        # result = {
        #     'loss': loss
        # }
        result['loss'] = loss
        return result
    @torch.no_grad()
    def test_step(self, batch, **kwargs):
        self.eval()
        device = next(self.parameters()).device
        input_ids = batch['input_ids'].to(device)
        vis_feats = batch['vis_feats'].to(device)
        vis_pos = batch['boxes'].to(device)

        # img_order_ids = [0] * V_L + [1] * V_L
        # img_order_ids = torch.tensor(img_order_ids, dtype=torch.long, device=device)
        # img_order_ids = img_order_ids.view(1, 2*V_L).expand(B, -1)

        # obj_order_ids = torch.arange(V_L, dtype=torch.long, device=device)
        # obj_order_ids = obj_order_ids.view(1, 1, V_L).expand(B, 2, -1).contiguous().view(B, 2*V_L)
        
        result = {}
        if self.config.classifier:
            B = len(input_ids)

            decoder_input_ids = torch.ones(
                B, 1, dtype=torch.long, device=device) * self.config.decoder_start_token_id

            output = self(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos),
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=True,
                return_dict=True
            )

            last_layer_hidden_state = output.decoder_hidden_states[-1]
            last_hidden_state = last_layer_hidden_state.view(B, -1, self.config.d_model)[:, -1]

            # [B, num_answers]
            logit = self.answer_head(last_hidden_state)

            score, pred_ans_id = logit.max(1)
            pred_ans_id = pred_ans_id.cpu().numpy()
            pred_ans = [self.label2ans[ans_id] for ans_id in pred_ans_id]

            result['pred_ans_ids'] = pred_ans_id
            result['pred_ans'] = pred_ans
        else:
            #TO BE Implemented
            output = self.generate(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos),
                **kwargs
            )
            generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)
            result['token_ids'] = output
            result['pred_ans'] = generated_sents
        # decoder_input_ids = torch.ones(B, 1, dtype=torch.long, device=device) * self.config.decoder_start_token_id

        # output = self(
        #     input_ids=input_ids,
        #     vis_inputs=(vis_feats, vis_pos),
        #     decoder_input_ids=decoder_input_ids,
        #     return_dict=True
        # )

        # logits = output['logits'].detach()[:, 0]
        # logits = logits.view(B, self.lm_head.out_features)
        # true_logit = logits[:, self.true_id]
        # false_logit = logits[:, self.false_id]

        # pred_true = true_logit > false_logit
        # pred_true = pred_true.long().cpu().numpy()

        # result = {}
        # result['pred_ans_id'] = pred_true

        return result
