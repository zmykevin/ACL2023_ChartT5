from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modeling_t5 import VLT5, ChartT5

num_symbol = "<num_extra_id_0>"
MANTISSA_NORM = 10
EXPONENT_NORM = 20

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

class VLT5ChartQA(ChartT5):
    def __init__(self, config, num_answers=None, label2ans=None):
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

        #Load the visocrs and visocrs_bboxes information
        visocrs = batch.get('visocrs', None)
        #print(visocrs)
        if visocrs is not None:
            #print(visocrs)
            visocrs = visocrs.to(device)
        visocrs_bboxes = batch.get('visocrs_bboxes', None)
        if visocrs_bboxes is not None:
            #print(visocrs_bboxes)
            visocrs_bboxes = visocrs_bboxes.to(device)

        #Added by Mingyang for numerical modeling
        input_mantissa = batch.get('input_mantissa', None)
        if input_mantissa is not None:
            input_mantissa = input_mantissa.to(device)
        
        input_exponent = batch.get('input_exponent', None)
        if input_exponent is not None:
            input_exponent = input_exponent.to(device)
        
        visocrs_mantissa = batch.get('visocr_mantissa', None)
        if visocrs_mantissa is not None:
            visocrs_mantissa = visocrs_mantissa.to(device)
        
        visocrs_exponent = batch.get('visocr_exponent', None)
        if visocrs_exponent is not None:
            visocrs_exponent = visocrs_exponent.to(device)
        # input_mantissa = None
        # input_exponent = None
        # visocrs_mantissa = None
        # visocrs_exponent = None
        
        #Initialize the numerical values; 
        # if input_mantissa is not None:
        #     input_numericals = (input_mantissa, input_exponent)
        # else:
        #     input_numericals = None

        # if visocrs_mantissa is not None:
        #     visocrs_numericals = (visocrs_mantissa, visocrs_exponent)
        # else:
        #     visocrs_numericals = None

        if self.config.classifier:
            B = len(input_ids)

            decoder_input_ids = torch.ones(
                B, 1, dtype=torch.long, device=device) * self.config.decoder_start_token_id

            # output = self(
            #     input_ids=input_ids,
            #     vis_inputs=(vis_feats, vis_pos),
            #     visocrs_inputs=(visocrs, visocrs_bboxes), #Encode the visocr information
            #     decoder_input_ids=decoder_input_ids,
            #     output_hidden_states=True,
            #     return_dict=True
            # )
            
            output = self(
                input_ids=input_ids,
                input_numericals = (input_mantissa, input_exponent), #Encode the input nuemerical information. 
                vis_inputs=(vis_feats, vis_pos), 
                visocrs_inputs=(visocrs, visocrs_bboxes), #Encode the visocr information
                visocrs_numericals = (visocrs_mantissa, visocrs_exponent), #Encode the visocr numerical information
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

        else:
            target_mantissa = batch.get('target_mantissa', None)
            if target_mantissa is not None:
                target_mantissa = target_mantissa.to(device)
            
            target_exponent = batch.get('target_exponent', None)
            if target_exponent is not None:
                target_exponent = target_exponent.to(device)
            
            target_numericals_mask = batch.get('target_numericals_mask', None)
            if target_numericals_mask is not None:
                target_numericals_mask = target_numericals_mask.to(device)


            lm_labels = batch["target_ids"].to(device)
            lm_label_weights = torch.ones(self.tokenizer.vocab_size).to(device)
            lm_label_weights[-1] = 0.1
            output = self(
                input_ids=input_ids,
                input_numericals = (input_mantissa, input_exponent),
                vis_inputs=(vis_feats, vis_pos),
                visocrs_inputs=(visocrs, visocrs_bboxes), 
                visocrs_numericals = (visocrs_mantissa, visocrs_exponent),
                labels=lm_labels,
                labels_numericals = (target_mantissa, target_exponent),
                labels_numericals_mask = target_numericals_mask,
                lm_label_weights=lm_label_weights,
                return_dict=True
            )
            assert 'loss' in output
            #print(output.keys())
            lm_mask = (lm_labels != -100).float()
            B, L = lm_labels.size()

            loss = output['loss']
            
            # print(loss.size())
            # print(lm_mask.size())
            loss = loss.view(B, L) * lm_mask

            loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)  # B

            loss = loss * batch['scores'].to(device=device)

            loss = loss.mean()

            if 'numerical_loss' in output:
                #print(lm_labels)
                numerical_mask = (lm_labels == 32300).float()
                numerical_loss = output['numerical_loss']
                numerical_loss = numerical_loss.view(B, L).sum(dim=1) / numerical_mask.sum(dim=1).clamp(min=1)
                numerical_loss = numerical_loss * batch['scores'].to(device=device)
                numerical_loss = numerical_loss.mean()

                loss = loss + 10*numerical_loss

                result = {'loss': loss,
                          'numerical_loss': numerical_loss,
                         }
                return result

        result = {
            'loss': loss
        }

        return result

    @torch.no_grad()
    def test_step(self, batch, **kwargs):
        self.eval()
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)
        
        #Load the visocrs and visocrs_bboxes information
        visocrs = batch.get('visocrs', None)
        if visocrs is not None:
            #print(visocrs)
            visocrs = visocrs.to(device)
        visocrs_bboxes = batch.get('visocrs_bboxes', None)
        if visocrs_bboxes is not None:
            #print(visocrs_bboxes)
            visocrs_bboxes = visocrs_bboxes.to(device)
        
        #Added by Mingyang for numerical modeling
        input_mantissa = batch.get('input_mantissa', None)
        if input_mantissa is not None:
            input_mantissa = input_mantissa.to(device)
        
        input_exponent = batch.get('input_exponent', None)
        if input_exponent is not None:
            input_exponent = input_exponent.to(device)
        
        visocrs_mantissa = batch.get('visocr_mantissa', None)
        if visocrs_mantissa is not None:
            visocrs_mantissa = visocrs_mantissa.to(device)
        
        visocrs_exponent = batch.get('visocr_exponent', None)
        if visocrs_exponent is not None:
            visocrs_exponent = visocrs_exponent.to(device)


        result = {}
        if self.config.classifier:
            #print("classifier")
            B = len(input_ids)
            #print(B)
            decoder_input_ids = torch.ones(
                B, 1, dtype=torch.long, device=device) * self.config.decoder_start_token_id

            output = self(
                input_ids=input_ids,
                input_numericals = (input_mantissa, input_exponent), #Encode the input nuemerical information. 
                vis_inputs=(vis_feats, vis_pos), 
                visocrs_inputs=(visocrs, visocrs_bboxes), #Encode the visocr information
                visocrs_numericals = (visocrs_mantissa, visocrs_exponent), #Encode the visocr numerical information
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

            result['pred_ans'] = pred_ans

        else:
            #print("generator")
            # target_mantissa = batch.get('target_mantissa', None)
            # if target_mantissa is not None:
            #     target_mantissa = target_mantissa.unsqueeze(-1).to(device)
            
            # target_exponent = batch.get('target_exponent', None)
            # if target_exponent is not None:
            #     target_exponent = target_exponent.unsqueeze(-1).to(device)

            output = self.generate(
                input_ids=input_ids,
                input_numericals = (input_mantissa, input_exponent),
                vis_inputs=(vis_feats, vis_pos),
                visocrs_numericals = (visocrs_mantissa, visocrs_exponent),
                visocrs_inputs=(visocrs, visocrs_bboxes), #Encode the visocr information
                **kwargs
            )
            #print(batch['target_ids'])
            
            # print(output.size())
            # print the exponent and mantissa output

            #generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)
            
            
            #print(output.size())
            predicted_numericals = []
            if visocrs_mantissa is not None:
                numerical_prediction_output = self(
                    input_ids=input_ids,
                    input_numericals = (input_mantissa, input_exponent),
                    vis_inputs=(vis_feats, vis_pos),
                    visocrs_inputs=(visocrs, visocrs_bboxes), 
                    visocrs_numericals = (visocrs_mantissa, visocrs_exponent),
                    decoder_input_ids = output,
                    return_numerical_value = True,
                    return_dict=True
                )
                B, L = output.size()
                numerical_mask = (output == 32300).float()
                # print(numerical_mask.size())
                numerical_mantissa_prediction = (numerical_prediction_output['numerical_mantissa_prediction'].view(B,L)*numerical_mask).detach().cpu().numpy()
                numerical_exponent_prediction = (numerical_prediction_output['numerical_exponent_prediction'].view(B,L)*numerical_mask).detach().cpu().numpy()
                
                prediction_shape = numerical_mantissa_prediction.shape
                
                print(numerical_mantissa_prediction)
                predicted_numericals = []
                for i in range(prediction_shape[0]):
                    current_batch = []
                    for j in range(prediction_shape[1]):
                        if numerical_mantissa_prediction[i,j] != 0.0:
                            recovered_numerics = recover_numeric(numerical_mantissa_prediction[i,j], numerical_exponent_prediction[i,j])
                            current_batch.append(recovered_numerics)
                        else:
                            current_batch.append(0.0)
                    predicted_numericals.append(current_batch)
            print(predicted_numericals)
            if predicted_numericals:
                generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True, num_token_values=predicted_numericals)
            else:
                generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)
                
            #print(generated_sents)

                # print(numerical_prediction_output['numerical_mantissa_prediction'].size())
            #print(type(generated_sents))

            #     B = len(input_ids)
            #     #print(B)
            #     decoder_input_ids = torch.ones(
            #         B, 1, dtype=torch.long, device=device) * self.config.decoder_start_token_id
                # print(numerical_prediction_output['numerical_mantissa_prediction'].size())
                # print(output.size())
            #print(generated_sents)

            result['token_ids'] = output
            result['pred_ans'] = generated_sents

        return result

from modeling_bart import VLBart
class VLBartChartQA(VLBart):
    def __init__(self, config, num_answers=None, label2ans=None):
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

        if self.config.classifier:
            B = len(input_ids)

            decoder_input_ids = torch.tensor(
                [self.config.decoder_start_token_id, self.config.bos_token_id],
                dtype=torch.long, device=device).unsqueeze(0).expand(B, 2)

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

        else:
            lm_labels = batch["target_ids"].to(device)

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

            loss = loss * batch['scores'].to(device=device)

            loss = loss.mean()

        result = {
            'loss': loss
        }

        return result

    @torch.no_grad()
    def test_step(self, batch, **kwargs):
        self.eval()
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)

        result = {}
        if self.config.classifier:
            B = len(input_ids)

            decoder_input_ids = torch.tensor(
                [self.config.decoder_start_token_id, self.config.bos_token_id],
                dtype=torch.long, device=device).unsqueeze(0).expand(B, 2)

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

            result['pred_ans'] = pred_ans

        else:

            output = self.generate(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos),
                **kwargs
            )
            generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)
            result['token_ids'] = output
            result['pred_ans'] = generated_sents

        return result