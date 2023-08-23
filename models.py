import torch
from torch import nn
import torch.nn.functional as F
from transformers import T5Tokenizer,T5ForConditionalGeneration, T5Config
import os
import string
from data import processor as data_processor
from args import args

class T5Gen_Model(nn.Module):
    def __init__(self,model_path) -> None:
        super(T5Gen_Model,self).__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.T5_config = T5Config.from_pretrained(model_path)
        self.T5_config.dropout_rate = args.dropout_rate
        self.model = T5ForConditionalGeneration.from_pretrained(model_path,config=self.T5_config)
        if 'ATIS' in args.dataset_name:
            intent2nl = data_processor.atis_intent2lang
            slot2nl = data_processor.atis_slot2lang
        elif 'SNIPS' in args.dataset_name:
            intent2nl = data_processor.snips_intent2lang
            slot2nl = data_processor.snips_slot2lang
        self.nl2intent = {' '.join(intent2nl[key]):key for key in intent2nl}
        self.nl2slot = {' '.join(slot2nl[key]):key for key in slot2nl}

        num_of_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'The model has {num_of_params:,} trainable parameters.')

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        decoder_input_ids = None,
        decoder_attention_mask = None,
        head_mask = None,
        decoder_head_mask = None,
        cross_attn_head_mask = None,
        encoder_outputs = None,
        past_key_values = None,
        inputs_embeds = None,
        decoder_inputs_embeds = None,
        labels = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
    ):
        labels[labels == self.tokenizer.pad_token_id] = -100
        return self.model(input_ids=input_ids,attention_mask=attention_mask,labels=labels)

    def batch_forward(self,batch):
        i_outputs_loss = self.forward(
            input_ids=batch['intents_input_ids'],
            attention_mask=batch['intents_input_attention_mask'],
            labels=batch['intents_labels']
        ).loss
        s_outputs_loss = self.forward(
            input_ids=batch['slots_input_ids'],
            attention_mask=batch['slots_input_attention_mask'],
            labels=batch['slots_labels']
        ).loss
        p_outputs_loss = self.forward(
            input_ids=batch['pairs_input_ids'],
            attention_mask=batch['pairs_input_attention_mask'],
            labels=batch['pairs_labels']
        ).loss
        return i_outputs_loss,s_outputs_loss,p_outputs_loss

    def intents_prompt(self,text,intents=None,slots=None,pairs=None):
        return 'transfer sentence to intents : '+ text

    def slots_prompt(self,text,intents=None,slots=None,pairs=None):
        intents_utters = data_processor.intents_utters(intents)
        prefix = 'transfer sentence to slots with {} : '.format(intents_utters) 
        
        return prefix + text

    def pairs_prompt(self,text,intents=None,slots=None,pairs=None):
        intents_utters = data_processor.intents_utters(intents)
        prefix = 'transfer sentence to pairs with {} : '.format(intents_utters)

        return prefix + text

    def basic_decode(self,text,device='cpu'):
        temp = self.tokenizer(text,max_length=args.max_source_length, \
                                padding="max_length", \
                                truncation=True, \
                                return_tensors='pt')
        input_ids,attention_mask = temp.input_ids.to(device),temp.attention_mask.to(device)
        outputs = self.model.generate(input_ids=input_ids, \
                                        attention_mask=attention_mask, \
                                        max_length=args.max_target_length, \
                                        output_attentions=True, \
                                        return_dict_in_generate=True)
        outputs = outputs.sequences
        if isinstance(text,list):
            return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        else:
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def parse(self,text):
        if ', ' in text:
            parts = text.split(', ')
        else:
            parts = [text.strip()]
        res = []
        for part in parts:
            if ' : ' not in part:
                pass
            else:
                kv_tuple = part.split(' : ')
                if len(kv_tuple) != 2:
                    pass
                else:
                    k = kv_tuple[0].rstrip(string.digits).strip()
                    v = kv_tuple[1].strip()
                    res.append((k, v))
        return res

    def decode_intents(self,text,intents=None,slots=None,pairs=None,device='cpu'):
        def filter_intents(pred):
            kvs = self.parse(pred)
            output = []
            for i in kvs:
                if i[1] in self.nl2intent:
                    output.append(i[1].split(' '))
            return output

        if isinstance(text,list):
            if intents==None:
                intents = [None]*len(text)
            if slots==None:
                slots = [None]*len(text)
            if pairs==None:
                pairs = [None]*len(text)
            input_text = [self.intents_prompt(text=t,intents=i,slots=s,pairs=p) for t,i,s,p in zip(text,intents,slots,pairs)]
            pred = self.basic_decode(input_text,device=device)
            return [filter_intents(sample) for sample in pred],pred
        else:
            pred = self.basic_decode(self.intents_prompt(text=text,intents=intents,slots=slots,pairs=pairs),device=device)
            return filter_intents(pred),pred

    def decode_slots(self,text,intents=None,slots=None,pairs=None,device='cpu'):
        def filter_slots(pred):
            kvs = self.parse(pred)
            output = []
            for i in kvs:
                if i[1] in self.nl2slot:
                    output.append(i[1].split(' '))
            return output
        
        if isinstance(text,list):
            if intents == None:
                intents = [None]*len(text)
            if slots==None:
                slots = [None]*len(text)
            if pairs==None:
                pairs = [None]*len(text)
            input_text = [self.slots_prompt(text=t,intents=i,slots=s,pairs=p) for t,i,s,p in zip(text,intents,slots,pairs)]
            pred = self.basic_decode(input_text,device=device)
            return [filter_slots(sample) for sample in pred],pred
        else:
            pred = self.basic_decode(self.slots_prompt(text=text,intents=intents,slots=slots,pairs=pairs),device=device)
            return filter_slots(pred),pred

    def decode_pairs(self,text,intents=None,slots=None,pairs=None,device='cpu'):
        def filter_pairs(pred):
            kvs = self.parse(pred)
            output = []
            for i in kvs:
                if i[0] in self.nl2slot:
                    output.append([i[0].split(' '),i[1].split(' ')])
            return output

        if isinstance(text,list):
            if intents == None:
                intents = [None]*len(text)
            if slots == None:
                slots = [None]*len(text)
            if pairs == None:
                pairs = [None]*len(text)
            input_text = [self.pairs_prompt(text=t,intents=i,slots=s,pairs=p) for t,i,s,p in zip(text,intents,slots,pairs)]
            pred = self.basic_decode(input_text,device=device)
            return [filter_pairs(sample) for sample in pred],pred
        else:
            pred = self.basic_decode(self.pairs_prompt(text=text,intents=intents,slots=slots,pairs=pairs),device=device)
            return filter_pairs(pred),pred

    def decode(self,text,intents=None,slots=None,pairs=None,device='cpu'):
        new_intents = None
        new_slots = None
        new_pairs = None
        intents_pred = None
        slots_pred = None
        pairs_pred = None

        if intents == None:
            intents,intents_pred = self.decode_intents(text=text,intents=intents,slots=slots,pairs=pairs,device=device)
            new_intents = intents
        else:
            new_intents,intents_pred = self.decode_intents(text=text,intents=intents,slots=slots,pairs=pairs,device=device)
        if slots == None:
            slots,slots_pred = self.decode_slots(text=text,intents=intents,slots=slots,pairs=pairs,device=device)
            new_slots = slots
        else:
            new_slots,slots_pred = self.decode_slots(text=text,intents=intents,slots=slots,pairs=pairs,device=device)
        if pairs == None:
            pairs,pairs_pred = self.decode_pairs(text=text,intents=intents,slots=slots,pairs=pairs,device=device)
            new_pairs = pairs
        else:
            new_pairs,pairs_pred = self.decode_pairs(text=text,intents=intents,slots=slots,pairs=pairs,device=device)
        
        
        return new_pairs,new_slots,new_intents,pairs_pred,slots_pred,intents_pred
    
    def save_model(self,save_path,accelerator=None):
        os.makedirs(save_path,exist_ok=True)
        if accelerator != None:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(self.model)
            if accelerator.is_main_process:
                self.tokenizer.save_pretrained(save_path)
                unwrapped_model.save_pretrained(save_path, save_function=accelerator.save)
        else:
            unwrapped_model = self.model
            self.tokenizer.save_pretrained(save_path)
            unwrapped_model.save_pretrained(save_path)
        print('Successfully save checkpoint at \"{}\"'.format(save_path))
