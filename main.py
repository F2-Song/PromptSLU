from args import args
from models import T5Gen_Model
import os
# print(f'pid: {os.getpid()}')
import logging
import json
import math
import random
import string
import numpy as np
import torch
import datasets
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import f1_score

import transformers
from accelerate import Accelerator
from transformers import (
    AdamW,
    default_data_collator,
)
from data import processor as data_processor

global main_model
main_model = T5Gen_Model
logger = logging.getLogger(__name__)

def train(evaluate_func = None):
    accelerator = Accelerator()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    data_files = {}
    if args.train_file is not None:
        data_files["train"] = os.path.join('data',args.map,args.dataset_name,args.train_file)
    if args.validation_file is not None:
        data_files["valid"] = os.path.join('data',args.map,args.dataset_name,args.validation_file)
    extension = args.train_file.split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files)

    model = main_model(args.model_name_or_path)
    tokenizer = model.tokenizer
    column_names = raw_datasets["train"].column_names

    padding = "max_length"
    def preprocess_function(examples):
        intents_inputs = []
        slots_inputs = []
        pairs_inputs = []
        intents_tgts = []
        slots_tgts = []
        pairs_tgts = []
        for batch_index,text in enumerate(examples['text']):
            text = ' '.join(text)
            intents_input = model.intents_prompt(text=text,intents=examples['intents'][batch_index],slots=examples['slots'][batch_index],pairs=examples['pairs'][batch_index])
            slots_input = model.slots_prompt(text=text, intents=examples['intents'][batch_index],slots=examples['slots'][batch_index],pairs=examples['pairs'][batch_index])
            pairs_input = model.pairs_prompt(text=text, intents=examples['intents'][batch_index],slots=examples['slots'][batch_index],pairs=examples['pairs'][batch_index])
            
            intents_tgt = data_processor.intents_utters(examples['intents'][batch_index])
            slots_tgt = data_processor.slots_utters(examples['slots'][batch_index])
            pairs_tgt = data_processor.pairs_utters(examples['pairs'][batch_index])

            intents_inputs.append(intents_input)
            slots_inputs.append(slots_input)
            pairs_inputs.append(pairs_input)
            intents_tgts.append(intents_tgt)
            slots_tgts.append(slots_tgt)
            pairs_tgts.append(pairs_tgt)
        
        i = tokenizer(intents_inputs, max_length=args.max_source_length, padding=padding, truncation=True)
        s = tokenizer(slots_inputs, max_length=args.max_source_length, padding=padding, truncation=True)
        p = tokenizer(pairs_inputs, max_length=args.max_source_length, padding=padding, truncation=True)
        i_tgt = tokenizer(intents_tgts, max_length=args.max_target_length, padding=padding, truncation=True)
        s_tgt = tokenizer(slots_tgts, max_length=args.max_target_length, padding=padding, truncation=True)
        p_tgt = tokenizer(pairs_tgts, max_length=args.max_target_length, padding=padding, truncation=True)
        
        res = {
            'input_ids':i['input_ids']+s['input_ids']+p['input_ids'],
            'input_attention_mask':i['attention_mask']+s['attention_mask']+p['attention_mask'],
            'labels':i_tgt['input_ids']+s_tgt['input_ids']+p_tgt['input_ids']
        }
        return res

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
    data_collator = default_data_collator

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )
    
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    last_overall_acc = -1
    last_epoch = -1
    for epoch in range(args.num_train_epochs):
        os.makedirs(os.path.join(args.output_dir,str(epoch)), exist_ok=True)
        model.train()
        for step, batch in enumerate(train_dataloader):
            loss = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['input_attention_mask'],
                labels=batch['labels']
            ).loss
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            print('\nloss = {:.4f}'.format(loss.item()))
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break
        
        model.eval()
        with torch.no_grad():
            accelerator.wait_for_everyone()
            if evaluate_func != None:
                if accelerator.is_main_process:
                    print('\n[Epoch {}] dev:'.format(epoch))
                    _, _, current_overall_acc = evaluate_func(accelerator.unwrap_model(model),file='dev.json',device=accelerator.device)
                    if current_overall_acc > last_overall_acc:
                        if_update = True
                    else:
                        if_update = False
                    
                    if if_update:
                        print('[Epoch {}] test:'.format(epoch))
                        last_test_slot_f1,last_test_intent_acc,last_test_overall_acc = evaluate_func(accelerator.unwrap_model(model),file='test.json',device=accelerator.device)
                        test_slot_f1,test_intent_acc,test_overall_acc = last_test_slot_f1,last_test_intent_acc,last_test_overall_acc
                        accelerator.unwrap_model(model).save_model(os.path.join(args.output_dir,str(epoch)))
                        last_overall_acc = current_overall_acc
                        last_epoch = epoch
                        print('Checkpoint has been updated.')        
                    
        accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        os.symlink(str(last_epoch), os.path.join(args.output_dir, 'best_checkpoint'))
        print('\nAt epoch {}, the model acquires the best result on the validation set'.format(last_epoch))
        print('Corresponding result on the test set: ')
        print('         slot_f1 : {:.4f}'.format(test_slot_f1))
        print('      intent_acc : {:.4f}'.format(test_intent_acc))
        print('     overall_acc : {:.4f}'.format(test_overall_acc))
              
def evaluate(model=None,file=args.validation_file,device='cpu'):
    with open(os.path.join('data',args.map,args.dataset_name,file),'r') as f:
        raw_dev = f.readlines()
    dev = []
    for line in raw_dev:
        dev.append(json.loads(line))
    if args.mode != 'train':
        progress_bar = tqdm(range(1 + int(len(dev)/args.per_device_eval_batch_size)))

    goldens_pairs_num = 0.0
    pairs_num = 0.0
    pairs_valid_pred_num = 0.0
    
    golden_intents_binary = []
    intents_binary = []

    intent_acc = 0.0
    overall_acc = 0.0

    if model==None:
        model = main_model(args.output_dir).to(device)
    model.eval()
    input_buffer = []
    input_intents = []
    golden_buffer = []
    for index,sample in enumerate(dev):
        input_buffer.append(' '.join(sample['text']))
        input_intents.append(sample['intents'])
        golden_buffer.append((sample['pairs'],sample['slots'],sample['intents']))
        
        if (index+1) % args.per_device_eval_batch_size == 0 or (index+1) == len(dev):
            pairs_buffer,slots_buffer,intents_buffer,pairs_pred,slots_pred,intents_pred = model.decode(input_buffer,intents=None,device=device)
            pred = list(zip(pairs_buffer,intents_buffer))
            assert len(pred) == len(golden_buffer)

            for k in range(len(pred)):
                pairs,intents = pred[k]
                golden_pairs,golden_slots,golden_intents = golden_buffer[k]
                if intents != None:
                    golden_intents = list(set([' '.join(intent) for intent in golden_intents]))
                    intents = list(set([' '.join(intent) for intent in intents]))
                    if len(intents)>0 and args.dataset_name == 'SNIPS':
                        intents = intents[0:1]
                    golden_intents.sort()
                    intents.sort()

                    golden_intents_binary.append([0]*len(data_processor.lang2index))
                    intents_binary.append([0]*len(data_processor.lang2index))
                    for intent in golden_intents:
                        if intent in data_processor.lang2index:
                            golden_intents_binary[-1][data_processor.lang2index[intent]] = 1
                        else:
                            print("golden:",intent)
                            pass
                    for intent in intents:
                        if intent in data_processor.lang2index:
                            intents_binary[-1][data_processor.lang2index[intent]] = 1
                        else:
                            pass
                if pairs != None:
                    golden_pairs = list(set(['{}={}'.format(' '.join(pair[0]).rstrip(string.digits).strip(), ' '.join(pair[1])) for pair in golden_pairs]))
                    pairs = list(set(['{}={}'.format(' '.join(pair[0]).rstrip(string.digits).strip(), ' '.join(pair[1])) for pair in pairs]))
                    golden_pairs.sort()
                    pairs.sort()
                    left_in = 0
                    right_in = 0
                    for i in golden_pairs:
                        if i in pairs:
                            left_in += 1
                    for i in pairs:
                        if i in golden_pairs:
                            right_in += 1
                    assert left_in == right_in
                    pairs_valid_pred_num += left_in
                    goldens_pairs_num += len(golden_pairs)
                    pairs_num += len(pairs)
                
                if intents == golden_intents:
                    intent_acc += 1
                    if pairs == golden_pairs:
                        overall_acc += 1
            
            if args.mode != 'train':
                progress_bar.update(1)
                
            input_buffer = []
            input_intents = []
            golden_buffer = []
        else:
            pass

        
    overall_acc = overall_acc / len(dev)
    if len(intents_binary)>0:
        intent_acc = intent_acc / len(dev)
        intent_f1 = f1_score(golden_intents_binary,intents_binary,average='macro')
    else:
        intent_acc = 0
        intent_f1 = 0
    
    if pairs_num > 0:
        slot_precision = pairs_valid_pred_num / pairs_num
    else:
        slot_precision = 0
    if goldens_pairs_num > 0:
        slot_recall = pairs_valid_pred_num / goldens_pairs_num
    else:
        slot_recall = 0
    if (slot_precision+slot_recall) > 0:
        slot_f1 = 2*slot_precision*slot_recall / (slot_precision+slot_recall)
    else:
        slot_f1 = 0

    print('\n  slot_precision : {:.4f}'.format(slot_precision))
    print('     slot_recall : {:.4f}'.format(slot_recall))
    print('         slot_f1 : {:.4f}'.format(slot_f1))
    print('       intent_f1 : {:.4f}'.format(intent_f1))
    print('     overall_acc : {:.4f}'.format(overall_acc))
    print('      intent_acc : {:.4f}'.format(intent_acc))

    return slot_f1,intent_acc,overall_acc


if __name__ == "__main__":    
    print('hyper params:')
    for key in args.__dict__:
        print(f"\t{key}:{args.__dict__[key]}")
    print('\n*************************************\n')
    if args.mode == 'train':        
        train(evaluate)
    if args.mode == 'test' or args.mode == 'dev':
        evaluate(device='cuda')