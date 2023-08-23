import os
import json
import string
from args import args

class SlotValuePicker():
    def __init__(self) -> None:
        pass

    def __startOfChunk(self,prevTag, tag, prevTagType, tagType, chunkStart=False):
        if prevTag == 'B' and tag == 'B':
            chunkStart = True
        if prevTag == 'I' and tag == 'B':
            chunkStart = True
        if prevTag == 'O' and tag == 'B':
            chunkStart = True
        if prevTag == 'O' and tag == 'I':
            chunkStart = True

        if prevTag == 'E' and tag == 'E':
            chunkStart = True
        if prevTag == 'E' and tag == 'I':
            chunkStart = True
        if prevTag == 'O' and tag == 'E':
            chunkStart = True
        if prevTag == 'O' and tag == 'I':
            chunkStart = True

        if tag != 'O' and tag != '.' and prevTagType != tagType:
            chunkStart = True
        return chunkStart

    def __endOfChunk(self,prevTag, tag, prevTagType, tagType, chunkEnd=False):
        if prevTag == 'B' and tag == 'B':
            chunkEnd = True
        if prevTag == 'B' and tag == 'O':
            chunkEnd = True
        if prevTag == 'I' and tag == 'B':
            chunkEnd = True
        if prevTag == 'I' and tag == 'O':
            chunkEnd = True

        if prevTag == 'E' and tag == 'E':
            chunkEnd = True
        if prevTag == 'E' and tag == 'I':
            chunkEnd = True
        if prevTag == 'E' and tag == 'O':
            chunkEnd = True
        if prevTag == 'I' and tag == 'O':
            chunkEnd = True

        if prevTag != 'O' and prevTag != '.' and prevTagType != tagType:
            chunkEnd = True
        return chunkEnd

    def __splitTagType(self,tag):
        s = tag.split('-')
        if len(s) > 2 or len(s) == 0:
            raise ValueError('tag format wrong. it must be B-xxx.xxx')
        if len(s) == 1:
            tag = s[0]
            tagType = ""
        else:
            tag = s[0]
            tagType = s[1]
        return tag, tagType

    def pick_slots_and_values(self,words_list,labels_list):
        assert len(words_list) == len(labels_list)
        is_in = False
        lastTag = 'O'
        lastTagType = ''
        this_pair = None
        slot_value_pairs = []
        for token,label in zip(words_list,labels_list):
            tag,tagType = self.__splitTagType(label)
            if is_in:
                if self.__endOfChunk(lastTag,tag,lastTagType,tagType):
                    this_pair = (this_pair[0],' '.join(this_pair[1]))
                    slot_value_pairs.append(this_pair)
                    this_pair = None
                    is_in = False
                else:
                    this_pair[1].append(token)
            
            if self.__startOfChunk(lastTag,tag,lastTagType,tagType):
                this_pair = (tagType,[token])
                is_in = True
            
            lastTag = tag
            lastTagType = tagType
        
        if is_in:
            this_pair = (this_pair[0],' '.join(this_pair[1]))
            slot_value_pairs.append(this_pair)
        return slot_value_pairs

class DataProcess():
    def __init__(self) -> None:
        self.picker = SlotValuePicker()
        with open(os.path.join('data','map',args.map,'ATIS_map','intent_label2lang.json'),'r') as f:
            self.atis_intent2lang = json.load(f)
        with open(os.path.join('data','map',args.map,'ATIS_map','slot_label2lang.json'),'r') as f:
            self.atis_slot2lang = json.load(f)
        with open(os.path.join('data','map',args.map,'SNIPS_map','intent_label2lang.json'),'r') as f:
            self.snips_intent2lang = json.load(f)
        with open(os.path.join('data','map',args.map,'SNIPS_map','slot_label2lang.json'),'r') as f:
            self.snips_slot2lang = json.load(f)
        self.lang2index = {}
        if 'ATIS' in args.dataset_name:
            for index,old in enumerate(self.atis_intent2lang):
                self.lang2index[' '.join(self.atis_intent2lang[old])] = index
        else:
            for index,old in enumerate(self.snips_intent2lang):
                self.lang2index[' '.join(self.snips_intent2lang[old])] = index

    def read_file(self,dataset_name,file_name):
        texts, slots, intents = [], [], []
        samples = []
        text, slot = [], []

        with open(os.path.join('data','raw',dataset_name,file_name), 'r', encoding="utf8") as fr:
            for line in fr.readlines():
                items = line.strip().split()

                if len(items) == 1:
                    if "/" not in items[0]:
                        this_intents = items[0]
                    else:
                        this_intents = items[0].split("/")[1]
                    
                    if '#' in this_intents:
                        this_intents = this_intents.split('#')
                    else:
                        this_intents = [this_intents]
                    
                    texts.append(text)
                    intents.append(this_intents)
                    slots.append(slot)
                    # clear buffer lists.
                    text, slot = [], []

                elif len(items) == 2:
                    word = items[0].strip()
                    tag = items[1].strip()
                    text.append(word)
                    slot.append(tag)
        return texts,intents,slots

    def if_existed(self,prior_list,t):
        repeated_slots_num = 0
        for key,value in prior_list:
            if key.strip(string.digits) == t[0]:
                if value == t[1]:
                    return True, -1
                else:
                    repeated_slots_num += 1
        return False, repeated_slots_num


    def get_slot_value_pairs(self,words_list,labels_list):
        slot_value_pairs = self.picker.pick_slots_and_values(words_list,labels_list)
        results = []
        slots = []
        for index,pair in enumerate(slot_value_pairs):
            flag, num = self.if_existed(slot_value_pairs[:index],pair)
            if flag:
                pass
            else:
                results.append([pair[0],pair[1].split(' ')])
                if num == 0:
                    slots.append(pair[0])
        return results,slots

    def intents_utters(self,intents):
        intents_utters = []
        for index,intent in enumerate(intents):
            intents_utters.append(['intent',':'] + intent + ([','] if index<len(intents)-1 else []))
        intents_utters = sum(intents_utters,[])
        intents_utters = ' '.join(intents_utters)
        return intents_utters

    def slots_utters(self,slots):
        slots_utters = []
        for index,slot in enumerate(slots):
            slots_utters.append(['slot',':'] + slot + ([','] if index<len(slots)-1 else []))
        slots_utters = sum(slots_utters,[])
        slots_utters = ' '.join(slots_utters)
        return slots_utters

    def pairs_utters(self,pairs):
        pairs_utters = []
        for index,pair in enumerate(pairs):
            pairs_utters.append(pair[0] + [':'] + pair[1] + ([','] if index<len(pairs)-1 else []))
        pairs_utters = sum(pairs_utters,[])
        pairs_utters = ' '.join(pairs_utters)
        return pairs_utters
    
    def transform(self,sample,intent_label2lang,slot_label2lang,mode='train'):
        new_sample = {}
        new_sample['text'] = sample[0]
        new_sample['intents'] = [intent_label2lang[intent] for intent in sample[2]]
        slot_value_pairs,slots = self.get_slot_value_pairs(sample[0],sample[1])
        new_sample['slots'] = [slot_label2lang[slot] for slot in slots]
        
        new_sample['pairs'] = []
        for pair in slot_value_pairs:
            slot = slot_label2lang[pair[0]]
            value = pair[1]
            new_sample['pairs'].append([slot,value])
        
        return new_sample
    
    def add_instance(self,vocabulary, instance, multi_intent=False):
        if isinstance(instance, (list, tuple)):
            for element in instance:
                self.add_instance(vocabulary, element, multi_intent=multi_intent)
            return

        assert isinstance(instance, str)
        if multi_intent and '#' in instance:
            for element in instance.split('#'):
                self.add_instance(vocabulary, element, multi_intent=multi_intent)
            return

        if instance not in vocabulary:
            vocabulary.append(instance)

    def data_init(self,dataset_name):
        train_texts,train_intents,train_slots = self.read_file(dataset_name,'train.txt')
        dev_texts,dev_intents,dev_slots = self.read_file(dataset_name,'dev.txt')
        test_texts,test_intents,test_slots = self.read_file(dataset_name,'test.txt')
        
        index2slot = []
        index2intent = []
        self.add_instance(index2slot, train_slots)
        self.add_instance(index2slot, dev_slots)
        self.add_instance(index2slot, test_slots)
        self.add_instance(index2intent, train_intents, multi_intent=True)
        self.add_instance(index2intent, dev_intents, multi_intent=True)
        self.add_instance(index2intent, test_intents, multi_intent=True)
        os.makedirs(os.path.join('data',args.map,dataset_name),exist_ok=True)
        with open(os.path.join('data',args.map,dataset_name,'index2intent.json'),'w') as f:
            json.dump(index2intent,f)
        with open(os.path.join('data',args.map,dataset_name,'index2slot.json'),'w') as f:
            json.dump(index2slot,f)
        
        intent_label2lang = {}
        slot_label2lang = {}
        if 'ATIS' in dataset_name:
            intent_label2lang.update(self.atis_intent2lang)
            slot_label2lang.update(self.atis_slot2lang)
        if 'SNIPS' in dataset_name:
            intent_label2lang.update(self.snips_intent2lang)
            slot_label2lang.update(self.snips_slot2lang)
        train_samples = [self.transform(sample,intent_label2lang,slot_label2lang,mode='train') for sample in zip(train_texts,train_slots,train_intents)]
        dev_samples = [self.transform(sample,intent_label2lang,slot_label2lang,mode='test') for sample in zip(dev_texts,dev_slots,dev_intents)]
        test_samples = [self.transform(sample,intent_label2lang,slot_label2lang,mode='test') for sample in zip(test_texts,test_slots,test_intents)]
        
        with open(os.path.join('data',args.map,dataset_name,'train.json'),'w') as f:
            for output_sample in train_samples:
                f.write(json.dumps(output_sample)+'\n')
        with open(os.path.join('data',args.map,dataset_name,'dev.json'),'w') as f:
            for output_sample in dev_samples:
                f.write(json.dumps(output_sample)+'\n')
        with open(os.path.join('data',args.map,dataset_name,'test.json'),'w') as f:
            for output_sample in test_samples:
                f.write(json.dumps(output_sample)+'\n')

processor = DataProcess()
if args.mode == 'train':
    if not os.path.exists(os.path.join('data',args.map,args.dataset_name)):
        processor.data_init(args.dataset_name)
        print('******************* data is changed *******************')
    else:
        print('******************* data not changed *******************')