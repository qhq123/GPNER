import csv
import difflib
from multiprocessing.connection import answer_challenge
from random import sample
from tkinter import Label
import nltk
import time
import math
import torch

from seq2seq_model import Seq2SeqModel
from collections import defaultdict
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig, BartModel,T5Tokenizer,T5Model

# from utils_metrics import get_entities_bio, f1_score, classification_report, precision_score, recall_score
from seqeval.metrics import f1_score
from seqeval.metrics import precision_score
from seqeval.metrics import accuracy_score
from seqeval.metrics import recall_score
from seqeval.metrics import classification_report

from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(color_codes=True)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'



def cal_time(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


class InputExample():
    def __init__(self, words, labels):
        self.words = words
        self.labels = labels

class DiffPart():
    def __init__(self, start, end, span):
        self.start = start
        self.end = end
        self.span = span

def load_data(file_path):
    examples = []
    with open(file_path, "r", encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    examples.append(InputExample(words=words, labels=labels))
                    words = []
                    labels = []
            else:
                splits = line.split(" ")
                words.append(splits[0])
                if len(splits) > 1:
                    label = splits[-1].replace("\n", "")
                    if label == 'O':
                        labels.append("O")
                    else:
                        labels.append(label.split('-')[0] + '-' + label_to_lab[label.split('-')[-1]])
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            examples.append(InputExample(words=words, labels=labels))
    return examples

def get_similarity(labelword_embeddings, candidates, threshold = 0.95):
# def get_similarity(labelword_embeddings, candidates, threshold = 0.80):
    max_cos_similarity = []
    label_indexs = []
    for c in candidates:
        inputs = tokenizer(c, return_tensors="pt")
        outputs = bart_model(**inputs)
        last_hidden_states = outputs.last_hidden_state  # (1*n*1024)
        last_hidden_states = last_hidden_states[0, 0, :]
        last_hidden_states.unsqueeze_(0)
        word_embedding = last_hidden_states
        cosine_similarity = torch.cosine_similarity(word_embedding, labelword_embeddings).tolist()
        max_value = max(cosine_similarity)
        max_cos_similarity.append(max_value)
        label_indexs.append(cosine_similarity.index(max_value))
    
    max_value = max(max_cos_similarity)
    idx = max_cos_similarity.index(max_value)
    label_index = label_indexs[idx]

    if max_value < threshold:
        return -1, max_value
    # print(candidate, label_index)
    return label_index, max_value

def BIO(source, answer):
    # 首先将原句的label_word 打标
    source_words = nltk.word_tokenize(source)
    answer_words = nltk.word_tokenize(answer)
    # source_words = [':'] + source_words
    source_words = source_words[4:]
    # source_words = source_words[1:]
    # source_words = [':'] + source_words
    # answer_words = [':'] + answer_words
    # answer_words = answer_words[3:]

    s = difflib.SequenceMatcher(None, source_words, answer_words)
    diff_s_list = []
    diff_a_list = []
    dif = s.get_matching_blocks()
    for i in range(len(dif) - 1):
        item1 = dif[i]
        item2 = dif[i+1]
        diff_s_list.append(DiffPart(item1.a + item1.size, item2.a - 1, source_words[item1.a + item1.size:item2.a]))
        diff_a_list.append(DiffPart(item1.b + item1.size, item2.b - 1, answer_words[item1.b + item1.size:item2.b]))
    diff_s_list.pop()
    diff_a_list.pop()
    dif.pop()
    BIO_seq = ['O'] * len(source_words)

    # for diff_s, diff_a in zip(diff_s_list, diff_a_list):   
    #     label_index_list = [-1] * len(diff_s)
    #     for word in diff_a.span:
    #         label_index = get_similarity(labelword_embeddings, [word])
    #         label_index_list.append(label_index)

    for diff_s, diff_a in zip(diff_s_list, diff_a_list):
        # label被预测成多个词了
        # for i, word in enumerate(diff_a.span):
        #     word = word.capitalize()
        #     if i > 0:
        #         pre_label_index, pre_p = get_similarity(labelword_embeddings, [diff_a.span[i - 1]])
        #     label_index, p = get_similarity(labelword_embeddings, [word])
        #     if label_index == -1 and i > 0:
        #         merge_label_index, merge_max = get_similarity(labelword_embeddings, [diff_a.span[i-1] + word])
        #         if merge_label_index == pre_label_index and merge_max > pre_p:
        #             diff_a.span[i - 1] = diff_a.span[i] = diff_a.span[i-1] + word
        
        # label被预测成多个标签了
        # B-Geo B-Roc

        if diff_s.end - diff_s.start <= diff_a.end - diff_a.start:
            # 截断
            # end_word = answer_words[diff_a.end]
            # diff_a.end = diff_a.start + diff_s.end - diff_s.start
            # diff_a.span = answer_words[diff_a.start: diff_a.end]
            # diff_a.span.append(end_word)
            diff_a.end = diff_a.start + diff_s.end - diff_s.start
            diff_a.span = answer_words[diff_a.start: diff_a.end + 1]
        else:
            # 补齐
            word = diff_a.span[-1] if len(diff_a.span) > 0 else ''
            gap = (diff_s.end - diff_s.start) - (diff_a.end - diff_a.start)
            while gap > 0:
                diff_a.end += 1
                diff_a.span.append(word)
                gap -= 1


    for diff_s, diff_a in zip(diff_s_list, diff_a_list):
        label_index_list = []
        for word in diff_a.span:
            word = "".join(word[:1].upper() + word[1:]) #首字母大写
            label_index, _ = get_similarity(labelword_embeddings, [word])
            label_index_list.append(label_index)
        
        if len(label_index_list) > 1:
            for num, label_index in enumerate(label_index_list):
                if label_index == -1:
                    if num == 0:
                        # label_index_list[num] = label_index_list[num+1]
                        pass
                    else:
                        # if num == len(label_index_list) - 1:
                        #     break
                        label_index_list[num] = label_index_list[num-1]  
        
        for num, i in enumerate(range(diff_s.start, diff_s.end + 1)):
            label_index = label_index_list[num]
            if label_index == -1:
                BIO_seq[i] = 'O'
            else:
                if i == 0 or BIO_seq[i - 1] == 'O':
                    BIO_seq[i] = 'B-' + label_to_lab[label_words_to_labels[label_words[label_index]]]
                elif label_to_lab[label_words_to_labels[label_words[label_index]]] != BIO_seq[i-1].split('-')[-1]:
                    BIO_seq[i] = 'B-' + label_to_lab[label_words_to_labels[label_words[label_index]]]
                else:
                    BIO_seq[i] = 'I-' + label_to_lab[label_words_to_labels[label_words[label_index]]]
    
    replace_answer_list = []
    if len(dif) > 0 and dif[0].a == 0:
        for i in range(len(diff_a_list)):
            same_part = source_words[dif[i].a: dif[i].a + dif[i].size]
            dif_part = diff_a_list[i].span
            replace_answer_list += same_part
            replace_answer_list += dif_part
        if len(dif) > len(diff_a_list):
            same_part = source_words[dif[len(diff_a_list)].a: dif[len(diff_a_list)].a + dif[len(diff_a_list)].size]
            replace_answer_list += same_part
    
    # elif len(diff_a_list) > 0 and diff_a_list[0].start == 0:
    #     for i in range(len(diff_a_list)):
    #         dif_part = diff_a_list[i].span
    #         same_part = source_words[dif[i].a: dif[i].a + dif[i].size]
    #         replace_answer_list += dif_part
    #         replace_answer_list += same_part
    #     if len(diff_a_list) > len(dif):
    #         dif_part = diff_a_list[len(dif)].span
    #         replace_answer_list += dif_part

    return BIO_seq[1:], source_words, replace_answer_list

dataset_name = 'water_io_50-shot'
# dataset_name = 'test'
check_name = dataset_name
model_name = 'outputs_' + check_name
# model_name = 'outputs_water_io_40-shot'

model = Seq2SeqModel(
    encoder_decoder_type="bart",
    encoder_decoder_name=model_name+"/best_model",
)


# label_words = ['Slave craton', 'Archean']
# labels = {0: 'LOC', 1: 'SCA'}
#  5:'Stratigraphic_group_type', 6:'Geologic_age', -1:'O'}

label_words_to_labels = {'Location': 'Location', 'AquiferType': 'Aquifer_type', 'GroundwaterType': 'Groundwater_type','HydrochemicalType': 'Hydrochemical_type',
              'RockType': 'Rock_type', 'StratigraphicGroupType': 'Stratigraphic_group_type','GeologicAge':'Geologic_age'}
label_to_lab = {'Location': 'Loc', 'Aquifer_type': 'Aqu', 'Groundwater_type': 'Gro','Hydrochemical_type': 'Hyd',
              'Rock_type': 'Roc', 'Stratigraphic_group_type': 'Str','Geologic_age': 'Geo'}
label_words = {0: 'Location', 1: 'AquiferType', 2:'GroundwaterType', 3:'HydrochemicalType', 
4:'RockType', 5:'StratigraphicGroupType', 6:'GeologicAge', -1:'O'}

# label_words_to_labels = {'LabelA': 'Location', 'LabelB': 'Aquifer_type', 'LabelC': 'Groundwater_type','LabelD': 'Hydrochemical_type',
#               'LabelE': 'Rock_type', 'LabelF': 'Stratigraphic_group_type','LabelG':'Geologic_age'}
# label_to_lab = {'Location': 'Loc', 'Aquifer_type': 'Aqu', 'Groundwater_type': 'Gro','Hydrochemical_type': 'Hyd',
#               'Rock_type': 'Roc', 'Stratigraphic_group_type': 'Str','Geologic_age': 'Geo'}
# label_words = {0: 'LabelA', 1: 'LabelB', 2:'LabelC', 3:'LabelD', 4:'LabelE',
#  5:'LabelF', 6:'LabelG', -1:'O'}

# label_words_to_labels = {'AquiferType': 'Location', 'GroundwaterType': 'Aquifer_type', 'HydrochemicalType': 'Groundwater_type','RockType': 'Hydrochemical_type',
#               'StratigraphicGroupType': 'Rock_type', 'GeologicAge': 'Stratigraphic_group_type','Location':'Geologic_age'}
# label_to_lab = {'Location': 'Loc', 'Aquifer_type': 'Aqu', 'Groundwater_type': 'Gro','Hydrochemical_type': 'Hyd',
#               'Rock_type': 'Roc', 'Stratigraphic_group_type': 'Str','Geologic_age': 'Geo'}
# label_words = {0: 'AquiferType', 1: 'GroundwaterType', 2:'HydrochemicalType', 3:'RockType', 4:'StratigraphicGroupType',
#  5:'GeologicAge', 6:'Location', -1:'O'}

tokenizer = BartTokenizer.from_pretrained(model_name+"/best_model")
bart_model = BartModel.from_pretrained(model_name+"/best_model")
labelword_embeddings = []
for label_word in label_words_to_labels.keys():
    inputs = tokenizer(label_word, return_tensors="pt")
    outputs = bart_model(**inputs)
    last_hidden_states = outputs.last_hidden_state # (1*n*1024)
    last_hidden_states = last_hidden_states[0, 0, :]
    last_hidden_states.unsqueeze_(0)
    # print(last_hidden_states.shape, last_hidden_states.dim())
    labelword_embeddings.append(last_hidden_states)

labelword_embeddings = torch.cat(labelword_embeddings, dim=0)

# get_similarity(labelword_embeddings, ["Slave sample", "Slave"], threshold=0.90)



file_path = dataset_name + "/test.txt"
# file_path = 'test' + "/test.txt"
examples = load_data(file_path)
num_01 = len(examples)
num_point = 0
start = time.time()

trues_list = []
preds_list = []

# examples_pred = []
# with open('my_uw/result.csv', encoding='utf-8') as csvfile:
#     reader=csv.reader(csvfile)
#     examples_pred=[row[2] for row in reader]

# for example in examples:
for i, example in enumerate(examples):
    print('%d/%d (%s)' % (num_point + 1, num_01, cal_time(start)))
    # source = ' '.join(example.words)
    source = 'replace entity to label : ' + ' '.join(examples[i].words)
    # source = 'input : ' + ' '.join(examples[i].words)
    # source = ' '.join(examples[i].words)
    # source = ' '.join(examples[i].words)
    # answer = examples_pred[1:][i]
    answer = model.predict([source])[0]
    index = answer.find(':')
    answer = answer[index:]
    # answer = ': ' + answer
    predicted, replace_source_list, replace_answer_list = BIO(source, answer)
    preds_list.append(predicted)
    trues_list.append(example.labels)

    print_preds = ""
    print_trues = ""
    for i, j, k in zip(example.words, preds_list[num_point], trues_list[num_point]):
        print_preds += j.rjust(6)
        print_trues += k.rjust(6)
    print('source:',source)
    print('answer:',answer)
    print('words:', example.words)
    print('Pred:', print_preds)
    print('Gold:', print_trues)
    with open( dataset_name + '/' + check_name+ 'result.txt','a',encoding='utf-8') as f:
        f.write('source:' + str(source) + '\n')
        f.write('answer:' + str(answer) + '\n')
        f.write('source_list:' + str(replace_source_list) + '\n')
        f.write('answer_list:' + str(replace_answer_list) + '\n')
        f.write('words:' + str(example.words) + '\n')
        f.write('Pred:' + str(print_preds) + '\n')
        f.write('Gold:' + str(print_trues) + '\n')
        f.write('\n')
    num_point += 1

y_true = trues_list
y_pred = preds_list
print("accuary: ", accuracy_score(y_true, y_pred))
print("p: ", precision_score(y_true, y_pred))
print("r: ", recall_score(y_true, y_pred))
print("f1: ", f1_score(y_true, y_pred))
print("classification report: ")
print(classification_report(y_true, y_pred))


y_pred = sum(y_pred,[])
y_true = sum(y_true,[])
for i in range(len(y_pred)):
    s=y_pred[i].split('-')
    if len(s)>1:
        y_pred[i]=s[1]
for i in range(len(y_true)):
    s=y_true[i].split('-')
    if len(s)>1:
        y_true[i]=s[1]


cf_matrix = confusion_matrix(y_true, y_pred,labels=['O','Loc','Aqu','Gro','Hyd','Roc','Str','Geo'])

ax = sns.heatmap(cf_matrix / np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues',annot_kws={"fontsize":8})
ax.title.set_text("Confusion Matrix (percent)")
ax.set_xlabel("y_pred")
ax.set_ylabel("y_true")
ax.set_xticklabels(['O','Loc','Aqu','Gro','Hyd','Roc','Str','Geo'])
ax.set_yticklabels(['O','Loc','Aqu','Gro','Hyd','Roc','Str','Geo'])

plt.show()
plt.savefig('./1.png')



plt.clf()
cf_matrix = confusion_matrix(y_true, y_pred,labels=['Loc','Aqu','Gro','Hyd','Roc','Str','Geo'])

ax = sns.heatmap(cf_matrix / np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues',annot_kws={"fontsize":8})
ax.title.set_text("Confusion Matrix (percent)")
ax.set_xlabel("y_pred")
ax.set_ylabel("y_true")
ax.set_xticklabels(['Loc','Aqu','Gro','Hyd','Roc','Str','Geo'])
ax.set_yticklabels(['Loc','Aqu','Gro','Hyd','Roc','Str','Geo'])

plt.show()
plt.savefig('./2.png')
# true_entities = get_entities_bio(trues_list)
# pred_entities = get_entities_bio(preds_list)

# results = {
#     "p": precision_score(true_entities, pred_entities),
#     "r": recall_score(true_entities, pred_entities),
#     "f1": f1_score(true_entities, pred_entities),
# }
# print(results)  

# labels_true_entities = []
# labels_pred_entities = []
# labels_score = []
# for i in range(7):
#     labels_true_entities.append(set())
#     labels_pred_entities.append(set())


# labels = ['Loc', 'Aqu', 'Gro','Hyd', 'Roc', 'Str','Geo']
# for true_entity in true_entities:
#     label = true_entity[0]
#     span = (true_entity[1], true_entity[2])
#     index = labels.index(label)
#     labels_true_entities[index].add(span)

# for pred_entity in pred_entities:
#     label = pred_entity[0]
#     span = (pred_entity[1], pred_entity[2])
#     index = labels.index(label)
#     labels_pred_entities[index].add(span)

# for index, label in enumerate(labels):
#     label_true_entities = labels_true_entities[index]
#     label_pred_entities = labels_pred_entities[index]
#     label_pscore = precision_score(label_true_entities, label_pred_entities)
#     label_rscore = recall_score(label_true_entities, label_pred_entities)
#     label_fscore = f1_score(label_true_entities, label_pred_entities)
#     print(label, label_pscore, label_rscore, label_fscore)





