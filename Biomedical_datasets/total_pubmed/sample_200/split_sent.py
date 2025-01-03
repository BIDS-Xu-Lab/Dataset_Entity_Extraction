import json
import os
import spacy
nlp = spacy.load("en_core_web_sm")
import string


with open('/home/gy237/project/Biomedical_datasets/extraction_pipeline/total_pubmed/sample_200/sample_300_abstract.json','r', encoding='utf-8') as file:
    data = json.load(file)
print(len(data))


def ends_with_punctuation(text):
    return text[-1] in string.punctuation if text else False

empty = []
new_data = []
for i in data:
    pmid = i['pmid']
    doc = nlp(i['abstract'])
    abstracts = [sent.text for sent in doc.sents]
    if len(abstracts) == 0:
        empty.append(pmid)
    else:
        new_data.append({'id':pmid, 'abstracts':abstracts})

    new_abstracts = []
    for j in abstracts:
        j = j.strip().split('\n')   # 有些内部会有\n
        for k in j:
            if len(k.split(' ')) > 5 and ends_with_punctuation(k):
                new_abstracts.append(k)
    new_data.append({'id':i['id'], 'abstracts':new_new_})
print(len(empty))
print(len(new_data))




new_new = []
for i in new_data:
    new_new_ = []
    for j in i['abstracts']:
        j = j.strip().split('\n')   # 有些内部会有\n
        for k in j:
            if len(k.split(' ')) > 5 and ends_with_punctuation(k):
                new_new_.append(k)
    new_new.append({'id':i['id'], 'abstracts':new_new_})


for i in new_new[:100]:
    pmid = i['id']
    doc = i['abstracts']
    with open(f'/home/gy237/project/Biomedical_datasets/extraction_pipeline/total_pubmed/sample_200/Gui_collections/{pmid}.txt', 'w', encoding='utf-8') as f:
        for j in doc:
            f.write(j + '\n')

for i in new_new[100:200]:
    pmid = i['id']
    doc = i['abstracts']
    with open(f'/home/gy237/project/Biomedical_datasets/extraction_pipeline/total_pubmed/sample_200/Kalpana_collections/{pmid}.txt', 'w', encoding='utf-8') as f:
        for j in doc:
            f.write(j + '\n')