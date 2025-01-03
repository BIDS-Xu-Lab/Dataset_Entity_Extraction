import json


cas = 'Infectious Disease'
input_file = '/home/gy237/project/llama3/summary/html.txt'
summary = '/home/gy237/project/llama3/summary/summary.json'


with open(summary, 'r', encoding='utf-8') as file:
    data = json.load(file)

with open(input_file, 'r', encoding='utf-8') as file:
    inpt = file.read()

inpt_list = inpt.split('issue-item_title-link animation-underline')
# print(len(inpt_list))

title_list = [i.split(' ')[1] for i in inpt_list[1:]]

title = []
for i in title_list:
    month, year = i.split('-')
    month = month.strip()
    year = year.strip()
    if year.endswith(':'):
        year = year[:-1]
    year_n = int(year)
    month_n = int(month)
    if len(month) == 1:
        month = '0' + month
    
    title.append(f'nejm-case-{year}-{month}')
print(len(title))
print(len(set(title)))
# print(title)


for i in data:
    id_ = i['id']
    assert cas not in i['topic']
    for tit in title:
        if tit == id_:
            i['topic'].append(cas)

with open(summary, 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)