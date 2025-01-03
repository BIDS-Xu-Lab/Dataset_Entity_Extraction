# "presentation_of_case
# 把名字去掉，用4o改写一下，换一种方式表述同样的意思

import pandas as pd
from openai import OpenAI
import json
from tqdm import trange
import numpy as np
from multiprocessing import Pool
import time
import random


def generate(index, prompt, question):
    """
    # Determine whether the each_diagnosis and true_diagnosis is a same diagnosis
    # each_diagnosis: generated diagnosis, str
    # true_diagnosis: true diagnosis, str
    """
    # time.sleep(random.randint(1, 3))
    # return 'Y'    # use, when you check the diagnoses list
    client = OpenAI()
        
    chat_return = client.chat.completions.create(model='gpt-4o',temperature=0.0, messages=[{"role": "system", "content": prompt}, {"role":"user", "content": question}])

    result = chat_return.choices[0].message.content
    return result


data = pd.read_csv("/home/gy237/project/llama3/new_data/not_selected_nejm_12.2/not_selected_nejm (1).csv")

prompt = 'Please thoroughly rephrase the provided case presentation without losing information and professionalism.'

text = '''A 36-year-old man was admitted to the hospital because of chest pain, dysphagia, dyspnea, and pleural and mediastinal calcifications.
He had been well until approximately 6 years before admission, when a 182-kg weight fell on his shoulders while he was lifting weights at a gym. Intermittent left-sided chest pain occurred thereafter, gradually increased in severity, and responded transiently to nonsteroidal antiinflammatory medications and ice packs. Two and a half years before admission, dysphagia developed in association with solid foods and gradually increased in severity to include liquids; he lost 4.5 kg in weight. An esophagogram reportedly showed minimal tertiary contraction in the distal esophagus, with possible associated reflux and esophageal spasm, and no mucosal abnormality. Twenty-two months before admission, a chest radiograph at another hospital showed left pleural calcifications, which had not been present 3 years earlier.
A complete blood count and serum levels of glucose, electrolytes, calcium, phosphorus, magnesium, protein, and albumin were normal. The alkaline phosphatase level was 236 U per liter (reference range, 45 to 115). Results of pulmonary-function studies are shown in Table 1. A tuberculin skin test was nonreactive. A modified barium-swallow study showed high-grade stenosis of the distal esophagus at the gastroesophageal junction. Computed tomography (CT) of the chest revealed posterior mediastinal calcifications encircling the aorta and esophagus, extensive left pleural calcification, and right diaphragmatic plaques. The patient was admitted to the other hospital. Esophagoscopy revealed a tight circumferential extrinsic compression at 40 cm, impassable to the endoscope, and no mucosal lesions. A left rib resection and pleural biopsy were performed. Pathological examination of the biopsy specimen revealed reactive fibrous tissue with ossification, no organisms, and no cancer. Cytologic examination of the pleural fluid revealed mesothelial cells and lymphocytes. A subculture of the specimen in anaerobic broth grew Propionibacterium acnes; other cultures were sterile.
Eighteen months before the current admission, the patient was admitted to this hospital for resection of the extrinsic esophageal stricture. On examination, breath sounds and chest-wall excursions were markedly diminished on the left side. A complete blood count; serum levels of glucose, electrolytes, calcium, phosphorus, magnesium, protein, albumin, vitamin D, parathyroid hormone, iron, iron-binding capacity, vitamin B12, folate, bilirubin, and alanine and aspartate aminotransferases; tests of coagulation and renal function; and serum protein electrophoresis were normal. Two units of blood drawn for future autotransfusion were positive on screening for antibody to the human T-cell lymphotropic virus type I (HTLV-I). The results of a Western blot test were indeterminate. Testing for antibody to human immunodeficiency virus (HIV) was negative.
At operation, an incision in the left sixth intercostal space disclosed pleural calcification, more than 1 cm thick, which prevented exposure of the left lung. A right thoracotomy revealed patches of wafer-thin calcification on the visceral pleura of the lower lobe, subpleural bone formation in the diaphragm, and a posterior mediastinal calcific mass surrounding the esophagus, inferior pulmonary ligament, vena cava, and aorta. A holmium laser was used to make an incision in the calcified tissue surrounding the esophagus, resulting in a release of the esophagus. After the release, a 60-French dilator was passed through the esophagus without resistance. A wedge biopsy of the right lower lobe was performed; pathological examination of the tissue revealed a dense fibroinflammatory process, with acute and chronic inflammation, focal necrosis, and ossification and no microorganisms. Cultures of the specimen grew propionibacterium species in the tube of thioglycollate broth and one colony of penicillium in fungal culture. Testing for antinuclear antibodies, anti–double-stranded DNA antibodies, and antibodies to anti–Scl-70 antibodies, and antibodies to Aspergillus fumigatus, Thermoactinomyces sacchari, T. candidus, T. vulgaris, Saccharomonospora viridis, Saccharopolyspora rectivirgula (formerly Micropolyspora faeni), and pigeon serum was negative.
Dysphagia improved for 1 month and then recurred, with increasing dyspnea. Fiberoptic esophagoscopy at the other hospital identified a recurrent lower-esophageal stricture. Dilation was unsuccessful and resulted in a superficial mucosal injury. An electrocardiogram revealed sinus rhythm, nondiagnostic inferior Q waves, an incomplete right bundle-branch block, and nonspecific ST-segment and T-wave abnormalities. Bone densitometry was normal. One month later, the patient was readmitted to this hospital; right thoracotomy with resection of posterior mediastinal calcification, lateral suspension of the lower esophagus, and esophageal plication were performed. Pathological examination of the tissue showed fibrosis extending into the soft tissues of the chest wall, with multiple foci of active endochondral ossification adjacent to tufts of smooth muscle. Staining for keratin was negative. The pleural fluid revealed a total protein level of 4.1 g per deciliter and a triglyceride level of 30 mg per deciliter (0.3 mmol per liter). After discharge, a 4-week course of oral corticosteroids was administered. Eleven months before admission, dysphagia and dyspnea gradually recurred. During the next 9 months, esophagogastroduodenoscopy with dilatation was performed on four occasions. Dyspnea increased, and the patient was unable to continue working. There was no cough or sputum production. Results of pulmonary-function testing 4 months before admission are shown in Table 1. Echocardiography revealed no abnormalities.
Three months before admission, the patient began to have episodes of light-headedness lasting 4 to 5 minutes, without loss of consciousness and associated with weakness in the arms and legs, and he was admitted to another hospital. Orthostatic vital signs revealed mild postural changes. Electrocardiography showed sinus tachycardia. An echocardiogram at rest and magnetic resonance imaging (MRI) of the brain and cervical spine were reportedly normal. Telemetry, electroencephalography, and levels of cortisol, creatine kinase, and troponin were normal. He was transferred to this hospital. Electroencephalography was normal, and MRI of the heart and aorta with angiography showed the calcifications previously seen on CT and a free-flowing pericardial effusion, with no evidence of pericardial constriction or abnormal enhancement of the pericardium. Olanzapine was prescribed for anxiety.
One month later, the patient was readmitted to this hospital. He had a history of optical migraine headaches and a lipoma of the right chest wall. He was married with two children. He worked in construction and had worked in a motorcycle and bicycle repair shop as a teenager. He had traveled to the Caribbean in the past, smoked briefly as a teenager, and had stopped drinking alcohol several years earlier. His maternal grandmother had breast cancer in her early 50s; his parents and siblings were well. Oxycodone had caused tachycardia; there were no other known allergies. Medications included olanzapine, lorazepam, and ibuprofen. On examination, the pulse was 76 beats per minute, the blood pressure 150/70 mm Hg, the respiratory rate 16 breaths per minute, the oxygen saturation 99% while the patient was breathing oxygen (5 liters), and the temperature 35.8°C. The remainder of the examination was unchanged.
A diagnostic procedure was performed.'''

for index, row in data.iterrows():
    if index == 122:
        question = text
    else:
        question = data.loc[index, 'presentation_of_case']

    # 修改presentation_of_case中医生名字开头的case reports
    if not question.startswith('A'):
        question = ':'.join(question.split(':')[1:])
    assert '-year-old' in question or '-month-old' in question or '-day-old' in question or 'A newborn' in question or 'A woman in her 90s' in question or 'A pediatric surgeon at this hospital was contacted by a nonprofit organization to' in question or 'A male infant was admitted to this hospital at the age of 5.5 months,' in question, data.loc[index, 'id']

    data.loc[index, 'presentation_of_case'] = question



def process_chunk(index_split, prompts_split, questions_split):
    results = []
    for i in trange(len(index_split)):
        refined_case_report = generate(index_split[i], prompts_split[i],questions_split[i])
        results.append((index_split[i], refined_case_report))
    return results

num_tasks = 20

index = data.index.tolist()
questions = data['presentation_of_case'].tolist()
prompts = [prompt]*len(index)

index_split = np.array_split(index, num_tasks)
questions_split = np.array_split(questions, num_tasks)
prompts_split = np.array_split(prompts, num_tasks)


with Pool(num_tasks) as pool:
    results = pool.starmap(process_chunk, zip(index_split, prompts_split, questions_split))

for chunk in results:
    for i in chunk:
        data.loc[i[0], 'refined_case_report'] = i[1]

# data.to_csv("/home/gy237/project/llama3/new_data/not_selected_nejm_12.2/not_selected_nejm_refined.tsv", sep='\t', index=False)