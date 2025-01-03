import json
from datasets import load_dataset
# from unsloth import FastLanguageModel
# max_seq_length = 8192 # Choose any! We auto support RoPE Scaling internally!
# dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
# load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
# model, tokenizer = FastLanguageModel.from_pretrained(
#         model_name = "unsloth/llama-3-8b-bnb-4bit", # YOUR MODEL YOU USED FOR TRAINING
#         max_seq_length = max_seq_length,
#         dtype = dtype,
#         load_in_4bit = load_in_4bit,
#     )

# def filtering(data, inn, out):
#     new_data = []
#     for i in data:
#         id_ = i['id']
#         prompt, inpt = i["query"].split('INPUT:')
#         oupt = i["answer"]

#         inputs = tokenizer([inpt], return_tensors = "pt")
#         input_len = len(inputs['input_ids'][0])
#         outputs = tokenizer([oupt], return_tensors = "pt")
#         output_len = len(outputs['input_ids'][0])

#         if inn< input_len < 8100 and out < output_len < 8100:
#             new_data.append(i)
    
#     return new_data




# with open('llama3/new_data/upload_hug/GENE_OMIM_SY_train.json', 'r', encoding='utf-8') as f:
#     GENE_OMIM_SY_train = json.load(f)
# print(f'GENE_OMIM_SY_train: {len(GENE_OMIM_SY_train)}')
# GENE_OMIM_SY_train = filtering(GENE_OMIM_SY_train, 10, 100)
# print(len(GENE_OMIM_SY_train))
# with open(f'/home/gy237/project/llama3/new_data/final_filter/GENE_OMIM_SY_train.json', 'w', encoding='utf-8') as file:
#     json.dump(GENE_OMIM_SY_train, file, ensure_ascii=False, indent=4)


# with open('/home/gy237/project/llama3/new_data/upload_hug/GI_Reasoning_train.json', 'r', encoding='utf-8') as file:
#     GI_Reasoning_train = json.load(file)
# print(f'GI_Reasoning_train: {len(GI_Reasoning_train)}')
# GI_Reasoning_train = filtering(GI_Reasoning_train, 100, 1)
# print(len(GI_Reasoning_train))
# with open(f'/home/gy237/project/llama3/new_data/final_filter/GI_Reasoning_train.json', 'w', encoding='utf-8') as file:
#     json.dump(GI_Reasoning_train, file, ensure_ascii=False, indent=4)


# MultiCaRe_PMC_Patients_PMC_CaseReport        gpt-4o mini to generate diagnosis list
# prompt4 = '''Your task is to provide at least 10 accurate and distinct patient diagnoses based on the input case report.   Ensure you provide at least 10 most likely diagnoses, listed in order of likelihood, and cover a wide range of unique possibilities.  \n Follow the guidelines for a generation: 1.   Each diagnosis should be precise and unique, ensuring a variety of at least 10 possibilities.   2.   List one diagnosis per line.   3.   Generate 10 differential diagnoses related to the input case report.   Think step by step.  \n \n***Output format***:Differential diagnosis: 1.   \n2.   \n3.\n4.   \n5.   \n6.   \n7.   \n8.   \n9.   \n10.'''

# with open('/home/gy237/project/llama3/new_data/final_filter/PMC_Patients_diagnosis.json', 'r', encoding='utf-8') as file:
#     PMC_Patients_diagnosis = json.load(file)
# with open('/home/gy237/project/llama3/new_data/temporary_data/PMC_Patients.json', 'r', encoding='utf-8') as file:
#     PMC_Patients_diagnosis_ = json.load(file)
# print(f'PMC_Patients_diagnosis: {len(PMC_Patients_diagnosis)}, Before using 4o-mini to decide whether it is case report: {len(PMC_Patients_diagnosis_)}')
# for i in PMC_Patients_diagnosis:
#     prompt, inot = i['query'].split('INPUT:')
#     i['query'] = 'INPUT:'.join([prompt4, inot])
# # PMC_Patients_diagnosis = filtering(PMC_Patients_diagnosis, 100, 0)
# print(len(PMC_Patients_diagnosis))
# with open(f'/home/gy237/project/llama3/new_data/final_filter/PMC_Patients_diagnosis.json', 'w', encoding='utf-8') as file:
#     json.dump(PMC_Patients_diagnosis, file, ensure_ascii=False, indent=4)


# with open('/home/gy237/project/llama3/new_data/temporary_data/GENE_REVIEW_SY_train.json', 'r', encoding='utf-8') as file:
#     GENE_REVIEW_SY_train = json.load(file)
# print(f'GENE_REVIEW_SY_train: {len(GENE_REVIEW_SY_train)}')
# GENE_REVIEW_SY_train = filtering(GENE_REVIEW_SY_train, 10, 100)
# print(len(GENE_REVIEW_SY_train))
# with open(f'/home/gy237/project/llama3/new_data/final_filter/GENE_REVIEW_SY_train.json', 'w', encoding='utf-8') as file:
#     json.dump(GENE_REVIEW_SY_train, file, ensure_ascii=False, indent=4)


# with open('/home/gy237/project/llama3/new_data/temporary_data/JAMA_Reasoning_Rare_train.json', 'r', encoding='utf-8') as file:
#     JAMA_Reasoning_Rare_train = json.load(file)
# print(f'JAMA_Reasoning_Rare_train: {len(JAMA_Reasoning_Rare_train)}')
# JAMA_Reasoning_Rare_train = filtering(JAMA_Reasoning_Rare_train, 100, 100)
# print(len(JAMA_Reasoning_Rare_train))
# with open(f'/home/gy237/project/llama3/new_data/final_filter/JAMA_Reasoning_Rare_train.json', 'w', encoding='utf-8') as file:
#     json.dump(JAMA_Reasoning_Rare_train, file, ensure_ascii=False, indent=4)


# with open('/home/gy237/project/llama3/new_data/temporary_data/medical_book_train.json', 'r', encoding='utf-8') as file:
#     medical_book_train = json.load(file)
# print(f'medical_book_train: {len(medical_book_train)}')
# medical_book_train = filtering(medical_book_train, 10, 70)
# print(len(medical_book_train))
# with open(f'/home/gy237/project/llama3/new_data/final_filter/medical_book_train.json', 'w', encoding='utf-8') as file:
#     json.dump(medical_book_train, file, ensure_ascii=False, indent=4)


# with open('/home/gy237/project/llama3/new_data/temporary_data/MedInstruct_train.json', 'r', encoding='utf-8') as file:
#     MedInstruct_train = json.load(file)
# print(f'MedInstruct_train: {len(MedInstruct_train)}')
# MedInstruct_train = filtering(MedInstruct_train, 10, 100)
# print(len(MedInstruct_train))
# with open(f'/home/gy237/project/llama3/new_data/final_filter/MedInstruct_train.json', 'w', encoding='utf-8') as file:
#     json.dump(MedInstruct_train, file, ensure_ascii=False, indent=4)


# with open('/home/gy237/project/llama3/new_data/temporary_data/MedQA_Reasoning_train.json', 'r', encoding='utf-8') as file:
#     MedQA_Reasoning_train = json.load(file)
# print(f'MedQA_Reasoning_train: {len(MedQA_Reasoning_train)}')
# MedQA_Reasoning_train = filtering(MedQA_Reasoning_train, 100, 1)
# print(len(MedQA_Reasoning_train))
# with open(f'/home/gy237/project/llama3/new_data/final_filter/MedQA_Reasoning_train.json', 'w', encoding='utf-8') as file:
#     json.dump(MedQA_Reasoning_train, file, ensure_ascii=False, indent=4)


# with open('/home/gy237/project/llama3/new_data/DDXPlus_Reasoning_train_gui_processed.json', 'r', encoding='utf-8') as file:
#     DDXPlus_Reasoning_train = json.load(file)
# print(f'DDXPlus_Reasoning_train: {len(DDXPlus_Reasoning_train)}')
# DDXPlus_Reasoning_train = filtering(DDXPlus_Reasoning_train, 100, 5)
# print(len(DDXPlus_Reasoning_train))
# with open(f'/home/gy237/project/llama3/new_data/final_filter/DDXPlus_Reasoning_train.json', 'w', encoding='utf-8') as file:
#     json.dump(DDXPlus_Reasoning_train, file, ensure_ascii=False, indent=4)


# # MultiCaRe_PMC_Patients_PMC_CaseReport        gpt-4o mini to generate diagnosis list
# prompt4 = '''Your task is to provide at least 10 accurate and distinct patient diagnoses based on the input case report.   Ensure you provide at least 10 most likely diagnoses, listed in order of likelihood, and cover a wide range of unique possibilities.  \n Follow the guidelines for a generation: 1.   Each diagnosis should be precise and unique, ensuring a variety of at least 10 possibilities.   2.   List one diagnosis per line.   3.   Generate 10 differential diagnoses related to the input case report.   Think step by step.  \n \n***Output format***:Differential diagnosis: 1.   \n2.   \n3.\n4.   \n5.   \n6.   \n7.   \n8.   \n9.   \n10.'''

# with open('/home/gy237/project/llama3/new_data/upload_hug/MultiCaRe_Reasoning_test_diagnosis.json', 'r', encoding='utf-8') as file:
#     MultiCaRe_Reasoning_test_diagnosis = json.load(file)
# print(f'MultiCaRe_Reasoning_test_diagnosis: {len(MultiCaRe_Reasoning_test_diagnosis)}')
# for i in MultiCaRe_Reasoning_test_diagnosis:
#     prompt, inot = i['query'].split('INPUT:')
#     i['query'] = 'INPUT:'.join([prompt4, inot])
# # MultiCaRe_Reasoning_test_diagnosis = filtering(MultiCaRe_Reasoning_test_diagnosis, 100, 1)
# print(len(MultiCaRe_Reasoning_test_diagnosis))
# with open(f'/home/gy237/project/llama3/new_data/final_filter/MultiCaRe_Reasoning_test_diagnosis.json', 'w', encoding='utf-8') as file:
#     json.dump(MultiCaRe_Reasoning_test_diagnosis, file, ensure_ascii=False, indent=4)



# ds1 = load_dataset("YBXL/liveqa_train", cache_dir='/home/gy237/project/download_data')
# ds2 = load_dataset("YBXL/medical_meadow_mediqa_train", cache_dir='/home/gy237/project/download_data')
# ds3 = load_dataset("YBXL/HealthCareMagic_train", cache_dir='/home/gy237/project/download_data')
# ds4 = load_dataset("YBXL/iCliniq_train", cache_dir='/home/gy237/project/download_data')
# ds5 = load_dataset("YBXL/Alpaca_train", cache_dir='/home/gy237/project/download_data')
# ds6 = load_dataset("YBXL/Dolly_train", cache_dir='/home/gy237/project/download_data')

# # print(ds1)
# def sav(name):
#     ds = load_dataset(f"YBXL/{name}", cache_dir='/home/gy237/project/download_data')
#     data = ds['train']
#     qa_list = []
#     for i in data:
#         query = i['conversations'][0]['content']
#         answer = i['conversations'][1]['content']
#         dic = {"id": i['id'], "query": query, "answer": answer}
#         qa_list.append(dic)

#     with open(f'/home/gy237/project/llama3/new_data/final_filter/{name}.json', 'w', encoding='utf-8') as file:
#         json.dump(qa_list, file, ensure_ascii=False, indent=4)


# name_list = ['liveqa_train', 'medical_meadow_mediqa_train', 'HealthCareMagic_train', 'iCliniq_train', 'Alpaca_train', 'Dolly_train']
# for i in name_list:
#     sav(i)


# with open(f'/home/gy237/project/llama3/new_data/temporary_data/GENE_REVIEW_SY_train.json', 'w', encoding='utf-8') as file:
#     json.dump(upload_data2, file, ensure_ascii=False, indent=4)
# upload_data.append({"id": _id[i], "query": inpt[i], "answer": oupt[i]})
