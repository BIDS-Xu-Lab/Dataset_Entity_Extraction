import os
import json


target_names = ['MultiCaRe_Reasoning_test_diagnosis', 'PMC_Patients_diagnosis']
print(len(target_names))


folder_path = '/home/gy237/project/llama3/new_data/download_openai/multicare_pmc_explation'
file_names = os.listdir(folder_path)
input_names = [i.split('.')[0] for i in file_names if i.endswith('jsonl')]
print(len(input_names))
# exit()

prompt = '''As a meticulous and evidence-driven physician, your task is to generate exactly 10 accurate and distinct differential diagnoses based on the diagnosis information mentioned in the input case report and its title. Each diagnosis must be followed by a concise, evidence-informed explanation. Your diagnoses should span a wide range of possibilities, listed in order of likelihood, with the most likely diagnosis appearing first. Focus on clinical findings, considering their diagnostic accuracy and relevance as emphasized in evidence-based physical diagnosis practices. Ensure each diagnosis reflects key physical findings relevant to the patient's symptoms, with emphasis on findings that significantly alter the likelihood of specific conditions.\n**Output format**:\nDifferential diagnosis:\n1. [Diagnosis 1]: [Evidence-informed One-sentence explanation]\n2. [Diagnosis 2]: [Evidence-informed One-sentence explanation]\n3. [Diagnosis 3]: [Evidence-informed One-sentence explanation]\n4. [Diagnosis 4]: [Evidence-informed One-sentence explanation]\n5. [Diagnosis 5]: [Evidence-informed One-sentence explanation]\n6. [Diagnosis 6]: [Evidence-informed One-sentence explanation]\n7. [Diagnosis 7]: [Evidence-informed One-sentence explanation]\n8. [Diagnosis 8]: [Evidence-informed One-sentence explanation]\n9. [Diagnosis 9]: [Evidence-informed One-sentence explanation]\n10. [Diagnosis 10]: [Evidence-informed One-sentence explanation]\n**Example**:\nDifferential diagnosis:\n1. Mitochondrial disease: Considering the patient's history of celiac disease, fatigue, and somatic symptoms, a mitochondrial disorder could be a possibility, especially with the presence of a prolonged QT interval.\nPlease generate exactly 10 differential diagnoses with corresponding evidence-based one-sentence explanations, adhering strictly to the output format without including any other outputs.\n'''


for name in target_names:
    upload_data = []
    for in_name in input_names:
        if name in in_name:
            input_file = f"/home/gy237/project/llama3/new_data/download_openai/multicare_pmc_explation/{in_name}.jsonl"
            target_file = f"/home/gy237/project/llama3/new_data/final_data/{name}.json"
            output_file = f"/home/gy237/project/llama3/new_data/final_data/explanation/{name}_explanation.json"

            data = []
            with open(input_file,'r', encoding='utf-8') as file:
                for line in file:
                    data.append(json.loads(line))
            with open(target_file, 'r', encoding='utf-8') as file:
                target = json.load(file)


            assert len([i for i in data if i["response"]["body"]["choices"][0]["finish_reason"] != 'stop' or i["error"]]) == 0, 'Error'
            
            print(f'{name}: {len(target)}, {in_name}: {len(data)}')

            for i in data:
                id_ = i["custom_id"]
                flag = True
                # print(i)
                for j in target:
                    # print(j)
                    # exit()
                    if j['id'] == id_:
                        flag = False
                        inpt = j['query'].split('INPUT:')[1]
                        j['query'] = 'INPUT:'.join([prompt, inpt])
                        explanation = i["response"]["body"]["choices"][0]["message"]["content"]
                        j['answer'] = explanation
                        upload_data.append(j)
                        # print(j)
                        # exit()
                if flag:
                    print("Error, id don't match")

    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(upload_data, file, ensure_ascii=False, indent=4)
    print(f'{name}: {len(target)}, upload_data: {len(upload_data)}')