import json

type_ = 'train'


data = []
with open(f'/home/gy237/project/llama3/Cleared/{type_}.json', 'r', encoding='utf-8') as file:
    for line in file:
        data.append(json.loads(line.strip()))
for i in data:
    inpt = i['conversations'][1]['content']
    target_part = inpt.split('Please summarize the function of the synthetic biology component named ')[1].split(' from the provided text. ')[0]
    assert target_part.startswith('BBa')
    prompt100 = f'''#Background#
1. We will provide you with a text after 'Here is the text:', which is converted from web page file, and may contain excessive and non-essential information.
2. The first line of the text is the name of a synthetic biology component, '{target_part}'.
3. The content of this web page is very irregular at present. Originally this page was supposed to describe only the function of '{target_part}', but now it contains descriptions of many other operating components related to '{target_part}', or of the entire project. It is necessary to extract the functional description and operating environment of '{target_part}' from the web page in order to better understand the characteristics and applicability of '{target_part}'.
#Task#
1. Please summarize the function of the synthetic biology component named '{target_part}' from the provided text, and provide a clear and concise description focusing on the function and usage of '{target_part}'. Your extraction should be no more than 7 sentences.
2. Please provide a detailed summary of the necessary operating environments for the use of '{target_part}', focusing on the strains employed. Describe the essential settings, conditions, and biological systems required for its effective use, especially any unique environmental parameters and relevant microbial or genetic strains.
3. Please ignore and remove the HTML contents that are not removed, and response with natural, fluid and accurate language.
4. Please remove meaningless and confusing symbols around words, such as '_', '\\', etc
#Role#
You are an excellent synthetic biologist, especially good at summarizing the functions of synthetic biological components, and you are always able to accurately summarize the functions of components and the required chemical and operating environment. 
#Profile#
You have the the ability to work as a researcher or analyst in the field with an in-depth understanding of synthetic biology components and their applications.
#Skills#
You are familiar with synthetic biology terminology, understanding web information extraction techniques, bioinformatics background. You can accurately distinguish between the description of '{target_part}' and the description of other components.
#Goals#
Accurately extract the functional description and operating environment of '{target_part}' from the web page. If the function or operating environment of '{target_part}' is related to other components, feel free to write other components' contents, functions, characteristics, etc.
#Constrains#
1. You should ensure that the extracted information is accurate to avoid any possible misinterpretation or information loss. 
2. If the function or operating environment of '{target_part}' is related to other components, feel free to write other components' contents, functions, characteristics, etc.
3. Please try to use the original sentences and words to summarize and keep professional terms.
#OutputFormat#
1. Please response with "FUNCTION: (functional description of '{target_part}') \nEnvironment: (operating environment of '{target_part}')"
2. Your answer should only contain the functional description and operating environment, no need for any other words.
3. If the function details are not clear, respond with 'No description of the function'.
4. If information on the operating environment or strains is insufficient, respond with 'No description of the environment'.
#Workflow#
1. Identify and locate the functional description and operating environment related to '{target_part}' in the web page.
2. Extract and organize information to ensure its accuracy and completeness.
3. If you cannot extract enough information, just output 'No description of the function' or 'No description of the environment'.
Here is the text:'''

    query = prompt100 + inpt.split('Here is the text:')[-1]
    answer = 'FUNCTION: ' + i['conversations'][2]['content'] + '\nEnvironment: ' + i['conversations'][4]['content']
    # print(target_part)
    # print(query)
    # print(answer)
    # exit()
    qa = {"text": f"<human>: {query}\n<bot>: {answer}", "metadata": {"source": "unified_chip2"}}
    # print(qa)
    # exit()
    with open(f'/home/gy237/project/llama3/unsloth/ifdata/{type_}.json', 'a', encoding='utf-8') as f:
        f.write(json.dumps(qa, ensure_ascii=False) + '\n')

