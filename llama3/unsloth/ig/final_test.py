import subprocess
import os
import json
from transformers import TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
import io
import logging
logging.getLogger("transformers.utils.hub").setLevel(logging.CRITICAL+1)
from unsloth import FastLanguageModel
import torch
from tqdm import tqdm, trange
import re
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"

# /home/gy237/project/llama3/unsloth/ig/output/checkpoint-500, 1500 5500, 9500, 10500, 12500, 13000, 14000, 21000, 60000
# 9500， 52000, 10500, 500, 1500 
# MODEL_NAME = '/home/gy237/project/llama3/unsloth/ig/output/checkpoint-21000'

model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = None,
        dtype = None,
        load_in_4bit = True,
    )
FastLanguageModel.for_inference(model)

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = tuple(set(stop_token_ids))
    pass

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return input_ids[0][-1].item() in self.stop_token_ids
    pass
pass

def async_process_chatbot(message, history):
    
    eos_token = tokenizer.eos_token
    stop_on_tokens = StopOnTokens([eos_token,])
    text_streamer  = TextIteratorStreamer(tokenizer, skip_prompt = True)

    history_transformer_format = history + [[message, ""]]
    messages = []
    for item in history_transformer_format:
        if item:
            messages.append({"role": "user",      "content": item[0]})
            messages.append({"role": "assistant", "content": item[1]})
    pass
    # print(message)
    # exit()

    # Remove last assistant and instead use add_generation_prompt
    messages.pop(-1)
   
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True,
        return_tensors = "pt",
    ).to("cuda", non_blocking = True)
    
    outputs = model.generate(
        input_ids, 
        streamer = text_streamer,
        max_new_tokens = 102400, 
        stopping_criteria = StoppingCriteriaList([stop_on_tokens,]),
        temperature = 0.7,
        do_sample = True,
        use_cache = True)
    outputs = tokenizer.batch_decode(outputs)
    outputs = outputs[0]

    result = outputs.split('<|end_header_id|>')[-1].split('<|eot_id|>')[0]
    # 正则表达式模式
    pattern_function = r'function:\s*(.*?)(?=\s*Environment:|\Z)'
    pattern_environment = r'environment:\s*(.*)'

    # 使用 re.IGNORECASE 标志来忽略大小写
    match_function = re.search(pattern_function, result, re.IGNORECASE)
    match_environment = re.search(pattern_environment, result, re.IGNORECASE)

    # 获取匹配结果
    fun = match_function.group(1) if match_function else 'None'
    env = match_environment.group(1) if match_environment else 'None'
    return fun, env


with open('/home/gy237/project/llama3/unsloth/ig/total_part/part_recatagorized.json', 'r', encoding='utf-8') as file:
    part_data = json.load(file)

miss = 0
for i in trange(len(part_data), desc='Steps'):
    target_part = part_data[i]['name']
    # target_part = 'BBa_B0020' # 'BBa_B0030'
    flag = False
    try:
        with open(f'/home/gy237/project/llama3/unsloth/ig/total_part/part/Part^{target_part}.txt', 'r', encoding='utf-8') as file:
            page = file.read()
        flag = True
    except:
        miss += 1

    if flag:
        prompt100 = f'''#Background#
1. We will provide you with a text after 'Here is the text:', which is converted from web page file, and may contain excessive and non-essential information.
2. The first line of the text is the name of a synthetic biology component, '{target_part}'.
3. The content of this web page is very irregular at present. Originally this page was supposed to describe only the function of '{target_part}', but now it contains descriptions of many other operating components related to '{target_part}', or of the entire project. It is necessary to extract the functional description and operating environment of '{target_part}' from the web page in order to better understand the characteristics and applicability of '{target_part}'.
#Task#
1. Please summarize the function of the synthetic biology component named '{target_part}' from the provided text, and provide a clear and concise description focusing on the function and usage of '{target_part}'. Your extraction should be no more than 7 sentences. The function here includes how can it affect how specific proteins are expressed, what strains it works, how do it interact with other elements, and any synthetic biological function that the element itself plays. Be careful to output few numbers, more technical terms, and summarize the content of the text.
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
4. Please do not include any web links or html text in the output.
5. In the process of summary, try to use more technical terms and comparative words from the original text, less use of numbers.
6. If there is not much information in the text, do not expand the sentence and add information yourself.
7. If the text is short and contains no more than 3 sentences of useful information, please output all the useful sentences.
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
        fun, env = async_process_chatbot(f'{prompt100}\n{page}', [])

        while fun == 'None' or env == 'None':
            fun, env = async_process_chatbot(f'{prompt100}\n{page}', [])

        part_data[i]['function_llama3.1'] = fun
        part_data[i]['environment_llama3.1'] = env

        print('-------------------------------------------------------')
        print(part_data[i]['function_llama3.1'])
        print('environment_llama3')
        print(part_data[i]['environment_llama3.1'])
        exit()

    
    if miss%50 == 0:
        with open('/home/gy237/project/llama3/unsloth/ig/total_part/part.json', 'w', encoding='utf-8') as file:
            json.dump(part_data, file, ensure_ascii=False, indent=4)

print(miss)
