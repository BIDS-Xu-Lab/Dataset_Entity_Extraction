# """
# EXAMPLE:
# (llama3) [s211505003@localhost llama3]$ python llama_test.py
# Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████| 4/4 [00:06<00:00,  1.69s/it]
# user:  Make E. coli fluoresce green.
# Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
# Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
# reply: [{"type": "Promoters", "description": "to initiate gene expression in E. coli."}, {"type": "Protein_coding_sequences", "description": "encode the green fluorescent protein."}, {"type": "Terminators", "description": "to terminate gene expression and prevent read-through."}]
# user:  To make yeast cells produce trehalose in large quantities.
# Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
# Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
# reply: [{"type": "Promoters", "description": "to start the gene expression in the yeast."}, {"type": "Protein_coding_sequences", "description": "to encode the trehalase or trehalose synthase enzyme."}, {"type": "Terminators", "description": "to stop the transcription and translation of the trehalose-producing enzyme."}, {"type": "Plasmid_backbones", "description": "to provide a platform for the genetic components to be maintained and replicated in yeast cells."}]
# user:  idk what i need.
# Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
# Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
# reply: [{"type": "ERROR", "description": "ERROR"}]
# user:  exit
# """
# import transformers
# import torch
# import json
# import difflib

# model_id = "/home/s211505003/iGEM2/llama3/Meta-Llama-3.1-8B-Instruct"

# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model_id,
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     device_map="auto",
# )

# catalog = [
#     {
#         "name": "Promoter",
#         "description": "A promoter is a DNA sequence that can recruit transcriptional machinery and lead to transcription of the downstream DNA sequence. The specific sequence of the promoter determines the strength of the promoter (a strong promoter leads to a high rate of transcription initiation).In addition to sequences that \"promote\" transcription, a promoter may include additional sequences known as operators that control the strength of the promoter. For example, a promoter may include a binding site for a protein that attracts or obstructs the RNAP binding to the promoter. The presence or absence of the protein will affect the strength of the promoter. Such a promoter is known as a regulated promoter."
#     },
#     {
#         "name": "Ribosome_Binding_Site",
#         "description": "The bacterial ribosome binds to particular sequences on an mRNA, primarily the Ribosome Binding Site (RBS) and the start codon (AUG). The RBS base-pairs with an RNA molecule that forms part of the bacterial ribosome (the 16s rRNA), while the start codon base-pairs with the initiator tRNA which is bound to the ribosome. In addition to the sequences of the RBS and the start codon being important, these two sequences need to be positioned approximately 6-7 nucleotides apart so they can both make contact with the appropriate parts of the ribosome complex. In the next section, we talk more about the specific interaction between the RBS and the ribosome."
#     },
#     {
#         "name": "Protein_Domain",
#         "description": "Protein domains encode portions of proteins and can be assembled together to form translational units, a genetic part spanning from translational initiation (the RBS) to translational termination (the stop codon)."
#     },
#     {
#         "name": "Protein_Coding_Sequence",
#         "description": "Protein coding sequences are DNA sequences that are transcribed into mRNA and in which the corresponding mRNA molecules are translated into a polypeptide chain. Every three nucleotides, termed a codon, in a protein coding sequence encodes 1 amino acid in the polypeptide chain. In some cases, different chassis may either map a given codon to a different sequence or may use different codons more or less frequently. Therefore some protein coding sequences may be optimized for use in a particular chassis."
#     },
#     {
#         "name": "Translational_Unit",
#         "description": "Translational units begin with the RBS, the site of ribosome binding and translational initiation, and end with a stop codon, the site of translational termination. Every translational unit in the Registry consists of at least three parts, a Translational start, one or more Internal Domains including Special Internal Domains, and a Tail Domain. Thus translational units can, in some sense, be thought of as a composite part made up of three or more parts. Protein coding sequences, in contrast, begin with a start codon and end with a stop codon."
#     },
#     {
#         "name": "Terminator",
#         "description": "Terminators are genetic parts that usually occur at the end of a gene or operon and cause transcription to stop. In prokaryotes, terminators usually fall into two categories (1) rho-independent terminators and (2) rho-dependent terminators. Rho-independent terminators are generally composed of palindromic sequence that forms a stem loop rich in G-C base pairs followed by several T bases. The conventional model of transcriptional termination is that the stem loop causes RNA polymerase to pause and transcription of the poly-A tail causes the RNA:DNA duplex to unwind and dissociate from RNA polymerase. All the E. coli terminators in the Registry are rho-independent terminators. Rho-dependent terminators are not included, because rho-dependent terminators are not specified by sequence."
#     },
#     {
#         "name": "DNA",
#         "description": "Nearly all BioBrick parts are specified as DNA sequences, but that does not make all BioBrick parts have type \"DNA\". Protein coding regions, promoters, ribosome binding sites, terminators, plasmid backbones, among others, also do NOT have type \"DNA\". Similarly, primers, being single-stranded synthetic DNA, do not have type \"DNA\" and instead have their own part type \"Primer\". You can browse the Registry primer collection separately."
#     },
#     {
#         "name": "Plasmid_Backbone",
#         "description": "Plasmids are circular, double-stranded DNA molecules typically containing a few thousand base pairs that replicate within the cell independently of the chromosomal DNA. Plasmid DNA is easily purified from cells, manipulated using common lab techniques and incorporated into cells. Most BioBrick parts in the Registry are maintained and propagated on plasmids. Thus, construction of BioBrick parts, devices and systems usually requires working with plasmids."
#     },
#     {
#         "name": "Primer",
#         "description": "A primer is a short single-stranded DNA sequences used as a starting point for PCR amplification or sequencing. Although primers are not actually available via the Registry distribution, we include commonly used primer sequences here."
#     }
# ]
# catalog_name = [i["name"] for i in catalog]

# system_prompt = {"role": "system", "content": f"""
# #IMPORTANT RULES#
# DO NOT REPEAT the descriptions of the catagories given to you in Section 'Allowed component types'. You only need to extract information from user input. You can make some reasonable deductions, to deduce what components the user may want.
# PLEASE make sure that your output clearly represents and contains the user input. You SHOULD contain Promoter AND Protein Coding Sequences. You SHOULD contain Promoter AND Protein Coding Sequences. You SHOULD contain Promoter AND Protein Coding Sequences.
# YOUR OUTPUT SHOULD FULLY CONTAIN EVERY REQUIREMENTS GIVEN BY USER, especially some proper nouns like names for parts or plasmids, such as E.coli. YOUR OUTPUT SHOULD FULLY CONTAIN EVERY REQUIREMENTS GIVEN BY USER.
# #Background#
# There is an interface that can match the corresponding component according to the function description of the synthetic biology component. This interface accepts inputs (component class, component function description). You will format the data for this interface, summarizing the required component categories and functions from a text that describes the function of the plasmid.
# #Character#
# You are an expert in synthetic biology who is good at summarizing the function of plasmids and giving the categories and functions of the individual components they contain. You take input from non-expert users in plain language and give professional-oriented answers, and you respond to requests in language that is professional, accurate, and concise, including the specialized academic vocabulary of synthetic biology.
# #Task#
# You are about to receive a text that describes the function of a synthetic plasmid. You should analyze the text describing the class of synthetic biological parts contained in the plasmid and give the function of each class of synthetic biological parts. For each component in the plasmid, you will select one of the given categories and give the corresponding function description. Example: There may be a category of parts called promoters whose function is described as initiating gene expression in yeast; Another category of parts is the protein coding sequence, the functional description of which encodes the protein that emits green fluorescence.
# #Steps#
# 1. Select the text describing the function of the plasmid from the input and summarize it into a new paragraph;
# 2. According to the text, determine which types of components the function of the plasmid comes from;
# 3. For each type of component required, extract the function completed by this component from the summarized text describing the function of the plasmid;
# 4. Summarize the functions of all components obtained, and infer whether the combination of their functions conforms to the description of the function of the plasmid in the original text. If yes, output the result; if no, re-perform the above steps and modify the answer;
# 5. You only need to output json format text, used to describe the component category and function description you think you need, should strictly conform to the output format, do NOT output any superfluous content;
# 6. In the absence of the necessary information to complete the above steps, do not output the final answer, only output what additional supplementary information is required, and repeat the above steps based on the original and supplementary information.
# #Allowed component types#
# {json.dumps(catalog)}
# #Input#
# You will be given a text that describes plasmids made up of multiple synthetic biology parts. These texts may contain the function and environment of the desired plasmid, from which you need to extract the information you need to indicate the category of components and functional description of the plasmid, or report insufficient information. Input must be in English.
# #Output#
# You need to give a string in json format as your answer. You ALWAYS reply in English. You MUST follow the '[{{"type": "type of part 1", "description": "description of part 1"}}, {{"type": "type of part 2", "description": "description of part 2"}}]' format. The component category must be one of the known types and you always answer the user in English. You MUST NOT output any extra content. Example: '[{{"type": "Promoters", "description": "to start the gene expression in the yeast."}}, {{"type": "Protein_coding_sequences", "description": "code of green fluorescent protein."}}]'
# #IMPORTANT RULES#
# DO NOT REPEAT the descriptions of the catagories given to you in Section 'Allowed component types'. You only need to extract information from user input. You can make some reasonable deductions, to deduce what components the user may want.
# PLEASE make sure that your output clearly represents and contains the user input. You SHOULD contain Promoter AND Protein Coding Sequences. You SHOULD contain Promoter AND Protein Coding Sequences. You SHOULD contain Promoter AND Protein Coding Sequences.
# YOUR OUTPUT SHOULD FULLY CONTAIN EVERY REQUIREMENTS GIVEN BY USER, especially some proper nouns like names for parts or plasmids. YOUR OUTPUT SHOULD FULLY CONTAIN EVERY REQUIREMENTS GIVEN BY USER.
# """}

# json_system_prompt = {"role": "system", "content": """
# #Role#
# You are an expert at extracting json objects from text.
# #Task#
# You will get a piece of text, you need to output the json object at the end of the text in one line, and there can be no additional output. You should do this task directly, NOT output a script to do this task. If you cannot find a valid json object, please report an error.
# #Input#
# A piece of text that contains a json object at the end.
# #Output#
# A json object in one line. No other content is allowed.
# """}


# def recommend(type):
#     similarities = [(i, difflib.SequenceMatcher(None, type, i).ratio()) for i in catalog_name]
#     similarities.sort(key=lambda x: x[1], reverse=True)
#     return similarities[0][0]

# def get_json(target: str) -> str:
#     messages = [system_prompt, {"role": "user", "content": target}]
#     outputs = pipeline(messages, max_new_tokens=1024)
#     reply = outputs[0]["generated_text"][-1]["content"]
#     # print(f"================================\n{reply}\n================================")
#     messages = [json_system_prompt, {"role": "user", "content": reply}]
#     outputs = pipeline(messages, max_new_tokens=1024)
#     reply = outputs[0]["generated_text"][-1]["content"]
#     if reply[0] == "{":
#         reply = "[" + reply.replace("}", "},", reply.count("}") - 1) + "]"
#     # print(f"================================\n{reply}\n================================")
#     try:
#         reply = json.loads(reply)
#         for i in reply:
#             i["type"] = recommend(i["type"])
#     except:
#         reply = [{"type": "ERROR", "description": "ERROR"}]
#     return json.dumps(reply)

# while True:
#     target = input("user:  ")
#     if target == "exit":
#         break
#     # Make E. coli fluoresce green.
#     # To make yeast cells produce trehalose in large quantities.
#     # target = "Make E. coli fluoresce green."
#     print(f"reply: {get_json(target)}")

import json
import os
from tqdm import tqdm, trange
with open('/home/gy237/project/llama3/unsloth/ig/total_part/part_recatagorized.json', 'r', encoding='utf-8') as file:
    part_data = json.load(file)

miss = 0
count = 0
count_ = 0
for i in trange(len(part_data), desc='Steps'):
    target_part = part_data[i]['name']
    try:
        with open(f'/home/gy237/project/llama3/unsloth/ig/total_part/part/Part^{target_part}.txt', 'r', encoding='utf-8') as file:
            page = file.read()
        if 'Sequence and Features' not in page:
            count_ += 1
        if f'# Part:{target_part}' not in page:
            count += 1
    except:
        miss += 1
    # if count>1:
    #     print(page)
    #     print(target_part)
    #     exit()
print(count)
print(count_)

