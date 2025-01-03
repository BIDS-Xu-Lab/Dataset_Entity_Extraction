# %%
import joblib
import pandas as pd
import random
import json
import re
import spacy
nlp = spacy.load("en_core_web_sm")
import string
import os

# %%
# 加载.joblib文件
old_data = joblib.load('/home/gy237/project/Biomedical_datasets/total_pubmed/abstracts_1-1252.joblib')
print(old_data.columns)
# ['pmid', 'title', 'abstract', 'journal', 'pubdate', 'authors','mesh_terms']

# %%
new_data = joblib.load('/home/gy237/project/Biomedical_datasets/total_pubmed/abstracts_1253-1575.joblib')
print(new_data.columns)

# %%
combined_df = pd.concat([old_data, new_data], axis=0)
print('去重前：', len(combined_df))
data = combined_df.drop_duplicates(subset=['pmid'], keep='last')
print('去重后：', len(data))
print(data.columns)
# 去重前： 39390916
# 去重后： 37814430

# %%
general_repositories = [
        " data", "repository", "survey", "questionnaire", "PubMed"," NCBI ","Scopus","Kaggle","knowledge base",
        "Mendeley","Education Resources Information Center","Science Open","Web of Science","Cochrane Library","EMbase",
        "Chinese Biomedical Literature Database","Medline",
        "Kaggle", "Google Dataset Search", "Zenodo", "Dryad", "Figshare",
        "Open Data Portal", "World Bank Open Data", "UN Data", "Data.gov", 
        "DataHub", "GitHub", "IEEE DataPort", "Mendeley Data", 
        "Open Science Framework", "AWS Open Data Registry", "Harvard Dataverse",
        "ICPSR","re3data.org", "PLOS Data Repository", "UK Data Service", 
        "Humanitarian Data Exchange)", "UCI Machine Learning Repository", 
        "Statista", "Quandl", "OpenStreetMap", "EarthData", "Global Health Observatory", 
        "GBIF", "GCMD", "Eurostat", "OECD Data", 
        "Climate Data Store", "FAOSTAT", "Geonames", "NCEI", 
        "Copernicus Open Access Hub", "Landsat Data Repository", 
        "National Cancer Institute Genomic Data Commons", 
        "Census Bureau Data", "IMF Data", "OpenAIRE", 
        "NCES", "PERSEE", "StatBank Denmark", 
        "Australian Data Archive", "China Statistical Yearbook", 
        "Open Knowledge Foundation CKAN", "BigQuery Public Datasets", 
        "UNESCO Institute for Statistics Data Centre", "Linked Data Platform"
]

life_science_repository = [
        "ClinicalTrials", "clinical trials",
        "GenBank","Gene Expression Omnibus","PubChem","Protein Data Bank","UniProt",
        "Genotype-Tissue Expression","Bioproject","dbSNP","ClinVar","PhysioNet","National Alzheimer's Coordinating Center",
        "Sequence Read Archive","LINCS","ImmPort","dbGaP","The Cancer Imaging Archive","CellChat",
        "FlyBase","BioPortal","Mouse Genome Informatics","National COVID Cohort Collaborative","Saccharomyces Genome Database",
        "Genomic Data Commons","PeptideAtlas","WormBase","International Mouse Phenotyping Consortium",
        "BindingDB","CHILDES","NITRC","Rat Genome Database","Immunological Genome Project",
        "Investigational New Drug Applications","ICPSR","HOMD","AphasiaBank","OpenNeuro",
        "Metabolomics","4D-Nucleome","NDEx","Mouse Phenome Database","BioLINCC",
        "National Sleep Research Resource","Xenbase",
        "NIH Genetic Testing Registry","BMRB","Kids First Data Resource",
        "Monarch Initiative","dbVar","ZFIN",
]
total_target = general_repositories + life_science_repository
print(len(total_target))

# %%
# 以data作为keyword进行筛选abstract
print('Total data len:',len(data))

pattern = re.compile("|".join(re.escape(word) for word in total_target), re.IGNORECASE)  # 创建一个匹配所有关键词的正则模式
# 使用正则表达式在一行中完成筛选
from multiprocessing import Pool

def match_pattern(text):
    return bool(pattern.search(text)) if pd.notna(text) else False

with Pool() as pool:
    results = pool.map(match_pattern, data['abstract'])
filtered_df = data[pd.Series(results)]


# " data" 筛选后有 5446108
# 全部筛选后有 

# %%
joblib.dump(filtered_df, '/home/gy237/project/Biomedical_datasets/total_pubmed/abstracts_filtere_1-1575.joblib')