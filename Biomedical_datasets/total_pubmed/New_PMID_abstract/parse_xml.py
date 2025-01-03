import os
import gzip
import xml.etree.ElementTree as ET
import json
from tqdm import tqdm
import pandas as pd
import joblib
from xml.etree.ElementTree import tostring


def get_xml(folder_path):
    with gzip.open(file_path, 'rb') as f:
        # 读取解压后的 XML 内容
        xml_content = f.read().decode('utf-8')
    # print(xml_content)
    # exit()
    return xml_content


# 提取出版日期
def extract_pub_date(article):
    # 首先尝试使用 Year 和 Month 提取
    pub_year = article.findtext('.//JournalIssue/PubDate/Year')
    pub_month = article.findtext('.//JournalIssue/PubDate/Month')

    if pub_year and pub_month:
        # 如果同时有年份和月份，直接返回格式化的日期
        return pub_year, pub_month
    
    # 如果没有 Year 和 Month，尝试使用 MedlineDate 提取
    medline_date = article.findtext('.//JournalIssue/PubDate/MedlineDate')
    
    if medline_date:
        if ' ' in medline_date:
            pub_year = medline_date.split(' ')[0]
            pub_month = medline_date.split(' ')[1].split('-')[0]
        else:
            pub_year = medline_date.split('-')[0]

        return pub_year, pub_month
    
    # 如果都没有找到，返回 None 或一个默认值
    return pub_year, pub_month


# 函数：将作者信息转化为期望格式
def format_authors(authors):
    formatted_authors = []
    for author in authors:
        last_name = author.findtext('LastName')
        fore_name = author.findtext('ForeName')
        initials = author.findtext('Initials')
        if last_name and fore_name and initials:
            formatted_authors.append(f"{last_name}|{fore_name}|{initials}|")
        elif last_name and initials:
            formatted_authors.append(f"{last_name}|{initials}|")
    return ";".join(formatted_authors)

# 函数：将Mesh terms格式化
def format_mesh_terms(mesh_terms):
    formatted_terms = []
    for mesh_heading in mesh_terms:
        descriptor = mesh_heading.findtext('DescriptorName')
        ui = mesh_heading.find('DescriptorName').get('UI')
        if descriptor and ui:
            formatted_terms.append(f"{ui}:{descriptor}")
    return "; ".join(formatted_terms)



# 文件夹
folder_path = '/home/gy237/project/Biomedical_datasets/total_pubmed/New_PMID_abstract/abstracts_1253-1575'


# 初始化提取的字段列表
articles_data = []
for filename in tqdm(os.listdir(folder_path), ncols = 100):
    if filename.endswith('.xml.gz'):
        file_path = os.path.join(folder_path, filename)
        xml_content = get_xml(file_path)
        # 将解压后的 XML 内容解析为 ElementTree 对象
        root = ET.fromstring(xml_content)

        # 遍历每篇文章
        for article in root.findall('PubmedArticle'):
            article_data = {}
            
            # 提取PMID
            pmid = article.findtext('.//PMID')
            article_data['pmid'] = pmid

            # 提取文章标题
            title = article.findtext('.//ArticleTitle')
            article_data['title'] = title

            # 提取摘要（如存在）
            # abstract_texts = article.findall('.//AbstractText')
            # abstract = ' '.join([abstract_text.text for abstract_text in abstract_texts if abstract_text.text])
            # article_data['abstract'] = abstract if abstract else ''
            # 导致37701329, 31804926内容不全

            abstract_texts = article.findall('.//AbstractText')
            abstract = ' '.join([tostring(abstract_text, encoding='unicode', method='text').strip() for abstract_text in abstract_texts])
            article_data['abstract'] = abstract if abstract else ''

            # 提取期刊名
            journal = article.findtext('.//Journal/Title')
            article_data['journal'] = journal

            # 提取出版日期
            pub_year, pub_month = extract_pub_date(article)
            article_data['pub_year'] = pub_year
            article_data['pub_month'] = pub_month

            # 提取作者列表并格式化
            authors = article.findall('.//Author')
            article_data['authors'] = format_authors(authors)

            # 提取Mesh Terms并格式化
            mesh_terms = article.findall('.//MeshHeading')
            article_data['mesh_terms'] = format_mesh_terms(mesh_terms)

            assert article_data.get("pub_year") is not None

            # 将结果添加到文章数据列表中
            articles_data.append(article_data)

        # with open('/home/gy237/project/Biomedical_datasets/total_pubmed/New_PMID_abstract/test.json', 'w', encoding='utf-8') as file:
        #     json.dump(articles_data[:100], file, ensure_ascii=False, indent=4)
        # exit()


# 保存数据
df = pd.DataFrame(articles_data)
joblib.dump(df, '/home/gy237/project/Biomedical_datasets/total_pubmed/New_PMID_abstract/abstracts_and_pubdata/abstracts_1253-1575_2.joblib')


# {"pmid":"1","title":"Formate assay in body fluids: application in methanol poisoning.","abstract":"","journal":"Biochemical medicine","pubdate":"1975","authors":"Makar|A B|AB|;McMartin|K E|KE|;Palese|M|M|;Tephly|T R|TR|","mesh_terms":"D000445:Aldehyde Oxidoreductases; D000818:Animals; D001826:Body Fluids; D002245:Carbon Dioxide; D005561:Formates; D000882:Haplorhini; D006801:Humans; D006863:Hydrogen-Ion Concentration; D007700:Kinetics; D000432:Methanol; D008722:Methods; D011549:Pseudomonas"}