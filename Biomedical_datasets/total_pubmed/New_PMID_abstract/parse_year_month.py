import os
import gzip
import xml.etree.ElementTree as ET
import json
from tqdm import tqdm
import pandas as pd
import joblib
import re


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
    if article.findtext('.//JournalIssue/PubDate'):
        pub_year = article.findtext('.//JournalIssue/PubDate/Year')
        pub_month = article.findtext('.//JournalIssue/PubDate/Month')

        if pub_year:
            # 如果同时有年份和月份，直接返回格式化的日期
            return pub_year, pub_month, '1321'
        
        # 如果没有 Year 和 Month，尝试使用 MedlineDate 提取
        medline_date = article.findtext('.//JournalIssue/PubDate/MedlineDate')
        data = article.findtext('.//JournalIssue/PubDate')
        
        if medline_date:
            pattern = r"(\b\d{4}\b)|([A-Za-z]+)"
            matches = re.findall(pattern, medline_date)
            pub_year = next((match[0] for match in matches if match[0]), None)
            pub_month = next((match[1] for match in matches if match[1]), None)

            # if ' ' in medline_date:
            #     if medline_date.split(' ')[0] in ['Spring', 'Winter', 'Summer', 'Autumn', 'Fall', 'fall']: # Fall 2017
            #         pub_year = medline_date.split(' ')[1]
            #         pub_month = medline_date.split(' ')[0]
            #     elif '-' in medline_date.split(' ')[0]:             # 2005-2006 Winter
            #         pub_year = medline_date.split(' ')[0].split('-')[0]
            #         pub_month = medline_date.split(' ')[1]
            #     else:
            #         pub_year = medline_date.split(' ')[0]
            #         pub_month = medline_date.split(' ')[1].split('-')[0]
            # else:
            #     pub_year = medline_date.split('-')[0]
    
    elif article.find(".//PubmedData/History/PubMedPubDate[@PubStatus='pubmed']") is not None:
        pub_year = date.find("Year").text if date.find("Year") is not None else None
        pub_month = date.find("Month").text if date.find("Month") is not None else None
    
    elif article.find(".//PubmedData/History/PubMedPubDate[@PubStatus='medline']") is not None:
        pub_year = date.find("Year").text if date.find("Year") is not None else None
        pub_month = date.find("Month").text if date.find("Month") is not None else None

    return pub_year, pub_month, data



# 文件夹
folder_path = '/gpfs/gibbs/project/huan_he/shared/datasets/pubmed/raw/updatefiles'
# /gpfs/gibbs/project/huan_he/shared/datasets/pubmed/raw/updatefiles    1220-1252
# /gpfs/gibbs/project/huan_he/shared/datasets/pubmed/raw/baseline       1-1219

# 初始化提取的字段列表
articles_data = []
flag = False
for filename in tqdm(os.listdir(folder_path), ncols = 100):
    if filename.endswith('.xml.gz'):
        try:
            file_path = os.path.join(folder_path, filename)
            xml_content = get_xml(file_path)
            # 将解压后的 XML 内容解析为 ElementTree 对象
            root = ET.fromstring(xml_content)
        except:
            print(filename)
            # exit()

        # 遍历每篇文章
        for article in root.findall('PubmedArticle'):
            article_data = {}
            # 提取PMID
            pmid = article.findtext('.//PMID')
            article_data['pmid'] = pmid

            # 提取出版日期
            pub_year, pub_month, date = extract_pub_date(article)
            article_data['pub_year'] = pub_year
            article_data['pub_month'] = pub_month

            if len(pub_year) != 4:
                print('adwdwqd')
                flag = True
            try:
                x = int(pub_year)
            except:
                print('ddddddddd')
                flag = True
            if flag:
                print(pub_year)
                article_str = ET.tostring(article, encoding='utf-8').decode('utf-8')
                print(article_str)
                exit()


            # 将结果添加到文章数据列表中
            articles_data.append(article_data)

# 保存数据
df = pd.DataFrame(articles_data)
joblib.dump(df, '/home/gy237/project/Biomedical_datasets/total_pubmed/New_PMID_abstract/abstracts_and_pubdata/pubdate_1220-1252.joblib')