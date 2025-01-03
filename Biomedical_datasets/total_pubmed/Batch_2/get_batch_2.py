import os
import shutil


folder_name = '/home/gy237/project/Biomedical_datasets/total_pubmed/Batch_1/Gui_collections'


file_list = os.listdir(folder_name)
file_list = [i.split('.')[0] for i in file_list]

for pmid in file_list[:100]:
    # 源文件路径
    source_file = f'{folder_name}/{pmid}.txt'
    # 目标文件夹路径
    destination_folder = '/home/gy237/project/Biomedical_datasets/total_pubmed/Batch_2/Batch_2'

    # 执行复制操作
    shutil.copy(source_file, destination_folder)