Here is all the code for the Gui.
Created on Jan/3/2025

The folder, “Biomedical_datasets”, is the “Dataset and Repository Ecosystem” project folder.
The following are useful scripts, data and results.

1. The folder, “Biomedical_datasets/NER”, contains scripts, data and results to fine-tune BERT models to conduct NER tasks.
1.1. The folder, “Biomedical_datasets/NER/nerbert” contains scripts and results to fine-tune BERT models to conduct NER tasks.
1.1.1. “Biomedical_datasets/NER/nerbert/nerbert.py” and “Biomedical_datasets/NER/nerbert/nerbert_nospacy.py” are the pipelines to fine-tune BERT models to conduct NER tasks.
1.1.1.1 “Biomedical_datasets/data_update/precessed_data/train_data.json”  and “Biomedical_datasets/data_update/precessed_data/test_data.json” are the training and testing data for “nerbert_nospacy.py”.
1.1.1.2. “Biomedical_datasets/NER/data/train_data.json” and “Biomedical_datasets/NER/data/test_data.json” are the training and testing data for “nerbert.py”.
1.1.1.3. “Biomedical_datasets/NER/nerbert/{_name}_output/time_{time}'” is the output path. _name is the name of the model. time is the the times for fine-tuning the model. Please see the details in the pipeline. 
1.1.2. “Biomedical_datasets/NER/nerbert/biobert-large-cased-v1.1-squad_output/time_1/hyperparameter.json” is the hyperparameter for fine-tune “biobert-large-cased-v1.1-squad” model, the 1 time. And the f1 score for this fine-tuning is “Biomedical_datasets/NER/nerbert/biobert-large-cased-v1.1-squad_output/time_1/performance.csv”.
2. The folder, “Biomedical_datasets/RE”, contains scripts, data and results to fine-tune BERT models to conduct RE tasks.

3. “Biomedical_datasets/total_pubmed” is the pipeline for getting the currently annotated Batches files.
3.1. “Biomedical_datasets/total_pubmed/Sampled_from_total_PubMed_specific_name_12_16” is useful and the others are ueseless.
3.2. “Biomedical_datasets/total_pubmed/Sampled_from_total_PubMed_specific_name_12_16/example_batch.ipynb” is the pipeline to use GPT-4o to filter the abstracts and get the files to be annotated.
3.3. “/home/gy237/project/Biomedical_datasets/total_pubmed/Sampled_from_total_PubMed_specific_name_12_16/abstracts_nonull_1-1575.joblib” is the total PubMed abstracts file and it has the data: “['pmid', 'title', 'abstract', 'journal', 'pubdate', 'authors', 'mesh_terms', 'pub_year', 'pub_month']”
