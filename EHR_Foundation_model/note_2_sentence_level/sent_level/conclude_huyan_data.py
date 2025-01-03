import os
import pandas as pd
import argparse

def extract_entities(tokens, labels):
    """Extracts entities from BIO-annotated sequences of tokens and labels."""
    entities = []
    entity_id = 1
    current_entity = None

    for idx, (token, label) in enumerate(zip(tokens, labels)):
        if label.startswith("B-"):  # Beginning of a new entity
            if current_entity:
                entities.append(current_entity)
            
            entity_type = label[2:]  # Extract the entity type (e.g., "Disease")
            current_entity = {
                "entity id": entity_id,
                "entity type": entity_type,
                "entity start index": idx,
                "entity end index": idx,
                "entity tokens": [token],
            }
            entity_id += 1
        
        elif label.startswith("I-") and current_entity and label[2:] == current_entity["entity type"]:
            current_entity["entity end index"] = idx
            current_entity["entity tokens"].append(token)
        
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    
    if current_entity:
        entities.append(current_entity)
    
    return pd.DataFrame(entities, columns=[
        "entity id", "entity type", 
        "entity start index", "entity end index", 
        "entity tokens"
    ])


def create_split_df(root_path):
    splits = ["train", "valid", "test"]
    data = []
    for split in splits:
        folder_path = os.path.join(root_path, split)
        if os.path.exists(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith(".bio"):  # Only process .bio files
                    file_id = int(os.path.splitext(file)[0])  # Remove the file extension to get the ID
                    data.append({"index": file_id, "id": file, "split": split})
    df = pd.DataFrame(data)
    return df

def process_bio_files(input_dir, split_infer_dir):
    split_df = create_split_df(split_infer_dir)
    bio_files = [f for f in os.listdir(input_dir) if f.endswith('.bio')]
    
    info_row_list = []
    local_df_span_df_list = []

    for _, file_name in enumerate(bio_files):
        file_path = os.path.join(input_dir, file_name)

        file_df = pd.read_csv(file_path, sep='\t', names=['token', 'tag'], na_filter=False)
        
        info_row = {'id': file_name, 'file': file_path, 'text': " ".join(file_df['token'].to_list())}
        info_row_list.append(info_row)

        # Extract entities and include file-related information
        local_df = extract_entities(file_df['token'].to_list(), file_df['tag'].to_list())
        local_df.insert(0, 'id', file_name)
        local_df['file'] = file_path
        
        # Add context info (full sentence for inspection purposes)
        local_df['sentence_text'] = " ".join(file_df['token'].to_list())
        
        local_df_span_df_list.append(local_df)
    
    local_token_span_df = pd.concat(local_df_span_df_list, ignore_index=True)
    local_token_span_df = pd.merge(local_token_span_df, split_df, on='id', how='left').sort_values('index')

    output_path = os.path.join(input_dir, 'local_token_span_df.csv')
    local_token_span_df.to_csv(output_path, index=False)

    info_df = pd.DataFrame(info_row_list)
    info_df = pd.merge(info_df, split_df, on='id', how='left').sort_values('index')
    output_path = os.path.join(input_dir, 'info_df.csv')
    info_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process BIO-annotated files and extract entities.")
    parser.add_argument('--data_dir', type=str, default='/home/jupyter/20000360102458359xu/Weipeng/symptomllm/data/huyan', help="Directory containing the '.bio' files (default: %(default)s)")
    parser.add_argument('--split_infer_dir', type=str, default='/home/jupyter/20000360102458359xu/Weipeng/symptomllm/Yan_NER/Clinical_Entity_Recognition_Using_GPT_models/data/MTSamples', help="Directory containing the '.bio' files (default: %(default)s)")
    args = parser.parse_args()
    process_bio_files(args.data_dir, args.split_infer_dir)