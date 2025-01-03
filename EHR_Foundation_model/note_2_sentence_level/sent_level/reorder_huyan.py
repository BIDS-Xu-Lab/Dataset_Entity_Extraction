import os
import pandas as pd
import argparse
import numpy as np

pd.set_option('display.max_columns', 6)

def calculate_entity_density(span_file):
    """Calculate entity count, size, and density for each sentence."""

    span_df = pd.read_csv(span_file, usecols=['id', 'entity id', 'sentence_text'])
    span_df['sentence_size'] = span_df['sentence_text'].apply(len)
    
    density_df = (
        span_df.groupby(['id'])
        .agg(entity_count=('entity id', 'count'), sentence_size=('sentence_size', 'first'))
        .reset_index()
    )
    
    density_df['density'] = density_df.apply(
        lambda row: row['entity_count'] / row['sentence_size'] if row['sentence_size'] > 0 else 0,
        axis=1
    )
    
    return density_df

def prepare_mydf(span_file, info_file):
    """Merge span density data with additional info."""

    density_df = calculate_entity_density(span_file)
    info_df = pd.read_csv(info_file)
    
    mydf = info_df.merge(density_df[['id', 'density']], how='left', on='id').fillna(0)
    return mydf

def generate_notes(mydf, n_note, output_dir, order, n_sent_in_note):
    """Generate and save combined notes with specified density sorting."""
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(n_note):
        no_entity_df = mydf[mydf['density'] == 0]
        has_entity_df = mydf[mydf['density'] > 0]
        
        subgroup_size = int(n_sent_in_note/2)
        sample_no_entity = no_entity_df.sample(n=subgroup_size, replace=True, random_state=i)
        sample_has_entity = has_entity_df.sample(n=subgroup_size, replace=True, random_state=i)
        
        sampled_mydf = pd.concat([sample_no_entity, sample_has_entity]).sort_values(
            by=['density'], ascending=(order == 'ascending')
        )
        
        combined_bio = ""
        for _, row in sampled_mydf.iterrows():
            file_path = row['file']
            with open(file_path, 'r') as bio_file:
                combined_bio += bio_file.read()
        
        output_path = os.path.join(output_dir, f'note_{i + 1}.bio')
        print(f'Saved in {output_path} and note length is {len(combined_bio)}')
        with open(output_path, 'w') as output_file:
            output_file.write(combined_bio)
    
    print(f"Generated notes saved in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine Huyan notes into chunks.")
    parser.add_argument( '--huyan_location', default='/home/jupyter/20000360102458359xu/Weipeng/symptomllm/data/huyan', help='')
    parser.add_argument('--n_note', default=10, help='')
    parser.add_argument('--n_sent_in_note', default=200, help='')
    parser.add_argument('--order', default='ascending', choices=['ascending', 'descending'], help='')
    
    args = parser.parse_args()
    huyan_location = args.huyan_location
    
    local_span_file = os.path.join(huyan_location, 'local_token_span_df.csv')
    info_file = os.path.join(huyan_location, 'info_df.csv')
    output_dir = os.path.join(huyan_location + '_' + args.order)
    
    mydf = prepare_mydf(local_span_file, info_file)
    generate_notes(mydf, n_note=args.n_note, output_dir=output_dir, order=args.order, n_sent_in_note=args.n_sent_in_note)
