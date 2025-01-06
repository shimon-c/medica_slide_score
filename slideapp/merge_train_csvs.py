import os
import pandas as pd
import argparse

def get_args():
    ap = argparse.ArgumentParser("Merge train csvs")
    ap.add_argument('--file_name', type=str,required=True, help="Path to a file that constains csv")
    args = ap.parse_args()
    return args

def merge_list(csv_file_names=None):
    csv_names = []
    with open(csv_file_names, 'r') as file:
        # Read each line in the file
        for line in file:
            csv_names.append(line)
    df = pd.DataFrame()
    dir = None
    for csv in csv_names:
        csv = csv.strip('\n')
        if os.path.exists(csv):
            dir = os.path.dirname(csv)
            df1 = pd.read_csv(csv)
            df = pd.concat([df,df1])
    dir = os.path.dirname(csv_file_names)
    df1 = df[["file_name", "class"]]
    df = df1
    return df,dir
 #example --file_name="/mnt/medica/medica_data/medica_train_combind.txt"

if __name__ == '__main__':
    args = get_args()
    df,dir = merge_list(args.file_name)
    csv_name = os.path.join(dir,'merged_train.csv')
    df.to_csv(csv_name)
    print(f'csv:{csv_name}')

