import pandas as pd
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-i',
        '--input',
        help='Path of RetinaNet annotations file',
    )
    parser.add_argument(
        '-o',
        '--output',
        help='directory to save training and evaluation annotations',
    )
    parser.add_argument(
        '-l',
        '--labels',
        default=['bike3','boat5','car11','building3','car8','person10', 'person18','wakeboard3','truck3','uav3'],
        help='List of string corresponding of evaluation labels',
    )
    return parser.parse_args()

def main():
    args = parse_args()

    evaluation_labels = args.labels
    anno_to_split_path = args.input
    output_path = args.output

    if not os.path.exists(output_path):
        os.mkdir(output_path)


    df = pd.read_csv(anno_to_split_path, names=['path', 'x1','x2', 'y1', 'y2', 'label'])
    n_rows = 10
    df = df.iloc[::n_rows, :]


    evaluation_df = pd.DataFrame(columns=['path', 'x1','x2', 'y1', 'y2', 'label'])
    for label in evaluation_labels:
        evaluation_df = pd.concat([evaluation_df,df[df['path'].str.contains(label)]])
        df = df.drop(df[df['path'].str.contains(label)].index)

    evaluation_df.to_csv(output_path + '/evaluation_annotations_2.csv', header=False, index=False, float_format='%.0f', na_rep='')
    df.to_csv(output_path + '/training_annotations_2.csv', header=False, index=False, float_format='%.0f', na_rep='')

    print("Split annotations done and saved to : ", output_path)

if __name__ == '__main__':
    main()