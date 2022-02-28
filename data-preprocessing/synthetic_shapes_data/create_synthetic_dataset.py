import argparse
import yaml
import os
import logging


from SyntheticDataset_gaussian import SyntheticDataset_gaussian as Dataset


def create_synthetic_dataset(config):
    task = config['data']['dataset']

    train_set = Dataset(
        task = 'train',
        **config['data'],
    )

    val_set = Dataset(
        task = 'val',
        **config['data'],
    )

def create_synthetic_anno():
    synthetic_shapes_dataset_path =  "/home/baudoin/data/synthetic_shapes/"
    synthetic_folders = ['draw_lines', 'draw_polygon', 'draw_multiple_polygons', 'draw_ellipses', 'draw_star', 'draw_checkerboard', 'draw_stripes', 'draw_cube', 'gaussian_noise']
    sets = ['training', 'validation', 'test']
    for set in sets:
        save_dir = '/home/baudoin/pytorch-retinanet-pipeline/synthetic_shapes_anno/anno_'+ set +'.csv'
        with open(save_dir, 'w') as csv_file:
            for shapes_path in synthetic_folders:
                images_path = synthetic_shapes_dataset_path + shapes_path + '/images/' + set +'/'
                for path in sorted(os.listdir(images_path)):
                    full_path = os.path.join(images_path, path)
                    if os.path.isfile(full_path):
                        csv_file.write(images_path + path + ',' + ','.join(['','','', '']) + ',' + '' + '\n')

        import random
        fid = open(save_dir, "r")
        li = fid.readlines()
        fid.close()
        random.shuffle(li)
        shuffled_anno = '/home/baudoin/pytorch-retinanet-pipeline/synthetic_shapes_anno/anno_shuffled_'+ set +'.csv'
        fid = open(shuffled_anno, "w")
        fid.writelines(li)
        fid.close()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)

    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    create_synthetic_dataset(config)
    create_synthetic_anno()