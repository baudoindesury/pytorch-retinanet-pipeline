import argparse
import torch
from torchvision import transforms
import numpy as np

from retinanet import model
from retinanet.dataloader import CSVDataset, Resizer, Normalizer
from retinanet import csv_eval

# Inmport superpoint
from retinanet.networks.superpoint_pytorch import SuperPointFrontend

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--csv_annotations_path', help='Path to CSV annotations')
    parser.add_argument('--model_path', help='Path to model', type=str)
    parser.add_argument('--images_path',help='Path to images directory',type=str)
    parser.add_argument('--class_list_path',help='Path to classlist csv',type=str)
    parser.add_argument('--iou_threshold',help='IOU threshold used for evaluation',type=str, default='0.5')
    parser = parser.parse_args(args)

    #dataset_val = CocoDataset(parser.coco_path, set_name='val2017',transform=transforms.Compose([Normalizer(), Resizer()]))
    dataset_val = CSVDataset(parser.csv_annotations_path,parser.class_list_path,transform=transforms.Compose([Normalizer(), Resizer()]))
    
    #TODO Save checkpoints without Dataparallel
    # Create the model
    retinanet = model.resnet50(num_classes=dataset_val.num_classes(), pretrained=True)
    retinanet = torch.nn.DataParallel(retinanet)
    
    # Modify load checkpoints
    checkpoint = torch.load(parser.model_path)
    retinanet.load_state_dict(checkpoint['model'])

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        #retinanet.load_state_dict(torch.load(parser.model_path))
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet.load_state_dict(torch.load(parser.model_path))
        retinanet = torch.nn.DataParallel(retinanet)
    
    retinanet.training = False
    retinanet.eval()
    #retinanet.module.freeze_bn()

    # Load pre-trained SuperPoint teacher model
    superpoint = SuperPointFrontend(project_root='') #Modify project_root

    average_precisions, _ = csv_eval.evaluate(dataset_val, retinanet, superpoint, iou_threshold=float(parser.iou_threshold))
    average_precisions = np.array([list(ele) for ele in average_precisions.values()])
    mAP = np.sum(average_precisions[:,0] * average_precisions[:,1]) / np.sum(average_precisions[:,1])
    print('average mAP = ', mAP)




if __name__ == '__main__':
    main()
