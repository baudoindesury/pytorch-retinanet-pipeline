import argparse
import collections

import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms

from torch.utils.tensorboard import SummaryWriter


from retinanet import model
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from torch.utils.data import DataLoader

from retinanet import coco_eval
from retinanet import csv_eval

# Inmport superpoint
from retinanet.networks.superpoint_pytorch import SuperPointFrontend
from retinanet.losses import *


assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)

    parser.add_argument('--model_path', help='model path to resume training', default = None)

    parser = parser.parse_args(args)

    # Create the data loaders
    if parser.dataset == 'coco':

        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')

        dataset_train = CocoDataset(parser.coco_path, set_name='train2017',
                                    transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))

    elif parser.dataset == 'csv':

        if parser.csv_train is None:
            raise ValueError('Must provide --csv_train when training on COCO,')

        if parser.csv_classes is None:
            raise ValueError('Must provide --csv_classes when training on COCO,')

        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                   transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        print(dataset_train)

        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                     transform=transforms.Compose([Normalizer(), Resizer()]))

    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=2, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

    # Activate Tensorboard
    writer_root = '../results/training_results'
    writer = SummaryWriter(writer_root)

    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    # Load pre-trained SuperPoint teacher model
    superpoint = SuperPointFrontend(project_root='') #Modify project_root


    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.module.freeze_bn()



    print('Num training images: {}'.format(len(dataset_train)))

    iter_global = 0

    for epoch_num in range(parser.epochs):

        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            #try:
            optimizer.zero_grad()

            # Forward pass
            if torch.cuda.is_available():
                output = retinanet([data['img'].cuda().float(), data['annot']])
            else:
                output = retinanet([data['img'].float(), data['annot']])
            
            # Focal loss of retinanet
            classification_loss, regression_loss = output['focalLoss'][0], output['focalLoss'][1]
            
            # SuperPoint output Tensors
            output_desc = output['desc'].type(torch.FloatTensor)#.to(device)
            output_semi = output['semi'].type(torch.FloatTensor)#.to(device)

            # SuperPoint label with teacher model
            print('image gray :', data['img_gray'].size())
            output_superpoint = superpoint.run(data['img_gray'])
            desc_teacher = torch.from_numpy(output_superpoint['local_descriptor_map']).type(torch.FloatTensor)#.to(device)
            dect_teacher = torch.from_numpy(output_superpoint['dense_scores']).type(torch.FloatTensor)#.to(device)
            
            # Compute SuperPoint Losses
            desc_l_t = descriptor_local_loss(output_desc, desc_teacher)
            detc_l_t = detector_loss(output_semi, dect_teacher)

            # Compute RetinaNet Losses
            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()
            
            # Compute total loss
            loss_retina = classification_loss + regression_loss
            loss_superpoint = desc_l_t + detc_l_t
            loss = loss_retina + loss_superpoint
            
            writer.add_scalar('Train_Classification_Loss/Iteration', classification_loss, iter_global+1)
            writer.add_scalar('Train_Regression_Loss/Iteration', regression_loss, iter_global+1)
            writer.add_scalar('Train_SuperPoint_Descriptor/Iteration', desc_l_t, iter_global+1)
            writer.add_scalar('Train_SuperPoint_Detector/Iteration', detc_l_t, iter_global+1)
            writer.add_scalar('Train_Loss/Iteration', loss, iter_global+1)
            

            if bool(loss == 0):
                continue

            loss.backward()

            torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

            optimizer.step()

            loss_hist.append(float(loss))

            epoch_loss.append(float(loss))

            print(
                'Epoch: {} | Iteration: {}'.format(epoch_num, iter_num))
            print(
                'RetinaNet | Classification loss: {:1.5f} | Regression loss: {:1.5f}'.format(
                    float(classification_loss), float(regression_loss)))
            print(
                'SuperPoint | Detector loss: {:1.5f} | Descriptor loss: {:1.5f}'.format(
                    float(detc_l_t), float(desc_l_t)))
            print('Running Loss: {:1.5f}\n'.format(np.mean(loss_hist)))
            
            del classification_loss
            del regression_loss
            del desc_l_t
            del detc_l_t
            #except Exception as e:
            #    print('Exception : ', e)
            #    continue
            iter_global+=1
            

        if parser.dataset == 'coco':

            print('Evaluating dataset')

            coco_eval.evaluate_coco(dataset_val, retinanet)

        elif parser.dataset == 'csv' and parser.csv_val is not None:

            print('Evaluating dataset')

            mAP, losses = csv_eval.evaluate(dataset_val, retinanet, superpoint)

            #writer.add_scalar('Eval_RetinaLoss/Epoch', losses['loss_retina'], epoch_num+1)
            writer.add_scalar('Eval_SuperpointLoss/Epoch', losses['loss_superpoint'], epoch_num+1)

            #writer.add_scalar('Evaluation bike mAP/Epoch', mAP[0], epoch_num+1)
            #writer.add_scalar('Evaluation bird mAP/Epoch', mAP[1], epoch_num+1)
            #writer.add_scalar('Evaluation boat mAP/Epoch', mAP[2], epoch_num+1)
            #writer.add_scalar('Evaluation building mAP/Epoch', mAP[3], epoch_num+1)
            #writer.add_scalar('Evaluation car mAP/Epoch', mAP[4], epoch_num+1)
            #writer.add_scalar('Evaluation group mAP/Epoch', mAP[5], epoch_num+1)
            #writer.add_scalar('Evaluation person mAP/Epoch', mAP[6], epoch_num+1)
            #writer.add_scalar('Evaluation truck mAP/Epoch', mAP[7], epoch_num+1)
            #writer.add_scalar('Evaluation uav mAP/Epoch', mAP[8], epoch_num+1)
            #writer.add_scalar('Evaluation wakeboard mAP/Epoch', mAP[9], epoch_num+1)
        scheduler.step(np.mean(epoch_loss))
        

        writer.add_scalar('Train_Loss/Epoch', loss, epoch_num+1)

        # Save checkpoints
        if epoch_num%10 == 0:
            torch.save(retinanet.module, 'checkpoints/{}_retinanet_{}.pt'.format(parser.dataset, epoch_num))

    retinanet.eval()

    torch.save(retinanet, 'checkpoints/model_final.pt')
    writer.close()

import matplotlib.pyplot as plt
def plot(img, fig = 0):
    plt.figure(fig)
    plt.imshow(img) 
    plt.show()  # display it

if __name__ == '__main__':
    main()
