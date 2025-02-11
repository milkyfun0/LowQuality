from __future__ import print_function
import os
import sys
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from tensorboardX import SummaryWriter
from loss.pairwise_loss import Pairwise_Loss
from loss.contrastive_loss import Contrastive_Loss
from utils.tool import *
from model.alexnet_wtopk_twovq_fc import AlexNet
from timm.scheduler.cosine_lr import CosineLRScheduler


def parse_option():
    parser = argparse.ArgumentParser('argument for paper')

    parser.add_argument('--test_interval', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--learning_rate', type=float, default=0.0001)  # 0.05
    parser.add_argument('--weight_decay', type=float, default=5e-4)  # 0.0005
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--bit', default=32, type=int)
    parser.add_argument('--severity', default=3, type=int)
    parser.add_argument('--num_embedding', default=1000, type=int)
    parser.add_argument('--topk', default=10, type=int)
    parser.add_argument('--data_type', default="vq")
    parser.add_argument('--cor_name', default="gaussian_noise",
                        type=str)  # gaussian_noise  gaussian_blur fog motion_blur
    parser.add_argument('--method',
                        default="alexnet_wtopk10_twovq(5000afterhash)_fc_1.0pairwise_motion_blur(s3)_NWPU_adam",
                        type=str)  # alexnet_pairwise(margin1.0)_ucmd
    parser.add_argument('--dataset', type=str, default='UCMD')
    parser.add_argument('--root_path', default='../data/UCMD/original/')
    parser.add_argument('--txtfile_path', default='txtfile/')
    parser.add_argument('--save_path', default='/home/lkshpc/huangmengluan/low-quality/log/UCMD/')
    parser.add_argument('--gpu', type=str, default='9')

    # loss setting
    parser.add_argument('--margin', type=float, default=0.7)
    parser.add_argument('--lamda', type=float, default=0.01)
    args = parser.parse_args()
    return args


def train(train_loader, test_loader, dataset_loader, bit, args):
    net = AlexNet(hash_bit=bit, classes=21, num_embedding=args.num_embedding, topk=args.topk).cuda()

    n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")

    criterion_cls = nn.CrossEntropyLoss().cuda()
    criterion_pairwise = Pairwise_Loss(margin=args.margin).cuda()
    contristive_pairwise = Contrastive_Loss().cuda()

    # optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    Best_mAP = 0

    current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
    logger.info("[%s] bit:%d,  dataset:%s, training...." % (current_time, bit, args.dataset), end="")
    last_model_name = ""
    for epoch in range(args.epochs):
        net.train()
        train_loss = 0
        train_pair_loss = 0
        train_cls_loss = 0
        train_vq_loss = 0
        train_contrastive_loss = 0
        for idx, (image, distort_image, label, ind) in enumerate(tqdm(train_loader, desc=f"train {epoch}")):
            image = image.cuda()
            distort_image = distort_image.cuda()
            label = np.argmax(label, axis=1).cuda()
            optimizer.zero_grad()
            features = torch.cat([image, distort_image], dim=0)
            hash_, e_q_loss_clean, e_q_loss_distort, logit, index = net(features)

            # print(features.shape)
            con_loss_1 = contristive_pairwise(hash_[:len(label)], label)
            con_loss_2 = contristive_pairwise(hash_[len(label):], label)
            labels = torch.cat([label, label])
            # print(labels.shape)
            loss_pair = criterion_pairwise(hash_, labels)
            loss_cls = criterion_cls(logit, labels)

            loss = loss_pair + loss_cls + e_q_loss_clean + e_q_loss_distort + con_loss_1 * 0.05 + con_loss_2 * 0.05

            train_vq_loss += (e_q_loss_clean.item() + e_q_loss_distort.item())
            train_pair_loss += loss_pair.item()
            train_cls_loss += loss_cls.item()
            train_contrastive_loss += con_loss_1.item() + con_loss_2.item()
            train_loss += loss.item()

            loss.backward()
            # if epoch >= 100:
            #     torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=2)
            # if 100 < epoch <= 103:
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] /= 2

            optimizer.step()

        train_pair_loss = train_pair_loss / len(train_loader)
        train_cls_loss = train_cls_loss / len(train_loader)
        train_vq_loss = train_vq_loss / len(train_loader)
        train_loss = train_loss / len(train_loader)

        logger.info(
            '[epoch:{}/{}][pair_loss:{:.6f}][cls_loss:{:.6f}][vq_loss:{:.6f}][total_loss:{:.6f}][train_contrastive_loss:{:6f}]'.format(
                epoch + 1, args.epochs, train_pair_loss, train_cls_loss, train_vq_loss, train_loss,
                train_contrastive_loss
            )
        )
        if (epoch + 1) % args.test_interval == 0:
            test_binary, test_label = compute_hashcode(test_loader, net, args)
            database_binary, database_label = compute_hashcode(dataset_loader, net, args)
            mAP = CalcMap(database_binary.numpy(), test_binary.numpy(), database_label.numpy(), test_label.numpy())
            if mAP > Best_mAP:
                Best_mAP = mAP
                dir_name = args.log_path + "/" + args.dataset + f"-{epoch}-" + str(bit) + "-" + str(
                    Best_mAP) + "-model.pt"
                torch.save(net.state_dict(), dir_name)
                if os.path.isfile(last_model_name):
                    os.remove(last_model_name)
                    print('delete model in: %s' % last_model_name)
                last_model_name = dir_name

            logger.info("epoch:%d, bit:%d, dataset:%s, MAP:%.4f, Best MAP: %.4f" % (
                epoch + 1, bit, args.dataset, mAP, Best_mAP))


def main():
    args = parse_option()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    args.log_path = os.path.join(args.save_path, args.method)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    logger.add(args.log_path + '/' + 'train.log')
    logger.info(args)
    # writer = SummaryWriter(log_dir=log_path)

    train_loader, test_loader, dataset_loader = get_dataloader(args)
    train(train_loader, test_loader, dataset_loader, args.bit, args)


if __name__ == "__main__":
    main()
