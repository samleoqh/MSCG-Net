from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import time

import torchvision.utils as vutils
from lib.loss.acw_loss import *
from tensorboardX import SummaryWriter
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader

from config.configs_kf import *
from lib.utils.lookahead import *
from lib.utils.lr import init_params_lr
from lib.utils.measure import *
from lib.utils.visual import *

from tools.model import load_model

cudnn.benchmark = True

prepare_gt(VAL_ROOT)
prepare_gt(TRAIN_ROOT)

train_args = agriculture_configs(net_name='MSCG-Rx101',
                                 data='Agriculture',
                                 bands_list=['NIR', 'RGB'],
                                 kf=0, k_folder=0,
                                 note='reproduce'
                                 )

train_args.input_size = [512, 512]
train_args.scale_rate = 1.  # 256./512.  # 448.0/512.0 #1.0/1.0
train_args.val_size = [512, 512]
train_args.node_size = (32, 32)
train_args.train_batch = 7
train_args.val_batch = 7

train_args.lr = 2.18e-4/np.sqrt(3)
train_args.weight_decay = 2e-5

train_args.lr_decay = 0.9
train_args.max_iter = 1e8

train_args.snapshot = ''
train_args.print_freq = 100
train_args.save_pred = False

# output training configuration to a text file
train_args.write2txt()
writer = SummaryWriter(os.path.join(train_args.save_path, 'tblog'))
visualize, restore = get_visualize(train_args)


# Remember to use num_workers=0 when creating the DataBunch.
def random_seed(seed_value, use_cuda=True):
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False


def main():
    random_seed(train_args.seeds)

    train_args.write2txt()
    net = load_model(name=train_args.model, classes=train_args.nb_classes,
                     node_size=train_args.node_size)

    net, start_epoch = train_args.resume_train(net)
    net.cuda()
    net.train()

    train_set, val_set = train_args.get_dataset()
    train_loader = DataLoader(dataset=train_set, batch_size=train_args.train_batch, num_workers=0, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=train_args.val_batch, num_workers=0)

    criterion = ACW_loss().cuda()

    params = init_params_lr(net, train_args)

    # first train with Adam for around 10 epoch, then manually change to SGD
    # to continue the rest train, Note: need resume train from the saved snapshot
    base_optimizer = optim.Adam(params, amsgrad=True)
    # base_optimizer = optim.SGD(params, momentum=train_args.momentum, nesterov=True)
    optimizer = Lookahead(base_optimizer, k=6)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 60, 1.18e-6)

    new_ep = 0

    while True:
        starttime = time.time()
        train_main_loss = AverageMeter()
        aux_train_loss = AverageMeter()
        cls_trian_loss = AverageMeter()

        start_lr = train_args.lr
        train_args.lr = optimizer.param_groups[0]['lr']
        num_iter = len(train_loader)
        curr_iter = ((start_epoch + new_ep) - 1) * num_iter
        print('---curr_iter: {}, num_iter per epoch: {}---'.format(curr_iter, num_iter))

        for i, (inputs, labels) in enumerate(train_loader):
            sys.stdout.flush()

            inputs, labels = inputs.cuda(), labels.cuda(),
            N = inputs.size(0) * inputs.size(2) * inputs.size(3)
            optimizer.zero_grad()
            outputs, cost = net(inputs)

            main_loss = criterion(outputs, labels)

            loss = main_loss + cost

            loss.backward()
            optimizer.step()
            lr_scheduler.step(epoch=(start_epoch + new_ep))

            train_main_loss.update(main_loss.item(), N)
            aux_train_loss.update(cost.item(), inputs.size(0))

            curr_iter += 1
            writer.add_scalar('main_loss', train_main_loss.avg, curr_iter)
            writer.add_scalar('aux_loss', aux_train_loss.avg, curr_iter)
            # writer.add_scalar('cls_loss', cls_trian_loss.avg, curr_iter)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], curr_iter)

            if (i + 1) % train_args.print_freq == 0:
                newtime = time.time()

                print('[epoch %d], [iter %d / %d], [loss %.5f, aux %.5f, cls %.5f], [lr %.10f], [time %.3f]' %
                      (start_epoch + new_ep, i + 1, num_iter, train_main_loss.avg, aux_train_loss.avg,
                       cls_trian_loss.avg,
                       optimizer.param_groups[0]['lr'], newtime - starttime))

                starttime = newtime

        validate(net, val_set, val_loader, criterion, optimizer, start_epoch + new_ep, new_ep)

        new_ep += 1


def validate(net, val_set, val_loader, criterion, optimizer, epoch, new_ep):
    net.eval()
    val_loss = AverageMeter()
    inputs_all, gts_all, predictions_all = [], [], []

    with torch.no_grad():
        for vi, (inputs, gts) in enumerate(val_loader):
            inputs, gts = inputs.cuda(), gts.cuda()
            N = inputs.size(0) * inputs.size(2) * inputs.size(3)
            outputs = net(inputs)

            val_loss.update(criterion(outputs, gts).item(), N)
            # val_loss.update(criterion(gts, outputs).item(), N)
            if random.random() > train_args.save_rate:
                inputs_all.append(None)
            else:
                inputs_all.append(inputs.data.squeeze(0).cpu())

            gts_all.append(gts.data.squeeze(0).cpu().numpy())
            predictions = outputs.data.max(1)[1].squeeze(1).squeeze(0).cpu().numpy()
            predictions_all.append(predictions)

    update_ckpt(net, optimizer, epoch, new_ep, val_loss,
                inputs_all, gts_all, predictions_all)

    net.train()
    return val_loss, inputs_all, gts_all, predictions_all


def update_ckpt(net, optimizer, epoch, new_ep, val_loss,
                inputs_all, gts_all, predictions_all):
    avg_loss = val_loss.avg

    acc, acc_cls, mean_iu, fwavacc, f1 = evaluate(predictions_all, gts_all, train_args.nb_classes)

    writer.add_scalar('val_loss', avg_loss, epoch)
    writer.add_scalar('acc', acc, epoch)
    writer.add_scalar('acc_cls', acc_cls, epoch)
    writer.add_scalar('mean_iu', mean_iu, epoch)
    writer.add_scalar('fwavacc', fwavacc, epoch)
    writer.add_scalar('f1_score', f1, epoch)

    updated = train_args.update_best_record(epoch, avg_loss, acc, acc_cls, mean_iu, fwavacc, f1)

    # save best record and snapshot prameters
    val_visual = []

    snapshot_name = 'epoch_%d_loss_%.5f_acc_%.5f_acc-cls_%.5f_mean-iu_%.5f_fwavacc_%.5f_f1_%.5f_lr_%.10f' % (
        epoch, avg_loss, acc, acc_cls, mean_iu, fwavacc, f1, optimizer.param_groups[0]['lr']
    )

    if updated or (train_args.best_record['val_loss'] > avg_loss):
        torch.save(net.state_dict(), os.path.join(train_args.save_path, snapshot_name + '.pth'))
        # train_args.update_best_record(epoch, val_loss.avg, acc, acc_cls, mean_iu, fwavacc, f1)
    if train_args.save_pred:
        if updated or (new_ep % 5 == 0):
            val_visual = visual_ckpt(epoch, new_ep, inputs_all, gts_all, predictions_all)

    if len(val_visual) > 0:
        val_visual = torch.stack(val_visual, 0)
        val_visual = vutils.make_grid(val_visual, nrow=3, padding=5)
        writer.add_image(snapshot_name, val_visual)


def visual_ckpt(epoch, new_ep, inputs_all, gts_all, predictions_all):
    val_visual = []
    if train_args.save_pred:
        to_save_dir = os.path.join(train_args.save_path, str(epoch) + '_' + str(new_ep))
        check_mkdir(to_save_dir)

    for idx, data in enumerate(zip(inputs_all, gts_all, predictions_all)):
        if data[0] is None:
            continue

        if train_args.val_batch == 1:
            input_pil = restore(data[0][0:3, :, :])
            gt_pil = colorize_mask(data[1], train_args.palette)
            predictions_pil = colorize_mask(data[2], train_args.palette)
        else:
            input_pil = restore(data[0][0][0:3, :, :])  # only for the first 3 bands
            # input_pil = restore(data[0][0])
            gt_pil = colorize_mask(data[1][0], train_args.palette)
            predictions_pil = colorize_mask(data[2][0], train_args.palette)

        # if train_args['val_save_to_img_file']:
        if train_args.save_pred:
            input_pil.save(os.path.join(to_save_dir, '%d_input.png' % idx))
            predictions_pil.save(os.path.join(to_save_dir, '%d_prediction.png' % idx))
            gt_pil.save(os.path.join(to_save_dir, '%d_gt.png' % idx))

        val_visual.extend([visualize(input_pil.convert('RGB')), visualize(gt_pil.convert('RGB')),
                           visualize(predictions_pil.convert('RGB'))])
    return val_visual


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


if __name__ == '__main__':
    main()
