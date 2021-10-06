import os
import os.path as osp
import time
from datetime import datetime

import torch
from tensorboardX import SummaryWriter

import lr_scheduler
import setproctitle
from torch.utils.data import Dataset, DataLoader

import sys
import torch.cuda.amp as amp

import metric
import prefetch

best_acc1 = 0


def train_model_split(model, trainset, trainset_gt=None, out_path=None, blackbox=None, batch_size=10, criterion_train=None,
                      criterion_test=None, testset=None,device=None, num_workers=10, lr=0.1, momentum=0.5, lr_step=30, lr_gamma=0.1, resume=None,
                      epochs=100, log_interval=10, weighted_loss=False, checkpoint_suffix='', optimizer=None, scheduler=None,
                      writer=None, s_m=None, args=None):

    best_acc_all = [0, 0, 0, 0, 0, 0]

    # Data loaders
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    if testset is not None:
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    else:
        test_loader = None

    # scale learning rate based on global batch size
    if args.is_linear_lr:
        args = lr_scheduler.scale_lr_and_momentum(args)

    # If we traing the model seperately, all the number of loops will be one.
    # It is similar as split_factor = 1
    args.loop_factor = 1 if args.is_train_sep else args.split_factor

    # use distributed training or not
    # args.is_distributed = args.world_size > 1 or args.multiprocessing_distributed

    global best_acc1
    # args.gpu = gpu

    # set the name of the process
    setproctitle.setproctitle(args.proc_name + '_rank{}'.format(args.rank))

    # define loss function (criterion) and optimizer
    # criterion = torch.nn.CrossEntropyLoss()

    # create model
    if args.pretrained:
        model_info = "INFO:PyTorch: using pre-trained model '{}'".format(args.arch)
    else:
        model_info = "INFO:PyTorch: creating model '{}'".format(args.arch)
    print(model_info)

    # print the number of parameters in the model
    print("INFO:PyTorch: The number of parameters in the model is {}".format(metric.get_the_number_of_params(model)))

    # DataParallel will divide and allocate batch_size to all available GPUs

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model)

    # optimizer
    param_groups = model.parameters() if args.is_wd_all else lr_scheduler.get_parameter_groups(model)
    if args.is_wd_all:
        print("INFO:PyTorch: Applying weight decay to all learnable parameters in the model.")

    if args.optimizer == 'SGD':
        print("INFO:PyTorch: using SGD optimizer.")
        optimizer = torch.optim.SGD(param_groups,
                                    args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    nesterov=True if args.is_nesterov else False
                                    )
    # elif args.optimizer == "AdamW":
    #     print("INFO:PyTorch: using AdamW optimizer.")
    #     optimizer = torch.optim.AdamW(param_groups, lr=args.lr,
    #                                   betas=(0.9, 0.999),
    #                                   eps=1e-4,
    #                                   weight_decay=args.weight_decay)
    #
    # elif args.optimizer == "RMSprop":
    #     # See efficientNet at https://github.com/tensorflow/tpu/
    #     print("INFO:PyTorch: using RMSprop optimizer.")
    #     optimizer = torch.optim.RMSprop(param_groups, lr=args.lr,
    #                                     alpha=0.9,
    #                                     weight_decay=args.weight_decay,
    #                                     momentum=0.9)
    #
    # elif args.optimizer == "RMSpropTF":
    #     # https://github.com/rwightman/pytorch-image-models/blob/fcb6258877/timm/optim/rmsprop_tf.py
    #     print("INFO:PyTorch: using RMSpropTF optimizer.")
    #     optimizer = rmsprop_tf.RMSpropTF(param_groups, lr=args.lr,
    #                                      alpha=0.9,
    #                                      eps=0.001,
    #                                      weight_decay=args.weight_decay,
    #                                      momentum=0.9,
    #                                      decoupled_decay=False)
    else:
        raise NotImplementedError

    # PyTorch AMP loss scaler
    scaler = None if not args.is_amp else amp.GradScaler()

    # accelarate the training
    torch.backends.cudnn.benchmark = True

    # Data loading code
    # data_split_factor = args.loop_factor if args.is_diff_data_train else 1
    # print("INFO:PyTorch: => The number of views of train data is '{}'".format(data_split_factor))
    # train_loader, train_sampler = factory.get_data_loader(args.data,
    #                                                       split_factor=data_split_factor,
    #                                                       batch_size=args.batch_size,
    #                                                       crop_size=args.crop_size,
    #                                                       dataset=args.dataset,
    #                                                       split="train",
    #                                                       is_distributed=args.is_distributed,
    #                                                       is_autoaugment=args.is_autoaugment,
    #                                                       randaa=args.randaa,
    #                                                       is_cutout=args.is_cutout,
    #                                                       erase_p=args.erase_p,
    #                                                       num_workers=args.workers)
    # val_loader = factory.get_data_loader(args.data,
    #                                      batch_size=args.eval_batch_size,
    #                                      crop_size=args.crop_size,
    #                                      dataset=args.dataset,
    #                                      split="val",
    #                                      num_workers=args.workers)

    # learning rate scheduler
    scheduler = lr_scheduler.lr_scheduler(mode=args.lr_mode,
                                          init_lr=args.lr,
                                          num_epochs=args.epochs,
                                          iters_per_epoch=len(train_loader),
                                          lr_milestones=args.lr_milestones,
                                          lr_step_multiplier=args.lr_step_multiplier,
                                          slow_start_epochs=args.slow_start_epochs,
                                          slow_start_lr=args.slow_start_lr,
                                          end_lr=args.end_lr,
                                          multiplier=args.lr_multiplier,
                                          decay_factor=args.decay_factor,
                                          decay_epochs=args.decay_epochs,
                                          staircase=True
                                          )

    if args.evaluate:
        validate(test_loader, model, args, device, blackbox)
        return None

    saved_ckpt_filenames = []

    streams = None
    # streams = [torch.cuda.Stream() for i in range(args.loop_factor)]
    start_epoch = 1
    # val_writer = SummaryWriter(log_dir=os.path.join(out_path, 'val'))

    log_path = osp.join(out_path, '{}.log.tsv'.format(s_m))
    if not osp.exists(log_path):
        with open(log_path, 'w') as wf:
            # columns = ['run_id', 'epoch', 'split', 'loss', 'accuracy', 'best_accuracy']
            columns = ['run_id', 'epochs', 'query_number', 'train_loss', 'train_acc', 'acc@1', 'acc@5', 'fidelity@1', 'fidelity@5', 'black_acc@1', 'black_acc@5']
            wf.write('\t'.join(columns) + '\n')
    run_id = str(datetime.now())

    best_interval = 0
    for epoch in range(start_epoch, epochs + 1):
        # train for one epoch
        print("[epoch] ", epoch)
        train_loss, train_acc = train(train_loader, model, optimizer, scheduler, epoch, args, device, streams, scaler=scaler, blackbox=blackbox)

        if (epoch + 1) % args.eval_per_epoch == 0: #eval_per_epoch=1
            # evaluate on validation set
            acc_all = validate(test_loader, model, args, device, blackbox=blackbox)

            # remember best acc@1 and save checkpoint
            # is_best = acc_all[0] > best_acc1
            # best_acc1 = max(acc_all[0], best_acc1)
            is_best = acc_all[2] > best_acc1 #fidelity
            if not is_best:
                best_interval += 1
            else:
                best_interval = 0
            best_acc1 = max(acc_all[2], best_acc1)
            if acc_all[2] > best_acc_all[2]:
                best_acc_all = acc_all
            # save checkpoint
            if not args.multiprocessing_distributed:
                # summary per epoch
                # val_writer.add_scalar('avg_acc1', acc_all[0], global_step=epoch)
                # if args.dataset == 'imagenet':
                #     val_writer.add_scalar('avg_acc5', acc_all[1], global_step=epoch)
                #
                # for i in range(2, args.loop_factor + 2):
                #     val_writer.add_scalar('{}_acc1'.format(i - 1), acc_all[i], global_step=epoch)
                #
                # val_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step=epoch)
                # val_writer.add_scalar('best_acc1', best_acc1, global_step=epoch)

                # save checkpoints
                filename = "checkpoint_{0}.pth.tar".format(epoch)
                saved_ckpt_filenames.append(filename)
                # remove the oldest file if the number of saved ckpts is greater than args.max_ckpt_nums
                if len(saved_ckpt_filenames) > args.max_ckpt_nums:
                    os.remove(os.path.join(out_path, saved_ckpt_filenames.pop(0)))

                ckpt_dict = {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                }

                if args.is_amp:
                    ckpt_dict['scaler'] = scaler.state_dict()

                metric.save_checkpoint(ckpt_dict, is_best, out_path, filename=filename)

            if best_interval > 100:
                break

                # here, save
    with open(log_path, 'a') as af:
        train_cols = [run_id, epoch, checkpoint_suffix, train_loss, train_acc, best_acc_all[0], best_acc_all[1], best_acc_all[2], best_acc_all[3],
                     best_acc_all[4], best_acc_all[5]]
        af.write('\t'.join([str(c) for c in train_cols]) + '\n')

    # clean GPU cache
    # torch.cuda.empty_cache()
    # sys.exit(0)


def train(train_loader, model, optimizer, scheduler, epoch, args, device, streams=None, scaler=None, blackbox=None):
    """training function"""
    batch_time = metric.AverageMeter('Time', ':6.3f')
    data_time = metric.AverageMeter('Data', ':6.3f')
    avg_ce_loss = metric.AverageMeter('ce_loss', ':.4e')
    avg_cot_loss = metric.AverageMeter('cot_loss', ':.4e')

    # record the top1 accuray of each small network
    top1_all = []
    for i in range(args.loop_factor):
        # ce_losses_l.append(metric.AverageMeter('{}_CE_Loss'.format(i), ':.4e'))
        top1_all.append(metric.AverageMeter('{}_Acc@1'.format(i), ':6.2f'))
    avg_top1 = metric.AverageMeter('Avg_Acc@1', ':6.2f')
    # if args.dataset == 'imagenet':
    #	avg_top5 = metric.AverageMeter('Avg_Acc@1', ':6.2f')

    # show all
    total_iters = len(train_loader)
    progress = metric.ProgressMeter(total_iters, batch_time, data_time, avg_ce_loss, avg_cot_loss,
                                    *top1_all, avg_top1, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    end = time.time()

    # prefetch data
    # prefetcher = prefetch.data_prefetcher(train_loader)
    # images, target = prefetcher.next()
    i = 0

    """Another way to load the data
    for i, (images, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
    """
    optimizer.zero_grad()
    # while images is not None:
    train_loss_avg = 0
    batch_number = 1
    for batch_idx, (images, target) in enumerate(train_loader):
        images, target = images.to(device), target.to(device)
        # measure data loading time
        data_time.update(time.time() - end)
        # adjust the lr first
        scheduler(optimizer, i, epoch)
        i += 1

        # compute outputs and losses
        if args.is_amp:
            # Runs the forward pass with autocasting.
            with amp.autocast():
                ensemble_output, outputs, ce_loss, cot_loss = model(x=images,
                                                                    target=target,
                                                                    mode='train',
                                                                    epoch=epoch,
                                                                    streams=streams)
        else:
            ensemble_output, outputs, ce_loss, cot_loss = model(x=images,
                                                                target=target,
                                                                mode='train',
                                                                epoch=epoch,
                                                                streams=streams)

        # measure accuracy and record loss
        batch_size_now = images.size(0)
        # notice the index i and j, avoid contradictory
        # print("outputs.shape", outputs.shape)
        for j in range(args.loop_factor):
            acc1 = metric.accuracy(outputs[j, ...], target, topk=(1,))
            top1_all[j].update(acc1[0].item(), batch_size_now)

        # simply average outputs of small networks
        avg_acc1 = metric.accuracy(ensemble_output, target, topk=(1,))
        avg_top1.update(avg_acc1[0].item(), batch_size_now)
        # avg_top5.update(avg_acc1[0].item(), batch_size_now)

        avg_ce_loss.update(ce_loss.mean().item(), batch_size_now)
        avg_cot_loss.update(cot_loss.mean().item(), batch_size_now)
        train_loss_avg += ce_loss.mean().item()
        batch_number = batch_idx

        # compute gradient and do SGD step
        total_loss = (ce_loss + cot_loss) / args.iters_to_accumulate

        if args.is_amp:
            # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
            # Backward passes under autocast are not recommended.
            # Backward ops run in the same dtype autocast chose for corresponding forward ops.
            scaler.scale(total_loss).backward()

            if i % args.iters_to_accumulate == 0 or i == total_iters:
                # scaler.step() first unscales the gradients of the optimizer's assigned params.
                # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
                # otherwise, optimizer.step() is skipped.
                scaler.step(optimizer)
                # Updates the scale for next iteration.
                scaler.update()
                optimizer.zero_grad()
        else:
            total_loss.backward()
            if i % args.iters_to_accumulate == 0 or i == total_iters:
                optimizer.step()
                optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print("why not print,",i)
        if not args.multiprocessing_distributed or (args.rank % args.ngpus_per_node == 0):
            if i % (args.print_freq * args.iters_to_accumulate) == 0:
                progress.print(i)
        # images, target = prefetcher.next()


    return train_loss_avg/batch_number, avg_acc1[0].item()


# clean GPU cache
# torch.cuda.empty_cache()


def validate(val_loader, model, args, device, streams=None, blackbox=None):
    """validate function"""
    batch_time = metric.AverageMeter('Time', ':6.3f')
    avg_ce_loss = metric.AverageMeter('ce_loss', ':.4e')

    # record the top1 accuray of each small network
    top1_all = []
    blaxk_top1_all = []
    for i in range(args.loop_factor):
        top1_all.append(metric.AverageMeter('{}_Acc@1'.format(i), ':6.2f'))
        blaxk_top1_all.append(metric.AverageMeter('{}_Black_Acc@1'.format(i), ':6.2f'))

    avg_top1 = metric.AverageMeter('Avg_Acc@1', ':6.2f')
    avg_top5 = metric.AverageMeter('Avg_Acc@5', ':6.2f')
    black_avg_top1 = metric.AverageMeter('Black_Avg_Acc@1', ':6.2f')
    black_avg_top5 = metric.AverageMeter('Black_Avg_Acc@5', ':6.2f')

    fidelity_top1 = metric.AverageMeter('Fidelity@1', ':6.2f')
    fidelity_top5 = metric.AverageMeter('Fidelity@5', ':6.2f')
    progress = metric.ProgressMeter(len(val_loader), batch_time, avg_ce_loss, *top1_all,
                                    avg_top1, avg_top5, fidelity_top1, fidelity_top5, black_avg_top1, black_avg_top5, prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute outputs and losses
            if args.is_amp:
                with amp.autocast():
                    ensemble_output, outputs, ce_loss = model(images,
                                                              target=target,
                                                              mode='val'
                                                              )
            else:
                ensemble_output, outputs, ce_loss = model(images, target=target, mode='val')

            if blackbox is not None:
                b_ensemble_output = blackbox(images)

            # measure accuracy and record loss
            batch_size_now = images.size(0)
            for j in range(args.loop_factor):
                acc1, acc5 = metric.val_accuracy(outputs[j, ...], target, topk=(1, 5))
                top1_all[j].update(acc1[0].item(), batch_size_now)

            # simply average outputs of small networks
            avg_acc1, avg_acc5 = metric.val_accuracy(ensemble_output, target, topk=(1, 5))
            if blackbox is not None:
                fidelity, fidelity5 = metric.fidelity(ensemble_output, b_ensemble_output, topk=(1, 5))
                # blackbox_acc1, blackbox_acc5 = metric.val_accuracy(b_ensemble_output, target, topk=(1, 5))
            else:
                fidelity = torch.tensor([0., 0.])
                fidelity5 = torch.tensor([0., 0.])

            avg_top1.update(avg_acc1[0].item(), batch_size_now)
            avg_top5.update(avg_acc5[0].item(), batch_size_now)
            fidelity_top1.update(fidelity[0].item(), batch_size_now)
            fidelity_top5.update(fidelity5[0].item(), batch_size_now)

            # black_avg_top1.update(blackbox_acc1[0].item(), batch_size_now)
            # black_avg_top5.update(blackbox_acc5[0].item(), batch_size_now)
            avg_ce_loss.update(ce_loss.mean().item(), batch_size_now)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i)

        acc_all = []
        acc_all.append(avg_top1.avg)
        acc_all.append(avg_top5.avg)
        acc_all.append(fidelity_top1.avg)
        acc_all.append(fidelity_top5.avg)
        acc_all.append(black_avg_top1.avg)
        acc_all.append(black_avg_top5.avg)
        acc_info = '* Acc@1 {:.3f} Acc@5 {:.3f} \n Fidelity@1 ' \
                   '{:.3f} Fidelity@5 {:.3f} \n Black_Acc@1 {:.3f} Black_Acc@5 {:.3f}'.format(acc_all[0], acc_all[1], acc_all[2], acc_all[3],
                                                                                              acc_all[4], acc_all[5])
        for j in range(args.loop_factor):
            acc_all.append(top1_all[j].avg)
            acc_info += '\t {}_acc@1 {:.3f}'.format(j, top1_all[j].avg)

        print(acc_info)

    # torch.cuda.empty_cache()
    return acc_all

