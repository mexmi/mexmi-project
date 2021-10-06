
def add_parser_params(parser):
    parser.add_argument('--arch', type=str, default='wide_resnet28_10')
    parser.add_argument('--num_classes', default=10, type=int, metavar='N',
                        help='The number of classes.')

    parser.add_argument('--norm_mode', type=str, default='batch',
                        choices=['batch', 'group', 'layer', 'instance', 'none'],
                        help='The style of the batchnormalization (default: batch)')

    parser.add_argument('--epochs', default=250, type=int, metavar='N',
                        help='number of total epochs to run (default: 300)')

    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'imagenet', 'svhn'],
                        help='dataset name (default: pascal)')
    # split factor
    parser.add_argument('--split_factor', default=1, type=int,
                        help='split one big network into split_factor small networks')

    parser.add_argument('--is_train_sep', default=0, type=int,
                        help='Train small models seperately.')

    parser.add_argument('--output_stride', default=8, type=int,
                        help='output_stride = (resolution of input) / (resolution of output)'
                             '(before global pooling layer)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--is_identical_init', default=0, type=int,
                        help='initialize the small networks identically or not')

    parser.add_argument('--is_ensembled_after_softmax', default=0, type=int,
                        help='whether ensemble the output after softmax')

    parser.add_argument('--is_linear_lr', default=0, type=int,
                        help='using linear scaling lr with batch_size strategy or not')

    # learning rate
    parser.add_argument('--lr_mode', type=str,
                        choices=['cos', 'step', 'poly', 'HTD', 'exponential'],
                        default='cos',
                        help='strategy of the learning rate')

    parser.add_argument('--lr', '--learning_rate', default=0.03, type=float,
                        metavar='LR', help='initial learning rate (default: 0.1)',
                        dest='lr') #0.1

    #optimizer
    parser.add_argument('--optimizer', type=str, default='SGD',
                        choices=['SGD', 'AdamW', 'RMSprop', 'RMSpropTF'],
                        help='The optimizer.')

    parser.add_argument('--lr_milestones', nargs='+', type=int,
                        default=[100, 150],
                        help='epochs at which we take a learning-rate step '
                             '(default: [100, 150])')

    parser.add_argument('--lr_step_multiplier', default=0.1, type=float, metavar='M',
                        help='lr multiplier at lr_milestones (default: 0.1)')

    parser.add_argument('--lr_multiplier', type=float, default=1.0,
                        help='Learning rate multiplier for the unpretrained model.')

    parser.add_argument('--slow_start_lr', type=float, default=5e-3,
                        help='Learning rate employed during slow start.')

    parser.add_argument('--end_lr', type=float, default=1e-4,
                        help='The ending learning rate.')

    parser.add_argument('--slow_start_epochs', type=int, default=10,
                        help='Training model with small learning rate for few epochs.')

    # parameters of the optimizer
    parser.add_argument('--momentum', default=0.5, type=float, metavar='M',
                        help='optimizer momentum (default: 0.9)')

    parser.add_argument('--is_nesterov', default=1, type=int,
                        help='using Nesterov accelerated gradient or not')

    parser.add_argument('--decay_factor', default=0.97, type=float,
                        help='decay factor of exponetital lr')

    parser.add_argument('--decay_epochs', default=0.8, type=float,
                        help='decay epochs of exponetital lr')

    #evaluate
    parser.add_argument('--evaluate', default=False, dest='evaluate', action='store_true',
                        help='evaluate model on validation set')

    parser.add_argument('--is_wd_all', default=0, type=int,
                        help='apply weight to all learnable in the model, otherwise, only weights parameters.')

    # setting about the weight decay
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    # process info
    parser.add_argument('--proc_name', type=str, default='splitnet',
                        help='The name of the process.')

    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')

    parser.add_argument('--is_amp', default=1, type=int,
                        help='Using PyTorch Automatic Mixed Precision (AMP)')

    #distribution
    parser.add_argument('--multiprocessing_distributed', default=False, action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    #print
    parser.add_argument('--print_freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 20)')
    parser.add_argument('--eval_per_epoch', default=1, type=int,
                        help='run evaluation per eval_per_epoch')
    # gradient accumulate
    parser.add_argument('--iters_to_accumulate', default=1, type=int,
                        help="Gradient accumulation adds gradients "
                             "over an effective batch of size batch_per_iter * iters_to_accumulate")
    #save
    parser.add_argument('--max_ckpt_nums', default=5, type=int,
                        help='maximum number of ckpts.')

    # parse
    args = parser.parse_args()

    return args
