import os
import argparse
from easydict import EasyDict as Edict


# It is a much better idea to just register every variant of network instead of
# defining them adhoc. Easier for retrieval of concrete definition.

def add_distill_args(parser_):
    kd = parser_.add_argument_group('distill', 'arguments for knowledge distillation')
    kd.add_argument('--teachers', type=str, default='r100',
                    help='name of teacher model')
    kd.add_argument('--teachers_build', type=str, default=None,
                    help='builder file name for creating the model')
    kd.add_argument('--teachers_dir', type=str, default=None,
                    help='directory to pretrained teacher model')
    kd.add_argument('--d_lr', type=float, default=1e-4,
                    help='learning rate for discriminator in MarginDistillation')
    kd.add_argument('--d_wt', type=float, default=1.0,
                    help='weight for discriminator loss')
    kd.add_argument('--kl_temp', type=float, default=5,
                    help='softmax temperature for KL-Divergence loss')
    kd.add_argument('--ce_wt', default=1.0, type=float,
                    help='weight of cross entropy loss')
    kd.add_argument('--kl_wt', default=0.0, type=float,
                    help='weight of KL divergence loss')
    kd.add_argument('--em_wt', default=1e3, type=float,
                    help='weight of embedding loss')
    kd.add_argument('--ft_wt', default=1e3, type=float,
                    help='weight of feature loss')
    kd.add_argument('--cc_wt', default=1e4, type=float,
                    help='weight of correlation congruence loss')
    kd.add_argument('--feat_list', default='', type=str,
                    help='default list of features to be compared')
    kd.add_argument('--class_per_batch', default=16, type=int,
                    help='number of classes in a single batch')
    kd.add_argument('--ensemble_ratio', default='0.5,0.5', type=str,
                    help='ratio of each teacher model, sum should be 1')

"""
    1. Be cautious, if you set default value has the effect of turning on/off certain functionality
       it might be good to turn off as default. E.g., label smoothing
"""
def parse_args():
    parser = argparse.ArgumentParser(description='face verification training')
    parser.add_argument('--distill', type=str, default=None,
                        help='which knowledge distillation method to use')
    parser.add_argument('--network', '-n', type=str, default=None,
                        help='which network to use')
    parser.add_argument('--dataset', '-d', type=str, default=None,
                        help='which dataset to use')
    parser.add_argument('--dataset_info', type=str, default='multi_5sets.json',
                        help='directory to the file containing data info')
    parser.add_argument('--loss', '-l', type=str, default=None,
                        help='what loss function to use')
    parser.add_argument('--opt', '-o', type=str, default='sgd',
                        help='which optimizer to use, SGD or Adam or other')
    parser.add_argument('--learner', type=str, default='base',
                        help="specify which learner to use")
    args, rest = parser.parse_known_args()
    # generate_config(args.network, args.dataset, args.loss, args.optimizer)
    if args.distill:    # optionally add arguments of knowledge distillation
        add_distill_args(parser)

    # general
    parser.add_argument('--script', type=str, default=None,
                        help='Script file used for spawning training process')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Total batch size for non-distributed training'
                             'Batch size per GPU for distributed training')
    # parser.add_argument('--batch_per_device', type=int, default=128,
    #                     help='Batch size in each context')
    parser.add_argument('--suffix', nargs="+", type=str, default=None,
                        help='items to be included in suffix')
    parser.add_argument('--precision', type=str, default='fp32', choices=['fp32', 'fp16'],
                        help='use Automatic Mixed Precision or not')
    parser.add_argument('--resume_ckpt', type=str, default=None,
                        help='path to checkpoint to resume from')
    parser.add_argument('--print_freq', type=int, default=20,
                        help='steps between logging the training statistics')
    parser.add_argument('--ddp', type=str, default=None, choices=['apex', 'torch', None],
                        help='distributed faceid')
    parser.add_argument('--cudnn_benchmark', type=str, default='Y', choices=['Y', 'N'],
                        help='whether to use cudnn benchmark')
    parser.add_argument('--use_val', type=str, default='N', choices=['Y', 'N'],
                        help='whether to do validation at all')
    parser.add_argument('--val_freq', type=int, default=10000,
                        help='steps between validation')
    parser.add_argument('--data_root', type=str, default=None,
                        help='parent directory of all datasets')
    parser.add_argument('--mnt', type=str, default=None,
                        help='mount point for the disk to work with')
    parser.add_argument('--workspace', type=str, default=None,
                        help='directory for saving all the jobs')
    parser.add_argument('--model_path', type=str, default=None,
                        help='path to model for evaluation, only used in evaluation')
    parser.add_argument('--do-lr-search', action='store_true',
                        help='do learning rate search')
    parser.add_argument('--set-to-none', action='store_true',
                        help='whether to set gradient to none instead of zero when doing zero_grad()')

# =============================== Net configuration =============================
    parser.add_argument('--act_func', type=str, default='prelu',
                        help='network activation config')
    parser.add_argument('--use_ibn', type=str, default='N',
                        choices=['Y', 'N'], help='whether or not use ibn')
    parser.add_argument('--stoch_depth', type=float, default=0.9,
                        help='stochastic depth')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='model directory to load from')
    parser.add_argument('--remove_fc', action='store_true',
                        help='whether or not remove final fc layer for fine tuning'
                             'with other dataset')
    parser.add_argument('--strict_load', action='store_false',
                        help='loading mode: strict or not')
    parser.add_argument('--bn_mom', type=float, default=0.1,
                        help='')
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='')
    parser.add_argument('--use_partial_fc', type=str, default='Y',
                        help='Whether to use partial FC')

# ================================= Optimizer  ==================================
#     parser.add_argument('--warmup_epoch', type=int, default=0,
#                         help='number of warmup epochs')
#     parser.add_argument('--warmup_lr', type=float, default=0.001,
#                         help='learning rate for warmup')
#     parser.add_argument('--init_lr', type=float, default=config.init_lr,
#                         help='start learning rate')
    parser.add_argument('--lr_steps', type=str, default='200000,400000,600000,800000,1000000',
                        help='learning rate reduction steps')
    parser.add_argument('--beta_freeze', type=int, default=0,
                        help='')
    parser.add_argument('--fc7_wd_mult', type=float, default=1.0,
                        help='weight decay mult for fc7')

    parser.add_argument('--fc-lr-mult', type=float, default=1.0,
                        help="Scale learning rate for fc layers")

    # Optimizer parameters
    # parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
    #                     help='Optimizer (default: "sgd"')
    parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: None, use opt default)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='Optimizer momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=2e-5,
                        help='weight decay (default: 2e-5)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--clip-mode', type=str, default='norm',
                        help='Gradient clipping mode. One of ("norm", "value", "agc")')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "step"')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                        help='learning rate (default: 0.05)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                        help='learning rate cycle len multiplier (default: 1.0)')
    parser.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT',
                        help='amount to decay each learning rate cycle (default: 0.5)')
    parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                        help='learning rate cycle limit, cycles enabled if > 1')
    parser.add_argument('--lr-k-decay', type=float, default=1.0,
                        help='learning rate k-decay for cosine/poly (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                        help='warmup learning rate (default: 0.0001)')
    parser.add_argument('--min-lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 300)')
    parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                        help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
    parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    # parser.add_argument('--decay-epochs', type=float, default=100, metavar='N',
    #                     help='epoch interval to decay LR')
    parser.add_argument('--decay-epochs', type=float, default=None, nargs='+', metavar='N',
                        help='epoch interval to decay LR')  # LIUJIN: changed for multistep
    parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    parser.add_argument('--t-in-epochs', default=False, action='store_true',
                        help='to use time in epochs')

# =================================== Loss  =====================================
    # Loss Function
    # parser.add_argument('--margin_type', type=str, default=config.margin_type,
    #                     help='ArcFace, MultiMargin')
    parser.add_argument('--margin_s', type=float, default=64.0,
                        help='scale for feature')
    parser.add_argument('--margin_m_sphere', type=float, default=1.1,
                        help='margin for sphere loss')
    parser.add_argument('--margin_m_cos', type=float, default=0.2,
                        help='margin for cos loss')
    parser.add_argument('--margin_m_arc', type=float, default=0.3,
                        help='margin for arc loss')
    parser.add_argument('--easy_margin', type=int, default=0.0,
                        help='')
    parser.add_argument('--margin_smoothing', type=float, default=0.1,
                        help='param for label smoothing')
    # Regularization
    parser.add_argument('--ring_loss', type=int, default=-1,
                        help='RingLoss type: -1:wo RingLoss(default)'
                             '0:L2(same as paper), 1:auto, 2:L1')
    parser.add_argument('--ring_loss_weight', type=float, default=0.01,
                        help='param for label smoothing')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='param for label smoothing, suggested value: 0.1')

# =============================== Data & Preprocess =============================
    # Augmentation
    parser.add_argument('--scale_pixel', type=str, default='Y', choices=['Y', 'N'],
                        help='whether or not to scale pixel value from 0~255 to -1 to 1')
    parser.add_argument('--auto_aug', type=int, default=-1,
                        help='auto_aug: -1:default aug only, 0:UniAug')
    parser.add_argument('--cutoff', type=int, default=3,
                        help='cut off aug')
    parser.add_argument('--rand_mirror', type=int, default=1,
                        help='whether to do random mirror in training')
    parser.add_argument('--aug_color', type=float, default=0.0,
                        help='set value for color augmentation')
    parser.add_argument('--rand_crop', type=int, default=1,
                        help='if do random crop in training')
    parser.add_argument('--crop_ratio', type=float, default=0.9,
                        help='the ratio of crop')
    parser.add_argument('--mixup_type', type=int, default=0,
                        help='mix type: -1:no mix, 0:img wo/label, 1:img w/label'
                             '2:feat wo/label, 3:feat w/label, 4:cutmix wo/label'
                             '5:cutmix w/label')
    parser.add_argument('--mixup_prob', type=float, default=0.25,
                        help='mix probability')
    parser.add_argument('--mixup_alpha', type=float, default=0.1,
                        help='mix alpha')

    # Image Normalization Setting
    parser.add_argument('--part_no', type=int, default=0,
                        help='set normalize method')
    parser.add_argument('--norm_margin_ratio', type=float, default=0.1,
                        help='margin ratio added to training images')
    parser.add_argument('--norm_img_size', type=int, default=112,
                        help='normalized image size WITHOUT MARGIN')
    parser.add_argument('--size_in_arcface_norm', type=int, default=112,
                        help='cropped (or marginalized) size in ArcFace normalized image')

    # ISDA Setting
    parser.add_argument('--isda_batch1', type=int, default=1,
                        help='number of batch for memory reduction in ISDA')
    parser.add_argument('--isda_batch2', type=int, default=1,
                        help='number of batch for memory reduction in ISDA')
    parser.add_argument('--isda_ratio', type=float, default=0.0,
                        help='hyper-parameter lambda for ISDA loss')

    args = parser.parse_args()

    return args
