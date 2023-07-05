import os
import logging
from datetime import datetime
from os.path import basename, join, dirname, exists, abspath
from shutil import copytree, rmtree, ignore_patterns, copy2

from util.others import del_empty_job, set_logger
from util import dir_tool
from config import parse_args
from learner import BaseLearner, MultidataLearner

def get_learner(training=True):
    args = parse_args()
    time_now = datetime.now().strftime('%Y%m%d_%H')
    assert exists(args.workspace), 'Please create proper workspace!'

    if training:
        # -----------------------Set job_space & log file----------------------------
        # Naming a job by <network-loss-data-time>
        data_name = args.dataset if args.dataset else basename(args.data_info).strip('.json')
        job_name = '-'.join([args.network, args.loss, data_name])
        if args.resume_ckpt:
            job_name = 'resumed-' + job_name
        args.job_space = join(args.workspace, '-'.join([job_name, time_now]))
        log_file = '-'.join(['1-log', basename(args.job_space)]) + '.log'

        # -------------------  Remove empty job_space & archive code ----------------
        if args.ddp and not int(os.environ['OMPI_COMM_WORLD_RANK'])==0:
            # Some actions only need to be done by once for distributed training
            pass
        else:
            del_empty_job(args.workspace)   # delete empty previous job_space
            dir_tool.mk_dir(args.job_space)
            copy2(args.script, join(args.job_space, '0-{}'.format(basename(args.script))))
            set_logger(join(args.job_space, log_file))
            # Archive source code used for this training
            copytree(dirname(abspath(__file__)), join(args.job_space, 'code'),
                     ignore=ignore_patterns('*.pyc', 'tmp*'))
    else:
        args.job_space = dirname(args.model_path)

    # -------------------------------- Get Learner ------------------------------
    if args.learner == 'base':
        learner = BaseLearner(args, training=training)
    elif args.learner == 'multidata':
        learner = MultidataLearner(args, training=training)
    else:
        raise ValueError('Learner currently not defined!')
    print("Using learner: {}".format(type(learner).__name__))

    return learner

def main():
    learner = get_learner()
    learner.train()


if __name__ == '__main__':
    main()
