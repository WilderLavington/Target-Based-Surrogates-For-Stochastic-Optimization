

import argparse

def get_args():
    # grab parse.
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    # optimization args
    parser.add_argument('--algo', type=str, default='SGD', help='SGD,Adam,SGD_FMDOpt')
    parser.add_argument('--loss', type=str, default='CrossEntropyLoss')
    parser.add_argument('--project', type=str, default='TargetBasedSurrogateOptimization')
    parser.add_argument('--entity', type=str, default='wilderlavington')
    parser.add_argument('--log_lr', type=float, default=-4)
    parser.add_argument('--m', type=int, default=5)
    parser.add_argument('--init_step_size', type=float, default=1)
    parser.add_argument('--inner_opt', type=str, default='LSOpt')
    parser.add_argument('--c', type=float, default=0.1)
    parser.add_argument('--beta_update', type=float, default=0.9)
    parser.add_argument('--outer_c', type=float, default=0.1)
    parser.add_argument('--outer_beta_update', type=float, default=0.9)
    parser.add_argument('--expand_coeff', type=float, default=2.0)
    parser.add_argument('--use_optimal_stepsize', type=int, default=1)
    parser.add_argument('--eta_schedule', type=str, default='constant')
    parser.add_argument('--dataset_name', type=str, default='mushrooms')
    parser.add_argument('--file_name', type=str, default='example')
    parser.add_argument('--folder_name', type=str, default='examples/')
    parser.add_argument('--randomize_folder', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fullbatch', type=int, default=0)
    parser.add_argument('--group', type=str, default='main')
    parser.add_argument('--matfac_xdim', type=int, default=6)
    parser.add_argument('--matfac_ydim', type=int, default=10)
    parser.add_argument('--matfac_nsamples', type=int, default=1000)
    parser.add_argument('--matfac_condition_number', type=float, default=1e-10)
    parser.add_argument('--reset_lr_on_step', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--normalize_epochs_lengths', type=int, default=1)
    parser.add_argument('--min_epochs', type=int, default=1)
    parser.add_argument('--min_episodes', type=int, default=1)
    parser.add_argument('--label', type=str, default='ex')
    parser.add_argument('--log_dir', type=str, default='./wandb')
    parser.add_argument('--gulf2_prox_steps', type=int, default=25)
    parser.add_argument('--gulf2_alpha', type=float, default=0.3)
    # parse
    args, knk = parser.parse_known_args()
    print(knk)
    # assert len(knk) == 0
    #
    return args, parser
