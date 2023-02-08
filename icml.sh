
#===================================================================================
# svmlib
#===================================================================================
# SGD - constant
python offline_dir_sweep.py --yaml_file=./configs/aistats/svmlib/SGD/gridsearch/constant/fullbatch/baselines.yaml --job_name=fig_1a --machine=narval --account=rrg-kevinlb
python offline_dir_sweep.py --yaml_file=./configs/aistats/svmlib/SGD/gridsearch/constant/minibatch/baselines.yaml --job_name=fig_1b --machine=narval --account=rrg-kevinlb
# SGD - exponential
python offline_dir_sweep.py --yaml_file=./configs/aistats/svmlib/SGD/gridsearch/exponential/fullbatch/baselines.yaml --job_name=fig_1c --machine=narval --account=rrg-kevinlb
python offline_dir_sweep.py --yaml_file=./configs/aistats/svmlib/SGD/gridsearch/exponential/minibatch/baselines.yaml --job_name=fig_1d --machine=narval --account=rrg-kevinlb
# SGD - stochastic
python offline_dir_sweep.py --yaml_file=./configs/aistats/svmlib/SGD/gridsearch/stochastic/fullbatch/baselines.yaml --job_name=fig_1e --machine=narval --account=rrg-kevinlb
python offline_dir_sweep.py --yaml_file=./configs/aistats/svmlib/SGD/gridsearch/stochastic/minibatch/baselines.yaml --job_name=fig_1f --machine=narval --account=rrg-kevinlb

# SSO - constant
python offline_dir_sweep.py --yaml_file=./configs/aistats/svmlib/SGD_FMDOpt/gridsearch/constant/fullbatch/funcopt.yaml --job_name=fig_2a --machine=cedar --account=def-fwood
python offline_dir_sweep.py --yaml_file=./configs/aistats/svmlib/SGD_FMDOpt/gridsearch/constant/minibatch/funcopt.yaml --job_name=fig_2b --machine=cedar --account=def-fwood
# SSO - exponential
python offline_dir_sweep.py --yaml_file=./configs/aistats/svmlib/SGD_FMDOpt/gridsearch/exponential/fullbatch/funcopt.yaml --job_name=fig_2c --machine=cedar --account=def-fwood
python offline_dir_sweep.py --yaml_file=./configs/aistats/svmlib/SGD_FMDOpt/gridsearch/exponential/minibatch/funcopt.yaml --job_name=fig_2d --machine=cedar --account=def-schmidtm
# SSO - stochastic
python offline_dir_sweep.py --yaml_file=./configs/aistats/svmlib/SGD_FMDOpt/gridsearch/stochastic/fullbatch/funcopt.yaml --job_name=fig_2e --machine=cedar --account=def-schmidtm
python offline_dir_sweep.py --yaml_file=./configs/aistats/svmlib/SGD_FMDOpt/gridsearch/stochastic/minibatch/funcopt.yaml --job_name=fig_2f --machine=cedar --account=def-schmidtm

# SLS - constant
python offline_dir_sweep.py --yaml_file=./configs/aistats/svmlib/SLS/gridsearch/constant/fullbatch/baselines.yaml --job_name=fig_3a --machine=narval --account=rrg-kevinlb
python offline_dir_sweep.py --yaml_file=./configs/aistats/svmlib/SLS/gridsearch/constant/minibatch/baselines.yaml --job_name=fig_3b --machine=narval --account=rrg-kevinlb
# SLS - exponential
python offline_dir_sweep.py --yaml_file=./configs/aistats/svmlib/SLS/gridsearch/exponential/fullbatch/baselines.yaml --job_name=fig_3c --machine=narval --account=rrg-fwood
python offline_dir_sweep.py --yaml_file=./configs/aistats/svmlib/SLS/gridsearch/exponential/minibatch/baselines.yaml --job_name=fig_3d --machine=narval --account=rrg-fwood
# SLS - stochastic
python offline_dir_sweep.py --yaml_file=./configs/aistats/svmlib/SLS/gridsearch/stochastic/fullbatch/baselines.yaml --job_name=fig_3e --machine=narval --account=rrg-fwood
python offline_dir_sweep.py --yaml_file=./configs/aistats/svmlib/SLS/gridsearch/stochastic/minibatch/baselines.yaml --job_name=fig_3f --machine=narval --account=rrg-kevinlb

# SLS-SSO - constant
python offline_dir_sweep.py --yaml_file=./configs/aistats/svmlib/SLS_FMDOpt/gridsearch/constant/fullbatch/funcopt.yaml --job_name=fig_4a --machine=narval --account=rrg-kevinlb
python offline_dir_sweep.py --yaml_file=./configs/aistats/svmlib/SLS_FMDOpt/gridsearch/constant/minibatch/funcopt.yaml --job_name=fig_4b --machine=narval --account=rrg-fwood
# SLS-SSO - exponential
python offline_dir_sweep.py --yaml_file=./configs/aistats/svmlib/SLS_FMDOpt/gridsearch/exponential/fullbatch/funcopt.yaml --job_name=fig_4c --machine=narval --account=rrg-fwood
python offline_dir_sweep.py --yaml_file=./configs/aistats/svmlib/SLS_FMDOpt/gridsearch/exponential/minibatch/funcopt.yaml --job_name=fig_4d --machine=narval --account=rrg-fwood
# SLS-SSO - stochastic
python offline_dir_sweep.py --yaml_file=./configs/aistats/svmlib/SLS_FMDOpt/gridsearch/stochastic/fullbatch/funcopt.yaml --job_name=fig_4e --machine=narval --account=rrg-kevinlb
python offline_dir_sweep.py --yaml_file=./configs/aistats/svmlib/SLS_FMDOpt/gridsearch/stochastic/minibatch/funcopt.yaml --job_name=fig_4f --machine=narval --account=rrg-kevinlb

# Sadagrad - constant
python offline_dir_sweep.py --yaml_file=./configs/aistats/svmlib/Sadagrad/gridsearch/constant/fullbatch/baselines.yaml --job_name=fig_5a --machine=narval --account=rrg-kevinlb
python offline_dir_sweep.py --yaml_file=./configs/aistats/svmlib/Sadagrad/gridsearch/constant/minibatch/baselines.yaml --job_name=fig_5b --machine=narval --account=rrg-kevinlb

# Sadagrad-SSO - constant
python offline_dir_sweep.py --yaml_file=./configs/aistats/svmlib/Ada_FMDOpt/gridsearch/constant/fullbatch/funcopt.yaml --job_name=fig_6a --machine=narval --account=rrg-fwood
python offline_dir_sweep.py --yaml_file=./configs/aistats/svmlib/Ada_FMDOpt/gridsearch/constant/minibatch/funcopt.yaml --job_name=fig_6b --machine=narval --account=rrg-fwood

# Adagrad - constant
python offline_dir_sweep.py --yaml_file=./configs/aistats/svmlib/Adagrad/gridsearch/constant/fullbatch/baselines.yaml --job_name=fig_7a --machine=narval --account=rrg-fwood
python offline_dir_sweep.py --yaml_file=./configs/aistats/svmlib/Adagrad/gridsearch/constant/minibatch/baselines.yaml --job_name=fig_7b --machine=narval --account=rrg-fwood

# Adagrad-SSO - constant
python offline_dir_sweep.py --yaml_file=./configs/aistats/svmlib/Diag_Ada_FMDOpt/gridsearch/constant/fullbatch/funcopt.yaml --job_name=fig_8a --machine=narval --account=rrg-kevinlb
python offline_dir_sweep.py --yaml_file=./configs/aistats/svmlib/Diag_Ada_FMDOpt/gridsearch/constant/minibatch/funcopt.yaml --job_name=fig_8b --machine=narval --account=rrg-kevinlb

# ON-SSO - constant
python offline_dir_sweep.py --yaml_file=./configs/aistats/svmlib/Online_Newton_FMDOpt/gridsearch/constant/fullbatch/funcopt.yaml --job_name=fig_9a --machine=narval --account=rrg-fwood
python offline_dir_sweep.py --yaml_file=./configs/aistats/svmlib/Online_Newton_FMDOpt/gridsearch/constant/minibatch/funcopt.yaml --job_name=fig_9b --machine=narval --account=rrg-fwood
python offline_dir_sweep.py --yaml_file=./configs/aistats/svmlib/Online_Newton_FMDOpt/gridsearch/stochastic/fullbatch/funcopt.yaml --job_name=fig_9c --machine=narval --account=rrg-fwood
python offline_dir_sweep.py --yaml_file=./configs/aistats/svmlib/Online_Newton_FMDOpt/gridsearch/stochastic/minibatch/funcopt.yaml --job_name=fig_9d --machine=narval --account=rrg-fwood
python offline_dir_sweep.py --yaml_file=./configs/aistats/svmlib/Online_Newton_FMDOpt/gridsearch/exponential/fullbatch/funcopt.yaml --job_name=fig_9e
python offline_dir_sweep.py --yaml_file=./configs/aistats/svmlib/Online_Newton_FMDOpt/gridsearch/exponential/minibatch/funcopt.yaml --job_name=fig_9f

# adam - constant
python offline_dir_sweep.py --yaml_file=./configs/aistats/svmlib/Adam/gridsearch/constant/fullbatch/baselines.yaml --job_name=fig_10a  --machine=narval --account=rrg-kevinlb
python offline_dir_sweep.py --yaml_file=./configs/aistats/svmlib/Adam/gridsearch/constant/minibatch/baselines.yaml --job_name=fig_10b  --machine=narval --account=rrg-kevinlb

# gulf2 - constant
python offline_dir_sweep.py --yaml_file=./configs/aistats/svmlib/GULF2/gridsearch/constant/fullbatch/baselines.yaml --job_name=fig_10c
python offline_dir_sweep.py --yaml_file=./configs/aistats/svmlib/GULF2/gridsearch/constant/minibatch/baselines.yaml --job_name=fig_10d

#===================================================================================
# matrix factorization
#===================================================================================

# MFAC-gridsearch
python offline_dir_sweep.py --yaml_file=./configs/aistats/mfac/gridsearch/minibatch/funcopt.yaml --job_name=fig_11a
python offline_dir_sweep.py --yaml_file=./configs/aistats/mfac/gridsearch/minibatch/baselines.yaml --job_name=fig_11b
python offline_dir_sweep.py --yaml_file=./configs/aistats/mfac/gridsearch/fullbatch/funcopt.yaml --job_name=fig_11c
python offline_dir_sweep.py --yaml_file=./configs/aistats/mfac/gridsearch/fullbatch/baselines.yaml --job_name=fig_11d

#===================================================================================
# Atari
#===================================================================================

# Atari-gridsearch
python offline_dir_sweep.py --yaml_file=./configs/aistats/atari/baselines.yaml --job_name=fig_12a --account=plai  --machine=borg --max_per=3
python offline_dir_sweep.py --yaml_file=./configs/aistats/atari/funcopt.yaml --job_name=fig_12b  --account=plai --machine=borg --max_per=3
python offline_dir_sweep.py --yaml_file=./configs/aistats/atari/baselines.yaml --job_name=fig_12a --account=plai  --machine=borg --max_per=3
python offline_dir_sweep.py --yaml_file=./configs/aistats/atari/funcopt.yaml --job_name=fig_12b  --account=plai --machine=borg --max_per=3


#===================================================================================
# Mujoco
#===================================================================================

# mujoco-gridsearch
python offline_dir_sweep.py --yaml_file=./configs/aistats/mujoco/funcopt.yaml --job_name=fig_13a \
  --directory=/ubc/cs/research/plai-scratch/wlaving/FunctionalStochasticOptimization --account=plai --machine=borg
python offline_dir_sweep.py --yaml_file=./configs/aistats/mujoco/baselines.yaml --job_name=fig_13b \
  --directory=/ubc/cs/research/plai-scratch/wlaving/FunctionalStochasticOptimization --account=plai --machine=borg

#===================================================================================
# MNIST
#===================================================================================

# mnist-gridsearch
python offline_dir_sweep.py --yaml_file=./configs/aistats/mnist/gridsearch/minibatch/funcopt.yaml --job_name=fig_14a --machine=borg --account=plai --max_per=24
python offline_dir_sweep.py --yaml_file=./configs/aistats/mnist/gridsearch/minibatch/baselines.yaml --job_name=fig_14b --machine=borg --account=plai --max_per=24
python offline_dir_sweep.py --yaml_file=./configs/aistats/mnist/gridsearch/fullbatch/funcopt.yaml --job_name=fig_14c --machine=borg --account=plai --max_per=24
python offline_dir_sweep.py --yaml_file=./configs/aistats/mnist/gridsearch/fullbatch/baselines.yaml --job_name=fig_14d --machine=borg --account=plai --max_per=24
