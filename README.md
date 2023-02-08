# Target Based Surrogates For Stochastic Optimization
The following is codebase allows for the reproduction of results following the paper "Target Based Surrogates For Stochastic Optimization". For any questions or concerns, please email ``jola2372 at cs.ubc.ca``. In order to execute the experiments discussed in the paper, please install the requirements file attached (using python 3.8), and execute commands listed below. The experiments are generated using weights and biases sweeps, which can be viewed in a separate web client.

## ICML Experiments -- Mujoco
In order to generate the desired experiment sweeps for Mujoco, execute the following commands:
```
# baselines
wandb sweep ./configs/icml/mujoco/baselines.yaml
# this will produce something like the following:
# wandb: Creating sweep from: ./configs/icml/mujoco/baselines.yaml
# wandb: Created sweep with ID: <sweep-id>
# wandb: View sweep at: https://wandb.ai/<user-id>/<project>/sweeps/<sweep-id>
# wandb: Run sweep agent with: wandb agent <user-id>/<project>/<sweep-id>

# target based surrogate optimizers
wandb sweep ./configs/icml/mujoco/funcopt.yaml

# this will produce something like the following:
# wandb: Creating sweep from: ./configs/icml/mujoco/baselines.yaml
# wandb: Created sweep with ID: <sweep-id>
# wandb: View sweep at: https://wandb.ai/<user-id>/<project>/sweeps/<sweep-id>
# wandb: Run sweep agent with: wandb agent <user-id>/<project>/<sweep-id>

```

## ICML Experiments -- SVMLib
In order to generate the desired experiment sweeps for SVMLib, execute the following commands:
```
# target based surrogate optimizers SLS (mini-batch)
wandb sweep ./configs/icml/svmlib/SLS_FMDOpt/gridsearch/constant/minibatch/funcopt.yaml

# this will produce something like the following:
# wandb: Creating sweep from: ./configs/icml/svmlib/SLS_FMDOpt/gridsearch/constant/minibatch/funcopt.yaml
# wandb: Created sweep with ID: <sweep-id>
# wandb: View sweep at: https://wandb.ai/<user-id>/<project>/sweeps/<sweep-id>
# wandb: Run sweep agent with: wandb agent <user-id>/<project>/<sweep-id>

# target based surrogate optimizers SGD (mini-batch)
wandb sweep ./configs/icml/svmlib/SGD_FMDOpt/gridsearch/constant/minibatch/funcopt.yaml

# this will produce something like the following:
# wandb: Creating sweep from: ./configs/icml/svmlib/SGD_FMDOpt/gridsearch/constant/minibatch/funcopt.yaml
# wandb: Created sweep with ID: <sweep-id>
# wandb: View sweep at: https://wandb.ai/<user-id>/<project>/sweeps/<sweep-id>
# wandb: Run sweep agent with: wandb agent <user-id>/<project>/<sweep-id>

```
The experiments in the appendix are also included in the configs files and you need only change the configuration folder path you are interested in to test them out.
