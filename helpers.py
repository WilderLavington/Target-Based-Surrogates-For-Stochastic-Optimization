
import torch
import random
import string

def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

def update_stoch_lr(optimizer, steps, initial_lr):
    """Decreases the learning rate as 1 / sqrt(t) """
    for param_group in optimizer.param_groups:
        param_group['lr'] = initial_lr / torch.sqrt(steps)
        # print('update_stoch_lr', param_group['lr'])
    return optimizer

def update_exp_lr(optimizer, steps, total_steps, initial_lr):
    """Decreases the learning rate as  """
    for param_group in optimizer.param_groups:
        param_group['lr'] = initial_lr * torch.tensor((1/ total_steps)**(steps/ total_steps))
        # print('update_exp_lr', param_group['lr'])
    return optimizer

# helpers
def compute_grad_norm(grad_list):
    grad_norm = 0.
    grad_list = [g.reshape(-1) for g in grad_list]
    flat_grad = torch.cat(grad_list, dim=0).reshape(-1)
    grad_norm = torch.sqrt(flat_grad.pow(2).sum())
    return grad_norm
def get_grad_list(params):
    g_list = []
    for p in params:
        grad = p.grad
        if grad is None:
            grad = torch.tensor( 0., device='cuda')
        g_list += [grad]
    return g_list
def get_grad_norm(params):
    glist = get_grad_list(params)
    return compute_grad_norm(glist)

def replace_params(model, params):
    zipped = zip([m for m in params], [m for m in model.parameters()])
    for p_new, p_current in zipped:
        p_current.data = p_new
    return model
def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
# line-search conditions
def check_armijo_conditions(step_size, loss, grad_norm,
                  loss_next, c, beta_b):
    found = 0
    # computing the new break condition
    break_condition = loss_next - \
        (loss - (step_size) * c * grad_norm**2)
    if (break_condition <= 0):
        found = 1
    else:
        # decrease the step-size by a multiplicative factor
        step_size = step_size * beta_b
    return found, step_size
