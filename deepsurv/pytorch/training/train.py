"""
Training function for Deep Survival Toy model
"""
import numpy as np
import torch
from deepsurv.pytorch.training.utils import epoch_func, get_optimizer, eval_func




def fit_toy_model(n_epochs, model, train_loader, valid_loader,
                 patience=2000, patience_increase=2,
                  optimizer_type='SGD',
                  lr=2.2890139399051963e-05,
                  weight_decay=1.999):
    """

    :param n_epochs:
    :param model:
    :param train_loader:
    :param valid_loader:
    :param test_loader:
    :param patience:
    :param patience_increase:
    :param optimizer_type:
    :param lr:
    :param weight_decay:
    :return:
    """
    best_validation_loss = np.inf
    avg_training_loss = []
    avg_validation_loss = []
    training_ci = []
    validation_ci = []

    optimizer = get_optimizer(model, optimizer_type, lr, weight_decay)

    for epoch in range(n_epochs):

        # Perform training epoch
        train_loss, ci_train = epoch_func(model, optimizer, train_loader)
        avg_training_loss.append(train_loss)
        training_ci.append(ci_train)

        valid_loss, ci_valid = eval_func(model, valid_loader)
        avg_validation_loss.append(valid_loss)
        validation_ci.append(ci_valid)

        if valid_loss < np.min(avg_validation_loss):
            if validation_loss < best_validation_loss * improvement_threshold:
                patience = max(patience, epoch * patience_increase)

            best_validation_loss = valid_loss

        if patience <= epoch:
            break

    # Test model
    ci_test = model.get_concordance_index(
        torch.Tensor(valid_loader.dataset.e).cuda(),
        torch.Tensor(valid_loader.dataset.t).cuda(),
        torch.Tensor(valid_loader.dataset.x).cuda()
    )

    out = {'avg_training_loss': avg_training_loss,
           'avg_validation_loss': avg_validation_loss,
           'training_ci': training_ci,
           'validation_ci': validation_ci,
           'ci_test': ci_test,
           'stop_epoch': epoch}

    return out

def batch_size_optim_objective(batch_size, dset_path):
    """

    :param batch_size:
    :return:
    """
    # Define the list of hyper-param to optimise
    space = [Integer(4, 50),  # Number of hidden dimensions
             Integer(1, 3),  # Number of layers
             Integer(0, 1),  # SGD==0 ADAM==1
             Real(3.9e-4, 0.154), # Learning Rate
             Real(0, 9.9)]  # Regularisation strength

    datasets = get

    loaders = get_loaders(datasets, b_size=batch_size)
    inp_shape = loaders['train'].dataset.x.shape[1]

    def objective(params):

        hidden_dim = params[0]
        num_layers = params[1]
        optimizer = 'SGD' if params[2]==0 else 'ADAM'
        lr = params[3]
        wd = params[4]

        model = ToyModel(inp_shape, hidden_dim, num_layers)
        model.cuda()
        out = fit_toy_model(n_epochs, model, train_loader, valid_loader,
                      patience=2000, patience_increase=2,
                      optimizer_type=optimizer,
                      lr=lr,
                      weight_decay=wd)
        return -out['ci_test']

    res_min = forest_minimize(objective, space, n_calls=50, random_state=0)

    best_param = rest_min.x

    return res_min, best_param



