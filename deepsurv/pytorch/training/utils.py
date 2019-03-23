"""
Stochastic weight averaging training utilities
"""
import torch
import numpy as np
from torch.autograd import Variable
from torch import optim


def inverse_time_decay_LR(optimizer, epoch, lr_decay):
    """
    Inverse time decay to the learning rate
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] / (1 + epoch*lr_decay)

def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha

def sort_batch(e, t, x):
    """
    Sorts the batch for loss computation
    """
    t, indices = torch.sort(t, descending=True)
    e = e[indices]
    x = x[indices]
    return e, t, x

def inverse_time_decay_LR(optimizer, epoch, lr_decay):
    """
    Inverse time decay to the learning rate
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] / (1 + epoch * lr_decay)

def epoch_func(model, optimizer, train_loader):
    """
    :param model: pytorch model
    :param optimizer: pytorch optimizer
    :param train_loader: data loader
    :return: train_loss, ci_train
    """
    training_loss = []
    model.train()
    #         inverse_time_decay_LR(optimizer, epoch, lr_decay)
    for e, t, x in train_loader:
        e, t, x = sort_batch(e, t, x)
        e, t, x = Variable(e.cuda()), Variable(t.cuda()), Variable(x.cuda())

        risk = model(x)

        loss = model.negative_log_likelihood_loss(risk, e)
        optimizer.zero_grad()
        loss.backward()
        training_loss.append(loss.item())
        optimizer.step()

    # Compute average loss per epoch and append
    train_loss = np.average(training_loss)

    # Get training concordance index
    ci_train = model.get_concordance_index(
        torch.Tensor(train_loader.dataset.e).cuda(),
        torch.Tensor(train_loader.dataset.t).cuda(),
        torch.Tensor(train_loader.dataset.x).cuda()
    )
    return train_loss, ci_train

def eval_func(model, valid_loader):
    """

    :param model:
    :param valid_loader:
    :return: eval_loss, ci_valid
    """
    validation_loss = []
    model.eval()
    with torch.no_grad():
        for e, t, x in valid_loader:
            e, t, x = sort_batch(e, t, x)
            e, t, x = Variable(e.cuda()), Variable(t.cuda()), Variable(x.cuda())

            risk = model(x)
            loss = model.negative_log_likelihood_loss(risk, e)
            validation_loss.append(loss.item())

    valid_loss = np.average(validation_loss)
    ci_valid = model.get_concordance_index(
        torch.Tensor(valid_loader.dataset.e).cuda(),
        torch.Tensor(valid_loader.dataset.t).cuda(),
        torch.Tensor(valid_loader.dataset.x).cuda()
    )
    return valid_loss, ci_valid

def get_optimizer(model, optimizer_type, lr, weight_decay):
    """

    :param model: pytorch model
    :param optimizer_type: SGD or ADAM
    :param lr:
    :param weight_decay:
    :return:
    """
    if optimizer_type == 'SGD':
        return optim.SGD(model.parameters(), lr=lr,
                          weight_decay=weight_decay)
    if optimizer_type == 'ADAM':
        return optim.Adam(model.parameters(), lr=lr,
                          weight_decay=weight_decay)
    else:
        raise Exception('Wrong optimizer type {}'.format(optimizer_type))