"""
Script to run sensitivity analysis given the batch size
for different datasets
"""
import os
import datetime
from deepsurv.pytorch.datasets import *
from deepsurv.pytorch.deep_surv import *
from deepsurv.pytorch.training.train import *
from deepsurv.pytorch.training.utils import *

from skopt import forest_minimize
from skopt.space import Real, Integer
import pickle

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

    datasets = get_datasets(dset_path)
    loaders = get_loaders(datasets, b_size=batch_size)
    train_loader = loaders['train']
    valid_loader = loaders['test']
    inp_shape = loaders['train'].dataset.x.shape[1]

    def objective(params):

        hidden_dim = params[0]
        num_layers = params[1]
        optimizer = 'SGD' if params[2]==0 else 'ADAM'
        lr = params[3]
        wd = params[4]

        model = ToyModel(inp_shape, hidden_dim, num_layers)
        model.cuda()
        out = fit_toy_model(100, model, train_loader, valid_loader,
                      patience=2000, patience_increase=2,
                      optimizer_type=optimizer,
                      lr=lr,
                      weight_decay=wd)
        return -out['ci_test']

    res_min = forest_minimize(objective, space, n_calls=15, random_state=0)

    best_param = res_min.x

    return res_min, best_param

def main():
    """
    Main script to run sensitivity analysis
    :return:
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    path_data = '/home/jevjev/Dropbox/Projects/DeepSurv/experiments/data/'
    data_name = 'whas'
    print(data_name)
    data_file_name = 'whas_train_test.h5'
    data_path = os.path.join(path_data, data_name, data_file_name)

    results_path = '/home/jevjev/Dropbox/Projects/DeepSurv/sensitivity_results'
    time_stamp = datetime.datetime.today().strftime('%H-%M-%S-%Y-%m-%d').replace('-', '_')
    log_path = os.path.join(results_path, data_name, time_stamp)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    b_sizes = [3, 4, 6, 8, 10, 12, 16, 20, 40, 60, 80, 100, 300, 600]
    results_min = []
    best_parameters = []

    for b_size in b_sizes:
        print(b_size)
        res_min, best_param = batch_size_optim_objective(b_size, data_path)
        results_min.append(res_min.fun)
        best_parameters.append(best_param)

        with open(os.path.join(log_path, '{}results.pickle'.format(b_size)), 'wb') as handle:
            pickle.dump([b_sizes, results_min, best_parameters], handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
