"""
Pytorch implementation of the model
First implements functions that are general to surival model
"""
import torch
from torch import nn
from lifelines.utils import concordance_index
import numpy as np


def negative_log_likelihood_loss(risk, E):
    """
    Return the negative average log-likelihood of the prediction
    of this model under a given target distribution.

    :parameter risk: output of the NN for inputs
    :parameter E: binary tensor providing censor information

    :returns: partial cox negative log likelihood
    """
    hazard_ratio = torch.exp(risk)
    log_risk = torch.log(torch.cumsum(torch.squeeze(hazard_ratio), dim=0))
    uncensored_likelihood = risk - log_risk
    censored_likelihood = uncensored_likelihood * E
    num_observed_events = torch.sum(E.data)
    neg_likelihood = -torch.sum(censored_likelihood) / num_observed_events
    return neg_likelihood

def get_concordance_index(model_class, e, t, x):
    """
    Calculates the concordance index (C-index) between two series
    of event times. The first is the real survival times from
    the experimental data, and the other is the predicted survival
    times from a model of some kind.

    The concordance index is a value between 0 and 1 where,
    0.5 is the expected result from random predictions,
    1.0 is perfect concordance and,
    0.0 is perfect anti-concordance (multiply predictions with -1 to get 1.0)

    Score is usually 0.6-0.7 for survival models.

    See:
    Harrell FE, Lee KL, Mark DB. Multivariable prognostic models: issues in
    developing models, evaluating assumptions and adequacy, and measuring and
    reducing errors. Statistics in Medicine 1996;15(4):361-87.

    :param model_class: a pytorch defined model
    :param e: binary tensor providing censor information
    :param t:
    :param x:
    """
    assert hasattr(model_class, 'partial_hazard'), 'Your model does not have partial hazard'
    partial_hazards = model_class.partial_hazard(x).data.cpu()
    return concordance_index(t, partial_hazards, e)

def partial_hazard(model_class, x):
    """
    Computes partial hazards
    :param model_class: a pytorch defined model
    """
    assert hasattr(model_class, 'partial_hazard'), 'Your model does not have risk function'
    risk = model_class.risk(x)
    return -torch.exp(risk)


class ToyModel(nn.Module):
    """
    Toy Model with variable number of layers.
    """

    def __init__(self, inp_size, hid_size, num_layers,
                 include_bn=True):
        """

        :param num_layers: Number of layers
        :param include_bn: if batch norm should be included
        """
        super(ToyModel, self).__init__()

        encoder_l = np.linspace(inp_size, hid_size, num_layers + 1).astype(int).tolist()

        encoder_l = [(encoder_l[i], encoder_l[i + 1]) for i in range(len(encoder_l) - 1)]

        self.encoder = self.return_mlp(num_layers, encoder_l, include_bn)

    @staticmethod
    def return_mlp(num_layers, num_hidden, include_bn):
        """
        Function to return an mlp
        """
        # Creates layers in an order Linear, Tanh, Linear, Tanh,.. and so on.. using list comprehension
        if include_bn == True:
            layers = [[nn.Linear(num_hidden[i][0], num_hidden[i][1]), nn.BatchNorm1d(num_hidden[i][1]),
                       nn.ReLU()] for i in range(num_layers - 1)]
        else:
            layers = [[nn.Linear(num_hidden[i][0], num_hidden[i][1]),
                       nn.ReLU()] for i in range(num_layers - 1)]

        layers = [layer for sublist in layers for layer in sublist]

        # Append last layer whihc will be just Linear in this case
        layers.append(nn.Linear(num_hidden[num_layers - 1][0], 1))
        #         layers.append(nn.Sigmoid()) # According to deep surv last layer is just linear

        # Convert into model
        model = nn.Sequential(*layers)

        return model

    def forward(self, x):
        """
        Forward pass through the survival network
        """
        return self.risk(x)

    def risk(self, x):
        """
        Returns the output of network which is an
        observation's predicted risk.

        :parameter x: model inputs

        :returns risk: model prediction for risk
        """
        risk = self.encoder(x)
        return risk

    @staticmethod
    def negative_log_likelihood_loss(risk, E):
        """
        Return the negative average log-likelihood of the prediction
        of this model under a given target distribution.

        :parameter risk: output of the NN for inputs
        :parameter E: binary tensor providing censor information

        :returns: partial cox negative log likelihood
        """
        return negative_log_likelihood_loss(risk, E)

    def partial_hazard(self, x):
        """
        Computes partial hazards
        """
        return partial_hazard(self, x)

    def get_concordance_index(self, e, t, x):
        """
        Calculates the concordance index (C-index) between two series
        of event times. The first is the real survival times from
        the experimental data, and the other is the predicted survival
        times from a model of some kind.

        The concordance index is a value between 0 and 1 where,
        0.5 is the expected result from random predictions,
        1.0 is perfect concordance and,
        0.0 is perfect anti-concordance (multiply predictions with -1 to get 1.0)

        Score is usually 0.6-0.7 for survival models.

        See:
        Harrell FE, Lee KL, Mark DB. Multivariable prognostic models: issues in
        developing models, evaluating assumptions and adequacy, and measuring and
        reducing errors. Statistics in Medicine 1996;15(4):361-87.
        """
        return get_concordance_index(self, e, t, x)