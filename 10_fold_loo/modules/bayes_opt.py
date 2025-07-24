# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 12:12:01 2025

@author: spect
"""

# modules/bayes_opt.py

import numpy as np
from skopt import gp_minimize
from skopt.utils import use_named_args

def optimize_xgb(objective_fn, search_space, n_calls=50, random_state=42):
    """
    Runs a Gaussian-process–based Bayesian search (skopt.gp_minimize)
    over the given search_space to minimize `objective_fn`.
    
    Parameters
    ----------
    objective_fn : callable
        fn(**params) → validation loss (lower is better)
    search_space : list of skopt.space.Dimension
    n_calls      : int, number of trials
    """
    @use_named_args(search_space)
    def _wrapped_obj(**params):
        return objective_fn(**params)

    res = gp_minimize(
        func      = _wrapped_obj,
        dimensions= search_space,
        n_calls   = n_calls,
        random_state=random_state,
    )
    return res.x, res.fun  # best_params, best_loss
