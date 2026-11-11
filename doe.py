"""This module contains functions for design of experiments analysis for HW 4 in AE 498 Data-Driven Modeling and Optimization."""

import numpy as np
import matplotlib.pyplot as plt

def _check_responses(responses, n_replicates):
    """Check that the data have four treatment rows and the stated replicates."""
    if (
        not isinstance(n_replicates, (int, np.integer))
        or isinstance(n_replicates, (bool, np.bool_))
        or n_replicates < 1
    ):
        raise ValueError("n_replicates must be a positive integer.")

    responses = np.asarray(responses, dtype=float)
    if responses.shape != (4, n_replicates):
        raise ValueError("responses must have shape (4, n_replicates).")
    if not np.isfinite(responses).all():
        raise ValueError("responses must contain only finite values.")
    return responses

def effects(responses, n_replicates):
    """Function to calculate main and interaction effects for a 2^2 factorial experiment.

    Parameters
    ----------
    responses : array_like
        Input response data (4 x n_replications dimensions), where each row is an observed response for a condition (ordered in the standard order) and each column is a replicate.
        For example, responses[0,1] = response from replicate 1 for condition (1), responses[1,1] = response from replicate 1 for condition a.
    n_replicates : int
        Number of replications in the experiment data.

    Returns
    -------
    effects : array_like
        Calculated effects (3 x 1 dimensions) for A, B, and AB.

    Notes
    -----
    Standard order of experiment conditions (order of rows in responses) = (1), a, b, ab
    """

    responses = _check_responses(responses, n_replicates)
    total_1, total_a, total_b, total_ab = responses.sum(axis=1)
    
    # Compare the positive and negative groups for each contrast.
    effect_a = (-total_1 + total_a - total_b + total_ab) / (2 * n_replicates)
    effect_b = (-total_1 - total_a + total_b + total_ab) / (2 * n_replicates)
    effect_ab = (total_1 - total_a - total_b + total_ab) / (2 * n_replicates)
    
    return np.array([effect_a, effect_b, effect_ab])

def plot_effects(responses, n_replicates):
    """Function to plot effects for a 2^2 factorial experiment.

    Parameters
    ----------
    responses : array_like
        Input response data (4 x n_replications dimensions), where each row is an observed response for a condition (ordered in the standard order) and each column is a replicate.
        For example, responses[0,1] = response from replicate 1 for condition (1), responses[1,1] = response from replicate 1 for condition a.
    n_replicates : int
        Number of replications in the experiment data.

    Returns
    -------
    out : list
        List of artists added.

    Notes
    -----
    This function structure is based on matplotlib's suggested signature function for returning a plot:
    https://matplotlib.org/3.5.0/tutorials/introductory/usage.html#the-object-oriented-interface-and-the-pyplot-interface
    """

    responses = _check_responses(responses, n_replicates)
    mean_1, mean_a, mean_b, mean_ab = responses.mean(axis=1)

    fig, ax = plt.subplots()
    line_high_b = ax.plot(
        [-1, 1], [mean_b, mean_ab], "o--", label="B+", markerfacecolor="white"
    )
    line_low_b = ax.plot(
        [-1, 1], [mean_1, mean_a], "o--", label="B-", markerfacecolor="white"
    )
    ax.set_xticks([-1, 1], ["Low, -", "High, +"])
    ax.set_xlabel("Factor A")
    ax.set_ylabel("Mean response")
    ax.legend(title="Factor B")
    fig.tight_layout()

    return line_high_b + line_low_b


def plot_tornado(responses, n_replicates):
    """Function to make a tornado plot for a 2^2 factorial experiment.

    Parameters
    ----------
    responses : array_like
        Input response data (4 x n_replications dimensions), where each row is an observed response for a condition (ordered in the standard order) and each column is a replicate.
        For example, responses[0,1] = response from replicate 1 for condition (1), responses[1,1] = response from replicate 1 for condition a.
    n_replicates : int
        Number of replications in the experiment data.

    Returns
    -------
    out : list
        List of artists added.

    Notes
    -----
    This function structure is based on matplotlib's suggested signature function for returning a plot:
    https://matplotlib.org/3.5.0/tutorials/introductory/usage.html#the-object-oriented-interface-and-the-pyplot-interface
    """

    effect_values = effects(responses, n_replicates)
    labels = np.array(["A", "B", "AB"])
    order = np.argsort(-np.abs(effect_values), kind="stable")

    fig, ax = plt.subplots()
    bars = ax.barh(labels[order], effect_values[order], color="slateblue")
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Average effect")
    ax.set_ylabel("Factor or interaction")
    fig.tight_layout()

    return list(bars)