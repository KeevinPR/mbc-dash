# -*- coding: utf-8 -*-
"""
Independence tests and mutual information calculations for MBC learning

Based on Borchani et al. algorithms for MB-MBC (HITON-PC/MB with G² tests)
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2
from itertools import combinations
from math import log


def mutual_information(X, Y, base=2):
    """
    Calculate mutual information between two discrete variables
    
    Args:
        X, Y: array-like, discrete variables
        base: logarithm base (default 2 for bits)
    
    Returns:
        float: mutual information I(X;Y)
    """
    # Convert to pandas Series for easier handling
    if not isinstance(X, pd.Series):
        X = pd.Series(X)
    if not isinstance(Y, pd.Series):
        Y = pd.Series(Y)
    
    # Joint distribution
    joint_counts = pd.crosstab(X, Y)
    joint_probs = joint_counts / joint_counts.sum().sum()
    
    # Marginal distributions
    px = joint_probs.sum(axis=1)
    py = joint_probs.sum(axis=0)
    
    # Calculate MI
    mi = 0.0
    for i in joint_probs.index:
        for j in joint_probs.columns:
            if joint_probs.loc[i, j] > 0:
                mi += joint_probs.loc[i, j] * np.log(joint_probs.loc[i, j] / (px[i] * py[j])) / np.log(base)
    
    return mi


def conditional_mutual_information(X, Y, Z, base=2):
    """
    Calculate conditional mutual information I(X;Y|Z)
    
    Args:
        X, Y, Z: array-like, discrete variables
        base: logarithm base
    
    Returns:
        float: conditional mutual information I(X;Y|Z)
    """
    if not isinstance(X, pd.Series):
        X = pd.Series(X)
    if not isinstance(Y, pd.Series):
        Y = pd.Series(Y)
    if not isinstance(Z, pd.Series):
        Z = pd.Series(Z)
    
    # Calculate I(X;Y|Z) = H(X|Z) + H(Y|Z) - H(X,Y|Z) - H(Z)
    # Equivalent to: I(X;Y|Z) = I(X;Y,Z) - I(X;Z)
    
    # Create combined Z variable if Z has multiple columns
    if hasattr(Z, 'values') and len(Z.shape) > 1:
        Z_combined = Z.apply(lambda x: tuple(x), axis=1)
    else:
        Z_combined = Z
    
    cmi = 0.0
    for z_val in Z_combined.unique():
        # Get indices where Z = z_val
        z_mask = (Z_combined == z_val)
        if z_mask.sum() == 0:
            continue
            
        # Conditional probability P(Z=z)
        p_z = z_mask.mean()
        
        # Calculate MI for this conditioning value
        X_z = X[z_mask]
        Y_z = Y[z_mask]
        
        if len(X_z) > 1:  # Need at least 2 samples
            mi_z = mutual_information(X_z, Y_z, base)
            cmi += p_z * mi_z
    
    return cmi


def g_square_test(X, Y, Z=None, alpha=0.05):
    """
    G² (log-likelihood ratio) test for conditional independence
    
    Tests H0: X ⊥ Y | Z vs H1: X ⊥̸ Y | Z
    
    Args:
        X, Y: array-like, variables to test
        Z: array-like or None, conditioning set
        alpha: significance level
    
    Returns:
        dict: {
            'statistic': G² statistic,
            'p_value': p-value,
            'independent': bool, True if independent at level alpha,
            'df': degrees of freedom
        }
    """
    if not isinstance(X, pd.Series):
        X = pd.Series(X)
    if not isinstance(Y, pd.Series):
        Y = pd.Series(Y)
    
    if Z is None:
        # Unconditional test
        observed = pd.crosstab(X, Y)
        
        # Expected frequencies under independence
        row_totals = observed.sum(axis=1)
        col_totals = observed.sum(axis=0)
        n = observed.sum().sum()
        
        expected = np.outer(row_totals, col_totals) / n
        
        # G² statistic
        g2_stat = 0.0
        for i in range(observed.shape[0]):
            for j in range(observed.shape[1]):
                if observed.iloc[i, j] > 0 and expected[i, j] > 0:
                    g2_stat += 2 * observed.iloc[i, j] * log(observed.iloc[i, j] / expected[i, j])
        
        # Degrees of freedom
        df = (len(observed.index) - 1) * (len(observed.columns) - 1)
        
    else:
        # Conditional test
        if not isinstance(Z, pd.DataFrame):
            Z = pd.DataFrame(Z)
        
        g2_stat = 0.0
        df = 0
        
        # Group by conditioning variables
        for z_vals, group_idx in Z.groupby(list(Z.columns)).groups.items():
            if len(group_idx) < 2:  # Need at least 2 samples
                continue
                
            X_z = X.iloc[group_idx]
            Y_z = Y.iloc[group_idx]
            
            # Contingency table for this conditioning value
            observed = pd.crosstab(X_z, Y_z)
            
            if observed.shape[0] <= 1 or observed.shape[1] <= 1:
                continue
            
            # Expected frequencies under conditional independence
            row_totals = observed.sum(axis=1)
            col_totals = observed.sum(axis=0)
            n_z = observed.sum().sum()
            
            expected = np.outer(row_totals, col_totals) / n_z
            
            # Add to G² statistic
            for i in range(observed.shape[0]):
                for j in range(observed.shape[1]):
                    if observed.iloc[i, j] > 0 and expected[i, j] > 0:
                        g2_stat += 2 * observed.iloc[i, j] * log(observed.iloc[i, j] / expected[i, j])
            
            # Add to degrees of freedom
            df += (len(observed.index) - 1) * (len(observed.columns) - 1)
    
    # P-value from chi-square distribution
    if df > 0:
        p_value = 1 - chi2.cdf(g2_stat, df)
    else:
        p_value = 1.0
    
    return {
        'statistic': g2_stat,
        'p_value': p_value,
        'independent': p_value > alpha,
        'df': df
    }


def mi(X, Y):
    """Wrapper for mutual information (compatibility with existing code)"""
    return mutual_information(X, Y)


def cmi(X, Y, Z):
    """Wrapper for conditional mutual information"""
    return conditional_mutual_information(X, Y, Z)


def independence_test(df, X, Y, Z=None, test='g2', alpha=0.05):
    """
    General independence test interface
    
    Args:
        df: pandas DataFrame
        X, Y: int or str, variable names/indices
        Z: list of int/str or None, conditioning set
        test: str, test type ('g2', 'chi2')
        alpha: float, significance level
    
    Returns:
        bool: True if X and Y are independent given Z
    """
    # Extract variables
    if isinstance(X, (int, np.integer)):
        X_data = df.iloc[:, X]
    else:
        X_data = df[X]
        
    if isinstance(Y, (int, np.integer)):
        Y_data = df.iloc[:, Y]
    else:
        Y_data = df[Y]
    
    if Z is not None:
        if isinstance(Z, (list, tuple)):
            Z_data = df.iloc[:, Z] if all(isinstance(z, (int, np.integer)) for z in Z) else df[Z]
        else:
            Z_data = df.iloc[:, Z] if isinstance(Z, (int, np.integer)) else df[Z]
    else:
        Z_data = None
    
    if test == 'g2':
        result = g_square_test(X_data, Y_data, Z_data, alpha)
        return result['independent']
    else:
        raise ValueError(f"Unknown test type: {test}")


def find_markov_blanket_vars(df, target, alpha=0.05, max_cond_size=3):
    """
    Find variables in the Markov blanket of target variable
    Uses G² tests with increasing conditioning set sizes
    
    Args:
        df: pandas DataFrame
        target: int, target variable index
        alpha: significance level
        max_cond_size: maximum conditioning set size
    
    Returns:
        list: variables in Markov blanket
    """
    n_vars = df.shape[1]
    candidates = [i for i in range(n_vars) if i != target]
    mb_vars = []
    
    # Phase 1: Find variables dependent on target
    for var in candidates:
        if not independence_test(df, target, var, None, 'g2', alpha):
            mb_vars.append(var)
    
    # Phase 2: Remove variables that become independent when conditioned
    final_mb = mb_vars.copy()
    
    for var in mb_vars:
        # Try conditioning on subsets of other MB variables
        other_mb = [v for v in mb_vars if v != var]
        
        found_independence = False
        for cond_size in range(min(len(other_mb), max_cond_size) + 1):
            if found_independence:
                break
                
            if cond_size == 0:
                continue  # Already tested unconditional
                
            for cond_set in combinations(other_mb, cond_size):
                if independence_test(df, target, var, list(cond_set), 'g2', alpha):
                    final_mb.remove(var)
                    found_independence = True
                    break
    
    return final_mb
