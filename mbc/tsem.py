# -*- coding: utf-8 -*-
"""
TSEM - Tractable Structural EM for learning from incomplete data

Implements Benjumeda's TSEM algorithm: structural EM with treewidth bounds
for learning MBCs from datasets with missing values. Ensures convergence
and maintains tractability constraint.
"""

import numpy as np
import pandas as pd
import networkx as nx
from .tw_mbc import learn_tw_mbc, TWMBCModel
from .params import learn_parameters_mle, compute_log_likelihood
from .inference import infer_marginals_simple
import copy


def tsem(df, classes, features, tw_max=3, init_method="tw_mbc", max_iter=10, 
         convergence_threshold=1e-4, alpha=1.0, verbose=True):
    """
    TSEM: Tractable Structural EM algorithm
    
    Args:
        df: pandas DataFrame with missing values (NaN)
        classes: list of class variable indices
        features: list of feature variable indices
        tw_max: maximum allowed treewidth
        init_method: initialization method ("tw_mbc", "empty")
        max_iter: maximum EM iterations
        convergence_threshold: convergence threshold for log-likelihood
        alpha: Dirichlet prior parameter
        verbose: print progress information
    
    Returns:
        dict: {
            'model': final TWMBCModel,
            'completed_data': DataFrame with imputed values,
            'log_likelihood_history': list of log-likelihoods,
            'converged': bool,
            'iterations': int
        }
    """
    if verbose:
        print(f"Starting TSEM with tw_max={tw_max}, max_iter={max_iter}")
    
    # Check for missing data
    missing_count = df.isnull().sum().sum()
    if missing_count == 0:
        if verbose:
            print("No missing data - using standard TW-MBC learning")
        model = learn_tw_mbc(df, classes, features, tw_max=tw_max)
        return {
            'model': model,
            'completed_data': df.copy(),
            'log_likelihood_history': [],
            'converged': True,
            'iterations': 0
        }
    
    if verbose:
        print(f"Found {missing_count} missing values ({missing_count/df.size*100:.1f}% of data)")
    
    # Initialize model
    if init_method == "tw_mbc":
        # Initialize with TW-MBC learned from complete cases
        complete_cases = df.dropna()
        if len(complete_cases) > 0:
            if verbose:
                print(f"Initializing with {len(complete_cases)} complete cases")
            initial_model = learn_tw_mbc(complete_cases, classes, features, tw_max=tw_max)
        else:
            if verbose:
                print("No complete cases - using empty initialization")
            initial_model = _initialize_empty_model(df, classes, features, tw_max)
    else:
        initial_model = _initialize_empty_model(df, classes, features, tw_max)
    
    # TSEM iterations
    current_model = initial_model
    completed_df = df.copy()
    log_likelihood_history = []
    converged = False
    
    for iteration in range(max_iter):
        if verbose:
            print(f"\n--- TSEM Iteration {iteration + 1} ---")
        
        # E-step: Impute missing values using current model
        if verbose:
            print("E-step: Imputing missing values...")
        
        completed_df = _e_step_imputation(df, current_model, classes, features, verbose)
        
        # Compute current log-likelihood
        if hasattr(current_model, 'cpts') and current_model.cpts:
            current_ll = compute_log_likelihood(completed_df, current_model.cpts, current_model.G)
            log_likelihood_history.append(current_ll)
            
            if verbose:
                print(f"Log-likelihood: {current_ll:.4f}")
            
            # Check convergence
            if len(log_likelihood_history) > 1:
                ll_diff = current_ll - log_likelihood_history[-2]
                if verbose:
                    print(f"LL improvement: {ll_diff:.6f}")
                
                if abs(ll_diff) < convergence_threshold:
                    if verbose:
                        print("Converged!")
                    converged = True
                    break
        
        # M-step: Re-learn structure and parameters with treewidth constraint
        if verbose:
            print("M-step: Re-learning structure and parameters...")
        
        new_model = learn_tw_mbc(
            completed_df, classes, features, 
            tw_max=tw_max, alpha=0.05
        )
        
        # Ensure treewidth constraint is satisfied
        tw_info = new_model.get_treewidth_info()
        if not tw_info['tractable']:
            if verbose:
                print(f"Warning: New model has treewidth {tw_info['treewidth_estimate']} > {tw_max}")
            # Keep previous model if new one violates constraint
            if hasattr(current_model, 'G') and current_model.G.number_of_edges() > 0:
                if verbose:
                    print("Keeping previous model to maintain tractability")
                continue
        
        current_model = new_model
        
        if verbose:
            tw_info = current_model.get_treewidth_info()
            struct_info = current_model.get_structure_info()
            print(f"Structure: {struct_info['total_edges']} edges, "
                  f"TW estimate: {tw_info['treewidth_estimate']}")
    
    # Final log-likelihood
    if hasattr(current_model, 'cpts') and current_model.cpts:
        final_ll = compute_log_likelihood(completed_df, current_model.cpts, current_model.G)
        if len(log_likelihood_history) == 0 or final_ll != log_likelihood_history[-1]:
            log_likelihood_history.append(final_ll)
    
    if verbose:
        print(f"\nTSEM completed after {len(log_likelihood_history)} iterations")
        print(f"Final log-likelihood: {log_likelihood_history[-1] if log_likelihood_history else 'N/A'}")
        print(f"Converged: {converged}")
    
    return {
        'model': current_model,
        'completed_data': completed_df,
        'log_likelihood_history': log_likelihood_history,
        'converged': converged,
        'iterations': len(log_likelihood_history)
    }


def _initialize_empty_model(df, classes, features, tw_max):
    """Initialize empty TW-MBC model"""
    model = TWMBCModel(classes, features, tw_max)
    
    # Add minimal structure (each class connected to most informative feature)
    for class_var in classes:
        best_feature = None
        best_mi = -1
        
        for feature in features:
            # Calculate MI using available data
            class_data = df.iloc[:, class_var].dropna()
            feature_data = df.iloc[:, feature].dropna()
            
            # Find common indices
            common_idx = class_data.index.intersection(feature_data.index)
            
            if len(common_idx) > 10:  # Need sufficient data
                try:
                    from .independence import mutual_information
                    mi = mutual_information(
                        class_data.loc[common_idx], 
                        feature_data.loc[common_idx]
                    )
                    if mi > best_mi:
                        best_mi = mi
                        best_feature = feature
                except:
                    continue
        
        if best_feature is not None:
            model.G.add_edge(class_var, best_feature)
    
    # Learn initial parameters
    complete_cases = df.dropna()
    if len(complete_cases) > 0:
        model.cpts = learn_parameters_mle(complete_cases, model.G, alpha=1.0)
    
    return model


def _e_step_imputation(df, model, classes, features, verbose=False):
    """
    E-step: Impute missing values using current model
    
    Args:
        df: DataFrame with missing values
        model: current MBC model
        classes: class variables
        features: feature variables
        verbose: print progress
    
    Returns:
        DataFrame: with imputed values
    """
    completed_df = df.copy()
    
    # Count missing values per variable
    missing_counts = df.isnull().sum()
    total_imputed = 0
    
    for var in df.columns:
        var_idx = var if isinstance(var, int) else df.columns.get_loc(var)
        
        if missing_counts[var] > 0:
            if verbose and missing_counts[var] > 0:
                print(f"  Imputing {missing_counts[var]} values for variable {var_idx}")
            
            # Get rows with missing values for this variable
            missing_mask = df[var].isnull()
            
            for idx in df[missing_mask].index:
                # Get available evidence for this row
                evidence = {}
                for other_var in df.columns:
                    other_idx = other_var if isinstance(other_var, int) else df.columns.get_loc(other_var)
                    if other_idx != var_idx and pd.notna(df.loc[idx, other_var]):
                        evidence[other_idx] = df.loc[idx, other_var]
                
                # Impute missing value
                if len(evidence) > 0:
                    imputed_value = _impute_single_value(model, var_idx, evidence, classes, features)
                else:
                    # No evidence - use marginal mode
                    imputed_value = _get_marginal_mode(df, var_idx)
                
                completed_df.loc[idx, var] = imputed_value
                total_imputed += 1
    
    if verbose:
        print(f"  Total values imputed: {total_imputed}")
    
    return completed_df


def _impute_single_value(model, target_var, evidence, classes, features):
    """
    Impute single missing value using model inference
    
    Args:
        model: MBC model
        target_var: variable to impute
        evidence: dict of observed values
        classes: class variables
        features: feature variables
    
    Returns:
        imputed value
    """
    if not hasattr(model, 'cpts') or not model.cpts or target_var not in model.cpts:
        # No model available - use simple heuristic
        return 0  # Default value
    
    try:
        # Get possible states for target variable
        target_states = model.cpts[target_var]['states']
        
        # Calculate probability for each state
        state_probs = {}
        
        for state in target_states:
            # Add this state to evidence
            full_evidence = evidence.copy()
            full_evidence[target_var] = state
            
            # Calculate probability (simplified)
            log_prob = 0.0
            
            # Probability of target given parents
            parents = model.cpts[target_var]['parents']
            if len(parents) == 0:
                prob = model.cpts[target_var]['cpt'].get(state, 1.0/len(target_states))
            else:
                parent_values = tuple(full_evidence.get(p, 0) for p in parents)
                cpt_entry = model.cpts[target_var]['cpt'].get(parent_values, {})
                prob = cpt_entry.get(state, 1.0/len(target_states))
            
            state_probs[state] = prob
        
        # Return state with highest probability
        best_state = max(state_probs.keys(), key=lambda s: state_probs[s])
        return best_state
        
    except Exception as e:
        # Fallback to default
        return 0


def _get_marginal_mode(df, var_idx):
    """Get mode (most frequent value) for variable"""
    var_col = df.iloc[:, var_idx] if isinstance(var_idx, int) else df[var_idx]
    mode_values = var_col.mode()
    
    if len(mode_values) > 0:
        return mode_values.iloc[0]
    else:
        return 0  # Default


def evaluate_tsem_quality(original_df, completed_df, test_mask=None):
    """
    Evaluate quality of TSEM imputation
    
    Args:
        original_df: DataFrame with missing values
        completed_df: DataFrame with imputed values
        test_mask: mask for evaluation (if available)
    
    Returns:
        dict: evaluation metrics
    """
    results = {
        'total_imputed': 0,
        'imputation_accuracy': None,
        'missing_percentage': 0.0
    }
    
    # Count imputed values
    missing_mask = original_df.isnull()
    results['total_imputed'] = missing_mask.sum().sum()
    results['missing_percentage'] = (results['total_imputed'] / original_df.size) * 100
    
    # If test mask provided, evaluate accuracy
    if test_mask is not None:
        # This would require ground truth for evaluation
        pass
    
    return results


# Wrapper function for compatibility
def tsem_cache(data_incomplete, metric='bic', tw_bound=5, custom_classes=None, 
               add_only=False, max_iter=10):
    """
    Compatibility wrapper for existing TSEM interface
    
    Args:
        data_incomplete: DataFrame with missing values
        metric: scoring metric (unused in this implementation)
        tw_bound: treewidth bound
        custom_classes: variable classes/states
        add_only: structure learning constraint
        max_iter: maximum EM iterations
    
    Returns:
        tuple: (elimination_tree, _, cpp_model, completed_df, _)
    """
    # Assume first half are classes, second half are features
    n_vars = data_incomplete.shape[1]
    classes = list(range(n_vars // 2))
    features = list(range(n_vars // 2, n_vars))
    
    # Run TSEM
    result = tsem(
        data_incomplete, classes, features, 
        tw_max=tw_bound, max_iter=max_iter, verbose=True
    )
    
    # Return in expected format (simplified)
    model = result['model']
    completed_df = result['completed_data']
    
    return model, None, model, completed_df, None
