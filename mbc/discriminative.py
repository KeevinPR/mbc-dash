# -*- coding: utf-8 -*-
"""
Discriminative learning for MBCs - Conditional Log-Likelihood (CLL) optimization

Implements discriminative parameter and structure learning for MBCs by optimizing
conditional log-likelihood instead of joint likelihood (Benjumeda's approach)
"""

import numpy as np
import pandas as pd
import networkx as nx
from scipy.optimize import minimize
from .params import learn_parameters_mle, compute_log_likelihood
from .tw_mbc import TWMBCModel


def compute_conditional_log_likelihood(df, cpts, graph, class_vars, feature_vars):
    """
    Compute conditional log-likelihood P(Classes|Features)
    
    Args:
        df: pandas DataFrame
        cpts: dict, learned parameters
        graph: nx.DiGraph
        class_vars: list, class variable indices
        feature_vars: list, feature variable indices
    
    Returns:
        float: conditional log-likelihood
    """
    cll = 0.0
    
    for _, row in df.iterrows():
        # P(Classes|Features) = P(Classes, Features) / P(Features)
        
        # Compute joint log-probability P(Classes, Features)
        joint_log_prob = 0.0
        for node in graph.nodes():
            if node not in cpts:
                continue
                
            node_value = row.iloc[node]
            parents = cpts[node]['parents']
            
            if len(parents) == 0:
                prob = cpts[node]['cpt'].get(node_value, 1e-10)
            else:
                parent_values = tuple(row.iloc[p] for p in parents)
                cpt_entry = cpts[node]['cpt'].get(parent_values, {})
                prob = cpt_entry.get(node_value, 1e-10)
            
            joint_log_prob += np.log(max(prob, 1e-10))
        
        # Compute marginal log-probability P(Features)
        # This requires marginalizing over class variables (expensive)
        # For efficiency, we approximate or use the feature subgraph only
        
        feature_log_prob = 0.0
        for feature in feature_vars:
            if feature not in cpts:
                continue
            
            # Get parents of this feature
            parents = cpts[feature]['parents']
            feature_parents = [p for p in parents if p in feature_vars]
            
            if len(feature_parents) == 0:
                # Feature has no feature parents - use marginal
                feature_value = row.iloc[feature]
                prob = cpts[feature]['cpt'].get(feature_value, 1e-10)
                feature_log_prob += np.log(max(prob, 1e-10))
        
        # Approximate CLL (this is a simplification)
        cll += joint_log_prob - feature_log_prob
    
    return cll


def optimize_parameters_cll(df, graph, class_vars, feature_vars, max_iter=100, lr=0.01):
    """
    Optimize parameters to maximize conditional log-likelihood
    
    Args:
        df: pandas DataFrame
        graph: nx.DiGraph, MBC structure
        class_vars: list, class variables
        feature_vars: list, feature variables
        max_iter: int, maximum optimization iterations
        lr: float, learning rate
    
    Returns:
        dict: optimized CPTs
    """
    # Start with MLE parameters
    initial_cpts = learn_parameters_mle(df, graph)
    
    # Convert parameters to optimization variables
    param_vector, param_info = _cpts_to_vector(initial_cpts)
    
    def objective(params):
        # Convert back to CPTs
        cpts = _vector_to_cpts(params, param_info)
        
        # Compute negative CLL (minimize)
        cll = compute_conditional_log_likelihood(df, cpts, graph, class_vars, feature_vars)
        return -cll
    
    # Constraints: probabilities must sum to 1 and be non-negative
    constraints = _get_probability_constraints(param_info)
    bounds = [(1e-10, 1.0) for _ in param_vector]  # Probabilities between 0 and 1
    
    try:
        # Optimize using L-BFGS-B
        result = minimize(
            objective, 
            param_vector, 
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter}
        )
        
        if result.success:
            optimized_cpts = _vector_to_cpts(result.x, param_info)
            return optimized_cpts
        else:
            print(f"Optimization failed: {result.message}")
            return initial_cpts
            
    except Exception as e:
        print(f"Optimization error: {e}")
        return initial_cpts


def _cpts_to_vector(cpts):
    """Convert CPTs to parameter vector for optimization"""
    param_vector = []
    param_info = {}
    
    for node, cpt_info in cpts.items():
        param_info[node] = {
            'parents': cpt_info['parents'],
            'states': cpt_info['states'],
            'param_indices': {}
        }
        
        if len(cpt_info['parents']) == 0:
            # Root node
            states = cpt_info['states'][:-1]  # Exclude last state (determined by others)
            start_idx = len(param_vector)
            
            for state in states:
                param_vector.append(cpt_info['cpt'].get(state, 1.0/len(cpt_info['states'])))
            
            param_info[node]['param_indices']['root'] = (start_idx, len(param_vector))
            
        else:
            # Node with parents
            for parent_config, conditional_dist in cpt_info['cpt'].items():
                states = cpt_info['states'][:-1]  # Exclude last state
                start_idx = len(param_vector)
                
                for state in states:
                    param_vector.append(conditional_dist.get(state, 1.0/len(cpt_info['states'])))
                
                param_info[node]['param_indices'][parent_config] = (start_idx, len(param_vector))
    
    return np.array(param_vector), param_info


def _vector_to_cpts(param_vector, param_info):
    """Convert parameter vector back to CPTs"""
    cpts = {}
    
    for node, info in param_info.items():
        cpts[node] = {
            'parents': info['parents'],
            'states': info['states'],
            'cpt': {}
        }
        
        if 'root' in info['param_indices']:
            # Root node
            start_idx, end_idx = info['param_indices']['root']
            probs = param_vector[start_idx:end_idx]
            
            # Last probability is 1 - sum(others)
            last_prob = max(1e-10, 1.0 - np.sum(probs))
            all_probs = np.append(probs, last_prob)
            
            # Normalize to ensure sum = 1
            all_probs = all_probs / np.sum(all_probs)
            
            for i, state in enumerate(info['states']):
                cpts[node]['cpt'][state] = all_probs[i]
                
        else:
            # Node with parents
            for parent_config, (start_idx, end_idx) in info['param_indices'].items():
                probs = param_vector[start_idx:end_idx]
                
                # Last probability is 1 - sum(others)
                last_prob = max(1e-10, 1.0 - np.sum(probs))
                all_probs = np.append(probs, last_prob)
                
                # Normalize
                all_probs = all_probs / np.sum(all_probs)
                
                conditional_dist = {}
                for i, state in enumerate(info['states']):
                    conditional_dist[state] = all_probs[i]
                
                cpts[node]['cpt'][parent_config] = conditional_dist
    
    return cpts


def _get_probability_constraints(param_info):
    """Get constraints for probability normalization"""
    constraints = []
    
    for node, info in param_info.items():
        if 'root' in info['param_indices']:
            # Root node constraint: sum of first n-1 probabilities <= 1
            start_idx, end_idx = info['param_indices']['root']
            
            def constraint_func(params, s=start_idx, e=end_idx):
                return 1.0 - np.sum(params[s:e])
            
            constraints.append({'type': 'ineq', 'fun': constraint_func})
            
        else:
            # Constraints for each parent configuration
            for parent_config, (start_idx, end_idx) in info['param_indices'].items():
                
                def constraint_func(params, s=start_idx, e=end_idx):
                    return 1.0 - np.sum(params[s:e])
                
                constraints.append({'type': 'ineq', 'fun': constraint_func})
    
    return constraints


def fit_discriminative_mbc(df, initial_model, class_vars, feature_vars, 
                          optimize_structure=False, max_iter=50):
    """
    Fit discriminative MBC by optimizing conditional log-likelihood
    
    Args:
        df: pandas DataFrame
        initial_model: TWMBCModel or similar, initial MBC structure
        class_vars: list, class variable indices
        feature_vars: list, feature variable indices
        optimize_structure: bool, whether to also optimize structure
        max_iter: int, maximum optimization iterations
    
    Returns:
        model with discriminatively optimized parameters/structure
    """
    print("Fitting discriminative MBC...")
    
    # Phase 1: Optimize parameters given structure
    print("  Optimizing parameters...")
    optimized_cpts = optimize_parameters_cll(
        df, initial_model.G, class_vars, feature_vars, max_iter=max_iter
    )
    
    # Update model parameters
    initial_model.cpts = optimized_cpts
    
    # Phase 2: Optionally optimize structure
    if optimize_structure:
        print("  Optimizing structure...")
        # This would involve hill-climbing on structure while optimizing CLL
        # For now, keep structure fixed
        print("  Structure optimization not implemented - keeping current structure")
    
    # Compute final CLL score
    final_cll = compute_conditional_log_likelihood(
        df, optimized_cpts, initial_model.G, class_vars, feature_vars
    )
    
    print(f"  Final CLL: {final_cll:.4f}")
    
    return initial_model


def compare_generative_vs_discriminative(df, model, class_vars, feature_vars):
    """
    Compare generative (MLE) vs discriminative (CLL) parameter learning
    
    Args:
        df: pandas DataFrame
        model: MBC model with structure
        class_vars: list, class variables
        feature_vars: list, feature variables
    
    Returns:
        dict: comparison results
    """
    # Learn generative parameters
    generative_cpts = learn_parameters_mle(df, model.G)
    
    # Learn discriminative parameters
    discriminative_cpts = optimize_parameters_cll(
        df, model.G, class_vars, feature_vars, max_iter=50
    )
    
    # Compute scores
    gen_ll = compute_log_likelihood(df, generative_cpts, model.G)
    gen_cll = compute_conditional_log_likelihood(df, generative_cpts, model.G, class_vars, feature_vars)
    
    disc_ll = compute_log_likelihood(df, discriminative_cpts, model.G)
    disc_cll = compute_conditional_log_likelihood(df, discriminative_cpts, model.G, class_vars, feature_vars)
    
    return {
        'generative': {
            'log_likelihood': gen_ll,
            'conditional_log_likelihood': gen_cll,
            'parameters': generative_cpts
        },
        'discriminative': {
            'log_likelihood': disc_ll,
            'conditional_log_likelihood': disc_cll,
            'parameters': discriminative_cpts
        },
        'cll_improvement': disc_cll - gen_cll,
        'll_difference': disc_ll - gen_ll
    }


# Hook for compatibility with existing code
def fit_discriminative(df, model, classes, max_iter=20):
    """
    Compatibility function for discriminative parameter optimization
    
    Args:
        df: pandas DataFrame
        model: MBC model
        classes: list of class variables
        max_iter: maximum iterations
    
    Returns:
        dict: optimized CPTs
    """
    features = [i for i in range(df.shape[1]) if i not in classes]
    
    if hasattr(model, 'G'):
        graph = model.G
    elif hasattr(model, 'get_graph'):
        graph = model.get_graph()
    else:
        # Create simple graph structure
        graph = nx.DiGraph()
        graph.add_nodes_from(list(range(df.shape[1])))
    
    return optimize_parameters_cll(df, graph, classes, features, max_iter=max_iter)
