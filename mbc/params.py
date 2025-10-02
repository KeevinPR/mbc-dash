# -*- coding: utf-8 -*-
"""
Parameter learning for MBCs - MLE and Bayesian estimation

Implements Maximum Likelihood Estimation (MLE) and Bayesian parameter estimation
for Conditional Probability Tables (CPTs) in MBC structures
"""

import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict


def learn_parameters_mle(df, graph, alpha=0.0):
    """
    Learn Maximum Likelihood Estimation (MLE) parameters for MBC
    
    Args:
        df: pandas DataFrame with data
        graph: nx.DiGraph, learned MBC structure
        alpha: Dirichlet prior parameter (0 for pure MLE, >0 for MAP)
    
    Returns:
        dict: CPTs for each node {node: {'parents': [...], 'cpt': {...}, 'states': [...]}}
    """
    cpts = {}
    
    for node in graph.nodes():
        parents = list(graph.predecessors(node))
        node_data = df.iloc[:, node]
        
        # Get unique states for this variable
        states = sorted(node_data.unique())
        
        cpts[node] = {
            'parents': parents,
            'states': states,
            'cpt': {}
        }
        
        if len(parents) == 0:
            # Root node - marginal distribution
            counts = node_data.value_counts()
            total = len(node_data) + alpha * len(states)  # Add Dirichlet prior
            
            for state in states:
                cpts[node]['cpt'][state] = (counts.get(state, 0) + alpha) / total
                
        else:
            # Node with parents - conditional distribution
            parent_data = df.iloc[:, parents]
            
            # Get all observed parent configurations
            if len(parents) == 1:
                parent_configs = [(config,) for config in parent_data.unique()]
            else:
                parent_configs = [tuple(row) for row in parent_data.drop_duplicates().values]
            
            # Learn CPT for each parent configuration
            for parent_config in parent_configs:
                # Filter data for this parent configuration
                if len(parents) == 1:
                    mask = (parent_data.iloc[:, 0] == parent_config[0])
                else:
                    mask = (parent_data == list(parent_config)).all(axis=1)
                
                filtered_data = node_data[mask]
                
                if len(filtered_data) > 0:
                    counts = filtered_data.value_counts()
                    total = len(filtered_data) + alpha * len(states)
                    
                    cpt_entry = {}
                    for state in states:
                        cpt_entry[state] = (counts.get(state, 0) + alpha) / total
                    
                    cpts[node]['cpt'][parent_config] = cpt_entry
                else:
                    # No data for this configuration - use prior
                    uniform_prob = 1.0 / len(states)
                    cpts[node]['cpt'][parent_config] = {state: uniform_prob for state in states}
    
    return cpts


def learn_parameters_bayesian(df, graph, alpha=1.0):
    """
    Learn Bayesian parameters with Dirichlet prior
    
    Args:
        df: pandas DataFrame
        graph: nx.DiGraph
        alpha: Dirichlet hyperparameter
    
    Returns:
        dict: CPTs with Bayesian estimates
    """
    return learn_parameters_mle(df, graph, alpha=alpha)


def compute_log_likelihood(df, cpts, graph):
    """
    Compute log-likelihood of data given parameters
    
    Args:
        df: pandas DataFrame
        cpts: dict, learned parameters
        graph: nx.DiGraph
    
    Returns:
        float: log-likelihood
    """
    log_likelihood = 0.0
    
    for _, row in df.iterrows():
        for node in graph.nodes():
            if node not in cpts:
                continue
                
            node_value = row.iloc[node]
            parents = cpts[node]['parents']
            
            if len(parents) == 0:
                # Root node
                prob = cpts[node]['cpt'].get(node_value, 1e-10)
            else:
                # Node with parents
                parent_values = tuple(row.iloc[p] for p in parents)
                cpt_entry = cpts[node]['cpt'].get(parent_values, {})
                prob = cpt_entry.get(node_value, 1e-10)
            
            log_likelihood += np.log(max(prob, 1e-10))
    
    return log_likelihood


def compute_bic_score(df, cpts, graph):
    """
    Compute BIC score for model selection
    
    Args:
        df: pandas DataFrame
        cpts: dict, learned parameters
        graph: nx.DiGraph
    
    Returns:
        float: BIC score (higher is better)
    """
    log_likelihood = compute_log_likelihood(df, cpts, graph)
    n_samples = len(df)
    
    # Count parameters
    n_params = 0
    for node in graph.nodes():
        if node not in cpts:
            continue
            
        n_states = len(cpts[node]['states'])
        n_parents = len(cpts[node]['parents'])
        
        if n_parents == 0:
            # Root node: (n_states - 1) free parameters
            n_params += n_states - 1
        else:
            # Node with parents: product of parent states * (n_states - 1)
            parent_configs = len(cpts[node]['cpt'])
            n_params += parent_configs * (n_states - 1)
    
    bic_score = log_likelihood - 0.5 * n_params * np.log(n_samples)
    return bic_score


def compute_aic_score(df, cpts, graph):
    """
    Compute AIC score for model selection
    
    Args:
        df: pandas DataFrame
        cpts: dict, learned parameters
        graph: nx.DiGraph
    
    Returns:
        float: AIC score (higher is better)
    """
    log_likelihood = compute_log_likelihood(df, cpts, graph)
    
    # Count parameters
    n_params = 0
    for node in graph.nodes():
        if node not in cpts:
            continue
            
        n_states = len(cpts[node]['states'])
        n_parents = len(cpts[node]['parents'])
        
        if n_parents == 0:
            n_params += n_states - 1
        else:
            parent_configs = len(cpts[node]['cpt'])
            n_params += parent_configs * (n_states - 1)
    
    aic_score = log_likelihood - n_params
    return aic_score


def update_parameters_em_step(df, cpts, graph, missing_mask=None):
    """
    Update parameters using EM step (for missing data)
    
    Args:
        df: pandas DataFrame (may have missing values)
        cpts: dict, current parameters
        graph: nx.DiGraph
        missing_mask: pandas DataFrame, True where data is missing
    
    Returns:
        dict: updated CPTs
    """
    if missing_mask is None:
        # No missing data - just relearn parameters
        return learn_parameters_mle(df.dropna(), graph)
    
    # For missing data, this would require proper EM implementation
    # For now, return current parameters (placeholder)
    print("Warning: EM step for missing data not fully implemented")
    return cpts


def validate_cpts(cpts):
    """
    Validate that CPTs are properly normalized probability distributions
    
    Args:
        cpts: dict, learned CPTs
    
    Returns:
        dict: validation results
    """
    validation_results = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    for node, cpt_info in cpts.items():
        cpt = cpt_info['cpt']
        
        if len(cpt_info['parents']) == 0:
            # Root node - check marginal distribution sums to 1
            total_prob = sum(cpt.values())
            if abs(total_prob - 1.0) > 1e-6:
                validation_results['errors'].append(
                    f"Node {node}: marginal probabilities sum to {total_prob}, not 1.0"
                )
                validation_results['valid'] = False
        else:
            # Node with parents - check each conditional distribution
            for parent_config, conditional_dist in cpt.items():
                total_prob = sum(conditional_dist.values())
                if abs(total_prob - 1.0) > 1e-6:
                    validation_results['errors'].append(
                        f"Node {node}, parents={parent_config}: probabilities sum to {total_prob}, not 1.0"
                    )
                    validation_results['valid'] = False
        
        # Check for zero or negative probabilities
        if len(cpt_info['parents']) == 0:
            for state, prob in cpt.items():
                if prob <= 0:
                    validation_results['warnings'].append(
                        f"Node {node}, state {state}: probability is {prob} (should be > 0)"
                    )
        else:
            for parent_config, conditional_dist in cpt.items():
                for state, prob in conditional_dist.items():
                    if prob <= 0:
                        validation_results['warnings'].append(
                            f"Node {node}, parents={parent_config}, state {state}: probability is {prob}"
                        )
    
    return validation_results


def get_parameter_summary(cpts):
    """
    Get summary statistics about learned parameters
    
    Args:
        cpts: dict, learned CPTs
    
    Returns:
        dict: parameter summary
    """
    summary = {
        'total_nodes': len(cpts),
        'total_parameters': 0,
        'nodes_with_parents': 0,
        'root_nodes': 0,
        'parameter_details': {}
    }
    
    for node, cpt_info in cpts.items():
        n_states = len(cpt_info['states'])
        n_parents = len(cpt_info['parents'])
        
        if n_parents == 0:
            summary['root_nodes'] += 1
            n_params = n_states - 1
        else:
            summary['nodes_with_parents'] += 1
            parent_configs = len(cpt_info['cpt'])
            n_params = parent_configs * (n_states - 1)
        
        summary['total_parameters'] += n_params
        summary['parameter_details'][node] = {
            'states': n_states,
            'parents': n_parents,
            'parameters': n_params
        }
    
    return summary


def export_parameters_to_dict(cpts):
    """
    Export parameters to a serializable dictionary
    
    Args:
        cpts: dict, learned CPTs
    
    Returns:
        dict: serializable parameter representation
    """
    exported = {}
    
    for node, cpt_info in cpts.items():
        exported[str(node)] = {
            'parents': [str(p) for p in cpt_info['parents']],
            'states': [str(s) for s in cpt_info['states']],
            'cpt': {}
        }
        
        # Convert CPT to serializable format
        for key, value in cpt_info['cpt'].items():
            if isinstance(key, tuple):
                key_str = '_'.join(str(k) for k in key)
            else:
                key_str = str(key)
            
            if isinstance(value, dict):
                exported[str(node)]['cpt'][key_str] = {str(k): float(v) for k, v in value.items()}
            else:
                exported[str(node)]['cpt'][key_str] = float(value)
    
    return exported


def import_parameters_from_dict(param_dict):
    """
    Import parameters from serialized dictionary
    
    Args:
        param_dict: dict, serialized parameters
    
    Returns:
        dict: CPTs in standard format
    """
    cpts = {}
    
    for node_str, node_info in param_dict.items():
        node = int(node_str) if node_str.isdigit() else node_str
        
        cpts[node] = {
            'parents': [int(p) if p.isdigit() else p for p in node_info['parents']],
            'states': node_info['states'],
            'cpt': {}
        }
        
        # Convert CPT back from serialized format
        for key_str, value in node_info['cpt'].items():
            if '_' in key_str:
                # Tuple key
                key_parts = key_str.split('_')
                key = tuple(int(k) if k.isdigit() else k for k in key_parts)
            else:
                # Single key
                key = int(key_str) if key_str.isdigit() else key_str
            
            cpts[node]['cpt'][key] = value
    
    return cpts
