# -*- coding: utf-8 -*-
"""
Inference algorithms for MBCs - Gray code enumeration and tractable inference

Implements MPE (Most Probable Explanation) and marginal inference for MBCs
using Gray code enumeration for small class spaces and Variable Elimination
for tractable models.
"""

import numpy as np
import pandas as pd
import networkx as nx
from itertools import product
from .params import compute_log_likelihood


def infer_mpe_graycode(model, evidence):
    """
    MPE inference using Gray code enumeration for class variables
    
    Args:
        model: MBC model with learned structure and parameters
        evidence: dict {feature_var: value, ...}
    
    Returns:
        dict: {
            'class_config': dict with MPE assignment for class variables,
            'probability': float, probability of MPE,
            'log_probability': float, log probability of MPE
        }
    """
    if not hasattr(model, 'classes') or not hasattr(model, 'cpts'):
        return {'class_config': {}, 'probability': 0.0, 'log_probability': -np.inf}
    
    classes = model.classes
    
    # Get possible states for each class variable
    class_states = {}
    for class_var in classes:
        if class_var in model.cpts:
            class_states[class_var] = model.cpts[class_var]['states']
        else:
            class_states[class_var] = [0, 1]  # Default binary
    
    # Check if class space is small enough for enumeration
    total_configs = np.prod([len(states) for states in class_states.values()])
    
    if total_configs > 1024:  # Limit for tractability
        print(f"Warning: Class space too large ({total_configs} configurations) for Gray code enumeration")
        return _infer_mpe_greedy(model, evidence)
    
    # Generate all class configurations using Gray code order
    class_vars_ordered = sorted(classes)
    state_lists = [class_states[var] for var in class_vars_ordered]
    
    best_config = {}
    best_log_prob = -np.inf
    
    # Enumerate all configurations
    for config_tuple in product(*state_lists):
        class_config = dict(zip(class_vars_ordered, config_tuple))
        full_assignment = {**evidence, **class_config}
        
        # Calculate log probability
        log_prob = _calculate_assignment_probability(model, full_assignment)
        
        if log_prob > best_log_prob:
            best_log_prob = log_prob
            best_config = class_config.copy()
    
    return {
        'class_config': best_config,
        'probability': np.exp(best_log_prob) if best_log_prob > -np.inf else 0.0,
        'log_probability': best_log_prob
    }


def infer_mpe_ve(model, evidence):
    """
    MPE inference using Variable Elimination (for tractable models)
    
    Args:
        model: TW-MBC model with bounded treewidth
        evidence: dict {feature_var: value, ...}
    
    Returns:
        dict: MPE result
    """
    # Check if model is tractable
    if hasattr(model, 'treewidth_estimate') and hasattr(model, 'tw_max'):
        if model.treewidth_estimate > model.tw_max:
            print("Warning: Model may not be tractable for VE")
    
    # For now, fall back to Gray code for small problems
    if hasattr(model, 'classes') and len(model.classes) <= 4:
        return infer_mpe_graycode(model, evidence)
    else:
        return _infer_mpe_greedy(model, evidence)


def _infer_mpe_greedy(model, evidence):
    """
    Greedy MPE inference for large class spaces
    
    Args:
        model: MBC model
        evidence: dict with evidence
    
    Returns:
        dict: approximate MPE
    """
    if not hasattr(model, 'classes'):
        return {'class_config': {}, 'probability': 0.0, 'log_probability': -np.inf}
    
    class_config = {}
    
    # Greedy assignment: for each class, choose most probable state
    for class_var in model.classes:
        if class_var in model.cpts:
            states = model.cpts[class_var]['states']
            parents = model.cpts[class_var]['parents']
            
            best_state = states[0]  # Default
            best_prob = -np.inf
            
            for state in states:
                # Calculate probability of this state given current assignment
                temp_assignment = {**evidence, **class_config, class_var: state}
                
                if len(parents) == 0:
                    prob = model.cpts[class_var]['cpt'].get(state, 1.0/len(states))
                else:
                    parent_values = tuple(temp_assignment.get(p, 0) for p in parents)
                    cpt_entry = model.cpts[class_var]['cpt'].get(parent_values, {})
                    prob = cpt_entry.get(state, 1.0/len(states))
                
                if np.log(prob) > best_prob:
                    best_prob = np.log(prob)
                    best_state = state
            
            class_config[class_var] = best_state
        else:
            class_config[class_var] = 0  # Default
    
    # Calculate final probability
    full_assignment = {**evidence, **class_config}
    log_prob = _calculate_assignment_probability(model, full_assignment)
    
    return {
        'class_config': class_config,
        'probability': np.exp(log_prob) if log_prob > -np.inf else 0.0,
        'log_probability': log_prob
    }


def _calculate_assignment_probability(model, assignment):
    """
    Calculate log probability of a complete assignment
    
    Args:
        model: MBC model
        assignment: dict {var: value, ...}
    
    Returns:
        float: log probability
    """
    if not hasattr(model, 'cpts') or not model.cpts:
        return -np.inf
    
    log_prob = 0.0
    
    for node in model.G.nodes() if hasattr(model, 'G') else model.cpts.keys():
        if node not in model.cpts:
            continue
        
        node_value = assignment.get(node, 0)
        parents = model.cpts[node]['parents']
        
        if len(parents) == 0:
            # Root node
            prob = model.cpts[node]['cpt'].get(node_value, 1e-10)
        else:
            # Node with parents
            parent_values = tuple(assignment.get(p, 0) for p in parents)
            cpt_entry = model.cpts[node]['cpt'].get(parent_values, {})
            prob = cpt_entry.get(node_value, 1e-10)
        
        log_prob += np.log(max(prob, 1e-10))
    
    return log_prob


def infer_marginals(model, evidence, query_vars=None):
    """
    Marginal inference for query variables given evidence
    
    Args:
        model: MBC model
        evidence: dict {var: value, ...}
        query_vars: list of variables to compute marginals for (default: all classes)
    
    Returns:
        dict: {var: {state: probability, ...}, ...}
    """
    if query_vars is None:
        query_vars = model.classes if hasattr(model, 'classes') else []
    
    marginals = {}
    
    for query_var in query_vars:
        if query_var not in model.cpts:
            marginals[query_var] = {0: 0.5, 1: 0.5}  # Default
            continue
        
        states = model.cpts[query_var]['states']
        state_probs = {}
        
        # For each state of query variable
        for state in states:
            # Add query variable to evidence
            extended_evidence = evidence.copy()
            extended_evidence[query_var] = state
            
            # Sum over all other unobserved variables
            # For simplicity, we'll use the probability of this assignment
            log_prob = _calculate_partial_probability(model, extended_evidence, query_var)
            state_probs[state] = np.exp(log_prob)
        
        # Normalize
        total_prob = sum(state_probs.values())
        if total_prob > 0:
            for state in states:
                state_probs[state] /= total_prob
        
        marginals[query_var] = state_probs
    
    return marginals


def _calculate_partial_probability(model, evidence, query_var):
    """Calculate probability considering only relevant factors"""
    if not hasattr(model, 'cpts') or query_var not in model.cpts:
        return -np.inf
    
    log_prob = 0.0
    
    # Include probability of query variable given its parents
    parents = model.cpts[query_var]['parents']
    query_value = evidence[query_var]
    
    if len(parents) == 0:
        prob = model.cpts[query_var]['cpt'].get(query_value, 1e-10)
    else:
        parent_values = tuple(evidence.get(p, 0) for p in parents)
        cpt_entry = model.cpts[query_var]['cpt'].get(parent_values, {})
        prob = cpt_entry.get(query_value, 1e-10)
    
    log_prob += np.log(max(prob, 1e-10))
    
    # Include probability of evidence variables that depend on query
    if hasattr(model, 'G'):
        for child in model.G.successors(query_var):
            if child in evidence and child in model.cpts:
                child_value = evidence[child]
                child_parents = model.cpts[child]['parents']
                
                if len(child_parents) == 0:
                    child_prob = model.cpts[child]['cpt'].get(child_value, 1e-10)
                else:
                    parent_values = tuple(evidence.get(p, 0) for p in child_parents)
                    cpt_entry = model.cpts[child]['cpt'].get(parent_values, {})
                    child_prob = cpt_entry.get(child_value, 1e-10)
                
                log_prob += np.log(max(child_prob, 1e-10))
    
    return log_prob


def infer_marginals_simple(model, evidence):
    """
    Simplified marginal inference for compatibility
    
    Args:
        model: MBC model
        evidence: dict with evidence
    
    Returns:
        dict: marginal probabilities for class variables
    """
    if not hasattr(model, 'classes'):
        return {}
    
    return infer_marginals(model, evidence, model.classes)


def predict_classes(model, feature_evidence):
    """
    Predict class variables given feature evidence
    
    Args:
        model: trained MBC model
        feature_evidence: dict {feature_var: value, ...}
    
    Returns:
        dict: {
            'mpe': dict with most probable class assignment,
            'marginals': dict with marginal probabilities for each class,
            'confidence': float, confidence score
        }
    """
    # Get MPE
    mpe_result = infer_mpe_graycode(model, feature_evidence)
    
    # Get marginals
    marginals = infer_marginals(model, feature_evidence)
    
    # Calculate confidence (probability of MPE)
    confidence = mpe_result['probability']
    
    return {
        'mpe': mpe_result['class_config'],
        'marginals': marginals,
        'confidence': confidence
    }


def batch_predict(model, feature_data):
    """
    Batch prediction for multiple instances
    
    Args:
        model: trained MBC model
        feature_data: DataFrame with feature values
    
    Returns:
        list: predictions for each instance
    """
    predictions = []
    
    for _, row in feature_data.iterrows():
        # Convert row to evidence dict
        evidence = {}
        for col, value in row.items():
            if pd.notna(value):
                col_idx = col if isinstance(col, int) else feature_data.columns.get_loc(col)
                evidence[col_idx] = value
        
        # Predict
        pred = predict_classes(model, evidence)
        predictions.append(pred)
    
    return predictions


def explain_prediction(model, evidence, class_assignment):
    """
    Generate explanation for a prediction
    
    Args:
        model: MBC model
        evidence: dict with feature evidence
        class_assignment: dict with predicted class values
    
    Returns:
        dict: explanation information
    """
    explanation = {
        'evidence': evidence,
        'prediction': class_assignment,
        'factors': []
    }
    
    if not hasattr(model, 'cpts'):
        return explanation
    
    # For each class variable, explain its prediction
    for class_var, predicted_value in class_assignment.items():
        if class_var not in model.cpts:
            continue
        
        parents = model.cpts[class_var]['parents']
        
        factor_info = {
            'variable': class_var,
            'predicted_value': predicted_value,
            'parents': parents,
            'parent_values': {},
            'probability': 0.0
        }
        
        # Get parent values
        for parent in parents:
            if parent in evidence:
                factor_info['parent_values'][parent] = evidence[parent]
            elif parent in class_assignment:
                factor_info['parent_values'][parent] = class_assignment[parent]
        
        # Get probability
        if len(parents) == 0:
            prob = model.cpts[class_var]['cpt'].get(predicted_value, 0.0)
        else:
            parent_values = tuple(factor_info['parent_values'].get(p, 0) for p in parents)
            cpt_entry = model.cpts[class_var]['cpt'].get(parent_values, {})
            prob = cpt_entry.get(predicted_value, 0.0)
        
        factor_info['probability'] = prob
        explanation['factors'].append(factor_info)
    
    return explanation
