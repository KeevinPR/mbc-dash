# -*- coding: utf-8 -*-
"""
HITON-PC and HITON-MB algorithms for MB-MBC learning

Based on Borchani Chapter 6: MB-MBC Algorithm using HITON-PC + HITON-MB
with G² tests, configurable α, maxCS, and parent limit k
"""

import numpy as np
import pandas as pd
from .independence import g_square_test, mutual_information
from itertools import combinations


class HITONLearner:
    """
    HITON-PC and HITON-MB implementation for MBC structure learning
    """
    
    def __init__(self, alpha=0.05, max_cond_size=3, max_parents=5):
        """
        Initialize HITON learner
        
        Args:
            alpha: significance level for independence tests
            max_cond_size: maximum conditioning set size (maxCS)
            max_parents: maximum number of parents per node (k)
        """
        self.alpha = alpha
        self.max_cond_size = max_cond_size
        self.max_parents = max_parents
    
    def hiton_pc(self, df, target, candidate_vars=None):
        """
        HITON-PC: Learn parents and children of target variable
        
        Args:
            df: pandas DataFrame with data
            target: int, target variable index
            candidate_vars: list of candidate variables (default: all except target)
        
        Returns:
            list: parents and children of target
        """
        if candidate_vars is None:
            candidate_vars = [i for i in range(df.shape[1]) if i != target]
        
        # Phase 1: Growing phase - add variables associated with target
        pc_set = []
        
        for var in candidate_vars:
            # Test if var is associated with target given current PC set
            if len(pc_set) == 0:
                # Unconditional test
                result = g_square_test(df.iloc[:, target], df.iloc[:, var], alpha=self.alpha)
                if not result['independent']:
                    pc_set.append(var)
            else:
                # Test conditional independence given subsets of current PC set
                is_independent = False
                for cond_size in range(min(len(pc_set), self.max_cond_size) + 1):
                    if is_independent:
                        break
                    
                    if cond_size == 0:
                        # Unconditional test
                        result = g_square_test(df.iloc[:, target], df.iloc[:, var], alpha=self.alpha)
                        if result['independent']:
                            is_independent = True
                    else:
                        # Conditional tests
                        for cond_set in combinations(pc_set, cond_size):
                            Z_data = df.iloc[:, list(cond_set)] if len(cond_set) > 1 else df.iloc[:, cond_set[0]]
                            result = g_square_test(df.iloc[:, target], df.iloc[:, var], Z_data, self.alpha)
                            if result['independent']:
                                is_independent = True
                                break
                
                if not is_independent:
                    pc_set.append(var)
                    
                    # Limit number of parents
                    if len(pc_set) >= self.max_parents:
                        break
        
        # Phase 2: Shrinking phase - remove spurious associations
        final_pc = pc_set.copy()
        
        for var in pc_set:
            other_pc = [v for v in pc_set if v != var]
            
            # Test if var becomes independent given subsets of other PC variables
            for cond_size in range(min(len(other_pc), self.max_cond_size) + 1):
                if var not in final_pc:
                    break
                
                if cond_size == 0:
                    continue  # Already tested in growing phase
                
                for cond_set in combinations(other_pc, cond_size):
                    Z_data = df.iloc[:, list(cond_set)] if len(cond_set) > 1 else df.iloc[:, cond_set[0]]
                    result = g_square_test(df.iloc[:, target], df.iloc[:, var], Z_data, self.alpha)
                    
                    if result['independent'] and var in final_pc:
                        final_pc.remove(var)
                        break
        
        return final_pc
    
    def hiton_mb(self, df, target, candidate_vars=None):
        """
        HITON-MB: Learn Markov blanket of target variable
        
        Args:
            df: pandas DataFrame with data
            target: int, target variable index
            candidate_vars: list of candidate variables
        
        Returns:
            dict: {
                'parents_children': list of parents and children,
                'spouses': list of spouse variables (parents of children),
                'markov_blanket': complete Markov blanket
            }
        """
        if candidate_vars is None:
            candidate_vars = [i for i in range(df.shape[1]) if i != target]
        
        # Step 1: Get parents and children using HITON-PC
        pc_set = self.hiton_pc(df, target, candidate_vars)
        
        # Step 2: Find spouses (parents of children that are not parents of target)
        spouses = []
        remaining_vars = [v for v in candidate_vars if v not in pc_set]
        
        for var in remaining_vars:
            # Check if var is a spouse by testing if it's dependent on target
            # given some subset of PC(target) but becomes independent when
            # conditioned on PC(target) + some child of target
            
            is_spouse = False
            
            # Test if var is dependent on target given PC set
            if len(pc_set) > 0:
                for cond_size in range(min(len(pc_set), self.max_cond_size) + 1):
                    if is_spouse:
                        break
                    
                    for cond_set in combinations(pc_set, cond_size):
                        Z_data = df.iloc[:, list(cond_set)] if len(cond_set) > 1 else df.iloc[:, cond_set[0]]
                        result = g_square_test(df.iloc[:, target], df.iloc[:, var], Z_data, self.alpha)
                        
                        if not result['independent']:
                            # var is dependent on target given this conditioning set
                            # Check if there exists a child c such that target ⊥ var | cond_set ∪ {c}
                            for child in pc_set:
                                extended_cond = list(cond_set) + [child]
                                if len(extended_cond) <= self.max_cond_size + 1:
                                    Z_ext = df.iloc[:, extended_cond]
                                    result_ext = g_square_test(df.iloc[:, target], df.iloc[:, var], Z_ext, self.alpha)
                                    
                                    if result_ext['independent']:
                                        spouses.append(var)
                                        is_spouse = True
                                        break
                            
                            if is_spouse:
                                break
        
        markov_blanket = pc_set + spouses
        
        return {
            'parents_children': pc_set,
            'spouses': spouses,
            'markov_blanket': markov_blanket
        }
    
    def learn_mb_mbc_structure(self, df, class_vars, feature_vars):
        """
        Learn MB-MBC structure using HITON-PC for bridge subgraph
        
        Args:
            df: pandas DataFrame
            class_vars: list of class variable indices
            feature_vars: list of feature variable indices
        
        Returns:
            dict: MBC structure with class, bridge, and feature subgraphs
        """
        structure = {
            'class_subgraph': [],
            'bridge_subgraph': [],
            'feature_subgraph': [],
            'all_edges': []
        }
        
        # Learn class subgraph (edges between class variables)
        for class_var in class_vars:
            pc_class = self.hiton_pc(df, class_var, [c for c in class_vars if c != class_var])
            for parent in pc_class:
                if parent in class_vars:
                    structure['class_subgraph'].append((parent, class_var))
                    structure['all_edges'].append((parent, class_var))
        
        # Learn bridge subgraph (class -> feature edges)
        for feature_var in feature_vars:
            # Find class parents of each feature
            class_parents = []
            
            # Sort class variables by mutual information with feature
            class_mi_scores = []
            for class_var in class_vars:
                mi_score = mutual_information(df.iloc[:, class_var], df.iloc[:, feature_var])
                class_mi_scores.append((class_var, mi_score))
            
            # Sort by MI (descending)
            class_mi_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Test each class variable as potential parent
            for class_var, _ in class_mi_scores:
                if len(class_parents) >= self.max_parents:
                    break
                
                # Test if class_var -> feature_var given current class parents
                if len(class_parents) == 0:
                    result = g_square_test(df.iloc[:, feature_var], df.iloc[:, class_var], alpha=self.alpha)
                    if not result['independent']:
                        class_parents.append(class_var)
                else:
                    # Test with conditioning
                    is_independent = True
                    for cond_size in range(min(len(class_parents), self.max_cond_size) + 1):
                        if not is_independent:
                            break
                        
                        if cond_size == 0:
                            result = g_square_test(df.iloc[:, feature_var], df.iloc[:, class_var], alpha=self.alpha)
                            if not result['independent']:
                                is_independent = False
                        else:
                            for cond_set in combinations(class_parents, cond_size):
                                Z_data = df.iloc[:, list(cond_set)] if len(cond_set) > 1 else df.iloc[:, cond_set[0]]
                                result = g_square_test(df.iloc[:, feature_var], df.iloc[:, class_var], Z_data, self.alpha)
                                if not result['independent']:
                                    is_independent = False
                                    break
                    
                    if not is_independent:
                        class_parents.append(class_var)
            
            # Add bridge edges
            for class_parent in class_parents:
                structure['bridge_subgraph'].append((class_parent, feature_var))
                structure['all_edges'].append((class_parent, feature_var))
        
        # Learn feature subgraph (edges between features with class parents)
        connected_features = []
        for _, feature in structure['bridge_subgraph']:
            if feature not in connected_features:
                connected_features.append(feature)
        
        for feature_var in connected_features:
            # Only consider other connected features as potential parents
            other_connected = [f for f in connected_features if f != feature_var]
            
            if len(other_connected) > 0:
                pc_features = self.hiton_pc(df, feature_var, other_connected)
                
                for parent in pc_features:
                    if parent in connected_features:
                        structure['feature_subgraph'].append((parent, feature_var))
                        structure['all_edges'].append((parent, feature_var))
        
        return structure


def learn_mb_mbc(df, class_vars, feature_vars, alpha=0.05, max_cond_size=3, max_parents=5):
    """
    Convenience function to learn MB-MBC structure
    
    Args:
        df: pandas DataFrame
        class_vars: list of class variable indices
        feature_vars: list of feature variable indices
        alpha: significance level for G² tests
        max_cond_size: maximum conditioning set size (maxCS)
        max_parents: maximum number of parents per node (k)
    
    Returns:
        dict: learned MBC structure
    """
    learner = HITONLearner(alpha=alpha, max_cond_size=max_cond_size, max_parents=max_parents)
    return learner.learn_mb_mbc_structure(df, class_vars, feature_vars)
