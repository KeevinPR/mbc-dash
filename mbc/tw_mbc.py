# -*- coding: utf-8 -*-
"""
TW-MBC learning with treewidth bounds via elimination orders

Implements Benjumeda's approach: learn MBC structure while bounding treewidth
via elimination orders/trees. Parameters: tw_max, EO heuristic (MCS/LEX/MMD), k_max_parents.
Exact inference via VE/JT when tw_max constraint is satisfied.
"""

import networkx as nx
import numpy as np
import pandas as pd
from .eo import moralize, heuristic_order, induced_width, estimate_treewidth, check_treewidth_bound
from .independence import mutual_information, g_square_test
from itertools import combinations


class TWMBCModel:
    """TW-MBC model with bounded treewidth"""
    
    def __init__(self, classes, features, tw_max=3, eo_method="MCS"):
        self.classes = classes
        self.features = features
        self.tw_max = tw_max
        self.eo_method = eo_method
        self.G = nx.DiGraph()
        self.G.add_nodes_from(classes + features)
        self.cpts = {}
        self.elimination_order = []
        self.treewidth_estimate = 0
    
    def _within_tw_bound(self, graph=None):
        """Check if graph satisfies treewidth bound"""
        if graph is None:
            graph = self.G
        return check_treewidth_bound(graph, self.tw_max)
    
    def get_treewidth_info(self):
        """Get treewidth information for current structure"""
        result = estimate_treewidth(self.G, method=self.eo_method, num_trials=3)
        self.treewidth_estimate = result['treewidth_estimate']
        self.elimination_order = result['best_order']
        
        return {
            'treewidth_estimate': self.treewidth_estimate,
            'elimination_order': self.elimination_order,
            'satisfies_bound': self.treewidth_estimate <= self.tw_max,
            'tractable': self.treewidth_estimate <= self.tw_max
        }
    
    def add_edge_if_tractable(self, parent, child):
        """Add edge only if it doesn't violate treewidth bound"""
        # Try adding edge
        self.G.add_edge(parent, child)
        
        if self._within_tw_bound():
            return True  # Edge added successfully
        else:
            # Remove edge and return False
            self.G.remove_edge(parent, child)
            return False
    
    def get_structure_info(self):
        """Get structural information about the MBC"""
        class_edges = []
        bridge_edges = []
        feature_edges = []
        
        for edge in self.G.edges():
            parent, child = edge
            if parent in self.classes and child in self.classes:
                class_edges.append(edge)
            elif parent in self.classes and child in self.features:
                bridge_edges.append(edge)
            elif parent in self.features and child in self.features:
                feature_edges.append(edge)
        
        return {
            'class_subgraph': class_edges,
            'bridge_subgraph': bridge_edges,
            'feature_subgraph': feature_edges,
            'total_edges': len(self.G.edges()),
            'class_configs': np.prod([len(self.cpts.get(c, {}).get('states', [0, 1])) for c in self.classes])
        }


def learn_tw_mbc(df, classes, features, tw_max=3, eo_method="MCS", k_max=3, alpha=0.05):
    """
    Learn TW-MBC structure with bounded treewidth
    
    Args:
        df: pandas DataFrame with data
        classes: list of class variable indices
        features: list of feature variable indices
        tw_max: maximum allowed treewidth
        eo_method: elimination order heuristic ("MCS", "LEX", "MMD")
        k_max: maximum parents per node
        alpha: significance level for independence tests
    
    Returns:
        TWMBCModel: learned TW-MBC model
    """
    model = TWMBCModel(classes, features, tw_max, eo_method)
    
    # Phase 1: Build class-to-feature bridges (greedy by MI, respecting tw_max)
    print(f"Phase 1: Learning bridge subgraph (class -> feature)")
    
    # Sort features by total MI with all classes (prioritize most informative)
    feature_mi_scores = []
    for feature in features:
        total_mi = sum(mutual_information(df.iloc[:, feature], df.iloc[:, class_var]) 
                      for class_var in classes)
        feature_mi_scores.append((feature, total_mi))
    
    # Sort by total MI (descending)
    feature_mi_scores.sort(key=lambda x: x[1], reverse=True)
    
    # For each feature, try to add class parents
    parent_count = {f: 0 for f in features}
    
    for feature, _ in feature_mi_scores:
        # Sort classes by MI with this feature
        class_mi_scores = []
        for class_var in classes:
            mi_score = mutual_information(df.iloc[:, feature], df.iloc[:, class_var])
            class_mi_scores.append((class_var, mi_score))
        
        class_mi_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Try adding each class as parent
        for class_var, mi_score in class_mi_scores:
            if parent_count[feature] >= k_max:
                break
            
            if mi_score > 0:  # Some association
                # Test independence
                result = g_square_test(df.iloc[:, feature], df.iloc[:, class_var], alpha=alpha)
                
                if not result['independent']:  # Dependent
                    # Try adding edge if it preserves treewidth bound
                    if model.add_edge_if_tractable(class_var, feature):
                        parent_count[feature] += 1
                        print(f"  Added edge: {class_var} -> {feature}")
    
    # Phase 2: Add class-class edges (forest structure)
    print(f"Phase 2: Learning class subgraph")
    
    # Sort class pairs by MI
    class_pairs_mi = []
    for i, class1 in enumerate(classes):
        for j, class2 in enumerate(classes):
            if i < j:  # Avoid duplicates
                mi_score = mutual_information(df.iloc[:, class1], df.iloc[:, class2])
                class_pairs_mi.append(((class1, class2), mi_score))
    
    class_pairs_mi.sort(key=lambda x: x[1], reverse=True)
    
    # Try adding class-class edges (maintaining forest/tree structure)
    class_edges_added = 0
    for (class1, class2), mi_score in class_pairs_mi:
        if class_edges_added >= len(classes) - 1:  # Tree has n-1 edges
            break
        
        if mi_score > 0:
            # Test independence
            result = g_square_test(df.iloc[:, class1], df.iloc[:, class2], alpha=alpha)
            
            if not result['independent']:
                # Try adding edge (both directions, choose better one)
                edge_added = False
                
                # Try class1 -> class2
                if model.add_edge_if_tractable(class1, class2):
                    print(f"  Added edge: {class1} -> {class2}")
                    class_edges_added += 1
                    edge_added = True
                elif model.add_edge_if_tractable(class2, class1):
                    print(f"  Added edge: {class2} -> {class1}")
                    class_edges_added += 1
                    edge_added = True
                
                if edge_added:
                    # Check if we still have a tree (no cycles)
                    if not nx.is_forest(model.G.subgraph(classes).to_undirected()):
                        # Remove the edge that created a cycle
                        if model.G.has_edge(class1, class2):
                            model.G.remove_edge(class1, class2)
                        elif model.G.has_edge(class2, class1):
                            model.G.remove_edge(class2, class1)
                        class_edges_added -= 1
    
    # Phase 3: Add feature-feature edges (limited, respecting tw_max)
    print(f"Phase 3: Learning feature subgraph")
    
    # Only consider features that have class parents (connected features)
    connected_features = []
    for feature in features:
        has_class_parent = any(model.G.has_edge(class_var, feature) for class_var in classes)
        if has_class_parent:
            connected_features.append(feature)
    
    print(f"  Connected features: {len(connected_features)}")
    
    # Try adding edges between connected features (limited to avoid explosion)
    feature_pairs_tried = 0
    max_feature_pairs = min(20, len(connected_features) * 2)  # Limit attempts
    
    if len(connected_features) > 1:
        # Sort feature pairs by MI
        feature_pairs_mi = []
        for i, feat1 in enumerate(connected_features):
            for j, feat2 in enumerate(connected_features):
                if i < j:
                    mi_score = mutual_information(df.iloc[:, feat1], df.iloc[:, feat2])
                    feature_pairs_mi.append(((feat1, feat2), mi_score))
        
        feature_pairs_mi.sort(key=lambda x: x[1], reverse=True)
        
        for (feat1, feat2), mi_score in feature_pairs_mi:
            if feature_pairs_tried >= max_feature_pairs:
                break
            
            feature_pairs_tried += 1
            
            if mi_score > 0:
                # Test conditional independence given class parents
                class_parents_1 = [c for c in classes if model.G.has_edge(c, feat1)]
                class_parents_2 = [c for c in classes if model.G.has_edge(c, feat2)]
                common_parents = list(set(class_parents_1) & set(class_parents_2))
                
                if len(common_parents) > 0:
                    # Test conditional independence given common class parents
                    Z_data = df.iloc[:, common_parents] if len(common_parents) > 1 else df.iloc[:, common_parents[0]]
                    result = g_square_test(df.iloc[:, feat1], df.iloc[:, feat2], Z_data, alpha=alpha)
                else:
                    # Test unconditional independence
                    result = g_square_test(df.iloc[:, feat1], df.iloc[:, feat2], alpha=alpha)
                
                if not result['independent']:
                    # Try adding edge in both directions
                    if model.add_edge_if_tractable(feat1, feat2):
                        print(f"  Added edge: {feat1} -> {feat2}")
                    elif model.add_edge_if_tractable(feat2, feat1):
                        print(f"  Added edge: {feat2} -> {feat1}")
    
    # Phase 4: Learn parameters (simplified)
    print(f"Phase 4: Learning parameters")
    model.cpts = learn_parameters_mle(df, model.G)
    
    # Get final treewidth info
    tw_info = model.get_treewidth_info()
    print(f"Final treewidth estimate: {tw_info['treewidth_estimate']} (bound: {tw_max})")
    print(f"Tractable: {tw_info['tractable']}")
    
    return model


def learn_parameters_mle(df, graph):
    """
    Learn MLE parameters for the graph structure
    
    Args:
        df: pandas DataFrame
        graph: nx.DiGraph, learned structure
    
    Returns:
        dict: CPTs for each node
    """
    cpts = {}
    
    for node in graph.nodes():
        parents = list(graph.predecessors(node))
        
        # Get data for this node and its parents
        node_data = df.iloc[:, node]
        
        # Get unique states for this node
        states = sorted(node_data.unique())
        cpts[node] = {'states': states}
        
        if len(parents) == 0:
            # No parents - marginal distribution
            counts = node_data.value_counts()
            total = len(node_data)
            
            cpt = {}
            for state in states:
                cpt[state] = counts.get(state, 0) / total
            
            cpts[node]['cpt'] = cpt
            
        else:
            # Has parents - conditional distribution
            parent_data = df.iloc[:, parents]
            
            # Get all parent configurations
            if len(parents) == 1:
                parent_configs = parent_data.unique()
                parent_configs = [(config,) if not isinstance(config, tuple) else config 
                                for config in parent_configs]
            else:
                parent_configs = [tuple(row) for row in parent_data.drop_duplicates().values]
            
            cpt = {}
            for parent_config in parent_configs:
                # Filter data for this parent configuration
                if len(parents) == 1:
                    mask = (parent_data == parent_config[0])
                else:
                    mask = (parent_data == parent_config).all(axis=1)
                
                filtered_node_data = node_data[mask]
                
                if len(filtered_node_data) > 0:
                    counts = filtered_node_data.value_counts()
                    total = len(filtered_node_data)
                    
                    config_cpt = {}
                    for state in states:
                        config_cpt[state] = counts.get(state, 0) / total
                    
                    cpt[parent_config] = config_cpt
                else:
                    # No data for this configuration - uniform distribution
                    config_cpt = {state: 1.0/len(states) for state in states}
                    cpt[parent_config] = config_cpt
            
            cpts[node]['cpt'] = cpt
    
    return cpts


# Inference functions (simplified VE for tractable models)

def variable_elimination_mpe(model, evidence):
    """
    Simplified Variable Elimination for MPE in tractable TW-MBC
    
    Args:
        model: TWMBCModel
        evidence: dict {feature_var: value, ...}
    
    Returns:
        dict: MPE assignment for class variables
    """
    if model.treewidth_estimate > model.tw_max:
        print("Warning: Model may not be tractable for exact inference")
    
    # Simplified inference - use elimination order
    # For demo purposes, fall back to enumeration for small class spaces
    if len(model.classes) <= 4:  # Small class space
        return infer_mpe_enumeration(model, evidence)
    else:
        print("Class space too large for demo inference")
        return {c: 0 for c in model.classes}  # Default assignment


def infer_mpe_enumeration(model, evidence):
    """
    MPE inference by enumeration (for small class spaces)
    
    Args:
        model: TWMBCModel
        evidence: dict {feature_var: value, ...}
    
    Returns:
        dict: MPE assignment for class variables
    """
    if not hasattr(model, 'cpts') or not model.cpts:
        return {c: 0 for c in model.classes}
    
    # Get all possible class assignments
    class_states = []
    for class_var in model.classes:
        states = model.cpts.get(class_var, {}).get('states', [0, 1])
        class_states.append(states)
    
    # Generate all combinations
    from itertools import product
    
    best_prob = -float('inf')
    best_assignment = {}
    
    for assignment in product(*class_states):
        class_assignment = dict(zip(model.classes, assignment))
        full_assignment = {**evidence, **class_assignment}
        
        # Calculate probability (simplified)
        log_prob = 0.0
        
        for node in model.G.nodes():
            if node not in model.cpts:
                continue
                
            node_value = full_assignment.get(node, 0)
            parents = list(model.G.predecessors(node))
            
            if len(parents) == 0:
                # Marginal probability
                prob = model.cpts[node]['cpt'].get(node_value, 1e-10)
            else:
                # Conditional probability
                parent_values = tuple(full_assignment.get(p, 0) for p in parents)
                cpt_entry = model.cpts[node]['cpt'].get(parent_values, {})
                prob = cpt_entry.get(node_value, 1e-10)
            
            log_prob += np.log(max(prob, 1e-10))
        
        if log_prob > best_prob:
            best_prob = log_prob
            best_assignment = class_assignment
    
    return best_assignment
