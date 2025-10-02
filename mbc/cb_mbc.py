# -*- coding: utf-8 -*-
"""
CB-MBC (Wrapper) algorithm implementation

Based on Borchani Chapter 5: CB-MBC with 3-phase wrapper approach
- Phase I: SNB per class + shared children resolution
- Phase II: feature-feature edges if accuracy improves (up to T iterations)  
- Phase III: merge components if improvement
Metrics: global accuracy or mean-Hamming distance
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, hamming_loss
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
import networkx as nx
from itertools import combinations
from .independence import mutual_information


class CBMBCModel:
    """CB-MBC model with component structure and parameters"""
    
    def __init__(self, class_vars, feature_vars):
        self.class_vars = class_vars
        self.feature_vars = feature_vars
        self.components = []  # List of component subgraphs
        self.edges = []
        self.parameters = {}
        
    def add_component(self, class_nodes, feature_nodes, edges):
        """Add a component to the MBC"""
        component = {
            'class_nodes': class_nodes,
            'feature_nodes': feature_nodes,
            'edges': edges
        }
        self.components.append(component)
        self.edges.extend(edges)
    
    def get_graph(self):
        """Get NetworkX graph representation"""
        G = nx.DiGraph()
        G.add_nodes_from(self.class_vars + self.feature_vars)
        G.add_edges_from(self.edges)
        return G
    
    def predict(self, X_test):
        """Predict using the learned CB-MBC structure"""
        # Simplified prediction using component-wise inference
        predictions = []
        
        for _, row in X_test.iterrows():
            pred_row = {}
            
            for component in self.components:
                # For each component, predict class variables
                for class_var in component['class_nodes']:
                    # Simple prediction based on connected features
                    connected_features = []
                    for edge in component['edges']:
                        if edge[1] == class_var and edge[0] in self.feature_vars:
                            connected_features.append(edge[0])
                    
                    if connected_features and class_var in self.parameters:
                        # Use stored parameters for prediction
                        pred_row[class_var] = self.parameters[class_var].get('mode', 0)
                    else:
                        pred_row[class_var] = 0  # Default prediction
            
            predictions.append(pred_row)
        
        return predictions


class CBMBCLearner:
    """CB-MBC wrapper algorithm learner"""
    
    def __init__(self, metric='global', max_iterations=10, cv_folds=3):
        """
        Initialize CB-MBC learner
        
        Args:
            metric: 'global' for global accuracy or 'hamming' for mean-Hamming
            max_iterations: maximum iterations for Phase II (T)
            cv_folds: cross-validation folds for accuracy estimation
        """
        self.metric = metric
        self.max_iterations = max_iterations
        self.cv_folds = cv_folds
    
    def _evaluate_model(self, df, model, class_vars):
        """
        Evaluate model performance using cross-validation
        
        Args:
            df: pandas DataFrame
            model: CBMBCModel
            class_vars: list of class variables
        
        Returns:
            float: evaluation score (higher is better)
        """
        if len(model.components) == 0:
            return 0.0
        
        # Simple evaluation using Naive Bayes on each component
        total_score = 0.0
        
        for component in model.components:
            if len(component['class_nodes']) == 0 or len(component['feature_nodes']) == 0:
                continue
            
            # Get features connected to this component's classes
            connected_features = set()
            for edge in component['edges']:
                if edge[0] in component['feature_nodes'] and edge[1] in component['class_nodes']:
                    connected_features.add(edge[0])
            
            if len(connected_features) == 0:
                continue
            
            # Prepare data for this component
            X_component = df.iloc[:, list(connected_features)]
            y_component = df.iloc[:, component['class_nodes']]
            
            # Evaluate each class variable in the component
            component_scores = []
            for class_var in component['class_nodes']:
                try:
                    # Use simple Naive Bayes for evaluation
                    nb = MultinomialNB()
                    y_class = df.iloc[:, class_var]
                    
                    if len(X_component.columns) > 0 and len(y_class.unique()) > 1:
                        scores = cross_val_score(nb, X_component, y_class, cv=min(self.cv_folds, len(df)//2), scoring='accuracy')
                        component_scores.append(np.mean(scores))
                    else:
                        component_scores.append(0.0)
                except:
                    component_scores.append(0.0)
            
            if len(component_scores) > 0:
                if self.metric == 'hamming':
                    # For Hamming distance, we want to minimize (so negate)
                    total_score += np.mean(component_scores)
                else:
                    # Global accuracy
                    total_score += np.mean(component_scores)
        
        return total_score / max(len(model.components), 1)
    
    def _create_snb_per_class(self, df, class_vars, feature_vars):
        """
        Phase I: Create Selective Naive Bayes (SNB) for each class
        
        Args:
            df: pandas DataFrame
            class_vars: list of class variable indices
            feature_vars: list of feature variable indices
        
        Returns:
            dict: SNB structures per class
        """
        snb_structures = {}
        
        for class_var in class_vars:
            # Calculate mutual information between class and each feature
            mi_scores = []
            for feature_var in feature_vars:
                mi_score = mutual_information(df.iloc[:, class_var], df.iloc[:, feature_var])
                mi_scores.append((feature_var, mi_score))
            
            # Sort by MI (descending) and select top features
            mi_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Select features with MI > threshold or top k features
            selected_features = []
            mi_threshold = np.mean([score for _, score in mi_scores]) if mi_scores else 0
            
            for feature_var, mi_score in mi_scores:
                if mi_score > mi_threshold and len(selected_features) < 10:  # Limit features
                    selected_features.append(feature_var)
            
            # Ensure at least one feature if available
            if len(selected_features) == 0 and len(mi_scores) > 0:
                selected_features = [mi_scores[0][0]]
            
            snb_structures[class_var] = {
                'features': selected_features,
                'edges': [(f, class_var) for f in selected_features]
            }
        
        return snb_structures
    
    def _resolve_shared_children(self, snb_structures, class_vars, feature_vars):
        """
        Resolve shared children (features connected to multiple classes)
        
        Args:
            snb_structures: SNB structures from Phase I
            class_vars: list of class variables
            feature_vars: list of feature variables
        
        Returns:
            list: initial components after resolving shared children
        """
        # Find features that are children of multiple classes
        feature_parents = {}
        for class_var, structure in snb_structures.items():
            for feature in structure['features']:
                if feature not in feature_parents:
                    feature_parents[feature] = []
                feature_parents[feature].append(class_var)
        
        # Create initial components
        components = []
        used_classes = set()
        
        # Group classes that share features
        for feature, parent_classes in feature_parents.items():
            if len(parent_classes) > 1:
                # Multiple classes share this feature - create component
                component_classes = [c for c in parent_classes if c not in used_classes]
                if len(component_classes) > 0:
                    # Collect all features for these classes
                    component_features = set()
                    component_edges = []
                    
                    for class_var in component_classes:
                        for f in snb_structures[class_var]['features']:
                            component_features.add(f)
                            component_edges.append((f, class_var))
                    
                    components.append({
                        'class_nodes': component_classes,
                        'feature_nodes': list(component_features),
                        'edges': component_edges
                    })
                    
                    used_classes.update(component_classes)
        
        # Create components for remaining individual classes
        for class_var in class_vars:
            if class_var not in used_classes:
                components.append({
                    'class_nodes': [class_var],
                    'feature_nodes': snb_structures[class_var]['features'],
                    'edges': snb_structures[class_var]['edges']
                })
        
        return components
    
    def _add_feature_edges(self, df, model):
        """
        Phase II: Add edges between features if they improve accuracy
        
        Args:
            df: pandas DataFrame
            model: CBMBCModel
        
        Returns:
            CBMBCModel: improved model
        """
        current_score = self._evaluate_model(df, model, model.class_vars)
        improved = True
        iterations = 0
        
        while improved and iterations < self.max_iterations:
            improved = False
            best_edge = None
            best_score = current_score
            
            # Try adding edges between features in each component
            for comp_idx, component in enumerate(model.components):
                feature_nodes = component['feature_nodes']
                
                # Try all possible feature-feature edges
                for i, feature1 in enumerate(feature_nodes):
                    for j, feature2 in enumerate(feature_nodes):
                        if i != j:
                            edge = (feature1, feature2)
                            
                            # Check if edge already exists
                            if edge not in model.edges:
                                # Try adding this edge
                                model.edges.append(edge)
                                model.components[comp_idx]['edges'].append(edge)
                                
                                # Evaluate new model
                                new_score = self._evaluate_model(df, model, model.class_vars)
                                
                                if new_score > best_score:
                                    best_score = new_score
                                    best_edge = (edge, comp_idx)
                                    improved = True
                                
                                # Remove edge for next iteration
                                model.edges.remove(edge)
                                model.components[comp_idx]['edges'].remove(edge)
            
            # Add best edge if found
            if best_edge is not None:
                edge, comp_idx = best_edge
                model.edges.append(edge)
                model.components[comp_idx]['edges'].append(edge)
                current_score = best_score
            
            iterations += 1
        
        return model
    
    def _merge_components(self, df, model):
        """
        Phase III: Merge components if it improves performance
        
        Args:
            df: pandas DataFrame  
            model: CBMBCModel
        
        Returns:
            CBMBCModel: model with merged components
        """
        if len(model.components) <= 1:
            return model
        
        current_score = self._evaluate_model(df, model, model.class_vars)
        improved = True
        
        while improved and len(model.components) > 1:
            improved = False
            best_merge = None
            best_score = current_score
            
            # Try merging each pair of components
            for i in range(len(model.components)):
                for j in range(i + 1, len(model.components)):
                    # Create merged component
                    comp1 = model.components[i]
                    comp2 = model.components[j]
                    
                    merged_component = {
                        'class_nodes': comp1['class_nodes'] + comp2['class_nodes'],
                        'feature_nodes': list(set(comp1['feature_nodes'] + comp2['feature_nodes'])),
                        'edges': comp1['edges'] + comp2['edges']
                    }
                    
                    # Create temporary model with merged component
                    temp_model = CBMBCModel(model.class_vars, model.feature_vars)
                    temp_model.components = (model.components[:i] + 
                                           model.components[i+1:j] + 
                                           model.components[j+1:] + 
                                           [merged_component])
                    temp_model.edges = []
                    for comp in temp_model.components:
                        temp_model.edges.extend(comp['edges'])
                    
                    # Evaluate merged model
                    merge_score = self._evaluate_model(df, temp_model, model.class_vars)
                    
                    if merge_score > best_score:
                        best_score = merge_score
                        best_merge = (i, j, merged_component)
                        improved = True
            
            # Apply best merge if found
            if best_merge is not None:
                i, j, merged_component = best_merge
                # Remove original components and add merged one
                model.components = (model.components[:i] + 
                                  model.components[i+1:j] + 
                                  model.components[j+1:] + 
                                  [merged_component])
                # Update edges
                model.edges = []
                for comp in model.components:
                    model.edges.extend(comp['edges'])
                current_score = best_score
        
        return model
    
    def learn(self, df, class_vars, feature_vars):
        """
        Learn CB-MBC structure using 3-phase wrapper approach
        
        Args:
            df: pandas DataFrame
            class_vars: list of class variable indices
            feature_vars: list of feature variable indices
        
        Returns:
            CBMBCModel: learned CB-MBC model
        """
        # Phase I: Create SNB per class and resolve shared children
        snb_structures = self._create_snb_per_class(df, class_vars, feature_vars)
        initial_components = self._resolve_shared_children(snb_structures, class_vars, feature_vars)
        
        # Create initial model
        model = CBMBCModel(class_vars, feature_vars)
        for component in initial_components:
            model.add_component(component['class_nodes'], component['feature_nodes'], component['edges'])
        
        # Phase II: Add feature-feature edges if they improve accuracy
        model = self._add_feature_edges(df, model)
        
        # Phase III: Merge components if it improves performance
        model = self._merge_components(df, model)
        
        # Learn parameters (simplified)
        self._learn_parameters(df, model)
        
        return model
    
    def _learn_parameters(self, df, model):
        """Learn parameters for the CB-MBC model"""
        for class_var in model.class_vars:
            # Store simple statistics for each class variable
            class_data = df.iloc[:, class_var]
            model.parameters[class_var] = {
                'mode': class_data.mode().iloc[0] if len(class_data.mode()) > 0 else 0,
                'distribution': class_data.value_counts(normalize=True).to_dict()
            }


def learn_cb_mbc(df, class_vars, feature_vars, metric='global', max_iterations=10):
    """
    Convenience function to learn CB-MBC structure
    
    Args:
        df: pandas DataFrame
        class_vars: list of class variable indices
        feature_vars: list of feature variable indices
        metric: 'global' for global accuracy or 'hamming' for mean-Hamming
        max_iterations: maximum iterations for Phase II (T)
    
    Returns:
        CBMBCModel: learned CB-MBC model
    """
    learner = CBMBCLearner(metric=metric, max_iterations=max_iterations)
    return learner.learn(df, class_vars, feature_vars)
