# -*- coding: utf-8 -*-
"""
Elimination orders and treewidth approximation for TW-MBC

Implements heuristics for elimination orders (MCS/LEX/MMD), moralization,
and fast treewidth approximation for bounded learning
"""

import networkx as nx
import numpy as np
from collections import defaultdict, deque


def moralize(G):
    """
    Moralize a directed graph (connect parents and remove directions)
    
    Args:
        G: nx.DiGraph, directed graph
    
    Returns:
        nx.Graph: moralized undirected graph
    """
    UG = nx.Graph()
    UG.add_nodes_from(G.nodes)
    
    # Connect parents of each node (marry parents)
    for node in G.nodes:
        parents = list(G.predecessors(node))
        
        # Add edges between all pairs of parents
        for i in range(len(parents)):
            for j in range(i + 1, len(parents)):
                UG.add_edge(parents[i], parents[j])
        
        # Add edges from parents to child (remove directions)
        for parent in parents:
            UG.add_edge(parent, node)
    
    # Add existing undirected edges
    for edge in G.edges():
        UG.add_edge(edge[0], edge[1])
    
    return UG


def heuristic_order(UG, method="MCS"):
    """
    Generate elimination order using heuristic
    
    Args:
        UG: nx.Graph, undirected graph
        method: str, heuristic method ('MCS', 'LEX', 'MMD')
    
    Returns:
        list: elimination order (sequence of nodes)
    """
    if len(UG.nodes) == 0:
        return []
    
    H = UG.copy()
    order = []
    
    if method == "MCS":
        # Maximum Cardinality Search
        scores = {v: 0 for v in H.nodes}
        
        while H.nodes:
            # Select node with highest score (most neighbors already eliminated)
            v = max(scores.keys(), key=lambda x: scores[x])
            order.append(v)
            
            # Update scores of neighbors
            for u in H.neighbors(v):
                if u in scores:
                    scores[u] += 1
            
            # Remove node
            H.remove_node(v)
            del scores[v]
            
    elif method == "LEX":
        # LEX-BFS (Lexicographic Breadth-First Search)
        # Simplified version: prioritize by neighbor count and lexicographic order
        while H.nodes:
            # Sort by degree (ascending) then by node value
            candidates = list(H.nodes)
            candidates.sort(key=lambda x: (H.degree(x), x))
            v = candidates[0]
            order.append(v)
            H.remove_node(v)
            
    elif method == "MMD":
        # Minimum Degree (Min-Fill approximation)
        while H.nodes:
            # Select node with minimum degree
            v = min(H.nodes, key=lambda x: H.degree(x))
            order.append(v)
            H.remove_node(v)
            
    else:
        raise ValueError(f"Unknown heuristic method: {method}")
    
    return order


def induced_width(UG, order):
    """
    Calculate induced width (treewidth approximation) for given elimination order
    
    Args:
        UG: nx.Graph, undirected graph
        order: list, elimination order
    
    Returns:
        int: induced width (treewidth upper bound)
    """
    if len(order) == 0:
        return 0
    
    H = UG.copy()
    width = 0
    
    for v in order:
        if v not in H.nodes:
            continue
            
        # Get neighbors of v
        neighbors = list(H.neighbors(v))
        width = max(width, len(neighbors))
        
        # Fill-in: connect all pairs of neighbors
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                if not H.has_edge(neighbors[i], neighbors[j]):
                    H.add_edge(neighbors[i], neighbors[j])
        
        # Remove v
        H.remove_node(v)
    
    return width


def estimate_treewidth(G, method="MCS", num_trials=5):
    """
    Estimate treewidth using multiple heuristic trials
    
    Args:
        G: nx.DiGraph or nx.Graph, input graph
        method: str, elimination order heuristic
        num_trials: int, number of random trials
    
    Returns:
        dict: {
            'treewidth_estimate': best treewidth found,
            'best_order': best elimination order,
            'all_results': list of all trial results
        }
    """
    # Moralize if directed
    if isinstance(G, nx.DiGraph):
        UG = moralize(G)
    else:
        UG = G.copy()
    
    if len(UG.nodes) == 0:
        return {'treewidth_estimate': 0, 'best_order': [], 'all_results': []}
    
    best_width = float('inf')
    best_order = []
    all_results = []
    
    for trial in range(num_trials):
        # Add some randomization for multiple trials
        if trial > 0:
            # Randomly permute nodes with same degree
            nodes_by_degree = defaultdict(list)
            for node in UG.nodes:
                nodes_by_degree[UG.degree(node)].append(node)
            
            # Shuffle nodes within each degree class
            for degree_class in nodes_by_degree.values():
                np.random.shuffle(degree_class)
            
            # Create modified graph for this trial
            UG_trial = UG.copy()
        else:
            UG_trial = UG
        
        # Get elimination order
        order = heuristic_order(UG_trial, method)
        
        # Calculate induced width
        width = induced_width(UG, order)
        
        all_results.append({'order': order, 'width': width})
        
        if width < best_width:
            best_width = width
            best_order = order
    
    return {
        'treewidth_estimate': best_width,
        'best_order': best_order,
        'all_results': all_results
    }


def junction_tree_width(G):
    """
    Calculate exact treewidth using junction tree (for small graphs only)
    
    Args:
        G: nx.Graph, undirected graph
    
    Returns:
        int: exact treewidth (expensive for large graphs)
    """
    if len(G.nodes) <= 1:
        return 0
    
    # For small graphs, try all possible elimination orders (factorial complexity!)
    if len(G.nodes) > 8:
        # Fall back to heuristic for larger graphs
        result = estimate_treewidth(G, method="MCS", num_trials=10)
        return result['treewidth_estimate']
    
    from itertools import permutations
    
    best_width = float('inf')
    
    for order in permutations(G.nodes):
        width = induced_width(G, list(order))
        best_width = min(best_width, width)
    
    return best_width


def check_treewidth_bound(G, tw_bound):
    """
    Check if graph satisfies treewidth bound
    
    Args:
        G: nx.DiGraph or nx.Graph
        tw_bound: int, treewidth bound
    
    Returns:
        bool: True if treewidth <= tw_bound
    """
    result = estimate_treewidth(G, method="MCS", num_trials=3)
    return result['treewidth_estimate'] <= tw_bound


def get_elimination_tree(G, order):
    """
    Build elimination tree from elimination order
    
    Args:
        G: nx.Graph, undirected graph
        order: list, elimination order
    
    Returns:
        nx.DiGraph: elimination tree
    """
    H = G.copy()
    elim_tree = nx.DiGraph()
    elim_tree.add_nodes_from(order)
    
    for i, v in enumerate(order):
        if v not in H.nodes:
            continue
            
        neighbors = list(H.neighbors(v))
        
        # Fill-in edges
        for j in range(len(neighbors)):
            for k in range(j + 1, len(neighbors)):
                if not H.has_edge(neighbors[j], neighbors[k]):
                    H.add_edge(neighbors[j], neighbors[k])
        
        # Find parent in elimination tree (first neighbor in remaining order)
        remaining_neighbors = [n for n in neighbors if n in order[i+1:]]
        if remaining_neighbors:
            # Parent is the neighbor that appears first in remaining order
            parent_idx = min(order.index(n) for n in remaining_neighbors)
            parent = order[parent_idx]
            elim_tree.add_edge(parent, v)
        
        H.remove_node(v)
    
    return elim_tree


def decompose_graph(G, tw_bound=3):
    """
    Decompose graph into components respecting treewidth bound
    
    Args:
        G: nx.DiGraph or nx.Graph
        tw_bound: int, maximum allowed treewidth
    
    Returns:
        list: list of subgraphs, each with treewidth <= tw_bound
    """
    if isinstance(G, nx.DiGraph):
        UG = moralize(G)
    else:
        UG = G.copy()
    
    if check_treewidth_bound(UG, tw_bound):
        return [G]  # Graph already satisfies bound
    
    # Find good separator to split graph
    components = []
    
    # Simple approach: find articulation points and split
    if nx.is_connected(UG):
        articulation_points = list(nx.articulation_points(UG))
        
        if articulation_points:
            # Remove articulation point and split
            separator = articulation_points[0]
            UG_split = UG.copy()
            UG_split.remove_node(separator)
            
            # Get connected components
            for component_nodes in nx.connected_components(UG_split):
                # Add separator back to each component
                component_nodes = list(component_nodes) + [separator]
                subgraph = G.subgraph(component_nodes)
                
                # Recursively decompose if still too wide
                sub_components = decompose_graph(subgraph, tw_bound)
                components.extend(sub_components)
        else:
            # No articulation points, return original graph
            components = [G]
    else:
        # Graph already disconnected
        for component_nodes in nx.connected_components(UG):
            subgraph = G.subgraph(component_nodes)
            sub_components = decompose_graph(subgraph, tw_bound)
            components.extend(sub_components)
    
    return components if components else [G]
