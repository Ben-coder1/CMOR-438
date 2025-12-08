import networkx as nx
import random
import warnings
from collections import Counter

def _check_graph_simplicity(G):
    """
    Internal helper function to check if the graph is 'simple'.
    A simple graph has no self-loops and no multiple edges.
    
    This function is not intended to be exported (indicated by the _ prefix).

    Examples:
        >>> import networkx as nx
        >>> # 1. Valid Simple Graph
        >>> G = nx.Graph([(1, 2)])
        >>> _check_graph_simplicity(G)
        True

        >>> # 2. Graph with Self-Loop (Triggers Warning)
        >>> # Note: The warning is printed to stderr, so doctest won't see it.
        >>> # We only check that it returns False.
        >>> G_loop = nx.Graph()
        >>> G_loop.add_edge(1, 1)
        >>> _check_graph_simplicity(G_loop)
        False
    """
    # Check for self-loops
    if nx.number_of_selfloops(G) > 0:
        warnings.warn(
            "Graph contains self-loops. Label propagation may behave unpredictably.", 
            UserWarning
        )
        return False

    # Check for multi-edges (NetworkX MultiGraph or MultiDiGraph)
    if G.is_multigraph():
        warnings.warn(
            "Graph is a MultiGraph (allows parallel edges). "
            "Majority voting counts may be skewed by multiple edges.", 
            UserWarning
        )
        return False
        
    return True

def majority_label_propagation(G, label_attr='label', max_iter=100, freeze_seeds=True):
    """
    Runs Majority Label Propagation with robust error handling and validation.
    
    Args:
        G (nx.Graph): The input networkx graph.
        label_attr (str): The node attribute name holding the label.
        max_iter (int): Maximum number of iterations.
        freeze_seeds (bool): If True, pre-labeled nodes are immutable.
    
    Returns:
        nx.Graph: The graph with updated labels.
        
    Raises:
        TypeError: If G is not a valid NetworkX graph.
        ValueError: If max_iter is invalid.

    Examples:
        >>> import networkx as nx
        >>> # Setup a graph: 1(Red) -- 2(Unlabeled) -- 3(Red)
        >>> G = nx.Graph()
        >>> G.add_edges_from([(1, 2), (2, 3)])
        >>> G.nodes[1]['label'] = 'Red'
        >>> G.nodes[3]['label'] = 'Red'
        >>> 
        >>> # Run propagation
        >>> # Iteration 1 updates Node 2. Iteration 2 confirms no more changes.
        >>> G = majority_label_propagation(G, label_attr='label')
        Converged at iteration 2
        >>> 
        >>> # Node 2 should become Red (majority neighbor vote)
        >>> G.nodes[2]['label']
        'Red'
    """
    
    # --- Input Validation & Error Handling ---
    try:
        # 1. Validate Graph Object
        if not isinstance(G, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
            raise TypeError(f"Input must be a NetworkX Graph object, got {type(G)}.")
        
        # 2. Check for Simple Graph (Internal Helper)
        # We don't stop execution, but we warn the user.
        _check_graph_simplicity(G)

        # 3. Validate Parameters
        if max_iter < 1:
            raise ValueError("max_iter must be a positive integer.")
            
    except Exception as e:
        # Re-raise known errors, wrap unknown ones if strictly necessary, 
        # but here we mostly want to bubble up the specific validation errors.
        print(f"Initialization Error: {e}")
        raise e

    # --- Algorithm Execution ---
    try:
        # Identify fixed nodes
        fixed_nodes = set()
        if freeze_seeds:
            # Safe iteration in case node data is malformed
            fixed_nodes = {n for n, d in G.nodes(data=True) if label_attr in d}
        
        for i in range(max_iter):
            nodes = list(G.nodes())
            random.shuffle(nodes) 
            
            change_count = 0
            
            for node in nodes:
                if node in fixed_nodes:
                    continue
                
                # Retrieve neighbors safely
                neighbors = list(G.neighbors(node))
                if not neighbors:
                    continue
                
                # Collect valid neighbor labels
                neighbor_labels = []
                for n in neighbors:
                    val = G.nodes[n].get(label_attr)
                    if val is not None:
                        neighbor_labels.append(val)
                
                if not neighbor_labels:
                    continue
                
                # Majority Voting Logic
                label_counts = Counter(neighbor_labels)
                max_freq = max(label_counts.values())
                candidates = [l for l, count in label_counts.items() if count == max_freq]
                
                # Tie-Breaking
                new_label = random.choice(candidates)
                
                # Update
                current_label = G.nodes[node].get(label_attr)
                if current_label != new_label:
                    G.nodes[node][label_attr] = new_label
                    change_count += 1
            
            # Check Convergence
            if change_count == 0:
                print(f"Converged at iteration {i+1}")
                break
                
    except Exception as e:
        # Catch runtime errors during the loop (e.g. memory issues, interruption)
        print(f"Runtime Error during propagation: {e}")
        return None

    return G