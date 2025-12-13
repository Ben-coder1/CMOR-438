# Majority Label Propagation (LPA)

A robust, efficient Python implementation of the Majority Label Propagation Algorithm. This tool is designed for graph-based semi-supervised learning and unsupervised community detection.

📌 **Overview**  
The core principle of this algorithm is simple: **"Do as your neighbors do."**  
Nodes in a network update their labels based on the majority label of their immediate neighbors. By iterating this process, labels spread across the graph until a consensus is reached, revealing the underlying structure of the network.

---

## Key Capabilities

**Semi-Supervised Classification:**  
Infer labels for a whole graph based on a few "seed" examples (e.g., predicting paper topics based on citations).

**Unsupervised Community Detection:**  
Discover natural clusters in a network without any prior knowledge or training data.

---

## ⚙️ How It Works

The algorithm follows an iterative, asynchronous voting process:

- **Initialization:** Nodes are assigned an initial state (either a known label, a unique ID, or None).  
- **Propagation:** In every iteration, the order of nodes is shuffled (Asynchronous Update).  
- **Voting:** Each node queries its neighbors. It adopts the label that appears most frequently.  
- **Tie-Breaking:** If there is a tie between top labels, one is chosen continuously at random.  
- **Convergence:** The process stops when labels cease changing or a maximum iteration limit is reached.

---

## 🛠 Functionality & Modes

This specific implementation is controlled primarily by the `freeze_seeds` parameter, which toggles between the two main use cases.

### **Mode 1: Semi-Supervised Learning**

- **Goal:** You have a small set of labeled nodes (seeds) and want to classify the remaining unlabeled nodes.  
- **Configuration:** `freeze_seeds=True`.  
- **Behavior:** The algorithm locks the seed nodes so they never change their ground truth. Their influence radiates outward, filling in the `None` values of their neighbors.  
- **Use Case:** Classification of documents, fraud detection (guilt by association), interest prediction.

---

### **Mode 2: Unsupervised Community Detection**

- **Goal:** You have no labels and want to find natural groupings (clusters).  
- **Configuration:** `freeze_seeds=False` and initialize every node with a unique ID (e.g., its own node index).  
- **Behavior:** Every node competes. Over time, small unique labels die out as they are swallowed by larger, denser neighbor groups. The distinct labels that survive represent the discovered communities.  
- **Use Case:** Social network analysis, biological pathway discovery, organizing citation networks.

---

## 📝 API Reference

```python
majority_label_propagation(G, label_attr='label', max_iter=100, freeze_seeds=True)
```


## Parameters

| Parameter      | Type      | Description |
|----------------|-----------|-------------|
| `G`            | `nx.Graph` | The input NetworkX graph. |
| `label_attr`   | `str`      | The key in `G.nodes[n]` where the label is stored. |
| `max_iter`     | `int`      | Safety limit for iterations (prevents infinite loops). |
| `freeze_seeds` | `bool`     | True: Seeds are immutable (Semi-Supervised). False: All nodes update (Community Detection). |

---

## Robustness Features

- **Input Validation:** Ensures input is a valid NetworkX graph object.  
- **Graph Checks:** Includes an internal helper `_check_graph_simplicity` that issues warnings (without crashing) if the graph contains self-loops or multi-edges, which can skew voting results.  
- **Asynchronous Processing:** Nodes are shuffled every iteration to prevent the “oscillation” problem common in synchronous updates.

---

## ⚖️ Strengths & Limitations

### ✅ Where It Works Well

- **Scalability:** Near-linear time complexity \( O(m) \), making it one of the fastest options for massive graphs (millions of nodes).  
- **No Hyperparameters:** Unlike K-Means, you do not need to specify \( K \) (the number of communities) beforehand. The algorithm finds \( K \) automatically.  
- **Simplicity:** Requires no complex matrix factorization or training phases.

---

### ⚠️ Situations It Struggles In

- **The "Monster" Community:** A single massive community can consume smaller but valid communities if connections are too dense.  
- **Reproducibility:** Tie-breaking randomness and shuffled update order can lead to slightly different borders between clusters in different runs.  
- **Weak Connections:** If the graph is sparse or disconnected, labels cannot propagate to isolated components.