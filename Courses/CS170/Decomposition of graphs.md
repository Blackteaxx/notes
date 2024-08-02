Graph contains a pair of 2 sets

$$
G = (V, E) \\
E \subseteq V \times V \\
n = |V|, m = |E|
$$

- $V$ is the set of vertices
- $E$ is the set of edges, which are pairs of vertices
- Undirected graph: $(u, v) \in E \iff (v, u) \in E$
- directed

The Representation of Graphs, including 2 canconical ways:

- Adjacency matrix: $A_{ij} = 1 \iff (i, j) \in E$
- Adjacency list: $A[i] = \{j | (i, j) \in E\}$

## DFS

The template of Explore function is as follows:

```python
def Explore(G, v):
    visited[v] = True
    for u in G[v]:
        if not visited[u]:
            Explore(G, u)
```

> claim: `Explore(G, v)` visits all vertices reachable from $v$

> proof: Suppose not, let $a$ is a vertex reachable from $v$ but not visited by `Explore(G, v)`.
> It means that there is a path from $v$ to $a$. Let $b$ be **the first vertex** **on this path** that is not visited by `Explore(G, v)`. So it exists a vertex $u$ on the path such that $u$ is visited by `Explore(G, v)` and $(u, b) \in E$. But $b$ is not visited by `Explore(G, v)`, so it is not visited by `Explore(G, u)`.
> It contradicts the assumption that $b$ is the first vertex on the path that is not visited by `Explore(G, v)`.

The template of DFS is as follows:

```python
def DFS(G):
    visited = [False] * len(G)
    for v in range(len(G)):
        if not visited[v]:
            Explore(G, v)
```

### Runtime of DFS

We would better use counting to analyze the runtime of DFS other than using recurrence relation.

- $O(n)$ to initialize `visited`
- $O(n)$ to iterate over all vertices
- $O(m)$ to iterate over all edges

## Connectivity in undirected graphs

It is a equivalence relation, and in this problem, we want to get all the connected components.

```python
def number_of_components(G):
    visited = [False] * len(G)
    cc = [i for i in range(len(G))]
    components = 0
    for v in range(len(G)):
        if not visited[v]:
            Explore(G, v, components)
            components += 1

def Explore(G, v, components):
    visited[v] = True
    cc[v] = components
    for u in G[v]:
        if not visited[u]:
            Explore(G, u, components)
```

### The interval in DFS

**Maintain a global clock** for "time on stack" and "time off stack".

```python
def DFS(G):
    visited = [False] * len(G)
    pre = [0] * len(G)
    post = [0] * len(G)
    clock = 0
    for v in range(len(G)):
        if not visited[v]:
            Explore(G, v)

def Explore(G, v):
    visited[v] = True
    pre[v] = clock
    clock += 1
    for u in G[v]:
        if not visited[u]:
            Explore(G, u)
    post[v] = clock
    clock += 1
```

- if $(u,v) \in E$, then $pre(u) < pre(v) < post(v) < post(u)$

## Connectivity in directed graphs

- Tree edge: $(u, v)$ is a tree edge if $v$ is the first time visited by `Explore(G, u)`
- Back edge: $(u, v)$ is a back edge if $v$ is an ancestor of $u$ in the DFS tree
- Forward edge: $(u, v)$ is a forward edge if $v$ is a descendant of $u$ but not a tree edge
- Cross edge: $(u, v)$ is a cross edge if $u$ and $v$ are not ancestor of each other
