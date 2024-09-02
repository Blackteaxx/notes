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

**Maintain a global clock** for 2 events (first discovered and last departured) in the DFS.

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

    # Previsit
    pre[v] = clock
    clock += 1

    for u in G[v]:
        if not visited[u]:
            Explore(G, u)

    # Postvisit
    post[v] = clock
    clock += 1
```

_Property: for any node $u, v \in V$, the interval $[pre(u), post(u)]$, $[pre(v), post(v)]$ are either separate or one is contained within the other$_

- if $(u,v) \in E$, then $pre(u) < pre(v) < post(v) < post(u)$

## Connectivity in directed graphs

Huangyu P77, defined on the color of vertices.

- Tree edge: $(u, v)$ is a tree edge if $v$ is the first time visited by `Explore(G, u)`
- Back edge: $(u, v)$ is a back edge if $v$ is an ancestor of $u$ in the DFS tree
- Forward edge: $(u, v)$ is a forward edge if $v$ is a descendant of $u$ but not a tree edge
- Cross edge: $(u, v)$ is a cross edge if $u$ and $v$ are not ancestor of each other

## DAG

**D**irected **A**cyclic **G**raph is **a digraph with no cycles.**

- **Question: How to check if a digraph is a DAG?**

Huangyu P77

- Back edge implies a cycle in a digraph

```python
def IsAcyclic(G):
    Run DFS on G
    output False if there is a back edge
```

- We can review the definition of back edge: $(u, v)$ is a back edge if $v$ is an ancestor of $u$ in the DFS tree.
- and in the intervel

  - v:[ u:[ ] ] is a back edge, it shows that the u is the successor of v, but $(u, v) \in E$, which means there is a cycle in the graph.

- So we can propose that DAG **is equal to** there being no back edge in the graph. We should prove it.

**Claim: A digraph is a DAG if and only if there is no back edge in the DFS tree.**

> Proof:

- If there is a back edge, then there is a cycle in the graph, so it is not a DAG.
- If there is a cycle in the graph, let $v_1, v_2, \cdots, v_t$ be a cycle. Let **$v_i$ be the first vertex visited** by `Explore(G, v_1)` in the cycle. All the successors of $v_i$, maybe tree edges, back edges, forward edges, or cross edges. But there must be a node $v_j$ that is an ancestor of $v_i$(otherwise, $v_i$ is not the first vertex visited by `Explore(G, v_1)`). So $(v_j, v_i)$ is a back edge in the graph.

> end proof

## Topological ordering

Given G that is DAG, want to order the vertices(of index 0 to n-1) such that if $(u, v) \in E$, then $u$ comes before $v$ in the ordering.

```python
def InverseTopologicalOrdering(G):
    Run DFS on G
    output the vertices in post order(or post in interval)
```

It depends on the following lemma:

**Lemma: $(u,v) \in E \to post(u) > post(v)$**

**Property: The DAG has at least a source vertex and at least a sink vertex**

_Mark: The smallest post number comes last in the order, so it is the sink vertex. And conversely, the biggest post vertex is the source vertex._

## Connectivity for digraphs

**def: $u$ and $v$ are **strongly connected** if there is a path from $u$ to $v$ and a path from $v$ to $u$.**

And the strongly connected components(SCCs) is the equivalence relation of the strongly connected vertices.

**claim: every diagraph G is a DAG on the SCCs**

_it means if we shrink every SCC into a super vertex, then the graph is a DAG._

- We need to recyle something about DFS
  to find the SCCs.
- If we get into one **sink SCC**(that means the super vertex is the end), then we can get out of it.
- The critical point is that we can reverse the graph, and **the sink SCCs will become the source SCCs**.

_**Side Qusetion**: Given degraph G, find any v in Source SCC_

- The Last vertex in the post order of the graph must be in the source SCC.

Emphasis: So we can get the **algothrim** to find the SCCs:

```python
def SCC(G):
    Run DFS on G in post order, get Path p in a stack
    Reverse G_R
    each vertex in p.pop():
        if not visited[v]:
            Explore(G_R, v)
            output the SCC
```

Hints: Reverse is a common operation in graph algorithms to augment the graph.

## Shortest paths

### Shortest Path on graphs that are unweighted and undirected

_Q: Find distance of A to all other vertices_

- BFS, use a queue to store the vertices to be visited.

```python
def BFS(G, s):
    dist = [float.('inf')] * len(G)
    dist[s] = 0
    q = [s]
    while q:
        v = q.pop(0)
        for u in G[v]:
            if dist[u] == inf:
                dist[u] = dist[v] + 1
                q.append(u)
```

- The runtime of BFS is $O(n+m)$, like DFS.

### Shortest Path with distances

- Input: Graph G, source vertex s, lengths $l: E \to \mathbb{N}^+$
- Output: dist$[v]$ = dist$(s,v)$

_First Idea: Use BFS again_

- Split the Weighted Graph into Unweighted Graphs(egde to the unit weight)
- But the runtime is not polynomial.

  - if the weight $w \leq 2^a$, then the number of edges is $O(n2^a)$

So we need to find a better algorithm.

### Dijkstra

**State**

- estimates: There are some estimates of the shortest path from s to v, and we want to update the estimates when the program is going.
- Subset: $K = \{v | dist[v] = dist(s,v)\}$

**Invariant**

**Base Case**:

$K = \{s\}$

**Inductive step**:

All shortest path construct a tree(not proved).

And there are some vertices that are connected to the tree and not in the tree.

We want to enlarge the tree by adding the vertex with the smallest estimate.

- How to extend K(the tree)?

1. We want to prove that **$one$ shortest path is in one hop away from $K$.**

We Assume that $a$ is the vertex that is not in $K$ and is closest to $s$.

**Claim**: $a$ is $1$-hop away from $K$

![img](https://img2023.cnblogs.com/blog/3436855/202408/3436855-20240816221343217-1351458156.png)

> Proof: Suppose not, let $b$ be the first vertex on the shortest path from $s$ to $a$ that is not in $K$. So $b$ is not in $K$, and $b$ is the closest vertex to $s$ that is not in $K$. So $b$ is closer to $s$ than $a$. But $a$ is the closest vertex to $s$ that is not in $K$, so it is a contradiction.

So the algorithm is as follows:

```python
def Dijkstra(G, l, s):
    dist = [float('inf')] * len(G)
    dist[s] = 0
    K = {}
    while len(K) < len(G):
        pick the vertex v not in K with the smallest dist[v]
        K.add(v)
        for u in G[v]:
            dist[u] = min(dist[u], dist[v] + l[v][u])
```

**Runtime**:

- $O(n)$ to initialize dist
- $O(n)$ to get the minimal vertex
- $O(m)$ to update the dist

All the runtime is $O((n+m) \cdot \log n)$, by using the priority queue.

### The case of negative weights - Bellman-ford

The problem is not well-defined when it comes to negative cycles.

So we just consider the graph without negative cycles.

The claim in the positive weights is not true in the negative weights, there may not be a shortest path in one hop away from $K$.

We let `update(u, v) = min(dist[v], dist[u] + l[u][v])`

1. the update function is safe: $dist(s,v) \leq min(dist[v], dist[u] + l[u][v])$

2. if $s \to \cdots \to u \to v$ is the shortest path and the dist[u] is correct, then the `update` makes the dist[v] correct.

So if we update the $dist[v]$ in the order of the edges in the path, then the **$dist[v]$ is correct**.

In the basic case, the egde $(s,u) \in E$, we can immediately find the correct $dist[u]$.

And in the inductive step, we can update the $dist[v]$ in the order of the edges in the path.

So we can iterate $| V | - 1$ times to get the correct dist.(if there is no negative cycle)

```python
def BellmanFord(G, l, s):
    dist = [float('inf')] * len(G)
    dist[s] = 0
    for i in range(len(G) - 1):
        for e in G:
            u, v = e
            dist[v] = min(dist[v], dist[u] + l[u][v])
```

**Runtime**: $O(|V||E|)$
