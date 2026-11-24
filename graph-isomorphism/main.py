#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
solve_with_pybliss.py
Usage:
    python3 solve_with_pybliss.py graphs ans

Produces file 'ans' with n lines, i-th line is f(i) (1-based).
Requires: pybliss, numpy
"""

import sys
import numpy as np

def read_graphs_file(path):
    with open(path, 'r') as f:
        tokens = []
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            tokens.extend(line.split())
    if len(tokens) < 2:
        raise ValueError("输入文件格式错误：没有 n m")
    it = iter(tokens)
    n = int(next(it))
    m = int(next(it))
    expected = 2 + 2*m + 2*m
    if len(tokens) < expected:
        raise ValueError(f"输入文件太短：预期至少 {expected} 个数字，只有 {len(tokens)}")
    E1 = []
    for _ in range(m):
        u = int(next(it)) - 1
        v = int(next(it)) - 1
        E1.append((u, v))
    E2 = []
    for _ in range(m):
        u = int(next(it)) - 1
        v = int(next(it)) - 1
        E2.append((u, v))
    return n, m, E1, E2

def canonical_mapping_using_pybliss(n, E1, E2, timeout_nodes=None):
    import pybliss
    stats1 = pybliss.Stats()
    stats2 = pybliss.Stats()

    # Build graphs
    G = pybliss.Graph(n)
    Gp = pybliss.Graph(n)
    for (u,v) in E1:
        if u==v: continue
        G.add_edge(u, v)
    for (u,v) in E2:
        if u==v: continue
        Gp.add_edge(u, v)

    # Try to get permutations to canonical form.
    # Optionally, one could tune splitting heuristics or set component recursion.
    try:
        P1 = G.get_permutation_to_canonical_form(stats1)
        P2 = Gp.get_permutation_to_canonical_form(stats2)
        # P1 and P2 are numpy arrays length n, mapping old indices -> canonical indices.
        # To check if canonical graphs equal, we permute edges and compare.
        E1_np = np.array(E1, dtype=np.int64)
        E2_np = np.array(E2, dtype=np.int64)

        # Build canonical adjacency strings for both
        def edges_to_adj_signature(edges, perm):
            # perm maps original->canonical index
            mapped = [(int(perm[u]), int(perm[v])) for (u,v) in edges]
            # normalize undirected edges
            mapped_norm = [tuple(sorted(x)) for x in mapped]
            mapped_norm = sorted(mapped_norm)
            return tuple(mapped_norm)

        sig1 = edges_to_adj_signature(E1, P1)
        sig2 = edges_to_adj_signature(E2, P2)
        if sig1 == sig2:
            # graphs have identical canonical representation -> isomorphic
            # We want f mapping vertices of G -> vertices of G'.
            # P1: i (old index in G) -> c (canonical index)
            # P2: j (old index in G') -> c (canonical index)
            # So P2^{-1} maps canonical -> vertex in G'.
            invP2 = np.empty_like(P2)
            for i in range(len(P2)):
                invP2[int(P2[i])] = i
            # f = P2^{-1} o P1
            f = np.empty_like(P1)
            for i in range(len(P1)):
                f[i] = int(invP2[int(P1[i])])
            return True, f, stats1, stats2
        else:
            # canonical forms differ -> not isomorphic or different Bliss config
            return False, None, stats1, stats2
    except Exception as e:
        # pybliss可能抛出异常（极端情况下），返回失败
        return False, None, None, None

def heuristic_mapping_by_degree(n, E1, E2):
    # fallback: match vertices by (degree, second-order degree) signatures
    deg1 = [0]*n
    deg2 = [0]*n
    adj1 = [[] for _ in range(n)]
    adj2 = [[] for _ in range(n)]
    for (u,v) in E1:
        deg1[u]+=1; deg1[v]+=1
        adj1[u].append(v); adj1[v].append(u)
    for (u,v) in E2:
        deg2[u]+=1; deg2[v]+=1
        adj2[u].append(v); adj2[v].append(u)

    # Compute secondary features: sorted neighbor degrees
    neighsig1 = []
    neighsig2 = []
    for i in range(n):
        neighsig1.append(tuple(sorted([deg1[w] for w in adj1[i]])))
        neighsig2.append(tuple(sorted([deg2[w] for w in adj2[i]])))
    # create stable sort keys
    keys1 = [(deg1[i], neighsig1[i], i) for i in range(n)]
    keys2 = [(deg2[i], neighsig2[i], i) for i in range(n)]
    keys1_sorted = sorted(keys1)  # stable
    keys2_sorted = sorted(keys2)
    # Now map position-wise; when tie-multisets exist, this may be imperfect.
    f = [-1]*n
    for pos in range(n):
        f[keys1_sorted[pos][2]] = keys2_sorted[pos][2]
    return np.array(f, dtype=np.uint32)

def compute_match_score(n, E1, E2, f):
    # E2 as set of normalized edges
    E2set = set(tuple(sorted(e)) for e in E2)
    matched = 0
    for (u,v) in E1:
        a = int(f[u]); b = int(f[v])
        if tuple(sorted((a,b))) in E2set:
            matched += 1
    m = len(E1)
    x = matched / m if m>0 else 0.0
    score = 5.333 * x**3 - 4 * x**2 + 2.667 * x
    return matched, x, score

def write_ans_file(ans_path, f):
    # write 1-based mapping
    with open(ans_path, 'w') as fo:
        for i in range(len(f)):
            fo.write(str(int(f[i])+1) + "\n")

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 solve_with_pybliss.py graphs ans")
        sys.exit(1)
    input_path = sys.argv[1]
    ans_path = sys.argv[2]
    n, m, E1, E2 = read_graphs_file(input_path)
    print(f"Read n={n}, m={m}")

    # Try pybliss canonical mapping
    try:
        iso, f, stats1, stats2 = canonical_mapping_using_pybliss(n, E1, E2)
    except Exception as e:
        iso = False
        f = None
        stats1 = stats2 = None
        print("调用 pybliss 出现异常：", e)

    if iso and f is not None:
        print("Bliss 返回了同构映射（精确同构）")
        matched, x, score = compute_match_score(n, E1, E2, f)
        print(f"Matched edges: {matched}/{m}, x={x:.6f}, score={score:.6f}")
        write_ans_file(ans_path, f)
        print(f"结果已写入 {ans_path}")
        # Optionally print stats
        if stats1 is not None and stats2 is not None:
            print("Bliss stats for G:", stats1)
            print("Bliss stats for G':", stats2)
        return

    # else fallback heuristic
    print("未能通过 Bliss 得到相同的 canonical（可能非同构或配置不同）。使用启发式退化匹配。")
    f = heuristic_mapping_by_degree(n, E1, E2)
    matched, x, score = compute_match_score(n, E1, E2, f)
    print(f"Heuristic matched edges: {matched}/{m}, x={x:.6f}, score={score:.6f}")
    write_ans_file(ans_path, f)
    print(f"启发式结果已写入 {ans_path}")
    print("提示：若需提升结果，可以：1) 为 pybliss 设置不同的 splitting heuristic；2) 给顶点上颜色（例如基于度或顶点特征）再 canonicalize；3) 组合局部子图同构/最大重合子图方法。")

if __name__ == '__main__':
    main()
