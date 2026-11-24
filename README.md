# CSE 202 Algorithm Design and Analysis (UC San Diego, Fall 2025)

Selected work from CSE 202.

## Graph isomorphism

`graph-isomorphism/main.py`

Finds a vertex mapping between two isomorphic graphs given as edge lists, by computing canonical forms with pybliss and composing the two permutations, with a degree-signature heuristic as a fallback. Tested on a pair of graphs with 117,505 vertices and 205,061 edges. The input file is not included.

Usage: `python3 main.py graphs ans`

## Problem write-ups

`problem-writeups/`

Written solutions, with the idea, a correctness argument and the complexity, for five problems: Swim in Rising Water (LeetCode 778), Orderly Queue (899), Maximal Rectangle (85), Palindrome Partitioning II (132) and Find Median from Data Stream (295). These were written as group work.
