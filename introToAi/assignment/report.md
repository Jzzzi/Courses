# Assigment Report

刘锦坤 2022013352 行健-烽火2班

---

- [Assigment Report](#assigment-report)
  - [Finding a Fixed Food Dot](#finding-a-fixed-food-dot)
    - [深度优先搜索策略（DFS）](#深度优先搜索策略dfs)
    - [广度优先搜索策略（BFS）](#广度优先搜索策略bfs)
    - [一致代价搜索策略（UCS）](#一致代价搜索策略ucs)
    - [A\* with nullHeuristic策略](#a-with-nullheuristic策略)
    - [A\* with manhattanHeuristic策略](#a-with-manhattanheuristic策略)
  - [Finding All the Corners](#finding-all-the-corners)
    - [CornersProblem Search Problem](#cornersproblem-search-problem)

## Finding a Fixed Food Dot

这一部分的任务是给定起始位置、地图形状、以及目标食物位置，找寻到从起始位置前往目标食物位置的路径。算法采用图搜索的方式，通过记录已探索节点避免对已探索节点的重复搜索，在搜索策略上分别应用深度优先搜索策略（DFS）、广度优先搜索策略（BFS）、一致代价搜索策略（UCS）、和A*搜索策略进行搜索。在mediumMaze中，各搜索策略的结果如下表所示：

|         Strategy         | DFS | BFS | UCS | A* with<br /> nullHeuristic | A* with<br /> manhattanHeuristic |
| :----------------------: | :-: | :-: | :-: | :-------------------------: | :------------------------------: |
| **Nodes Expanded** | 144 | 267 | 267 |             267             |               221               |
|   **Total Cost**   | 130 | 68 | 68 |             68             |                68                |
|     **Score**     | 380 | 442 | 442 |             442             |               442             |

<div style="text-align: center; font-size: 12px;">
各搜索策略在mediumMaze中表现
</div>

### 深度优先搜索策略（DFS）

深度优先搜索策略优先向节点的后继节点搜索，空间复杂度较小，但是不能保证解的最优性，在迷宫寻径问题中，由于可以到达终点的路径较多，所以在本问题中深度优先搜索策略以最小的展开节点数114完成了路径搜索，但是由于深度优先搜索并不保证搜索结果的最优性，所以可以看到深度优先搜索策略所得结果的总花费为130，高于其他搜索策略搜索到的最优总花费68。

### 广度优先搜索策略（BFS）

广度优先搜索策略优先搜索深度较小的这些节点，因为在本问题中路径的花费事实上等价于对应节点的深度，所以在本问题中广度优先搜索策略可以保证结果的最优性，所以在最后结果中，广度优先搜索通过展开267个节点搜寻到了总花费为68的最优路径。

### 一致代价搜索策略（UCS）

一直代价搜索策略类似于广度优先搜索策略，但是优先展开待搜索节点中目前花费最小的那些节点，这个特性可以保证解的最优性。如之前所说，由于本问题中路径的花费等价于节点的深度，所以可以UCS和BFS在这个问题中是完全等价的，展开267个节点最终找到总花费为68的最优路径。

### A\* with nullHeuristic策略

不同于之前的搜索策略，A*搜索属于informed search的一种类型，由于额外信息的引入使得搜索更有方向性，在A\* with nullHeuristic搜索策略中，由于启发式函数（heuristic）使用的是平庸（trivial）的$h(n)=1$，根据A\*搜索的展开顺序$f(n)=g(n)+h(n)=g(n)+1$可以看出，采用平凡的启发式函数的A\*搜索就等价于一致代价搜索，在本问题中也等价于广度优先搜索策略，均展开267个节点找到总花费为68的最优路径。

### A\* with manhattanHeuristic策略

这里引入曼哈顿距离作为启发式函数，容易看出，这样的启发式函数有admissible和consistent两个性质，因此在图搜索形式的A*搜索可以保证解的最优性，又由于额外信息的引入有效的减少了展开节点的个数，所以A\* with manhattanHeuristic仅仅展开了221个节点（优于其他最优性算法267个节点）找到了总花费为68的最优路径。

## Finding All the Corners

在CornersProblem里，地图的四个角都有食物，需要寻找路径吃掉四个角所有的食物。

### CornersProblem Search Problem