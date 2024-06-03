#include <iostream>
#include <vector>
#include <stack>

bool hasCycle(std::vector<std::vector<int>> adjTable, int numNode, int* inDegree, int* outDegree){
    std::stack<int> toBeVisited;
    bool *visited = new bool[numNode];
    for(int i = 0; i < numNode; i++){
        visited[i] = false;
    }
    for(int i = 0; i < numNode; i++){
        if(inDegree[i] == 0){
            toBeVisited.push(i);
        }
    }
    while(!toBeVisited.empty()){
        int current = toBeVisited.top();
        toBeVisited.pop();
        visited[current] = true;
        for(int i = 0; i < adjTable[current].size(); i++){
            int next = adjTable[current][i];
            inDegree[next]--;
            if(inDegree[next] == 0){
                toBeVisited.push(next);
            }
        }
    }
    for(int i = 0; i < numNode; i++){
        if(!visited[i]){
            return true;
        }
    }
    return false;
}

bool isAncestor(int u, int v, std::vector<std::vector<int>> adjTable){
    // check if u is an ancestor of v
    std::stack<int> toBeVisited;
    toBeVisited.push(u);
    while(!toBeVisited.empty()){
        int current = toBeVisited.top();
        toBeVisited.pop();
        for(int i = 0; i < adjTable[current].size(); i++){
            int next = adjTable[current][i];
            if(next == v){
                return true;
            }
            toBeVisited.push(next);
        }
    }
    return false;

}

int main(){
    int numNode, numEdge, numQuery;
    std::cin >> numNode >> numEdge >> numQuery;
    std::vector<std::vector<int>> adjTable(numNode);
    int *inDegree = new int[numNode];
    int *outDegree = new int[numNode];
    for(int i = 0; i < numEdge; i++){
        int from, to;
        std::cin >> from >> to;
        outDegree[from]++;
        inDegree[to]++;
        adjTable[from].push_back(to);
    }
    if (hasCycle(adjTable, numNode, inDegree, outDegree)){
        std::cout << -1 << std::endl;
        return 0;
    }

    // the query part
    int *results = new int [numQuery];
    for(int i = 0; i < numQuery; i++){
        int u, v;
        std::cin >> u >> v;
        if(isAncestor(u, v, adjTable)){
            results[i] = 1;
        }
        else if(isAncestor(v, u, adjTable)){
            results[i] = -1;
        }
        else{
            results[i] = 0;
        }
    }
    for(int i = 0; i < numQuery; i++){
        std::cout << results[i] << std::endl;
    }
    return 0;
}