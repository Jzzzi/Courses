#include <iostream>
#include <vector>
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
    return 0;
}