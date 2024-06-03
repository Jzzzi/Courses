#include <iostream>
#include <queue>
using namespace std;

struct SearchNode{
    int x, y;
    int cost;
    int heuristic;
    SearchNode(int x, int y, int cost, int heuristic): x(x), y(y), cost(cost), heuristic(heuristic){}
    bool operator<(const SearchNode& other) const{
        return cost + heuristic > other.cost + other.heuristic;
    }
    bool operator>(const SearchNode& other) const{
        return cost + heuristic < other.cost + other.heuristic;
    }
};

// index = coordinate + 100
// coordinate = index - 100
bool valid[200][200];

int heuristic(int x, int y, int x_final, int y_final){
    return abs((x - x_final) + abs(y - y_final))/3;
    // return 0;
}

int AstarSearch(int x_final, int y_final, bool valid[200][200]){
    // priority queue
    priority_queue<SearchNode> pq;
    int x_start = 0;
    int y_start = 0;
    SearchNode start(x_start, y_start, 0, heuristic(x_start, y_start, x_final, y_final));
    pq.push(start);
    while(!pq.empty()){
        SearchNode current = pq.top();
        pq.pop();
        // set the valid position to false
        valid[current.x+100][current.y+100] = false;
        if(current.x == x_final && current.y == y_final){
            return current.cost;
        }
        // 8 possible moves
        int dx[8] = {1, 2, 2, 1, -1, -2, -2, -1};
        int dy[8] = {2, 1, -1, -2, -2, -1, 1, 2};
        for(int i = 0; i < 8; i++){
            int x_new = current.x + dx[i];
            int y_new = current.y + dy[i];
            if(x_new >= -100 && x_new <= 100 && y_new >= -100 && y_new <= 100 && valid[x_new+100][y_new+100]){
                SearchNode next(x_new, y_new, current.cost + 1, heuristic(x_new, y_new, x_final, y_final));
                pq.push(next);
            }
        }
    }
    return -1;
}


int main(){
    int x_final, y_final;
    cin >> x_final >> y_final;
    int n;
    cin >> n;
    for(int i = 0; i < 200; i++){
        for(int j = 0; j < 200; j++){
            valid[i][j] = true;
        }
    }
    for(int i = 0; i < n; i++){
        int x, y;
        cin >> x >> y;
        valid[x+100][y+100] = false;
    }
    int minStep = AstarSearch(x_final, y_final, valid);
    if (minStep == -1){
        cout << "fail" << endl;
        return 0;
    }
    cout << minStep << endl;
    return 0;
}