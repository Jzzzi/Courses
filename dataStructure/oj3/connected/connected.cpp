#include <iostream>
#include <vector>
using namespace std;
struct Position
{
    int x;
    int y;
};

int depthFirstSearch(vector<vector<int>> &figure, vector<vector<bool>> &Explored, int x, int y){
    int rows = figure.size();
    int cols = figure[0].size();
    if(x < 0 || x >= rows || y < 0 || y >= cols || figure[x][y] == 0){
        return 0;
    }
    if(Explored[x][y]){
        return 0;
    }
    Explored[x][y] = true;
    int count = 1;
    count += depthFirstSearch(figure, Explored, x - 1, y);
    count += depthFirstSearch(figure, Explored, x + 1, y);
    count += depthFirstSearch(figure, Explored, x, y - 1);
    count += depthFirstSearch(figure, Explored, x, y + 1);
    return count;
}

int main(){
    int rows, cols;
    cin >> rows >> cols;
    // input the figure
    vector<vector<int>> figure(rows, vector<int>(cols));
    char c;
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            cin >> c;
            figure[i][j] = c - '0';
        }
    }
    // find the maximun connected area
    int max = 0;
    vector<vector<bool>> Explored(rows, vector<bool>(cols, false));
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            int count = depthFirstSearch(figure, Explored, i, j);
            if(count > max){
                max = count;
            }
        }
    }
    cout << max << endl;
    return 0;
}
