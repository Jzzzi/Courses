#include <iostream>
#include <queue>
using namespace std;

int main(){
    int x_final, y_final;
    cin >> x_final >> y_final;
    int n;
    cin >> n;
    bool valid[200][200];
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
    
}