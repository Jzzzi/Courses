#include <iostream>
#include <queue>
#include <map>
#include <set>

using namespace std;

int main(){
    int n, d;
    cin >> n >> d;
    int *result = new int [n];
    // the number of all possible status of d dimensions
    int enum_num = 1 << d;
    // the data of all points
    int** data = new int* [n];
    for(int i = 0; i < n; i++){
        data[i] = new int [d];
    }
    // a large top pile
    multiset<int, greater<int>> *max_manhattan = new multiset<int, greater<int>> [enum_num];

    // the maximum Manhattan distance
    int max_manhattan_distance = 0;

    for(int i = 0; i < n; i++){
        int judge;
        cin >> judge;
        if(judge == 0){//Add a new point
            for(int j = 0; j < d; j++){
                cin >> data[i][j];
            }
            for(int j = 0; j < enum_num; j++){
                int sum = 0;
                for(int k = 0; k < d; k++){
                    int t = j & (1 << k);
                    if(t){
                        sum += data[i][k];
                    }
                    else{
                        sum -= data[i][k];
                    }
                }
                max_manhattan[j].insert(sum);
            }
        }
        else{//Delete a existed point
            int index;
            cin >> index;
            index--;
            for(int j = 0; j < enum_num; j++){
                int sum = 0;
                for(int k = 0; k < d; k++){
                    int t = j & (1 << k);
                    if(t){
                        sum += data[index][k];
                    }
                    else{
                        sum -= data[index][k];
                    }
                }
                max_manhattan[j].erase(max_manhattan[j].find(sum));
            }
        }
        // Calculate the Manhattan distance
        int max_manhattan_distance_now = 0;
        for(int j = 0; j < enum_num; j++){
            int manhattan_now = 0;
            if(max_manhattan[j].size() > 1){
                manhattan_now = *max_manhattan[j].begin() - *max_manhattan[j].rbegin();
            }
            if(manhattan_now > max_manhattan_distance_now){
                max_manhattan_distance_now = manhattan_now;
            }
        }
        result[i] = max_manhattan_distance_now;
    }
    // Output the result
    for(int i = 0; i < n; i++){
        cout << result[i] << endl;
    }
    return 0;
}