#include <iostream>
using namespace std;

// TO DO: Implement the drawTree function
void drawTree(int* parents, string* names, int N, int parent, string prefix, bool HasBrother = false){ 
    HasBrother = false;
    for(int i = 0; i < N; i++){
        if(parents[i] == parent){
            cout << prefix;
            cout << "+-" << names[i] << endl;
            for(int j = i + 1; j < N; j++){
                if(parents[j] == parent){
                    HasBrother = true;
                    break;
                }
                HasBrother = false;
            }
            drawTree(parents, names, N, i, prefix + (HasBrother ? "| " : "  "), false);
        }
    }
}

int main(){
    int N;
    cin >> N;
    int* parents = new int [N];
    for(int i = 0; i < N; i++){
        cin >> parents[i];
    }
    string* names = new string [N];
    for(int i = 0; i < N; i++){
        cin >> names[i];
    }
    drawTree(parents, names, N, -1, "");
    return 0;
}