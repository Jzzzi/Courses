#include <iostream>
#include <string>
#include <vector>
using namespace std;

struct TreeNode
{
    string name;
    vector<TreeNode*> childs;
    TreeNode(string name){
        this->name = name;
    }
    TreeNode(){
        // Empty constructor
    }
    
    void insertChild(string name){
        TreeNode* child = new TreeNode(name);
        childs.push_back(child);
    }
};

void drawTree(int N, int* parents, string* names){
    // Draw the tree
}

int main(){
    int N;
    cin >> N;
    int* parents = new int[N];
    string* names = new string[N];
    for(int i = 0; i < N; i++){
        cin >> parents[i];
    }
    for(int i = 0; i < N; i++){
        cin >> names[i];
    }
    for(int i = 0; i < N; i++){
        cout << names[i] << " " << parents[i] << endl;
    }
    // Build the tree
    return 0;
}