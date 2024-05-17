#include <iostream>

using namespace std;

struct TreeNode
{
    int data;
    TreeNode* leftChild = NULL;
    TreeNode* rightChild = NULL;
};

void insert(TreeNode*& root, int value){
    if(root == NULL){
        root = new TreeNode;
        root->data = value;
        return;
    }
    if(value < root->data){
        insert(root->leftChild, value);
    }
    else{
        insert(root->rightChild, value);
    }
    return;
}

int findAcestor(TreeNode* root, int val1, int val2){
    if(val1 > root->data && val2 > root->data){
        return findAcestor(root->rightChild, val1, val2);
    }
    else if(val1 < root->data && val2 < root->data){
        return findAcestor(root->leftChild, val1, val2);
    }
    else{
        return root->data;
    }
}

int main(){
    int N, M;
    cin >> N >> M;
    int* results = new int [M];
    // Build the tree
    TreeNode* root = NULL;
    for(int i = 0; i < N; i++){
        int value;
        cin >> value;
        insert(root, value);
    }
    // Find acestor in each sample
    for(int i = 0; i < M; i++){
        int val1, val2;
        cin >> val1 >> val2;
        results[i] = findAcestor(root, val1, val2);
    }
    // Print the results
    for(int i = 0; i < M; i++){
        cout << results[i] << endl;
    }
    return 0;
}