#include<iostream>

using namespace std;

struct TreeNode{
    int data;
    TreeNode* leftChild = NULL;
    TreeNode* rightChile = NULL;
};

int locateValue(int* array, int arrayLen, int value){
    for(int i = 0; i < arrayLen; i++){
        if(array[i] == value){
            return i;
        }
    }
}

void buildTree(int* preOrder, int* inOrder, int length, TreeNode*& root){
    // The boundary conditions
    if(length == 0){
        return;
    }
    if(root == NULL){
        root = new TreeNode;
    }
    root->data = preOrder[0];
    // cout << "Root is " << root->data << endl;
    int splitPoint = locateValue(inOrder, length, preOrder[0]);
    // cout << "SplitPoint is " << splitPoint << endl;
    // The left tree to build
    int* leftInOrder = inOrder;
    int* leftPreOrder = preOrder + 1;
    int leftLength = splitPoint;
    buildTree(leftPreOrder, leftInOrder, leftLength, root->leftChild);
    // The right tree to build
    int* rightInOrder = inOrder + splitPoint + 1;
    int* rightPreOrer = preOrder + leftLength + 1;
    int rightLength = length - splitPoint - 1;
    buildTree(rightPreOrer, rightInOrder, rightLength, root->rightChile);
    return;
}

void postOrder(TreeNode* root, int* result, int& index){
    if(root == NULL){
        return;
    }
    postOrder(root->leftChild, result, index);
    postOrder(root->rightChile, result, index);
    result[index] = root->data;
    // cout << "Index is " << index << endl;
    index += 1;
    return;
}

void delTree(TreeNode* root){
    if(root == NULL){
        return;
    }
    delTree(root->leftChild);
    delTree(root->rightChile);
    delete root;
    return;
}

int main(){
    // Number of samples
    int numSam;
    cin >> numSam;
    int** results = new int*[numSam];
    int* numElem = new int[numSam];
    // For each sample
    for(int i = 0; i < numSam; i++){
        cin >> numElem[i];
        int * preOrder = new int [numElem[i]];
        int * inOrder = new int [numElem[i]];
        for(int j = 0; j < numElem[i]; j++){
            cin >> preOrder[j];
        }
        for(int j = 0; j < numElem[i]; j++){
            cin >> inOrder[j];
        }
        results[i] = new int[numElem[i]];
        TreeNode* root = new TreeNode;
        buildTree(preOrder, inOrder, numElem[i], root);
        // PostOrder
        int index = 0;
        postOrder(root, results[i], index);
        delTree(root);
    }
    // Print the results
    for(int i = 0; i < numSam; i++){
        for(int j = 0; j < numElem[i]; j++){
            cout << results[i][j] << " ";
        }
        cout << endl;
    }
    return 0;
}