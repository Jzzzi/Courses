#include <iostream>
#include <stdio.h>
#include <math.h>
using namespace std;

struct Point{
    double cord[2];
    double dist(Point p){
        return sqrt(pow(cord[0] - p.cord[0], 2) + pow(cord[1] - p.cord[1], 2));
    }
    bool larger(Point p, int axis){
        return cord[axis] > p.cord[axis];
    }
};

struct TreeNode{
    Point data;
    int axis;
    TreeNode* leftChild = NULL;
    TreeNode* rightChild = NULL;
};

void sortTreeNode(Point* points, int numElem, int axis){
    // From small to large, use bubble sort
    for(int i = 0; i < numElem; i++){
        for(int j = 0; j < numElem - i - 1; j++){
            if(points[j].larger(points[j + 1], axis)){
                Point temp = points[j];
                points[j] = points[j + 1];
                points[j + 1] = temp;
            }
        }
    }
};

void buildTree(Point* points, int numElem, int axis, TreeNode*& root){
    if(numElem == 0){
        return;
    }
    if(numElem == 1){
        root = new TreeNode;
        root->data = points[0];
        root->axis = axis;
        return;
    }
    sortTreeNode(points, numElem, axis);
    int mid = (numElem - 1) / 2;
    root = new TreeNode;
    root->data = points[mid];
    root->axis = axis;
    buildTree(points, mid, (axis + 1) % 2, root->leftChild);
    buildTree(points + mid + 1, numElem - mid - 1, (axis + 1) % 2, root->rightChild);
};

double min(double a, double b){
    return a < b ? a : b;
};

double minDistance(TreeNode* root, Point target, double minDist = 1e9){
    if(root == NULL){
        return minDist;
    }    
    if(root->data.dist(target) < minDist){
        minDist = root->data.dist(target);
    }
    if(target.cord[root->axis] - root->data.cord[root->axis] < minDist){
        minDist = min(minDist, minDistance(root->leftChild, target, minDist));
    }
    if(root->data.cord[root->axis] - target.cord[root->axis] < minDist){
        minDist = min(minDist, minDistance(root->rightChild, target, minDist));
    }
    return minDist;
};

int main(){
    int numElem, numSam;
    cin >> numElem >> numSam;

    // Read points
    Point* points = new Point[numElem];
    for(int i = 0; i < numElem; i++){
        char name;
        cin >> points[i].cord[0] >> points[i].cord[1] >> name;
    }

    // Build the tree, 'root' is the pointer to the root node
    TreeNode* root = NULL;
    buildTree(points, numElem, 0, root);

    // Read the samples and save the results
    double* results = new double[numSam];
    for(int i = 0; i < numSam; i++){
        Point target;
        cin >> target.cord[0] >> target.cord[1];
        double minDist = minDistance(root, target);
        results[i] = minDist;
    }

    // Output the results
    for(int i = 0; i < numSam; i++){
        printf("%.4f\n", results[i]);
    }

    return 0;
};