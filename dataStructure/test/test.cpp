#include <iostream>
#include <queue>
using namespace std;
struct BinNode
{
    int data;
    BinNode *lc, *rc;
    // 构造函数
    BinNode(int x){
        data = x;
        lc = rc = NULL;
    }
    BinNode(int x, BinNode *l, BinNode *r){
        data = x;
        lc = l;
        rc = r;
    }
    BinNode(){
        data = 0;
        lc = rc = NULL;
    }
};

int widthTree(BinNode *x){
    // 通过递归实现
    if(!x) return 0;
    queue<BinNode*> q;
    q.push(x);
    int max_width = 1;
    while(!q.empty()){
        int size = q.size();
        while(size--){
            BinNode *node = q.front();
            q.pop();
            if(node->lc) q.push(node->lc);
            if(node->rc) q.push(node->rc);
        }
        max_width = max(max_width, (int)q.size());
    }
    return max_width;
}

int main(){
    int width = 0;
    BinNode *root = new BinNode(1);
    root->lc = new BinNode(2);
    root->rc = new BinNode(3);
    root->lc->lc = new BinNode(4);
    root->lc->rc = new BinNode(5);
    root->rc->lc = new BinNode(6);
    width = widthTree(root);
    // 理论上输出应该是3
    cout << width << endl;
    return width;
}