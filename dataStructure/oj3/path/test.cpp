#include "priorityqueue.h"
#include <iostream>
#include <vector>

using namespace std;

struct Point{
    double x;
    double y;
};

struct Node
{
    int t;
    float x;
    float y;
};

vector<Point> greedySearch(Point *points, int n){
    vector<Point> path;
    return path;
}

int main(){
    int n, l, q;
    cin >> n >> l >> q;
    Point *points = new Point[n];
    for(int i = 0; i < n; i++){
        cin >> points[i].x >> points[i].y;
    }
    vector<Point> path = greedySearch(points, n);
    delete [] points;
    return 0;
}

