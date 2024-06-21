#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

struct Point{
    double x;
    double y;
};

void greedySearch(Point *points, int n, vector<Point> &path){
    int got = 0;
    bool* explored = new bool[n];
    for(int i = 0; i < n; i++){
        explored[i] = false;
    }
    Point now;
    now.x = 0;
    now.y = 0;
    while (got < n)
    {
        double minDist = 1e9;
        int minIndex = -1;
        for(int i = 0; i < n; i++){
            if(explored[i]){
                continue;
            }
            double dist = (now.x - points[i].x) * (now.x - points[i].x) + (now.y - points[i].y) * (now.y - points[i].y);
            if(dist < minDist){
                minDist = dist;
                minIndex = i;
            }
        }
        minDist = sqrt(minDist);
        if(minDist <= 0.99){
            now = points[minIndex];
            explored[minIndex] = true;
            got++;
        }
        else{
            now.x = now.x + (points[minIndex].x - now.x) / minDist;
            now.y = now.y + (points[minIndex].y - now.y) / minDist;
            if(now.x == points[minIndex].x && now.y == points[minIndex].y){
                explored[minIndex] = true;
                got++;
            }
        }
        path.push_back(now);
    }
}

int main(){
    int n, l, q;
    cin >> n >> l >> q;
    Point *points = new Point[n];
    Point dogInit;
    for(int i = 0; i < n; i++){
        cin >> points[i].x >> points[i].y;
    }
    if(q == 1){
        cin >> dogInit.x >> dogInit.y;
    }
    vector<Point> path;
    greedySearch(points, n, path);
    double length = 0;
    for(int i = 1; i < path.size(); i++){
        length += sqrt((path[i].x - path[i - 1].x) * (path[i].x - path[i - 1].x) + (path[i].y - path[i - 1].y) * (path[i].y - path[i - 1].y));
    }
    if(length < l){
        cout << path.size() << endl;
        for(int i = 0; i < path.size(); i++){
            cout << path[i].x << " " << path[i].y << endl;
        }
    }
    else{
        cout << "failed" << endl;
    }
    return 0;
}

