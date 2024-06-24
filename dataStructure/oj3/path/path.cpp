# include <iostream>
# include <vector>
# include <cmath>
using namespace std;

struct Point{
    double x = 0;
    double y = 0;
};


void greedySearch(Point *points, int n, vector<Point> &path, bool hasDog, Point dogInit){
    Point dog = dogInit;
    Point dogNext = dog;
    Point person;
    Point personNext;
    double dogSpeed = 0.5;
    int got = 0;
    bool* explored = new bool[n];
    for(int i = 0; i < n; i++){
        explored[i] = false;
    }
    person.x = 0;
    person.y = 0;
    while (got < n)
    {
        double dist;
        double dogDist = 1e9;
        double minDist = 1e9;
        double minPriority = 1e9;
        int minIndex = -1;
        // the center of the unexplored points
        Point center;
        for(int i = 0; i < n; i++){
            if(!explored[i]){
                center.x += points[i].x;
                center.y += points[i].y;
            }
        }
        center.x /= (n - got);
        center.y /= (n - got);
        center.x = center.x - person.x;
        center.y = center.y - person.y;

        for(int i = 0; i < n; i++){
            if(explored[i]){
                continue;
            }
            double priority = 0;
            dist = sqrt((points[i].x - person.x) * (points[i].x - person.x) + (points[i].y - person.y) * (points[i].y - person.y));
            if(dist > 0.99){
                personNext.x = person.x + (points[i].x - person.x) / dist * 0.99;
                personNext.y = person.y + (points[i].y - person.y) / dist * 0.99;
            }
            else{
                personNext.x = points[i].x;
                personNext.y = points[i].y;
            }
            if(hasDog){
                dogDist = sqrt((personNext.x - dog.x) * (personNext.x - dog.x) + (personNext.y - dog.y) * (personNext.y - dog.y));
                if(dogDist <= dogSpeed){
                    priority = 1e9;
                }
                else{
                    priority = 1 / (dogDist - dogSpeed);
                }
            }
            priority += (points[i].x - person.x) * center.x + (points[i].y - person.y) * center.y;
            if(priority < minPriority){
                minPriority = priority;
                minIndex = i;
            }
        }
        minDist = sqrt((points[minIndex].x - person.x) * (points[minIndex].x - person.x) + (points[minIndex].y - person.y) * (points[minIndex].y - person.y));
        if(minDist > 1){
            person.x += (points[minIndex].x - person.x) / minDist * 0.99;
            person.y += (points[minIndex].y - person.y) / minDist * 0.99;
        }
        else{
            person.x = points[minIndex].x;
            person.y = points[minIndex].y;
            explored[minIndex] = true;
            got++;
        }
        // update the dog's position
        if(hasDog){
            dogNext.x = dog.x + (person.x - dog.x) / dogDist * dogSpeed;
            dogNext.y = dog.y + (person.y - dog.y) / dogDist * dogSpeed;
            dog.x = dogNext.x;
            dog.y = dogNext.y;
        }
        path.push_back(person);
    }
}

int main(){
    bool hasDog = false;
    int n, l, q;
    cin >> n >> l >> q;
    Point *points = new Point[n];
    Point dogInit;
    for(int i = 0; i < n; i++){
        cin >> points[i].x >> points[i].y;
    }
    if(q == 1){
        cin >> dogInit.x >> dogInit.y;
        hasDog = true;
    }
    vector<Point> path;
    greedySearch(points, n, path, hasDog, dogInit);
    // output the result
    cout << path.size() << endl;
    for(int i = 0; i < path.size(); i++){
        cout << path[i].x << " " << path[i].y << endl;
    }
    return 0;
}