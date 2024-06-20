#include <iostream>

#define MD 119997

using namespace std;

struct HashElem
{
    int point1;
    int point2;
    bool ocp = false;
};

struct Point
{
    int x;
    int y;
};

HashElem hashTable[MD];
int conflictCount = 0;

int hashKey(int x, int y){
    int key = ((x * 4217 + y * 8731) % MD+ MD) % MD;
    return key;
}

int searchCount(int key, int i, int j, Point* points){
    int count = 0;
    while (hashTable[key].ocp){
        int p = hashTable[key].point1;
        int q = hashTable[key].point2;
        if(points[i].x + points[j].x != points[p].x + points[q].x || points[i].y + points[j].y != points[p].y + points[q].y){
            key = (key + 1) % MD;
            continue;
        }
        if((points[p].x-points[q].x)*(points[i].y-points[j].y) != (points[i].x-points[j].x)*(points[p].y-points[q].y))
            count++;
        key = (key + 1) % MD;
    }
    hashTable[key].ocp = true;
    hashTable[key].point1 = i;
    hashTable[key].point2 = j;
    return count;
}

int main(){
    int n;
    scanf("%d", &n);
    int* results = new int[n];
    for(int indexN = 0; indexN < n; indexN++){
        // initialize hashTable
        for(int i = 0; i < MD; i++){
            hashTable[i].ocp = false;
        }
        int m;
        scanf("%d", &m);
        Point* points = new Point[m];
        for(int indexM = 0; indexM < m; indexM++){
            scanf("%d %d", &points[indexM].x, &points[indexM].y);
        }
        int sum = 0;
        int midX, midY, key;
        for(int i = 0; i < m; i++){
            for(int j = i + 1; j < m; j++){
                midX = (points[i].x + points[j].x);
                midY = (points[i].y + points[j].y);
                key = ((midX * 4217 + midY * 8731) % MD+ MD) % MD;
                sum += searchCount(key, i, j, points);
            }
        }
        results[indexN] = sum;
    }
    for(int indexN = 0; indexN < n; indexN++){
        printf("%d\n", results[indexN]);
    }
    return 0;
}