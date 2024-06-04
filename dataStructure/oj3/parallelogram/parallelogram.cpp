// WRONG ANSWER

#include<stdio.h>
#define MD 119997

struct Point{
    int x,y;
};

struct hashElem
{
    int point1, point2;
    bool occupied = false;
};


void hashing(int midx, int midy, int i, int j, hashElem* hashTable){
    int key = ((midx * 1000 + midy * 1234) % MD + MD) % MD;
    while(hashTable[key].occupied)
        key = (key + 1) % MD;
    hashTable[key].point1 = i;
    hashTable[key].point2 = j;
    hashTable[key].occupied = true;
}

int countParallelogram(int midx, int midy, int i, int j, Point* points, hashElem* hashTable){
    int count = 0;
    int key = ((midx * 1000 + midy * 1234) % MD + MD) % MD;
    while(hashTable[key].occupied){
        if((hashTable[key].point1 != i) && (hashTable[key].point1 != j) && (hashTable[key].point2 != i) && (hashTable[key].point2 != j)){
            // 确保4点不共线
            if((points[hashTable[key].point1].x - points[hashTable[key].point2].x) * (points[i].y - points[j].y) == (points[i].x - points[j].x) * (points[hashTable[key].point1].y - points[hashTable[key].point2].y)){
                key = (key + 1) % MD;
                continue;
            }
            count++;
            // print the 4 dots
            // printf("(%d, %d), (%d, %d), (%d, %d), (%d, %d)\n", points[i].x, points[i].y, points[j].x, points[j].y, points[hashTable[key].point1].x, points[hashTable[key].point1].y, points[hashTable[key].point2].x, points[hashTable[key].point2].y); 
        }
        key = (key + 1) % MD;
    }
    return count;
}

int totalCount(Point* points, int m){
    hashElem* hashTable = new hashElem[MD];
    int count = 0;
    for (int i = 0; i < m; i++){
        for (int j = i+1; j < m; j++){
            int midx = (points[i].x + points[j].x);
            int midy = (points[i].y + points[j].y);
            hashing(midx, midy, i, j, hashTable);
        }
    }
    for (int i = 0; i < m; i++){
        for (int j = i+1; j < m; j++){
            int midx = (points[i].x + points[j].x);
            int midy = (points[i].y + points[j].y);
            count += countParallelogram(midx, midy, i, j, points, hashTable);
        }
    }
    delete[] hashTable;
    return count/2;
}

int main(){
    int n;
    scanf("%d", &n);
    int* results = new int [n];
    for(int i = 0; i < n; i++){
        int m;
        scanf("%d", &m);
        Point* points = new Point[m];
        for(int j = 0; j < m; j++){
            scanf("%d %d", &points[j].x, &points[j].y);
        }
        results[i] = totalCount(points, m);
        delete[] points;
    }
    for(int i = 0; i < n; i++){
        printf("%d\n", results[i]);
    }
    return 0;
}