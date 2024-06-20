#include <iostream>
#include <cstdlib>

using namespace std;

struct Point{
    int x;
    int y;
};

static int image[1000][1000];
static int colOddSum[1000][1000];
static int colEvenSum[1000][1000];
static int imageHashOdd[1000][1000];
static int imageHashEven[1000][1000];
static int colHasher[1000];
static int rowHasher[1000];
static int imageSearch[1000][1000];

void hasherInit(int qn, int qm){
    for(int i = 0; i < qn; i++){
        rowHasher[i] = rand();
    }
    for(int i = 0; i < qm; i++){
        colHasher[i] = rand();
    }
}

void hashOriginImage(int qn, int qm, int n, int m){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < m - qm +1; j++){
            for(int k = 0; k < qm; k++){
                if(k % 2 == 0){
                    colEvenSum[i][j] += image[i][j + k] * colHasher[k];
                }
                else{
                    colOddSum[i][j] += image[i][j + k] * colHasher[k];
                }
            }
        }
    }
    for(int i = 0; i < n - qn + 1; i++){
        for(int j = 0; j < m - qm + 1; j++){
            for(int k = 0; k < qn; k++){
                imageHashEven[i][j] += colEvenSum[i + k][j] * rowHasher[k];
                imageHashOdd[i][j] += colOddSum[i + k][j] * rowHasher[k];
            }
        }
    }
}

int evenHashSearchImage(int qn, int qm){
    int searchHash = 0;
    for(int i = 0; i < qn; i++){
        for(int j = 0; j < qm; j++){
            if(j % 2 == 0){
                searchHash += imageSearch[i][j] * rowHasher[i] * colHasher[j];
            }
        }
    }
    return searchHash;
}

int oddHashSearchImage(int qn, int qm){
    int searchHash = 0;
    for(int i = 0; i < qn; i++){
        for(int j = 0; j < qm; j++){
            if(j % 2 == 1){
                searchHash += imageSearch[i][j] * rowHasher[i] * colHasher[j];
            }
        }
    }
    return searchHash;
}

bool checkSame(int x, int y, int qn, int qm){
    int conflict = 0;
    for(int i = 0; i < qn; i++){
        for(int j = 0; j < qm; j++){
            if(image[x + i][y + j] != imageSearch[i][j]){
                conflict++;
                if(conflict > 1){
                    return false;
                }
            }
        }
    }
    return true;
}

int main(){
    int n, m, q, qn, qm;
    cin >> n >> m >> q >> qn >> qm;
    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            char c;
            cin >> c;
            image[i][j] = c - '0';
        }
    }
    hasherInit(qn, qm);
    hashOriginImage(qn, qm, n, m);
    Point* points = new Point[q];
    for(int i = 0; i < q; i++){
        char Q;
        cin >> Q;
        if(Q == 'Q'){
            for (int j = 0; j < qn; j++){
                for (int k = 0; k < qm; k++){
                    char c;
                    cin >> c;
                    imageSearch[j][k] = c - '0';
                }
            }
        }
        else{
            exit(1);
        }
        // hash the search image
        int oddSearchHash = oddHashSearchImage(qn, qm);
        int evenSearchHash = evenHashSearchImage(qn, qm);
        // search the image
        int x, y;
        bool found = false;
        for(int j = 0; j < n - qn + 1; j++){
            for(int k = 0; k < m - qm + 1; k++){
                if(imageHashOdd[j][k] == oddSearchHash || imageHashEven[j][k] == evenSearchHash){
                    // check it is the same
                    bool same = checkSame(j, k, qn, qm);
                    if(same){
                        points[i].x = j;
                        points[i].y = k;
                        found = true;
                        break;
                    }
                }
            }
            if(found){
                break;
            }
        }

    }
    for(int i = 0; i < q; i++){
        cout << points[i].x << " " << points[i].y << endl;
    }
    return 0;
}