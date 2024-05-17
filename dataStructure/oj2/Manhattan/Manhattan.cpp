#include <iostream>
using namespace std;

int main(){
    // n is the number of the operation, d is the dimension of the point
    int n,d;
    cin >> n >> d;

    // Create a 2D array to store the points, the last element is to store whether the point is exist
    int** points = new int*[n];
    for(int i = 0; i < n; i++){
        points[i] = new int[d + 1];
    }

    // Read the points
    for(int i = 0; i < n; i++){
        int judge;
        cin >> judge;
        if(judge == 1){
            int index;
            cin >> index;
            points[index][d] = 0;
        }
        else{
            for(int j = 0; j < d; j++){
                cin >> points[i][j];
            }
            points[i][d] = 1;
        }
        // Enumerate the coordinates of the point
        
    }



}