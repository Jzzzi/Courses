#include <iostream>
int main(){
    // 计算卡特兰数
    int n;
    std::cin >> n;
    long long int catalan[n+1];
    catalan[0] = 1;
    catalan[1] = 1;
    for(int i = 2; i <= n; i++){
        catalan[i] = 0;
        for(int j = 0; j < i; j++){
            catalan[i] += catalan[j] * catalan[i-j-1];
        }
    }
    std::cout << catalan[n] << std::endl;
    return 0;
    }