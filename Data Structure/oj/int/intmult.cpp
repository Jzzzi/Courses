//实现两个大数的乘法
#include <iostream>
#include <string>
using namespace std;
int main(){
    string a,b;
    cin>>a>>b;
    int lena = a.size();
    for(int i=0; i<lena;i++)
        a[i] = a[i]-'0';
    int lenb=b.size();
    for(int i=0; i<lenb;i++)
        b[i] = b[i]-'0';
    int res[lena+lenb]={0};
    for(int i=0; i<lena; i++){
        for(int j=0; j<lenb; j++){
            res[i+j+1] += a[i]*b[j];
        }
    }
    for(int i=lena+lenb-1; i>0; i--){
        res[i-1] += res[i]/10;
        res[i] %= 10;
    }
    int i=0;
    while(i<lena+lenb && res[i]==0)
        i++;
    if(i==lena+lenb)
        cout<<0;
    else{
        for(; i<lena+lenb; i++)
            cout<<res[i];
    }
    return 0;
}