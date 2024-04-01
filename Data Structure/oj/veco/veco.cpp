//实现向量排序并去重输出
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;
//定义结构体并重载运算符
struct point{
    int x;
    int y;
    bool operator<(const point &a) const{
        if(x==a.x) return y<a.y;
        return x<a.x;
    }
    bool operator==(const point &a) const{
        return x==a.x&&y==a.y;
    }
};
//定义向量
vector<point> vec;
int main(){
    int n;
    cin>>n;
    for(int i=0; i<n; i++){
        point p;
        cin>>p.x>>p.y;
        vec.push_back(p);
    }
    //排序
    sort(vec.begin(), vec.end());
    //输出时去重
    for(int i=0; i<vec.size(); i++){
        if(i==0||!(vec[i]==vec[i-1])){
            cout<<vec[i].x<<' '<<vec[i].y<<endl;
        }
    }
}