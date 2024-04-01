//括号匹配
#include <stack>
#include <vector>
#include <iostream>
#include <stdio.h>
using namespace std;
stack<char> s;
vector<int> v;
int main(){
    int n;
    cin>>n;
    getchar();
    for(int i=0; i<n; i++){
        while(!s.empty()) s.pop();
        char c;
        while((c=getchar())!='\n'){
            if(c=='('||c=='['||c=='{')
                s.push(c);
            else if(c==')'){
                if(s.empty()||s.top()!='('){
                    v.push_back(0);
                    goto end;
                }
                s.pop();
            }
            else if(c==']'){
                if(s.empty()||s.top()!='['){
                    v.push_back(0);
                    goto end;
                }
                s.pop();
            }
            else if(c=='}'){
                if(s.empty()||s.top()!='{'){
                    v.push_back(0);
                    goto end;
                }
                s.pop();
            }
        }
        if(s.empty()) v.push_back(1);
        else v.push_back(0);
        end:;
        while(c!='\n') c=getchar();
    }
    for(int i=0; i<v.size(); i++)
        cout<<v[i]<<endl;
}