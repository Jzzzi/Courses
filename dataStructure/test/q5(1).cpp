#include <iostream>
#include <vector>
using namespace std;
// 打印当前轮的反应结果
void printAtoms(const vector<pair<int, int>>& atoms) {
    int i;
    for (i = 0; i < atoms.size() - 1; ++i) {
        cout << atoms[i].first << ' ';
    }
    cout << atoms[i + 1].first;
    std::cout << endl;
}
int main() {
    int n, k;
    cin >> n >> k;
    vector<pair<int, int>> atoms(n);// 原子的状态，第二个元素表示是否将被激活
    for (int i = 0; i < n; ++i) {
        cin >> atoms[i].first;
        atoms[i].second = 0;
    }// 初始化原子状态
    do {
        vector<pair<int, int>> nextAtoms;// 下一轮的原子状态
        for (int i = 0; i < atoms.size(); ++i) {
            if (atoms[i].first == 1) {
                // 激发顺时针第k个原子
                int nextIndexCW = (i + k) % atoms.size();// 下一个原子的索引
                if (nextIndexCW < 0)
                    nextIndexCW += atoms.size();
                    atoms[nextIndexCW].second = 1; // 标记顺时针第k个原子为将被激活
                // 激发逆时针第k个原子
                int nextIndexCCW = (i - k +atoms.size()) % atoms.size();// 下一个原子的索引
                if (nextIndexCCW < 0)
                    nextIndexCCW += atoms.size();
                    atoms[nextIndexCCW].second = 1; // 标记逆时针第k个原子为将被激活
            }
        }
        for (int i = 0; i < atoms.size(); ++i) {
            if (atoms[i].second == 1 && atoms[i].first != 1) {
                atoms[i].first = 1;// 激活原子
                nextAtoms.push_back(atoms[i]);// 将激活的原子加入下一轮的状态
            } 
            else if (atoms[i].first != 1)
                nextAtoms.push_back(atoms[i]);// 将未激活的原子加入下一轮的状态
        }
        if (atoms == nextAtoms) {
            cout<<endl;
            break;
            }// 如果下一轮的状态与当前状态相同，则不再进行下一轮
        else if (nextAtoms.size() == 0) {
            cout<<endl;
            break;
            }// 如果下一轮的状态为空，则不再进行下一轮
        else {
            atoms = nextAtoms; // 更新状态
            printAtoms(atoms); // 打印当前轮的状态
        }
        for (int i = 0; i < atoms.size(); ++i) {
            atoms[i].second = 0;
        }// 重置原子状态
    } while (atoms.size() > 0); // 当原子状态为空时结束
    return 0;
}