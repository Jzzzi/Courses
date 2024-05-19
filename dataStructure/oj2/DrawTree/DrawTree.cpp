#include <iostream>
#include <vector>
#include <string>

using namespace std;

void printTree(const vector<int>& parents, const vector<string>& names, int current, int depth, vector<bool>& visited) {
    cout << string(depth * 2, ' ') << "+-" << names[current] << endl;
    visited[current] = true;

    // 找到当前节点的子节点
    for (int i = 0; i < parents.size(); ++i) {
        if (parents[i] == current) {
            printTree(parents, names, i, depth + 1, visited);
        }
    }

    // 检查当前节点是否有未打印的兄弟节点
    for (int i = current + 1; i < parents.size(); ++i) {
        if (parents[i] == parents[current]) {
            if (!visited[i]) {
                cout << string(depth * 2, ' ') << "|" << endl;
                printTree(parents, names, i, depth, visited);
            }
            break;
        }
    }
}

int main() {
    int N;
    cin >> N;

    vector<int> parents(N);
    vector<string> names(N);
    vector<bool> visited(N, false);

    for (int i = 0; i < N; ++i) {
        cin >> parents[i];
    }
    for (int i = 0; i < N; ++i) {
        cin >> names[i];
    }

    // 找到根节点
    int root = -1;
    for (int i = 0; i < N; ++i) {
        if (parents[i] == -1) {
            root = i;
            break;
        }
    }

    // 输出树结构
    printTree(parents, names, root, 0, visited);

    return 0;
}
