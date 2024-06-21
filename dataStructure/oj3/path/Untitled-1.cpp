#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <tuple>

using namespace std;

struct Point {
    int x, y;
};

double euclidean_distance(const Point& p1, const Point& p2) {
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

pair<vector<int>, double> tsp(const vector<Point>& points) {
    int n = points.size();
    vector<int> indices(n);
    for (int i = 0; i < n; ++i) {
        indices[i] = i;
    }

    vector<int> best_path;
    double min_distance = numeric_limits<double>::infinity();

    do {
        double distance = 0;
        Point current_point = {0, 0};
        for (int idx : indices) {
            distance += euclidean_distance(current_point, points[idx]);
            current_point = points[idx];
        }
        distance += euclidean_distance(current_point, {0, 0});

        if (distance < min_distance) {
            min_distance = distance;
            best_path = indices;
        }
    } while (next_permutation(indices.begin(), indices.end()));

    return {best_path, min_distance};
}

vector<Point> generate_path(const vector<Point>& points, const vector<int>& order) {
    vector<Point> path;
    Point current = {0, 0};
    path.push_back(current);

    for (int idx : order) {
        Point target = points[idx];
        while (current.x != target.x || current.y != target.y) {
            if (current.x < target.x) current.x++;
            else if (current.x > target.x) current.x--;

            if (current.y < target.y) current.y++;
            else if (current.y > target.y) current.y--;

            path.push_back(current);
        }
    }
    return path;
}

int main() {
    int n, L, q;
    cin >> n >> L >> q;

    vector<Point> files(n);
    for (int i = 0; i < n; ++i) {
        cin >> files[i].x >> files[i].y;
    }

    Point dog;
    if (q == 1) {
        cin >> dog.x >> dog.y;
    }

    // Solve TSP to get the optimal path
    auto [order, min_distance] = tsp(files);

    if (min_distance > L) {
        cout << "Impossible to meet the length requirement." << endl;
        return 0;
    }

    // Generate the path taken each second
    vector<Point> result_path = generate_path(files, order);

    // Output the result
    cout << result_path.size() - 1 << endl;
    for (const auto& p : result_path) {
        cout << p.x << " " << p.y << endl;
    }

    return 0;
}