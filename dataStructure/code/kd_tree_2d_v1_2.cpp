#include <iostream>
#include <algorithm>
#include <vector>
#include <limits>
#include <cmath>

// ���ڱ�ʾ����
struct Coordinate
{
	double m_vector[2];

	Coordinate(double x = 0, double y = 0)
	{
		m_vector[0] = x;
		m_vector[1] = y;
	}

	bool isBigger(int dimension, const Coordinate& coor) const
	{
		return m_vector[dimension] > coor.m_vector[dimension];
	}

	bool isSmaller(int dimension, const Coordinate& coor) const
	{
		return m_vector[dimension] < coor.m_vector[dimension];
	}

	double distance(int dimension, const Coordinate& coor) const
	{
		return abs(m_vector[dimension] - coor.m_vector[dimension]);
	}

	double distance(const Coordinate& coor) const
	{
		double x_diff = m_vector[0] - coor.m_vector[0];
		double y_diff = m_vector[1] - coor.m_vector[1];
		return sqrt(x_diff * x_diff + y_diff * y_diff);
	}
};

// �����ݵ����� ������Ļ����ϼ��������� �������������������Ϊ�ַ���
struct PointStruct : public Coordinate
{
	char m_data;

	bool isEqual(PointStruct point)
	{
		return m_data == point.m_data && m_vector[0] == point.m_vector[0]
			&& m_vector[1] == point.m_vector[1];
	}
};

// ���ڵ� �̳��Ե����ݵ�����
struct TreeNode : public PointStruct
{
	// ���ڱ�ʾ�ռ仮�ֵķ��� 0������x���򻮷� 1������y���򻮷�
	int m_dimension;

	TreeNode* m_pLeftChild;
	TreeNode* m_pRightChild;

	TreeNode(const PointStruct& point)
		: PointStruct(point)
	{
		m_pLeftChild = nullptr;
		m_pRightChild = nullptr;
	}

	~TreeNode()
	{
		if (m_pLeftChild)
			delete m_pLeftChild;
		if (m_pRightChild)
			delete m_pRightChild;
	}
};

// ��ѯ�ľ�����������
struct RecArea
{
	// ��ž����������º���������
	Coordinate m_minCoor;
	Coordinate m_maxCoor;

	// Ĭ�Ͼ�������ķ�ΧΪ��������
	RecArea(double minX = -std::numeric_limits<double>::infinity(), double minY = -std::numeric_limits<double>::infinity()
		, double maxX = std::numeric_limits<double>::infinity(), double maxY = std::numeric_limits<double>::infinity())
	{
		m_minCoor = Coordinate(minX, minY);
		m_maxCoor = Coordinate(maxX, maxY);
	}

	// �ж�һ����dimension�����ϵĿռ仮���Ƿ�����������ཻ
	int intersect(int dimension, const Coordinate& hyperPlane) const
	{
		if (m_minCoor.isBigger(dimension, hyperPlane))
			return 1;
		else if (m_maxCoor.isSmaller(dimension, hyperPlane))
			return -1;
		else
			return 0;
	}

	// �ж�ĳһ���Ƿ����������������
	bool include(const Coordinate& point) const
	{
		return !m_minCoor.isBigger(0, point) && !m_maxCoor.isSmaller(0, point)
			&& !m_minCoor.isBigger(1, point) && !m_maxCoor.isSmaller(1, point);
	}

	// ���������򻮷�Ϊ������
	void split(int dimension, const Coordinate& hyperPlane, RecArea& smallerArea, RecArea& biggerArea) const
	{
		smallerArea = *this;
		smallerArea.m_maxCoor.m_vector[dimension] = hyperPlane.m_vector[dimension];
		biggerArea = *this;
		biggerArea.m_minCoor.m_vector[dimension] = hyperPlane.m_vector[dimension];
	}
};

struct KdTree
{
	TreeNode* m_pRoot;

	KdTree(std::vector<PointStruct>& pointList)
	{
		m_pRoot = buildTree(pointList, 0);
	}

	~KdTree()
	{
		if (m_pRoot)
			delete m_pRoot;
	}

	static TreeNode* buildTree(std::vector<PointStruct>& pointList, int depth);
	static PointStruct findMedialPoint(int dimension, std::vector<PointStruct>& pointList);
	static void recursionRectangleSearch(TreeNode* pNode, int depth, const RecArea& area, std::vector<PointStruct>& result);
	static void recursionNearestSearch(TreeNode* pNode, int depth, const Coordinate& point, PointStruct& nearest);

	void rectangleSearch(const RecArea& area, std::vector<PointStruct>& result)
	{
		result.clear();
		recursionRectangleSearch(m_pRoot, 0, area, result);
	}

	PointStruct nearestSearch(const Coordinate& point)
	{
		PointStruct nearest = (PointStruct)*m_pRoot;
		recursionNearestSearch(m_pRoot, 0, point, nearest);
		return nearest;
	}
};

TreeNode* KdTree::buildTree(std::vector<PointStruct>& pointList, int depth)
{
	if (pointList.empty())
		return nullptr;

	int dimension = depth % 2;
	PointStruct mediaPoint = findMedialPoint(dimension, pointList);

	std::vector<PointStruct> pointListLeft;
	std::vector<PointStruct> pointListRight;
	for (int i = 0; i < pointList.size(); i++)
		if (mediaPoint.isBigger(dimension, pointList[i]))
			pointListLeft.push_back(pointList[i]);
		else if (mediaPoint.isSmaller(dimension, pointList[i]))
			pointListRight.push_back(pointList[i]);
		else if (!mediaPoint.isEqual(pointList[i]))
			pointListRight.push_back(pointList[i]);

	// ������Ϣ�����ڱ�ʾ����������
	std::cout << depth << ' ' << mediaPoint.m_data << std::endl;
	TreeNode* pTreeNode = new TreeNode(mediaPoint);
	pTreeNode->m_pLeftChild = buildTree(pointListLeft, depth + 1);
	pTreeNode->m_pRightChild = buildTree(pointListRight, depth + 1);
	pTreeNode->m_dimension = dimension;

	return pTreeNode;
}

PointStruct KdTree::findMedialPoint(int dimension, std::vector<PointStruct>& pointList)
{
	std::sort(pointList.begin(), pointList.end(), [&](const PointStruct& point1, const PointStruct& point2){
		return point1.isSmaller(dimension, point2);
	});

	int mediaPosition = pointList.size() / 2;
	return pointList[mediaPosition];
}

void KdTree::recursionRectangleSearch(TreeNode* pNode, int depth, const RecArea& area, std::vector<PointStruct>& result)
{
	if (!pNode)
		return;

	int dimension = depth % 2;
	int flag = area.intersect(dimension, *pNode);
	if (flag < 0)
		recursionRectangleSearch(pNode->m_pLeftChild, depth + 1, area, result);
	else if (flag > 0)
		recursionRectangleSearch(pNode->m_pRightChild, depth + 1, area, result);
	else
	{
		RecArea smallerArea;
		RecArea biggerArea;
		area.split(dimension, *pNode, smallerArea, biggerArea);
		recursionRectangleSearch(pNode->m_pLeftChild, depth + 1, smallerArea, result);
		recursionRectangleSearch(pNode->m_pRightChild, depth + 1, biggerArea, result);
		if (area.include(*pNode))
			result.push_back(*pNode);
	}
}

void KdTree::recursionNearestSearch(TreeNode* pNode, int depth, const Coordinate& point, PointStruct& nearest)
{
	if (!pNode)
		return;

	int dimension = depth % 2;
	if (point.distance(*pNode) < point.distance(nearest))
		nearest = *pNode;

	if (point.isSmaller(dimension, *pNode))
	{
		recursionNearestSearch(pNode->m_pLeftChild, depth + 1, point, nearest);
		if (point.distance(nearest) >= point.distance(dimension, *pNode))
			recursionNearestSearch(pNode->m_pRightChild, depth + 1, point, nearest);
	}
	else
	{
		recursionNearestSearch(pNode->m_pRightChild, depth + 1, point, nearest);
		if (point.distance(nearest) >= point.distance(dimension, *pNode))
			recursionNearestSearch(pNode->m_pLeftChild, depth + 1, point, nearest);
	}
}

int main()
{
	int sampleCount;
	std::cout << "����������:";
	std::cin >> sampleCount;
	std::cout << "�����������ݣ���ʽΪ�� x���� y���� ���ݣ��ַ��ͣ�\n";
	std::vector<PointStruct> vec;
	while (sampleCount--)
	{
		PointStruct point;
		std::cin >> point.m_vector[0] >> point.m_vector[1];
		std::cin.ignore(1);
		std::cin >> point.m_data;
		vec.push_back(point);
	}

	std::cout << "������������ -��� -����\n";
	KdTree kdTree(vec);

	RecArea recArea;
	recArea.m_minCoor.m_vector[0] = -1;
	recArea.m_maxCoor.m_vector[0] = 1;
	std::cout << "��ѯ��x���귶Χ��" << recArea.m_minCoor.m_vector[0] << " ~ " << recArea.m_maxCoor.m_vector[0] << std::endl;
	std::cout << "��ѯ��y���귶Χ��" << recArea.m_minCoor.m_vector[1] << " ~ " << recArea.m_maxCoor.m_vector[1] << std::endl;
	std::vector<PointStruct> res;
	kdTree.rectangleSearch(recArea, res);

	std::cout << "�������������ݣ�";
	for (auto p : res)
	{
		std::cout << p.m_data << ' ';
	}
	std::cout << std::endl;

	Coordinate coor(4.9, 4.1);
	PointStruct nearest;
	nearest = kdTree.nearestSearch(coor);
	std::cout << "����㣺" << nearest.m_data << std::endl;
}