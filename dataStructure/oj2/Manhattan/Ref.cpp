#include <iostream>
#include <map>
#include <queue>
#include <set>
 
using namespace std;
 
int renew_prq(map<int, map<int, int> > p,int i,int j,int dem)
{
	int sum = 0;
		for (int k = 0; k < dem; k++) {      //dem维度，考察每个点、每个维度下的符号状态
			int t = i & (1 << k);      //i与2^k进行与运算，  
                                       //例如i=110 则k=0,运算结果是0; k=1ork=2 运算结果是1 
			                           //效果是判断i的第k为是否为1
			if (t) sum += p[j][k];     //如果为1，则累加入sum 反之，则减去
			else sum -= p[j][k];
		}
	return sum;
}
 
 
 
int main() {
	int n,dem;
    cin>>n>>dem;
	map<int, map<int, int> > Manhatdata;
	int data_now;
	int instruction;
	int ddem=1 <<dem;
	int *result = new int[n];
	//构造2^dem个 大顶堆,存放每一个维度符号状态下的数据点最大值和最小值(rbegin)
    multiset<int> *min_ms= new multiset<int>[ddem]; //小顶堆  begin是最小的
    //multiset<int, greater<int>> max_ms[ddem]; //begin 是最大的
 
	for(int ii=1;ii<=n;ii++)       //循环输入n个操作
	{
		int maxManhattan=0;
		cin>>instruction;
		if(instruction)           //如果指令是1，则删除第ist_i个指令插入的数
		{
			int ist_i;
			cin>>ist_i;
			//multiset<int>::iterator maxp,minp;
			for (int i = 0; i < ddem; i++) {         //用二进制i  遍历所有可能的维度符号状态 
				int sum = 0;
				for (int k = 0; k < dem; k++) {      //dem维度，考察每个点、每个维度下的符号状态
					int t = i & (1 << k);            //i与2^k进行与运算，  例如i=110 则k=0,运算结果是0; k=1,k=2 运算结果是1 
												     //效果是判断i的第k为是否为1
					if (t) sum += Manhatdata[ist_i][k];       //如果为1，则累加入sum 反之，则减去
					else sum -= Manhatdata[ist_i][k];
				}
				//max_ms[i].erase(tpn);
				min_ms[i].erase(min_ms[i].find(sum));
			}			
			Manhatdata.erase(ist_i);
			//cout<<maxManhattan<<endl;			
		}	
		else                      //如果指令是0，则插入数组data_now
		{   
			//multiset<int>::iterator maxp,minp;
			for(int dem_i=0;dem_i<dem;dem_i++)  {	
				cin>>data_now;
				Manhatdata[ii][dem_i]=data_now;
			}
			for (int i = 0; i < ddem; i++) {          //用二进制i  遍历所有可能的维度符号状态 
				int sum = 0;
				for (int k = 0; k < dem; k++) {      //dem维度，考察每个点、每个维度下的符号状态
					int t = i & (1 << k);            //i与2^k进行与运算，  例如i=110 则k=0,运算结果是0; k=1,k=2 运算结果是1 
												     //效果是判断i的第k为是否为1
					if (t) sum += Manhatdata[ii][k];       //如果为1，则累加入sum 反之，则减去
					else sum -= Manhatdata[ii][k];
				}
				min_ms[i].insert(sum);
			}
			//cout<<maxManhattan<<endl;	
		}
 
		for (int i = 0; i < ddem; i++)//插入或删除后，更新最大曼哈顿距离
		{
			int maxtan_now=(*min_ms[i].rbegin())-(*min_ms[i].begin());
			if(maxtan_now>maxManhattan) maxManhattan=maxtan_now;
		}			
		result[ii-1] = maxManhattan;
	}
 
	for (int k=0; k<n; k++)  printf("%d\n",result[k]);
    cin>>n;
	return 0;
}