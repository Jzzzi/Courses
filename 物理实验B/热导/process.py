import csv

def get_column(csv_file, column_number):
    column_data = []
    with open(csv_file, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        i = -1
        for row in csv_reader:
            i += 1
            # 检查行是否足够长，避免索引超出范围
            if len(row) >= column_number and i>0:
                column_data.append(row[column_number - 1])
    return column_data

# 示例用法
csv_file = '/home/jzi/Courses/物理实验B/热导/hhh0.csv'
column_number = 2  # 获取第2列的数据
ch1 = get_column(csv_file, column_number)
ch2 = get_column(csv_file, column_number+1)

total = 0
for i in range (len(ch1)):
    ch1[i] = float(ch1[i])
    ch2[i] = float(ch2[i])
    total += ch1[i]/0.05*ch2[i]*0.1

print(total)