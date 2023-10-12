from ucimlrepo import fetch_ucirepo
import pandas as pd
col_used = [1, 3 , 9, 14]  # indicate the 4 columns used in my analysis

# Apriori Generate Candidates
def generate_candidates(itemset, k):
    candidates = []
    for i in range(len(itemset)):
        for j in range(i + 1, len(itemset)):
            candidate = itemset[i] | itemset[j]
            if len(candidate) == k+1:
                candidates.append(candidate)
    return candidates


# Apriori Calculate support
def calculate_support(data, candidate):
    count = 0
    for transaction in data:
        flag = True  # if true, then itemset is a subset of transaction
        for i in candidate:
            if i not in transaction:
                flag = False
                break
        if flag:
            count += 1
    return count


# apriori algorithm
def Apriori(transactions, min_support, data):
    k = 1
    items = set()
    for i in range(len(transactions)):
        for transaction in transactions[i]:
            items.update({transaction})
    print(items)  # items: set of columns used, with no repeat in each set
    temp = list(items)
    itemset = []
    for i in temp:
        itemset.append({i})
    frequent_itemsets = []

    while True:
        candidates = generate_candidates(itemset, k)
        frequent_candidates = []
        for candidate in candidates:
            support = calculate_support(data, candidate)
            if support >= min_support:
                frequent_itemsets.append((candidate, support))
                frequent_candidates.append(candidate)
        k += 1
        print(k)
        print(frequent_candidates)
        itemset = frequent_candidates
        if len(frequent_candidates) == 0:
            break
    return frequent_itemsets


# fetch dataset
adult = fetch_ucirepo(id=2)
headers = []  # ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
# 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
# using 1workclass 3education 9sex 14income

# data preprocessing
# df: the dataset in pandas dataframe form
df = adult.data.original
data = []
transactions = []

for i in range(len(col_used)):
    transactions.append([])  # transactions:([], ... []), as many as the columns used
for i in df:
    headers.append(i)  # put all the headers in one list
print(headers)
# transform dataframe to list to store data
for row in df.iterrows():
    temp = []
    for i in range(len(col_used)):
        transactions[i].append(row[1][col_used[i]])
        temp.append(row[1][col_used[i]])
    data.append(temp)  # data:[[row1], [row2], [row3],...]
min_support = 5000
min_confidence = 0.7

# Apriori starts
frequent_itemsets = Apriori(transactions, min_support, data)
print("Frequent Itemsets:")
for itemset, support in frequent_itemsets:
    print(itemset, "Support:", support)
