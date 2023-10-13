from ucimlrepo import fetch_ucirepo
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth

col_used = [1, 3, 9, 14]  # indicate the 4 columns used in my analysis


# Apriori Generate Candidates
def generate_candidates(itemset, k):
    candidates = []
    for i in range(len(itemset)):
        for j in range(i + 1, len(itemset)):
            candidate = itemset[i] | itemset[j]
            if len(candidate) == k + 1:
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
        itemset = frequent_candidates
        if len(frequent_candidates) == 0:
            break
    return frequent_itemsets


# FP-growth functions
class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue  # node name
        self.count = numOccur  # counter
        self.nodeLink = None  # nodelink
        self.parent = parentNode  # needs to be updated
        self.children = {}  # children nodes

    def inc(self, numOccur):
        self.count += numOccur

    def disp(self, ind=1):
        print('  ' * ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind + 1)


def createTree(dataSet, minSup=1):
    headerTable = {}
    # scan the first time
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    headerTable = {k: v for k, v in headerTable.items() if v >= minSup}
    freqItemSet = set(headerTable.keys())
    # print ('freqItemSet: ',freqItemSet)
    if len(freqItemSet) == 0: return None, None
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]
    # print ('headerTable: ',headerTable)
    # scan the second time
    retTree = treeNode('Null Set', 1, None)
    for tranSet, count in dataSet.items():
        localD = {}
        for item in tranSet:  # put transaction items in order
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
            updateTree(orderedItems, retTree, headerTable, count)
    return retTree, headerTable


def updateTree(items, inTree, headerTable, count):
    if items[0] in inTree.children:
        inTree.children[items[0]].inc(count)
    else:
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        if headerTable[items[0]][1] == None:  # renew the head node
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)


def updateHeader(nodeToTest, targetNode):
    while (nodeToTest.nodeLink != None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode


def ascendTree(leafNode, prefixPath):
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)


def findPrefixPath(basePat, treeNode):  # treeNode comes from header table
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats


def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1][0])]
    for basePat in bigL:
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        print('finalFrequent Item: ', newFreqSet)
        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        print('condPattBases :', basePat, condPattBases)
        # 2. Create FP-tree
        myCondTree, myHead = createTree(condPattBases, minSup)
        #         print ('head from conditional tree: ', myHead)
        if myHead != None:  # 3 find conditional FP-tree
            print('conditional tree for: ', newFreqSet)
            myCondTree.disp(1)
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)


def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = retDict.get(frozenset(trans), 0) + 1
    return retDict


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
# transform dataframe to list to store data
for row in df.iterrows():
    temp = []
    for i in range(len(col_used)):
        transactions[i].append(row[1][col_used[i]])
        temp.append(row[1][col_used[i]])
    data.append(temp)  # data:[[row1], [row2], [row3],...]
min_support = 5000


# Apriori starts
# frequent_itemsets = Apriori(transactions, min_support, data)
# print("Frequent Itemsets:")
# for itemset, support in frequent_itemsets:
#     print(itemset, "Support:", support)

# FP-growth starts
initSet = createInitSet(data)
myFPtree, myHeaderTab = createTree(initSet, min_support)
myFPtree.disp()
myFreqList = []
mineTree(myFPtree, myHeaderTab, min_support, set([]), myFreqList)
