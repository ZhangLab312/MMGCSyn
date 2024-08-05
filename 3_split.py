import os
import random

import pandas


def get_indexs(dataset,combination, sets, type):
    random.seed(42)
    indexs = []
    # 五折交叉验证划分
    if type == '5-fold':
        lenth = combination.shape[0]
        pot = int(lenth / 5)
        random_num = random.sample(range(0, lenth), lenth)
        for i in range(5):
            train_num = random_num[:pot * i] + random_num[pot * (i + 1):]
            test_num = random_num[pot * i:pot * (i + 1)]
            indexs.append((f"{type}", train_num, test_num))
    # 留一法留下10%的药物组合
    elif type == 'leave_combination':
        lenth = len(sets['combinations'])
        test_drug_index = random.sample(range(0, lenth), int(lenth / 20))
        test_drug_name = [sets['combinations'][i] for i in test_drug_index]

        train_num = []
        test_num = []
        for i, row in combination.iterrows():

            drug1 = row['drug1']
            drug2 = row['drug2']
            cell = row['cell']
            label = row['label']

            if (drug1, drug2) in test_drug_name:
                test_num.append(i)
            else:
                train_num.append(i)
        indexs.append((f"{type}", train_num, test_num))
    # # 留一法留下10%的某种药物
    elif type == 'leave_drug':
        lenth = len(sets['drugs'])
        test_drug_index = random.sample(range(0, lenth), int(lenth / 20))
        test_drug_name = [sets['drugs'][i] for i in test_drug_index]

        train_num = []
        test_num = []
        for i, row in combination.iterrows():

            drug1 = row['drug1']
            drug2 = row['drug2']
            cell = row['cell']
            label = row['label']

            if drug1 in test_drug_name:
                test_num.append(i)
            elif drug2 in test_drug_name:
                test_num.append(i)
            else:
                train_num.append(i)
        indexs.append((f"{type}", train_num, test_num))
    # # 留一法留下10%的细胞系
    elif type == 'leave_cell':
        lenth = len(sets['cells'])
        test_drug_index = random.sample(range(0, lenth), int(lenth / 20))
        test_cell_name = [sets['cells'][i] for i in test_drug_index]

        train_num = []
        test_num = []
        for i, row in combination.iterrows():

            drug1 = row['drug1']
            drug2 = row['drug2']
            cell = row['cell']
            label = row['label']

            if cell in test_cell_name:
                test_num.append(i)
            else:
                train_num.append(i)
        indexs.append((f"{type}", train_num, test_num))
    for i, key in enumerate(indexs):
        name = key[0]
        train = key[1]
        test = key[2]
        if not os.path.exists(f'{dataset}/{name}'):
            os.mkdir(f'{dataset}/{name}')
        pandas.DataFrame(train).to_csv(f'{dataset}/{name}/{name}_train_{i}.csv', index=False, header=['id'])
        pandas.DataFrame(test).to_csv(f'{dataset}/{name}/{name}_test_{i}.csv', index=False, header=['id'])
    print(5)


if __name__ == '__main__':
    dataset = 'onei'

    combination = pandas.read_csv(f'{dataset}/synergy.csv')

    sets = {
        'drugs': [],
        'cells': [],
        'combinations': []
    }
    for i,row in combination.iterrows():

        drug1 = row['drug1']
        drug2 = row['drug2']
        cell = row['cell']
        label = row['label']

        if drug1 not in sets['drugs']:
            sets['drugs'].append(drug1)
        if drug2 not in sets['drugs']:
            sets['drugs'].append(drug2)
        if cell not in sets['cells']:
            sets['cells'].append(cell)
        if (drug1, drug2) not in sets['combinations']:
            sets['combinations'].append((drug1, drug2))

    types = ['5-fold', 'leave_drug', 'leave_combination', 'leave_cell']
    for type in types:
        get_indexs(dataset,combination, sets, type)
