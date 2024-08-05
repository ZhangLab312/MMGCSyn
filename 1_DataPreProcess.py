import os.path
import re

import pandas


def drop_non_smiles(synergy, drug_name):
    drug1 = synergy['drug1']
    drug2 = synergy['drug2']
    flag = drug1.isin(drug_name) * drug2.isin(drug_name)
    return synergy[flag]


def drop_non_cell(synergy, cell_name):
    cell_name = list(map(lambda x: x.split('_')[0], cell_name))
    # 去除cell列中的所有符号，仅保留字母和数字
    pattern = re.compile(r'[\W_]+')

    def keep_alphanumeric(text):
        return pattern.sub('', text)

    synergy['cell'] = synergy['cell'].apply(keep_alphanumeric)

    flag = synergy['cell'].isin(cell_name)
    return synergy[flag]


def score2label(data):
    if data > 10:
        return 1
    elif data < 0:
        return 0
    else:
        print("score is out of the threshold")


if __name__ == '__main__':
    root = 'drugcomb'

    synergy = pandas.read_csv(os.path.join(root, 'raw.csv'))
    drug = pandas.read_csv(os.path.join(root, 'drugInfo_deepdds.csv'))
    cell = pandas.read_csv('cell.csv').iloc[2:, :]
    # 去除含有nan的行
    synergy = synergy.dropna()
    # 根据药物smiles，去除无法找到药物smiles的行
    synergy = drop_non_smiles(synergy, drug['names'].tolist())
    # 根据cell expression，去除无法找到cell的行
    synergy = drop_non_cell(synergy, cell['gene_id'])
    # 根据阈值选取数据，去除噪声   （>10 正样本   < 0 负样本）
    synergy = synergy[(synergy['score'] > 10) | (synergy['score'] < 0)]
    # 保存文件
    result = pandas.DataFrame({})
    result['drug1'] = synergy['drug1']
    result['drug2'] = synergy['drug2']
    result['cell'] = synergy['cell']
    result['score'] = synergy['score']
    result['label'] = synergy['score'].apply(score2label)
    result.to_csv(os.path.join(root, 'synergy.csv'), index=False)

