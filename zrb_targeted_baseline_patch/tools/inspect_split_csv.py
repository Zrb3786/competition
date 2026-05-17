#!/usr/bin/env python3
from __future__ import annotations
import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split_csv', required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.split_csv)
    df = df[df['split'].astype(str).str.lower().eq('train')].copy()
    df['label3'] = df['label3'].astype(int)
    df['PHQ-9'] = df['PHQ-9'].astype(float)
    print('shape:', df.shape)
    print('\nlabel3 counts:')
    print(df['label3'].value_counts().sort_index())
    print('\nPHQ range by label3:')
    print(df.groupby('label3')['PHQ-9'].agg(['count','min','max','mean']))
    bad = df[((df['label3']==0)&(df['PHQ-9']>=5)) | ((df['label3']==1)&((df['PHQ-9']<5)|(df['PHQ-9']>=10))) | ((df['label3']==2)&(df['PHQ-9']<10))]
    print('\nlabel3/PHQ inconsistent rows:')
    print(bad[['ID','split','label2','label3','PHQ-9']] if len(bad) else bad)
    if len(bad):
        raise SystemExit(2)

if __name__ == '__main__':
    main()
