import os
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import gc

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils import seedset, dataload, FocalLoss, ECG_Dataset
from src.model import ResNetLead8, prevResNet, ResNet, LSTM
from src.process import Supervised
from src.config import diseaselabs

tqdm.pandas()

def Downstream(args):
    seedset(args.seed)
        
    if args.device != 'cpu':
        device = f'cuda:{args.device}'
    else:
        device = args.device
    
    print("Data Loading Start")
    with open('../usedata/traindata/diabetes.pkl', 'rb') as f:
        data = pickle.load(f)
        data['age'] = data['age'].astype(float)
        data = data[data['age'] >= 18]
        data['waveflag'] = data['waveform'].progress_apply(lambda x: 1 if np.isnan(x).sum() == 0 else 0)
        data = data[data['waveflag'] == 1][data.columns.difference(['waveflag'], sort=False)]
        data = data.sort_values(['PT_NO', 'date'])
        data['drank'] = data.groupby(['PT_NO']).cumcount()
        data = data[data['drank'] == 0][data.columns.difference(['drank'], sort=False)]
        args.disease = 'diabetes'
    
    cci = pd.read_csv('../usedata/cci.csv')
    def diseaselabel(v):
        if 'E1' in v:
            return 'diabetes'
        elif 'I2' in v:
            return 'MI'
        elif 'I50' in v:
            return 'HF'
        elif 'I7' in v:
            return 'peripheral'
        elif 'I6' in v:
            return 'cerebrovascular'
        elif 'F03' in v or 'G30' in v:
            return 'dementia'
        elif 'J' in v:
            return 'pulmonary'
        elif 'M' in v:
            return 'connective'
        elif 'K2' in v:
            return 'ulcer'
        elif 'B1' in v or 'K7' in v:
            return 'liver'
        elif 'G8' in v:
            return 'plegia'
        elif 'N1' in v:
            return 'renal'
        elif any(f'C{i}' in v for i in range(76, 81)):
            return 'metastatic'
        elif 'C' in v:
            return 'malignancy'
        elif 'B2' in v:
            return 'HIV'

    cci = pd.read_csv('/workspace/usedata/cci.csv')
    cci['disease'] = cci['ICD10코드'].apply(lambda x: diseaselabel(x))
    cci = cci[cci['disease'].notnull()]
    cci['cond_date'] = pd.to_datetime(cci['첫 진단일자'])
    cci = cci.sort_values(['PT_NO', 'disease', 'cond_date'])
    cci['drank'] = cci.groupby(['PT_NO', 'disease']).cumcount()+1
    cci = cci[cci['drank'] == 1][cci.columns.difference(['drank'], sort=False)]
    ccimerge = pd.merge(data, cci, on='PT_NO', how='left')
    ccimerge = ccimerge[ccimerge['disease'].notnull()][['PT_NO', 'date', 'disease', 'cond_date']]
    ccimerge['diff'] = (pd.to_datetime(ccimerge['date']) - pd.to_datetime(ccimerge['cond_date'])).dt.days
    ccimerge = ccimerge[ccimerge['diff'] > 0]
    ds = pd.unique(ccimerge['disease'])

    pd.set_option('mode.chained_assignment',  None)
    ccidfset = []
    for n, pid in tqdm(enumerate(pd.unique(ccimerge['PT_NO']))):
        ccidf = pd.DataFrame(columns=['PT_NO', 'date'] + list(ds))

        tmp = ccimerge[(ccimerge['PT_NO'] == pid)]
        tmp = tmp.groupby(['PT_NO', 'date']).apply(lambda x: list(set(x['disease'])))
        for _n, t in enumerate(tmp):
            ccidf.loc[_n] = [tmp.index[_n][0], tmp.index[_n][1]] + [d in t for d in ds]
        
        ccidfset.append(ccidf)
        
    ccidf = pd.concat(ccidfset)
    data = pd.merge(data, ccidf, on=['PT_NO', 'date'], how='left')
    data = data.fillna(False)
    bw, gn = data[data['site'] != 'KANGNAM CENTER'], data[data['site'] == 'KANGNAM CENTER']

    # with open('../usedata/traindata/diabetes_bd.pkl', 'rb') as f:
    #     conti_data_bd = pickle.load(f)
    #     conti_data_bd['age'] = conti_data_bd['age'].astype(float)
    #     conti_data_bd['waveflag'] = conti_data_bd['waveform'].progress_apply(lambda x: 1 if np.isnan(x[0]).sum() == 0 and np.isnan(x[1]).sum() == 0 else 0)
    #     conti_data_bd = conti_data_bd[conti_data_bd['waveflag'] == 1][conti_data_bd.columns.difference(['waveflag'], sort=False)]
    #     args.disease = 'diabetes'
    #     ds = ECG_Dataset(conti_data_bd, args.disease)
    #     bd_loader = DataLoader(ds, batch_size=args.bs, shuffle=True)

    train_loader, val_loader, test_loader = dataload(gn, args)
    ds = ECG_Dataset(bw, args.disease)
    bw_loader = DataLoader(ds, batch_size=args.bs, shuffle=True)
    with open('../tmp/valtest.pkl', 'wb') as f:
        pickle.dump((test_loader, bw_loader), f, pickle.HIGHEST_PROTOCOL)

    print("Done")
    # logpth = f'../results/{args.bs}/{args.disease}/logs/'
    logpth = f'../results/{args.bs}/{args.disease}/logs_lstm/'
    os.makedirs(logpth, exist_ok=True)
    for conti in (0,):
        args.conti = conti
        for lead in (12,):
            for hid_dim in (256,):
                args.hid_dim = hid_dim
                for which in ('both',):
                    args.lead, args.which = lead, which
                    if which == 'demo':
                        params = f'{args.which}'
                    else:
                        params = f'{args.conti}_{args.lead}_{args.which}_{args.hid_dim}'
                    if os.path.exists(os.path.join(logpth, f'{params}.log')): continue

                    savepath = f'../results/{args.bs}/{args.disease}/saved/{params}/'
                    os.makedirs(savepath, exist_ok=True)

                    logger = logging.getLogger()
                    logger.setLevel(logging.INFO)
                    handler = logging.FileHandler(os.path.join(logpth, f'{params}.log'))
                    handler.setLevel(logging.INFO)
                    formatter = logging.Formatter("%(asctime)s %(levelname)s:%(message)s")
                    handler.setFormatter(formatter)
                    logger.addHandler(handler)

                    # model = ResNet(fc_dim=args.hid_dim, lead=args.lead)
                    model = LSTM()
                    
                    # if args.conti:
                    #     model = prevResNet(fc_dim=args.hid_dim, lead=args.lead)
                    # else:
                    #     if args.which == 'demo':
                    #         model = ResNetLead8(fc_dim=args.hid_dim, lead=args.lead, demo=True)
                    #     else:
                    #         model = ResNetLead8(fc_dim=args.hid_dim, lead=args.lead)

                    criterion = nn.CrossEntropyLoss()
                    # criterion = FocalLoss(gamma=2, alpha=[0.05, 1-0.05], device=device)
                    
                    Trainer = Supervised(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        criterion=criterion,
                        args=args,
                        savepath=savepath,
                        logger=logger
                    )
                    
                    load_model, _ = Trainer.train()
                    model.load_state_dict(load_model['model'])
                        
                    logger.info("\nInternal Validation Score\n")
                    loss, roc_auc, pr_auc = Trainer.evaluation(model, test_loader)
                    logger.info("Loss: {:.4f}, AUROC: {:.4f}, AUPRC: {:.4f}".format(loss, roc_auc, pr_auc))

                    logger.info("\nExternal Validation Score\n")
                    loss, roc_auc, pr_auc = Trainer.evaluation(model, bw_loader)
                    logger.info("Loss: {:.4f}, AUROC: {:.4f}, AUPRC: {:.4f}".format(loss, roc_auc, pr_auc))


                    logger.removeHandler(handler)
                    del logger, handler, model
                    gc.collect()
