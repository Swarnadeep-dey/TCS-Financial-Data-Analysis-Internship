"""Week 4 — Financial Forecasting & Predictive Modeling.
Run from repository root:
    python src/week4_forecasting.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/cleaned/quarterly_cleaned.csv'
OUT=ROOT/'data/model_outputs'; OUT.mkdir(parents=True,exist_ok=True)
COMPANIES=['TCS','Infosys','HCLTech','Wipro']
QUARTERS=['Q1 FY24','Q2 FY24','Q3 FY24','Q4 FY24','Q1 FY25','Q2 FY25','Q3 FY25','Q4 FY25','Q1 FY26','Q2 FY26','Q3 FY26','Q4 FY26','Q1 FY27']
MODEL_NAMES=['Naive','Drift','Holt damped','ARIMA(1,1,0)','AutoReg(1)']

def fit_predict(name,y,h):
    y=np.asarray(y,dtype=float)
    if name=='Naive': return np.repeat(y[-1],h)
    if name=='Drift':
        x=np.arange(len(y)); coef=np.polyfit(x,y,1); return np.polyval(coef,np.arange(len(y),len(y)+h))
    if name=='Holt damped': return np.asarray(Holt(y,damped_trend=True,initialization_method='estimated').fit().forecast(h),dtype=float)
    if name=='ARIMA(1,1,0)': return np.asarray(ARIMA(y,order=(1,1,0),trend='t').fit().forecast(h),dtype=float)
    if name=='AutoReg(1)': return np.asarray(AutoReg(y,lags=1,trend='ct',old_names=False).fit().predict(start=len(y),end=len(y)+h-1),dtype=float)
    raise ValueError(name)

def metrics(a,p): return (mean_absolute_error(a,p),np.sqrt(mean_squared_error(a,p)),mean_absolute_percentage_error(a,p)*100)

def main():
    df=pd.read_csv(DATA)
    rows=[]; fc=[]
    for target in ['Revenue (INR Cr)','Operating Margin (%)']:
        for company in COMPANIES:
            y=df[df.Company==company].sort_values('Period End')[target].astype(float).to_numpy()
            train,test=y[:-3],y[-3:]
            scored=[]
            for name in MODEL_NAMES:
                p=fit_predict(name,train,3); mae,rmse,mape=metrics(test,p); scored.append((mae,rmse,mape,name))
            scored.sort()
            selected=scored[0][3]
            for mae,rmse,mape,name in scored:
                rows.append({'Target':target,'Company':company,'Model':name,'MAE':mae,'RMSE':rmse,'MAPE (%)':mape,'Selection Status':'SELECTED' if name==selected else ''})
            pred=fit_predict(selected,y,4)
            for q,v in zip(['Q2 FY27','Q3 FY27','Q4 FY27','Q1 FY28'],pred): fc.append({'Target':target,'Company':company,'Forecast Quarter':q,'Forecast':v,'Selected Model':selected})
    pd.DataFrame(rows).to_csv(OUT/'model_selection_validation.csv',index=False)
    pd.DataFrame(fc).to_csv(OUT/'forecast_outputs.csv',index=False)
    print('Forecasting outputs created in',OUT)
if __name__=='__main__': main()
