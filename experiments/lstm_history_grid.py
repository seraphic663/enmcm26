"""Full history-structure ablation for the q80 residual LSTM.

Fixed across every row:
- frozen Dynamic base forecast;
- tau=0.8 pinball loss;
- hidden size 32 per LSTM branch;
- monthly causal refit, last 7 prior days held out for validation;
- 3 fixed seeds;
- identical repository physical.solve_physical_day and P2 settlement;
- 334-day Feb-Dec evaluation with continuous SOC.

History structures:
A) all available prior days;
B) recent caps L in {7,14,28,56,84};
C) same-weekday caps K in {1,2,4,8,12};
D) every B x C dual-branch combination (25 architectures).

When less than L/K history exists, use all history causally available at that time.
"""

from __future__ import annotations
import csv, json, math, sys, types
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

ROOT = Path(__file__).resolve().parents[1]
pkg = types.ModuleType("mcm_code")
pkg.__path__ = [str(ROOT / "code")]
pkg.__package__ = "mcm_code"
sys.modules["mcm_code"] = pkg

from mcm_code.common import dataset as common
from mcm_code.common import physical
from mcm_code.result.problem2.solve_problem2_yesterday_dynamic import (
    FINAL_DYNAMIC_CONFIG, YESTERDAY_BASELINE_CONFIG, choose_candidate, run_config
)

SEEDS=(42,2026,3407)
RECENT=(7,14,28,56,84)
WEEKLY=(1,2,4,8,12)
TAU=0.8
HIDDEN=32
MAX_EPOCHS=80
PATIENCE=8
BATCH=16
LR=1e-3
WEIGHT_DECAY=1e-4
OUTPUT_START_DAY=31

def calendar_feature(day, dates):
    ts=dates[day]; doy=int(ts.dayofyear); wd=int(ts.dayofweek)
    out=np.zeros(9,np.float32)
    out[0]=math.sin(2*math.pi*doy/365.0); out[1]=math.cos(2*math.pi*doy/365.0)
    out[2+wd]=1.0
    return out

def hist_indices(day, dates, mode, cap=None):
    if day<=0: return np.array([],dtype=int)
    if mode=="all":
        return np.arange(day,dtype=int)
    if mode=="recent":
        return np.arange(max(0,day-int(cap)),day,dtype=int)
    if mode=="weekly":
        wd=dates[day].dayofweek
        idx=np.flatnonzero(dates[:day].dayofweek==wd)
        return idx[-int(cap):]
    raise ValueError(mode)

def arch_specs():
    xs=[{"name":"all","recent":None,"weekly":None,"all":True}]
    xs += [{"name":f"recent_{L}","recent":L,"weekly":None,"all":False} for L in RECENT]
    xs += [{"name":f"weekly_{K}","recent":None,"weekly":K,"all":False} for K in WEEKLY]
    xs += [{"name":f"recent_{L}+weekly_{K}","recent":L,"weekly":K,"all":False}
           for L in RECENT for K in WEEKLY]
    return xs

class HistQ80(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec=spec
        self.all_lstm = nn.LSTM(144,HIDDEN,batch_first=True) if spec["all"] else None
        self.recent_lstm = nn.LSTM(144,HIDDEN,batch_first=True) if spec["recent"] is not None else None
        self.weekly_lstm = nn.LSTM(144,HIDDEN,batch_first=True) if spec["weekly"] is not None else None
        branches=int(spec["all"])+int(spec["recent"] is not None)+int(spec["weekly"] is not None)
        self.head=nn.Sequential(nn.Linear(branches*HIDDEN+144+9,64),nn.ReLU(),nn.Linear(64,144))

    @staticmethod
    def encode(lstm, seqs):
        lens=torch.tensor([len(x) for x in seqs],dtype=torch.long)
        padded=pad_sequence(seqs,batch_first=True)
        packed=pack_padded_sequence(padded,lens.cpu(),batch_first=True,enforce_sorted=False)
        _,(h,_)=lstm(packed)
        return h[-1]

    def forward(self, all_s, recent_s, weekly_s, cal, base):
        z=[]
        if self.all_lstm is not None: z.append(self.encode(self.all_lstm,all_s))
        if self.recent_lstm is not None: z.append(self.encode(self.recent_lstm,recent_s))
        if self.weekly_lstm is not None: z.append(self.encode(self.weekly_lstm,weekly_s))
        z += [base,cal]
        return self.head(torch.cat(z,dim=1))

def pinball(pred,target):
    e=target-pred
    return torch.maximum(TAU*e,(TAU-1.0)*e).mean()

@dataclass
class Fitted:
    model: HistQ80
    rmean: np.ndarray
    rstd: np.ndarray
    nmean: np.ndarray
    nstd: np.ndarray
    best_epoch:int
    val_loss:float

def make_examples(days,spec,resid,base_net,net,dates,rmean,rstd,nmean,nstd):
    ex=[]
    for d in days:
        def seq(mode,cap):
            idx=hist_indices(int(d),dates,mode,cap)
            if len(idx)==0: raise RuntimeError(f"empty history day={d} mode={mode}")
            return torch.tensor((resid[idx]-rmean)/rstd,dtype=torch.float32)
        a=seq("all",None) if spec["all"] else None
        r=seq("recent",spec["recent"]) if spec["recent"] is not None else None
        w=seq("weekly",spec["weekly"]) if spec["weekly"] is not None else None
        cal=torch.tensor(calendar_feature(int(d),dates),dtype=torch.float32)
        base=torch.tensor((base_net[d]-nmean)/nstd,dtype=torch.float32)
        y=torch.tensor((resid[d]-rmean)/rstd,dtype=torch.float32)
        ex.append((a,r,w,cal,base,y))
    return ex

def batchify(ex,idx):
    rows=[ex[int(i)] for i in idx]
    def gather(k):
        vals=[x[k] for x in rows]
        return vals if vals[0] is not None else None
    return gather(0),gather(1),gather(2),torch.stack([x[3] for x in rows]),torch.stack([x[4] for x in rows]),torch.stack([x[5] for x in rows])

def fit_month(spec,resid,base_net,net,dates,start,seed):
    fit_end=start-7
    train_days=np.arange(1,fit_end,dtype=int)
    val_days=np.arange(fit_end,start,dtype=int)
    rmean=resid[:fit_end].mean(0); rstd=np.maximum(resid[:fit_end].std(0),1e-3)
    nmean=net[:fit_end].mean(0); nstd=np.maximum(net[:fit_end].std(0),1e-3)
    tr=make_examples(train_days,spec,resid,base_net,net,dates,rmean,rstd,nmean,nstd)
    va=make_examples(val_days,spec,resid,base_net,net,dates,rmean,rstd,nmean,nstd)
    torch.manual_seed(seed); np.random.seed(seed)
    model=HistQ80(spec); opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    gen=torch.Generator().manual_seed(seed+start)
    best=float("inf"); best_state=None; best_epoch=0; bad=0
    for epoch in range(1,MAX_EPOCHS+1):
        model.train()
        order=torch.randperm(len(tr),generator=gen)
        for pos in range(0,len(order),BATCH):
            aa,rr,ww,c,b,y=batchify(tr,order[pos:pos+BATCH])
            opt.zero_grad(set_to_none=True)
            p=model(aa,rr,ww,c,b); loss=pinball(p,y); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        model.eval()
        vals=[]
        with torch.no_grad():
            for pos in range(0,len(va),BATCH):
                ids=np.arange(pos,min(pos+BATCH,len(va)))
                aa,rr,ww,c,b,y=batchify(va,ids)
                vals.append(float(pinball(model(aa,rr,ww,c,b),y))*len(ids))
        vl=sum(vals)/len(va)
        if vl<best-1e-6:
            best=vl; best_epoch=epoch; best_state={k:v.detach().clone() for k,v in model.state_dict().items()}; bad=0
        else:
            bad+=1
            if bad>=PATIENCE: break
    model.load_state_dict(best_state); model.eval()
    return Fitted(model,rmean,rstd,nmean,nstd,best_epoch,best)

def predict_one(fit,spec,resid,base_net,net,dates,d):
    ex=make_examples(np.array([d]),spec,resid,base_net,net,dates,fit.rmean,fit.rstd,fit.nmean,fit.nstd)
    aa,rr,ww,c,b,_=batchify(ex,[0])
    with torch.no_grad(): z=fit.model(aa,rr,ww,c,b).numpy()[0]
    z=np.clip(z,-5,5)
    return base_net[d] + fit.rmean + fit.rstd*z

def month_ranges(dates):
    out=[]
    for m in range(2,13):
        s=int(np.flatnonzero((dates.month==m)&(dates.day==1))[0])
        e=int(np.flatnonzero((dates.month==m+1)&(dates.day==1))[0]) if m<12 else len(dates)
        out.append((m,s,e))
    return out

def build_pred(spec,seed,resid,base_net,net,dates):
    pred=np.full_like(net,np.nan)
    meta=[]
    for m,s,e in month_ranges(dates):
        fit=fit_month(spec,resid,base_net,net,dates,s,seed)
        for d in range(s,e):
            pred[d]=np.clip(predict_one(fit,spec,resid,base_net,net,dates,d),-2500,3000)
        meta.append({"month":m,"best_epoch":fit.best_epoch,"val_loss":fit.val_loss})
    return pred,meta

def solve_net(raw,prices,soc):
    load=np.maximum(raw,0.0); pv=np.maximum(-raw,0.0)
    try: return physical.solve_physical_day(load,pv,prices,soc,3.0)
    except RuntimeError: return physical.solve_physical_day(load,pv,prices,soc,20.0)

def simulate(pred,net,prices,initial_soc):
    soc=float(initial_soc); pc=ec=ek=0.0; days=0; daily=[]
    for d in range(OUTPUT_START_DAY,len(net)):
        plan=solve_net(pred[d],prices,soc)
        gap=net[d]+plan.charge-plan.x-plan.discharge
        emergency=np.maximum(gap,0.0)
        p=float(np.dot(prices,plan.x)); e=float(np.dot(5.0*prices,emergency))
        pc+=p; ec+=e; ek+=float(emergency.sum()); days+=int(emergency.sum()>1e-7); daily.append(p+e)
        soc=float(plan.soc[-1])
    actual=net[OUTPUT_START_DAY:]; pr=pred[OUTPUT_START_DAY:]; err=actual-pr
    return {
      "forecast_mae_kwh":float(np.mean(np.abs(err))),
      "forecast_rmse_kwh":float(np.sqrt(np.mean(err**2))),
      "coverage":float(np.mean(actual<=pr)),
      "underprediction_mean_kwh":float(np.mean(np.maximum(err,0))),
      "planned_cost_yuan":pc,"emergency_purchase_kwh":ek,"emergency_cost_yuan":ec,
      "total_cost_yuan":pc+ec,"days_with_emergency":days,"final_soc_kwh":soc,
      "max_daily_cost_yuan":float(max(daily))
    }

def main():
    torch.set_num_threads(2)
    paths=common.get_paths(); data=common.read_inputs(paths); cold_load,cold_pv=physical.read_cold_start(paths)
    y=run_config(data,cold_load,cold_pv,YESTERDAY_BASELINE_CONFIG,20.0)
    dyn=run_config(data,cold_load,cold_pv,FINAL_DYNAMIC_CONFIG,20.0)
    expected_y=22823747.29284119; expected_d=15069856.025188675; expected_e=318742.2694089983
    gate={
      "yesterday_total":float(y.summary["total_cost_yuan"]),
      "dynamic_total":float(dyn.summary["total_cost_yuan"]),
      "dynamic_emergency":float(dyn.summary["emergency_purchase_kwh"])
    }
    gate["yesterday_abs_diff"]=abs(gate["yesterday_total"]-expected_y)
    gate["dynamic_total_rel_diff"]=abs(gate["dynamic_total"]-expected_d)/expected_d
    gate["dynamic_emergency_rel_diff"]=abs(gate["dynamic_emergency"]-expected_e)/expected_e
    print("REPRO_JSON="+json.dumps(gate),flush=True)
    if gate["yesterday_abs_diff"]>1e-4 or gate["dynamic_total_rel_diff"]>1e-4 or gate["dynamic_emergency_rel_diff"]>1e-4:
        raise RuntimeError("reproduction gate failed")

    net=np.asarray(data.load_kwh-data.pv_kwh,float)
    base=np.zeros_like(net)
    for d in range(len(net)):
        ch=choose_candidate(d,data,cold_load,cold_pv,FINAL_DYNAMIC_CONFIG)
        base[d]=ch.load-ch.pv
    resid=net-base
    initial_soc=float(dyn.daily.iloc[OUTPUT_START_DAY]["initial_soc_kwh"])
    baseline={
      "total_cost_yuan":float(dyn.summary["total_cost_yuan"]),
      "emergency_purchase_kwh":float(dyn.summary["emergency_purchase_kwh"]),
      "planned_cost_yuan":float(dyn.summary["planned_cost_yuan"]),
      "emergency_cost_yuan":float(dyn.summary["emergency_cost_yuan"]),
      "forecast_mae_kwh":float(np.mean(np.abs(resid[OUTPUT_START_DAY:]))),
      "initial_soc_feb1":initial_soc
    }
    print("BASELINE_JSON="+json.dumps(baseline),flush=True)

    specs=arch_specs()
    print("ARCH_COUNT="+str(len(specs)),flush=True)
    results=[]
    for ai,spec in enumerate(specs,1):
        for seed in SEEDS:
            print(f"START arch={ai}/{len(specs)} {spec['name']} seed={seed}",flush=True)
            pred,meta=build_pred(spec,seed,resid,base,net,data.dates)
            met=simulate(pred,net,data.prices,initial_soc)
            row={"arch":spec["name"],"seed":seed,**spec,**met}
            results.append(row)
            print("RESULT_JSON="+json.dumps(row),flush=True)

    aggs=[]
    for spec in specs:
        rows=[r for r in results if r["arch"]==spec["name"]]
        a={"arch":spec["name"],**spec}
        for k in ("forecast_mae_kwh","forecast_rmse_kwh","coverage","underprediction_mean_kwh",
                  "planned_cost_yuan","emergency_purchase_kwh","emergency_cost_yuan","total_cost_yuan",
                  "days_with_emergency","max_daily_cost_yuan"):
            v=np.array([r[k] for r in rows],float)
            a[k+"_mean"]=float(v.mean()); a[k+"_std"]=float(v.std(ddof=1))
            a[k+"_min"]=float(v.min()); a[k+"_max"]=float(v.max())
        a["cost_vs_dynamic_pct"]=100*(a["total_cost_yuan_mean"]/baseline["total_cost_yuan"]-1)
        a["emergency_vs_dynamic_pct"]=100*(a["emergency_purchase_kwh_mean"]/baseline["emergency_purchase_kwh"]-1)
        aggs.append(a)
    aggs.sort(key=lambda x:x["total_cost_yuan_mean"])
    final={"protocol":{"seeds":SEEDS,"recent_caps":RECENT,"weekly_caps":WEEKLY,"tau":TAU,
                       "hidden_per_branch":HIDDEN,"architectures":len(specs),"runs":len(results),
                       "history_shortfall":"use all causally available history",
                       "period":["2025-02-01","2025-12-31"],"days":334},
           "reproduction":gate,"dynamic":baseline,"results":results,"aggregate":aggs}
    out=ROOT/"history-grid-results.json"
    out.write_text(json.dumps(final,ensure_ascii=False,indent=2),encoding="utf-8")
    csvp=ROOT/"history-grid-aggregate.csv"
    keys=list(aggs[0].keys())
    with csvp.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(aggs)
    print("TOP10_JSON="+json.dumps(aggs[:10]),flush=True)
    print("FINAL_SUMMARY_JSON="+json.dumps({"best":aggs[0],"arch_count":len(specs),"run_count":len(results)}),flush=True)

if __name__=="__main__": main()
