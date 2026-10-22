"""Rebuild the paper's tables from annual solver evidence, without stored values."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from ..common.paths import output_root, protected_output_path
from ..common import dataset as p2
from ..common.intervals import emergency_intervals

DATES = ['2025-03-20', '2025-06-21', '2025-09-23', '2025-12-21']
HOURS = [10, 12, 14, 16, 18, 20]
FINAL = 'dynamic_full_seasonal_bias75_dense_delta'


def build_tables(result_dir, output):
    output = protected_output_path(output)
    output.mkdir(parents=True, exist_ok=True)
    evidence = result_dir/'来源证据'
    def read(name): return pd.read_csv(evidence/name)
    def emit(name, rows):
        frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
        frame.to_csv(output/name, index=False, encoding='utf-8-sig')
    def arrays(name):
        with np.load(evidence/name, allow_pickle=False) as saved:
            return {k: saved[k] for k in saved.files}
    def period(frame): return frame[frame.date >= '2025-02-01'].copy()
    def selected(frame): return frame[frame.date.isin(DATES)].copy()
    def interval(t): return f'{t//6}:{(t%6)*10:02d}-{(t+1)//6}:{((t+1)%6)*10:02d}'

    p1 = read('problem1/audit/problem1_schedule-A.csv')
    purchases = [{'interval': interval(h*6), 'purchase_kwh': p1.iloc[h*6].purchase_kwh} for h in HOURS]
    purchases += [{'interval': '全天购电量', 'purchase_kwh': p1.purchase_kwh.sum()}, {'interval': '全天购电费', 'purchase_kwh': (p1.purchase_kwh*p1.price_yuan_per_kwh).sum()}]
    emit('p1_purchase.csv', purchases)
    storage = [{'interval': f'{b*4}:00-{(b+1)*4}:00', 'charge_kwh': p1.iloc[b*24:(b+1)*24].charge_kwh.sum(), 'discharge_kwh': p1.iloc[b*24:(b+1)*24].discharge_kwh.sum()} for b in range(6)]
    storage += [{'interval': '0:00储电量', 'charge_kwh': p1.iloc[0].soc_start_kwh}, {'interval': '24:00储电量', 'charge_kwh': p1.iloc[-1].soc_end_kwh}]
    emit('p1_storage.csv', storage)

    actual_a = read('problem2/audit/problem2_A_schedule.csv')
    a = period(actual_a)
    planned = float((a.planned_purchase_kwh*a.price_yuan_per_kwh).sum())
    ecost = float((5*a.emergency_purchase_kwh*a.price_yuan_per_kwh).sum())
    compare = [dict(model='版本 A', information='全年实际信息', planned_purchase_kwh=a.planned_purchase_kwh.sum(), planned_cost_yuan=planned, emergency_purchase_kwh=a.emergency_purchase_kwh.sum(), emergency_cost_yuan=ecost, total_cost_yuan=planned+ecost, pv_curtailment_kwh=a.curtailed_pv_kwh.sum(), grid_spill_kwh=0.)]
    daily = read('problem2/formal/dynamic/problem2_yesterday_dynamic_daily.csv')
    labels = {'yesterday_baseline': 'Yesterday-MILP', 'yesterday_baseline_plus70': 'Yesterday-MILP+70', FINAL: 'Dynamic'}
    for variant, frame in period(daily).groupby('variant', sort=False):
        information = '历史信息、日期、初始储电量及近期残差' if variant == FINAL else '昨日前历史信息'
        compare.append(dict(model=labels[variant], information=information, planned_purchase_kwh=frame.planned_purchase_kwh.sum(), planned_cost_yuan=frame.planned_cost_yuan.sum(), emergency_purchase_kwh=frame.actual_emergency_kwh.sum(), emergency_cost_yuan=frame.actual_emergency_cost_yuan.sum(), total_cost_yuan=frame.total_cost_yuan.sum(), pv_curtailment_kwh=frame.pv_curtailment_kwh.sum(), grid_spill_kwh=frame.grid_spill_kwh.sum()))
    emit('p2_compare.csv', compare)
    dynamic = daily[daily.variant == FINAL]
    dyn = arrays('problem2/formal/dynamic/dynamic_intervals.npz')
    dynamic_days = selected(dynamic).rename(columns={'selected_candidate': 'candidate', 'actual_emergency_kwh': 'emergency_purchase_kwh', 'actual_emergency_cost_yuan': 'emergency_cost_yuan'})
    emit('p2_selected_days_dynamic.csv', dynamic_days[['date','candidate','delta_kwh_per_interval','planned_purchase_kwh','planned_cost_yuan','emergency_purchase_kwh','emergency_cost_yuan','total_cost_yuan','initial_soc_kwh','terminal_soc_kwh']])
    day_compare = []
    for day in DATES:
        f = actual_a[actual_a.date == day]
        cost = float((f.planned_purchase_kwh*f.price_yuan_per_kwh).sum())
        e = float((5*f.emergency_purchase_kwh*f.price_yuan_per_kwh).sum())
        day_compare.append(dict(date=day, model='版本 A', planned_cost_yuan=cost, emergency_purchase_kwh=f.emergency_purchase_kwh.sum(), emergency_cost_yuan=e, total_cost_yuan=cost+e, initial_soc_kwh=f.iloc[0].soc_start_kwh, terminal_soc_kwh=f.iloc[-1].soc_end_kwh))
        d = dynamic_days[dynamic_days.date == day].iloc[0]
        fields = ('date','planned_cost_yuan','emergency_purchase_kwh','emergency_cost_yuan','total_cost_yuan','delta_kwh_per_interval','initial_soc_kwh','terminal_soc_kwh')
        day_compare.append(dict(model='Dynamic', **{key:d[key] for key in fields}))
    emit('p2_selected_days.csv', day_compare)

    rolling = arrays('problem3/audit/rolling/problem3_rolling_daily.npz')
    p3daily = read('problem3/audit/rolling/problem3_rolling_daily.csv')
    emit('p3_daily_totals.csv', selected(p3daily))
    for tag, saved, frame, plan_key in [('p2', dyn, dynamic, 'x'), ('p3', rolling, p3daily, 'x_plan')]:
        purchase, store, emergency = [], [], []
        for day in DATES:
            i = list(saved['dates'].astype(str)).index(day)
            record = frame[frame.date == day].iloc[0]
            purchase += [dict(date=day, interval=interval(h*6), **{('purchase_kwh' if tag=='p2' else 'planned_purchase_kwh'): saved[plan_key][i,h*6]}) for h in HOURS]
            if tag == 'p2':
                purchase += [dict(date=day, interval='全天购电量', purchase_kwh=saved['x'][i].sum()), dict(date=day, interval='全天购电费', purchase_kwh=record.planned_cost_yuan)]
            store += [dict(date=day, interval=f'{b*4}:00-{(b+1)*4}:00', charge_kwh=saved['charge'][i,b*24:(b+1)*24].sum(), discharge_kwh=saved['discharge'][i,b*24:(b+1)*24].sum()) for b in range(6)]
            if tag == 'p2':
                store += [dict(date=day, interval='0:00储电量', charge_kwh=saved['soc'][i,0]),dict(date=day, interval='24:00储电量', charge_kwh=saved['soc'][i,-1])]
            emergency += [dict(date=day, interval=label, emergency_kwh=value) for label,value in emergency_intervals(saved['emergency'][i], 1e-6)]
        emit('p2_purchase.csv' if tag=='p2' else 'p3_purchase_intervals.csv', purchase)
        emit(tag+'_storage.csv', store)
        emit('p2_emergency.csv' if tag=='p2' else 'p3_emergency_intervals.csv', emergency)

    p4_2 = {tag: period(read(f'problem4/audit/p4_2/p4_2_{tag}_daily.csv')) for tag in ['att1','yest','wd2']}
    price_labels = {'att1':'附件1固定电价法（Att1）','yest':'前一日电价法（Yest）','wd2':'同星期历史日均值法（WD2）'}
    emit('p4_2_price_compare.csv', [dict(method=price_labels[tag], price_forecast_mae=f.price_forecast_mae.mean(), planned_purchase_kwh=f.planned_purchase_kwh.sum(), planned_cost_yuan=f.planned_cost_yuan.sum(), emergency_purchase_kwh=f.emergency_kwh.sum(), emergency_cost_yuan=f.emergency_cost_yuan.sum(), total_cost_yuan=f.total_cost_yuan.sum()) for tag,f in p4_2.items()])
    emit('p4_2_selected_days.csv', selected(p4_2['wd2'])[['date','planned_purchase_kwh','planned_cost_yuan','emergency_kwh']])
    p4_3 = {tag: period(read(f'problem4/audit/p4_3/p4_3_{tag}_daily.csv')) for tag in ['a_wd2','b0_yest','bstar_wd2','bfreeze_wd2']}
    strategy_labels = {'a_wd2':'A：不调整','b0_yest':'B0：前一日电价滚动调整','bstar_wd2':'bstar：储能同步调整','bfreeze_wd2':'bfreeze：冻结储能滚动调整'}
    emit('p4_3_strategy_compare.csv', [dict(strategy=strategy_labels[tag], price_method='Yest' if tag=='b0_yest' else 'WD2', planned_purchase_kwh=f.planned_purchase_kwh.sum(), actual_purchase_kwh=f.actual_purchase_kwh.sum(), contract_cost_yuan=f.contract_cost_yuan.sum(), emergency_purchase_kwh=f.emergency_kwh.sum(), emergency_cost_yuan=f.emergency_cost_yuan.sum(), total_cost_yuan=f.total_cost_yuan.sum(), accepted_adjustments=f.adjustment_accept_count.sum()) for tag,f in p4_3.items()])
    a4 = p4_3['a_wd2'].set_index('date'); b4 = p4_3['bfreeze_wd2'].set_index('date')
    joined = pd.DataFrame(dict(a_total_cost_yuan=a4.total_cost_yuan,bfreeze_total_cost_yuan=b4.total_cost_yuan,cost_difference_yuan=a4.total_cost_yuan-b4.total_cost_yuan,a_emergency_kwh=a4.emergency_kwh,bfreeze_emergency_kwh=b4.emergency_kwh))
    monthly = joined.groupby(joined.index.str[:7]).sum(); monthly.index.name='month'
    emit('p4_monthly_compare.csv', monthly.reset_index())
    typical = joined.loc[DATES].copy(); typical['bfreeze_accepted_adjustments'] = b4.loc[DATES].adjustment_accept_count
    emit('p4_typical_days.csv', typical.reset_index())

    actual = p2.read_inputs(p2.get_paths())
    def metrics(saved):
        u,v,soc = (saved[k] for k in ['charge','discharge','soc'])
        x = saved['x_act'] if 'x_act' in saved else saved['x']
        deficit = actual.load_kwh + u-v-actual.pv_kwh-x
        emergency = saved['emergency']
        balance = x+emergency+actual.pv_kwh+v-actual.load_kwh-u-np.maximum(-deficit,0.)
        return dict(soc=float(np.max(np.abs(np.diff(soc,axis=1)-(.9*u-v/.9)))),balance=float(np.max(np.abs(balance))),emergency=float(np.max(np.abs(emergency-np.maximum(deficit,0.)))),simultaneous=int(np.count_nonzero((u>1e-6)&(v>1e-6))),bad_soc=int(np.count_nonzero((soc.min(axis=1)<1200-1e-5)|(soc.max(axis=1)>10800+1e-5))),minimum=float(soc.min()),maximum=float(soc.max()),charge=float(u.max()),discharge=float(v.max()))
    m3 = metrics(rolling)
    cost_residual = float(np.max(np.abs(p3daily.contract_cost_yuan.to_numpy()+5*(rolling['emergency']*actual.prices).sum(axis=1)-p3daily.total_cost_yuan.to_numpy())))
    emit('p3_check.csv', [dict(metric=label,value=value,unit=unit) for label,value,unit in [('储电量递推残差',m3['soc'],'kWh'),('实际电量平衡残差',m3['balance'],'kWh'),('紧急购电量计算残差',m3['emergency'],'kWh'),('每日费用计算误差',cost_residual,'元'),('同时充放电时段数量',m3['simultaneous'],'数量'),('储电量越界天数',m3['bad_soc'],'天')]])
    m2 = metrics(arrays('problem4/audit/p4_2/p4_2_wd2_arrays.npz')); m4 = metrics(arrays('problem4/audit/p4_3/p4_3_bfreeze_wd2_arrays.npz'))
    checks=[('统计天数',len(p4_2['wd2']),len(p4_3['bfreeze_wd2']),'天')]
    checks += [(label,m2[key],m4[key],unit) for label,key,unit in [('违反SOC约束天数','bad_soc','天'),('最大电量平衡残差','balance','kWh'),('SOC最小值','minimum','kWh'),('SOC最大值','maximum','kWh'),('最大单时段充电量','charge','kWh'),('最大单时段放电量','discharge','kWh'),('同时充放电时段数','simultaneous','数量')]]
    emit('p4_numerical_check.csv', [dict(metric=label,problem2_main=a,problem3_main=b,unit=unit) for label,a,b,unit in checks])
    print(f'Rebuilt 19 paper tables from annual evidence at {output}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--result-dir',type=Path,default=output_root()/'result')
    args=parser.parse_args()
    build_tables(args.result_dir.resolve(),args.output_dir.resolve())


if __name__=='__main__':
    main()
