"""Operational (new-fire) evaluation for the DUAL-branch model.

Same protocol as operational_stats.py -- TPR/FPR/lift vs top-k, for ALL fire and
for NEW fire only (>DILATE px from recent t-3 fire) -- but for ConvLSTMLitDual,
which consumes x_slow (144d @ 8d bins) and x_fast (14d daily) rather than a
single 30-day tensor.

The question this answers: does the 144-day drought window help predict
genuinely NEW ignition, where the 30-day model only reached 5.1x enrichment?
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, zarr, pandas as pd
from scipy import ndimage
from zarr_dual_datamodule import DualWindowDataset, DualPatchConfig
from conv_lstm_lit_dual import ConvLSTMLitDual

CKPT=os.environ["CKPT"]
N_PATCH=int(os.environ.get("N_PATCH",200))
# Evaluate at the CHECKPOINT'S OWN training patch size -- the fire-history
# distance-transform feature is window-size-sensitive (confirmed: it's why
# coastal/tile-boundary artifacts appeared in the 128px-tiled continent maps
# for a 256px-trained checkpoint). Mismatching eval patch size vs training
# patch size would reintroduce exactly that distortion into these numbers.
PATCH=int(os.environ.get("PATCH",128))
# Scale the fire-density filter with patch AREA, same convention as
# train_convlstm_dual.py's patch-size sweep (BASE_PATCH=256, BASE_MIN_POS=20 ->
# density 0.0305%). A fixed absolute pixel count (the old hardcoded 10) would
# make a 384px eval draw from much lower-density scenes than a 128px eval,
# inflating its lift numbers for reasons unrelated to model quality -- this is
# what made the raw 128px-vs-384px new-fire-lift comparison (5.4x vs 21.2x)
# not apples-to-apples.
_BASE_PATCH, _BASE_MIN_POS = 256, 20
MIN_POS = max(1, round(_BASE_MIN_POS * (PATCH / _BASE_PATCH) ** 2))
DILATE=int(os.environ.get("DILATE",3))
KS=[0.0001,0.0002,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.10]
EVAL_YEAR=os.environ.get('EVAL_YEAR','2020')
OUT_TAG=os.environ.get('OUT_TAG','')
LAT0,PX=-9.005000113999998,0.01
SEAS={12:'DJF',1:'DJF',2:'DJF',3:'MAM',4:'MAM',5:'MAM',6:'JJA',7:'JJA',8:'JJA',9:'SON',10:'SON',11:'SON'}

def tpr_fpr(prob,truth,land,f):
    v=prob[land]
    if v.size==0 or truth.sum()==0: return (np.nan,)*3
    k=max(1,int(round(f*v.size))); thr=np.partition(v,-k)[-k]
    sel=land&(prob>=thr)
    tp=float((truth&sel).sum()); fn=float((truth&~sel).sum())
    neg=land&~truth; fp=float((neg&sel).sum()); tn=float((neg&~sel).sum())
    prec=tp/max(float(sel.sum()),1); base=float(truth.sum())/float(land.sum())
    return tp/max(tp+fn,1), fp/max(fp+tn,1), (prec/base if base>0 else np.nan)

def main():
    torch.set_num_threads(8)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    g=zarr.open_group(f'cube_daily_smgrid_{EVAL_YEAR}.zarr',mode='r'); TIMES=list(g.attrs.get('time',[]))
    m=ConvLSTMLitDual.load_from_checkpoint(CKPT,map_location=device); m.eval(); m.to(device)
    print(f'[info] loaded {os.path.basename(CKPT)} on {device}, patch_size={PATCH}',flush=True)

    ds=DualWindowDataset(DualPatchConfig(
        zarr_paths=(f'cube_daily_smgrid_{EVAL_YEAR}.zarr',), stats_path='channel_stats_2015_2018.json',
        slow_cube_path='cube_slow_8day.zarr', day_offset={'2015':0,'2016':365,'2017':731,'2018':1096,'2019':1461,'2020':1826}[EVAL_YEAR],
        patch_size=PATCH, samples_per_epoch=N_PATCH*3, seed=21,
        min_pos_pixels=MIN_POS, pos_frac=1.0, deterministic=True,
        fire_history=True, fire_history_lags=(3,4,5), fire_history_distance=True,
        use_lightning=os.environ.get("USE_LIGHTNING", "0") == "1",
        use_elevation=os.environ.get("USE_ELEVATION", "0") == "1",
        use_slope_aspect=os.environ.get("USE_SLOPE_ASPECT", "0") == "1",
        use_fuel_age=os.environ.get("USE_FUEL_AGE", "0") == "1",
        use_wind_dir=os.environ.get("USE_WIND_DIR", "0") == "1"))
    print(f'[info] patch={PATCH} min_pos_pixels={MIN_POS} (density {100*MIN_POS/PATCH**2:.4f}%)', flush=True)
    rows=[]; ALL_P=[]; ALL_Y=[]
    for i in range(N_PATCH*3):
        if len(rows)>=N_PATCH: break
        b=ds[i]; land=b['mask'].numpy()>0.5
        truth=(b['y'][-1].numpy()>0)&land
        if truth.sum()<MIN_POS or land.mean()<0.5: continue
        # fire_hist_t-3 is the first appended fh channel on the fast branch (index 7)
        recent=b['x_fast'][-1,:,:,7].numpy()>0.5
        known=ndimage.binary_dilation(recent,iterations=DILATE)
        new_fire=truth&~known
        with torch.no_grad():
            p=torch.sigmoid(m.forward_seq(b['x_slow'].unsqueeze(0).to(device), b['x_fast'].unsqueeze(0).to(device),
                                          b['x_cat'].unsqueeze(0).to(device))[:,-1])[0].cpu().numpy()
        tt=int(b['t_end']); date=TIMES[tt] if tt<len(TIMES) else ''
        mth=int(date[5:7]) if date else 0
        lat=LAT0-(int(b['y0'])+PATCH/2)*PX
        r=dict(n_fire=int(truth.sum()),n_new=int(new_fire.sum()),date=date,
               season=SEAS.get(mth,'?'),
               region='Tropical N' if lat>-20 else ('Temperate S' if lat<-30 else 'Central'))
        for f in KS:
            k=f'{f:g}'
            t_,f_,l_=tpr_fpr(p,truth,land,f); r[f'tpr_{k}'],r[f'fpr_{k}'],r[f'lift_{k}']=t_,f_,l_
            if new_fire.sum()>=3:
                tn_,_,ln_=tpr_fpr(p,new_fire,land,f); r[f'tprNEW_{k}'],r[f'liftNEW_{k}']=tn_,ln_
        ALL_P.append(p[land]); ALL_Y.append(truth[land].astype(int))
        rows.append(r)
        if len(rows)%50==0: print(f'  {len(rows)}...',flush=True)
    df=pd.DataFrame(rows); df.to_csv(f'operational_stats_dual_{EVAL_YEAR}{OUT_TAG}.csv',index=False)
    from sklearn.metrics import average_precision_score, roc_auc_score
    P=np.concatenate(ALL_P); Y=np.concatenate(ALL_Y)
    print(f'\n*** POOLED {EVAL_YEAR} TEST METRICS (n={len(Y):,} land px, base rate {Y.mean():.5f}) ***')
    print(f'    AUC-PR  = {average_precision_score(Y,P):.4f}')
    print(f'    ROC-AUC = {roc_auc_score(Y,P):.4f}')
    print(f'\n=== DUAL model | n={len(df)} patches | mean fire {df.n_fire.mean():.0f}, NEW {df.n_new.mean():.0f} ({100*df.n_new.sum()/df.n_fire.sum():.0f}%) ===')
    print(f'\n{"top-k":>7} {"TPR(all)":>9} {"FPR":>8} {"lift(all)":>10} {"TPR(new)":>9} {"lift(new)":>10}')
    for f in KS:
        k=f'{f:g}'
        tn=df[f'tprNEW_{k}'].mean() if f'tprNEW_{k}' in df else np.nan
        ln=df[f'liftNEW_{k}'].mean() if f'liftNEW_{k}' in df else np.nan
        print(f'{100*f:6.1f}% {df[f"tpr_{k}"].mean():9.3f} {df[f"fpr_{k}"].mean():8.4f} {df[f"lift_{k}"].mean():10.1f} {tn:9.3f} {ln:10.1f}')
    ba=max(KS,key=lambda f: np.nanmean(df[f'lift_{f:g}']))
    bn=max([f for f in KS if f'liftNEW_{f:g}' in df],key=lambda f: np.nanmean(df[f'liftNEW_{f:g}']))
    print(f'\n  ALL-fire enrichment peaks at top-{100*ba:g}%  ({np.nanmean(df[f"lift_{ba:g}"]):.1f}x)')
    print(f'  NEW-fire enrichment peaks at top-{100*bn:g}%  ({np.nanmean(df[f"liftNEW_{bn:g}"]):.1f}x)')
    print(f'  [30-day unified reference: all 45.4x @0.5%, new 5.1x @2%]')
    for col,lab in (('season','season'),('region','region')):
        print(f'\n-- by {lab} (top-2%) --')
        print(f'  {"group":14} {"n":>4} {"TPR(all)":>9} {"TPR(new)":>9} {"lift(new)":>10}')
        for kk,sub in df.groupby(col):
            print(f'  {str(kk):14} {len(sub):4d} {sub["tpr_0.02"].mean():9.3f} '
                  f'{sub.get("tprNEW_0.02", pd.Series([np.nan])).mean():9.3f} {sub.get("liftNEW_0.02", pd.Series([np.nan])).mean():10.1f}')
    print(f'\nwrote operational_stats_dual_{EVAL_YEAR}{OUT_TAG}.csv')

    # ---- boxplots (TPR / FPR / lift at top-2%) ----
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    for strat,fname,order in (('season',f'fig_dual_{EVAL_YEAR}{OUT_TAG}_by_season.png',['DJF','MAM','JJA','SON']),
                              ('region',f'fig_dual_{EVAL_YEAR}{OUT_TAG}_by_region.png',['Tropical N','Central','Temperate S'])):
        groups=[g_ for g_ in order if g_ in set(df[strat])]
        if not groups: continue
        fig,axes=plt.subplots(1,3,figsize=(15,4.6))
        for ax,(met,lab) in zip(axes,[('tpr_0.02','TPR (all fire) @ top-2%'),
                                      ('tprNEW_0.02','TPR (NEW fire only) @ top-2%'),
                                      ('liftNEW_0.02','Enrichment, NEW fire @ top-2%')]):
            if met not in df: continue
            data=[df[df[strat]==g_][met].dropna() for g_ in groups]
            bp=ax.boxplot(data,positions=np.arange(len(groups)),widths=0.5,patch_artist=True,showfliers=False)
            for b_ in bp['boxes']: b_.set_facecolor('#2c7fb8'); b_.set_alpha(0.75)
            for w in bp['medians']: w.set_color('k')
            ax.set_xticks(np.arange(len(groups))); ax.set_xticklabels(groups,fontsize=9)
            ax.set_title(lab,fontsize=10,fontweight='bold'); ax.grid(alpha=0.25,axis='y')
        fig.suptitle(f'Dual-branch (144d lookback) skill by {strat} — {EVAL_YEAR} TEST year, n={len(df)} patches',
                     fontsize=12,fontweight='bold')
        fig.tight_layout(); fig.savefig(f'figures/{fname}',dpi=200,facecolor='white',bbox_inches='tight')
        print('wrote figures/'+fname)

if __name__=='__main__': main()
