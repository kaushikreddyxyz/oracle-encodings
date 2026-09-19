"""Generate standalone scientific figures and machine-readable aggregate tables."""
from pathlib import Path
import json,re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
LABEL={'relu':'ReLU SAE','batchtopk':'BatchTopK SAE','matryoshka':'Matryoshka SAE','mp':'Matching Pursuit SAE','difference_means':'Difference of means','ridge':'Ridge readout','logistic':'Logistic readout','generative_ols':'Generative OLS','graft':'Ground-truth graft','unedited':'Unedited'}
COLORS=dict(zip(LABEL,['#4477AA','#EE6677','#228833','#AA3377','#CCBB44','#66CCEE','#EE9944','#333333','#111111','#999999']))
METHODS=['relu','batchtopk','matryoshka','mp','difference_means','ridge','logistic','generative_ols']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'svg.fonttype':'none','savefig.facecolor':'white'})

def save(fig,name):
 fig.savefig(OUT/f'{name}.png',dpi=180,bbox_inches='tight');fig.savefig(OUT/f'{name}.svg',bbox_inches='tight');plt.close(fig)

def data():
 rows=[];ev=[]
 nulls=json.loads((ROOT/'subspace_nulls.json').read_text()) if (ROOT/'subspace_nulls.json').exists() else {}
 for p in sorted((ROOT/'results').glob('g*/g*-s*-*/*-geometry.json')):
  if int(p.parent.parent.name[1:])>10:continue
  tag=p.parent.name;match=re.fullmatch(r'g(\d+)-s(\d+)-(fixed|trainable|baseline)',tag)
  if not match:continue
  g,seed,cond=match.groups();d=json.loads(p.read_text());gauge=d.get('identifiable',{});t=d.get('training',[{}])[-1]
  row={'run':tag,'grid':int(g),'seed':int(seed),'condition':cond,'site':d['site'],'method':d['method'],
       'nearest_cosine':gauge.get('nearest_mean',np.nan),'label_cosine':gauge.get('label_selected_mean',np.nan),'assignment_cosine':gauge.get('assignment_mean',np.nan),
       'raw_label_cosine':d.get('raw',{}).get('label_selected_mean',np.nan),'fraction_above_08':gauge.get('frac_above_08',np.nan),'random_nearest':d.get('random_dictionary',{}).get('nearest_mean',np.nan),
       'shuffled_labels':d.get('shuffled_labels_mean',np.nan),'ev':t.get('ev',np.nan),'l0':t.get('l0',np.nan),'dead':t.get('dead',np.nan),'probe_accuracy':d.get('probe_accuracy',np.nan)}
  row['subspace_random_nearest']=nulls.get('/'.join([tag,d['site'],d['method']]),{}).get('nearest_mean',np.nan)
  ep=p.with_name(p.name.replace('-geometry.json','.json'))
  if ep.exists():
   dd=json.loads(ep.read_text());row.update(test_solved=dd.get('test_solved'),alpha=dd.get('alpha'))
   z=np.load(ep.with_name(ep.stem+'-eval.npz'))
   for i in range(len(z['maze'])):
    ev.append({'run':tag,'grid':int(g),'seed':int(seed),'condition':cond,'site':d['site'],'method':d['method'],'maze':int(z['maze'][i]),'level':int(z['level'][i]),'eps':float(z['eps'][i]),'distance':float(z['dist'][i])/(2*int(g)*(int(g)-1)),'solved':int(z['solved'][i]),'true_solved':int(z['true_solved'][i])})
  rows.append(row)
 return pd.DataFrame(rows),pd.DataFrame(ev)

def geometry_figs(df):
 for metric,title,name in [('label_cosine','Label-associated directions','direction_recovery'),('nearest_cosine','Best atom for each graft direction','nearest_atom_recovery')]:
  fig,axs=plt.subplots(2,2,figsize=(11,7),sharex=True,sharey=True)
  for i,site in enumerate(['injection','post0']):
   for j,cond in enumerate(['fixed','trainable']):
    ax=axs[i,j];sub=df[(df.site==site)&(df.condition==cond)]
    for m in METHODS:
     z=sub[sub.method==m].groupby('grid')[metric].agg(['mean','min','max'])
     if not len(z):continue
     ax.plot(z.index,z['mean'],label=LABEL[m],color=COLORS[m],lw=1.8,marker='o',ms=3,ls='--' if m=='generative_ols' else '-')
     ax.fill_between(z.index,z['min'],z['max'],color=COLORS[m],alpha=.13)
    null='shuffled_labels' if metric=='label_cosine' else ('subspace_random_nearest' if sub.subspace_random_nearest.notna().any() else 'random_nearest');z=sub[sub.method.isin(METHODS[:4])].groupby('grid')[null].mean()
    ax.plot(z.index,z,color='#777777',ls=':',lw=1.5,label='Shuffled edge labels' if metric=='label_cosine' else 'Random subspace dictionary')
    ax.set_title(f'{cond.capitalize()} graft · '+('immediately after injection' if site=='injection' else 'after block 0'))
    ax.set_ylim(0,1.04);ax.set_xticks(range(5,11))
    if j==0:ax.set_ylabel('Mean absolute cosine')
    if i==1:ax.set_xlabel('Maze grid width')
  handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,-.08),fontsize=9)
  fig.suptitle(title,fontsize=16,y=1.01);fig.text(.5,.955,'Free edges only; common-vector gauge removed. Shading spans two model seeds.',ha='center',fontsize=9)
  fig.tight_layout(rect=(0,.04,1,.94));save(fig,name)
 fig,axs=plt.subplots(1,2,figsize=(10,4))
 for m in METHODS[:4]:
  z=df[(df.site=='injection')&(df.method==m)&(df.condition!='baseline')]
  axs[0].scatter(z.ev,z.label_cosine,s=30,alpha=.7,color=COLORS[m],label=LABEL[m])
  for cond,ls in [('fixed','-'),('trainable','--')]:
   zz=z[z.condition==cond].groupby('grid').l0.mean();axs[1].plot(zz.index,zz,color=COLORS[m],ls=ls,marker='o',ms=3)
 axs[0].set(xlabel='Held-out explained variance',ylabel='Label-associated cosine',title='Reconstruction versus direction recovery',ylim=(0,1),xlim=(0,1.02))
 axs[1].set(xlabel='Maze grid width',ylabel='Mean active latents',title='Actual sparsity (solid: fixed; dashed: trained)',yscale='log')
 axs[0].legend(frameon=False,fontsize=8);fig.tight_layout();save(fig,'reconstruction_and_sparsity')

def interventions(df,ev):
 if ev.empty:return
 main=ev[(ev.site=='injection')&(ev.condition!='baseline')]
 fig,axs=plt.subplots(1,2,figsize=(11,4),sharey=True)
 for ax,cond in zip(axs,['fixed','trainable']):
  for m in ['graft','unedited']+METHODS:
   z=main[(main.condition==cond)&(main.method==m)].groupby(['run','grid']).solved.mean().reset_index().groupby('grid').solved.agg(['mean','min','max'])
   if len(z):
    ax.plot(z.index,z['mean'],color=COLORS[m],marker='o',ms=3,label=LABEL[m],ls='--' if m in ['unedited','generative_ols'] else '-')
    ax.fill_between(z.index,z['min'],z['max'],color=COLORS[m],alpha=.12)
  ax.set(title=f'{cond.capitalize()} graft',xlabel='Maze grid width',ylim=(0,1),xticks=range(5,11))
 axs[0].set_ylabel('Solve rate in target maze B');h,l=axs[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',bbox_to_anchor=(.5,-.17),ncol=4,frameon=False,fontsize=8)
 fig.suptitle('Maze-diff interventions on held-out base mazes',fontsize=15);fig.tight_layout();save(fig,'maze_diff_accuracy')
 # Secondary site: compare edits after block 0 with the same original graft control.
 fig,axs=plt.subplots(1,2,figsize=(11,4),sharey=True)
 for ax,cond in zip(axs,['fixed','trainable']):
  for m in ['graft','unedited']+METHODS:
   source=main if m in ['graft','unedited'] else ev[(ev.site=='post0')&(ev.condition!='baseline')]
   z=source[(source.condition==cond)&(source.method==m)].groupby(['run','grid']).solved.mean().reset_index().groupby('grid').solved.agg(['mean','min','max'])
   if len(z):
    ax.plot(z.index,z['mean'],color=COLORS[m],marker='o',ms=3,label=LABEL[m],ls='--' if m in ['unedited','generative_ols'] else '-')
    ax.fill_between(z.index,z['min'],z['max'],color=COLORS[m],alpha=.12)
  ax.set(title=f'{cond.capitalize()} graft',xlabel='Maze grid width',ylim=(0,1),xticks=range(5,11))
 axs[0].set_ylabel('Solve rate in target maze B');h,l=axs[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',bbox_to_anchor=(.5,-.17),ncol=4,frameon=False,fontsize=8)
 fig.suptitle('Secondary intervention site: after block 0',fontsize=15);fig.tight_layout();save(fig,'maze_diff_accuracy_post0')
 # Probability view, weighting each model equally within each nonempty bin.
 fig,axs=plt.subplots(1,2,figsize=(11,4),sharey=True)
 bins=np.array([0,.1,.25,.5,.75,1.00001]);centers=(bins[:-1]+bins[1:])/2
 for ax,cond in zip(axs,['fixed','trainable']):
  for m in ['graft','unedited']+METHODS:
   z=main[(main.condition==cond)&(main.method==m)].copy();z['bin']=np.digitize(z.eps,bins)-1
   means=z.groupby(['run','bin']).solved.mean().groupby('bin').mean()
   ax.plot([centers[int(b)] for b in means.index],means.values,color=COLORS[m],marker='o',ms=3,label=LABEL[m],ls='--' if m in ['unedited','generative_ols'] else '-')
  ax.set(title=f'{cond.capitalize()} graft',xlabel='Generation difference probability ε (binned)',ylim=(0,1),xlim=(0,1))
 axs[0].set_ylabel('Solve rate in target maze B');h,l=axs[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',bbox_to_anchor=(.5,-.17),ncol=4,frameon=False,fontsize=8);fig.suptitle('Intervention performance versus difference probability',fontsize=14);fig.tight_layout();save(fig,'maze_diff_probability')
 # Paired differences, bootstrap whole base-maze clusters within run.
 rows=[];rng=np.random.default_rng(53)
 for cond in ['fixed','trainable']:
  ref=main[(main.condition==cond)&(main.method=='graft')][['run','maze','level','solved']].rename(columns={'solved':'graft_solved'})
  for m in METHODS:
   q=main[(main.condition==cond)&(main.method==m)].merge(ref,on=['run','maze','level']);q['delta']=q.solved-q.graft_solved
   if q.empty:continue
   grouped=[g.groupby('maze').delta.agg(['sum','count']).to_numpy() for _,g in q.groupby('grid')]
   boot=[]
   for _ in range(1000):
    draws=[z[rng.integers(0,len(z),len(z))].sum(0) for z in grouped]
    boot.append(np.mean([a[0]/a[1] for a in draws]))
   boot=np.asarray(boot)
   rows.append(dict(condition=cond,method=m,delta=float(q.groupby('run').delta.mean().mean()),lo=float(np.quantile(boot,.025)),hi=float(np.quantile(boot,.975)),runs=int(q.run.nunique())))
 pd.DataFrame(rows).to_csv(ROOT/'intervention_comparison.csv',index=False)

def legacy_fig():
 rows=[]
 for p in (ROOT/'results/g11').glob('legacy-s*/*.json'):
  d=json.loads(p.read_text())
  if not isinstance(d,dict) or 'families' not in d:continue
  for fam,v in d['families'].items():rows.append({'run':p.parent.name,'method':d['method'],'layer':d['layer'],'stage':int(d['layer']!=0),'family':fam,**v})
 if not rows:return
 df=pd.DataFrame(rows);df.to_csv(ROOT/'legacy_summary.csv',index=False)
 fig,axs=plt.subplots(1,2,figsize=(11,4))
 for ax,fam,metric,title in [(axs[0],'detour','crossed_block','Detour: still crosses the blocked edge ↓'),(axs[1],'shortcut_open','used_open','Shortcut: uses the newly opened edge ↑')]:
  q=df[df.family==fam]
  for m in ['unedited']+METHODS:
   z=q[q.method==m].groupby('stage')[metric].mean()
   ax.plot(z.index,z,color=COLORS[m],marker='o',label=LABEL[m])
  ax.set(title=title,xlabel='Intervention site (resid_post)',ylim=(0,1),xticks=[0,1],xticklabels=['Layer 0','Best probe layer'])
 h,l=axs[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',bbox_to_anchor=(.5,-.16),ncol=3,frameon=False,fontsize=8);fig.suptitle('Archived counterfactual tasks · class-coordinate steering',fontsize=14);fig.tight_layout();save(fig,'legacy_steering')

def robustness_fig():
 rows=[]
 for p in (ROOT/'results/g12/robustness').glob('*.json'):
  d=json.loads(p.read_text());rows.append({**{k:d[k] for k in ['condition','method','sae_seed','k','l1']},'cosine':d['identifiable']['label_selected_mean'],'ev':d['training'][-1]['ev'],'l0':d['training'][-1]['l0']})
 if not rows:return
 df=pd.DataFrame(rows);df.to_csv(ROOT/'robustness_summary.csv',index=False)
 primary=pd.read_csv(ROOT/'direction_metrics.csv')
 fig,axs=plt.subplots(1,2,figsize=(10,4),sharey=True)
 for ax,cond in zip(axs,['fixed','trainable']):
  for i,m in enumerate(METHODS[:4]):
   q=df[(df.condition==cond)&(df.method==m)]
   base=primary[(primary.grid==10)&(primary.seed==0)&(primary.site=='injection')&(primary.condition==cond)&(primary.method==m)]
   if len(base):ax.scatter(i-.2,base.label_cosine.iloc[0],color=COLORS[m],marker='o',s=55)
   for _,row in q.iterrows():ax.scatter(i+(0 if row.k==16 else .2),row.cosine,color=COLORS[m],marker='^' if row.k==16 else 's',s=55)
  ax.set(title=cond.capitalize()+' graft',xticks=range(4),xticklabels=['ReLU','BatchTopK','Matryoshka','MP'],ylim=(0,1))
 axs[0].set_ylabel('Label-associated cosine');fig.suptitle('10×10: ○ primary; △ second seed; □ k=64 / ReLU L1=1',fontsize=13);fig.tight_layout();save(fig,'sensitivity')

if __name__=='__main__':
 df,ev=data();df.to_csv(ROOT/'direction_metrics.csv',index=False)
 if not ev.empty:ev.to_csv(ROOT/'intervention_trials.csv.gz',index=False,compression='gzip')
 geometry_figs(df);interventions(df,ev);legacy_fig();robustness_fig()
 print('Figures:',len(list(OUT.glob('*.png'))),'direction rows:',len(df),'intervention trials:',len(ev))
