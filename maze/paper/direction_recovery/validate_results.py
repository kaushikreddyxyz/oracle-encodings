"""Final artifact and split validation; fails if the campaign is incomplete."""
from pathlib import Path
import json,hashlib
import numpy as np
import torch
R=Path(__file__).resolve().parent
M=['generative_ols','difference_means','ridge','logistic','relu','batchtopk','matryoshka','mp']
validation={'splits':{},'models':{},'pods':{},'figures':[]}
for g in range(5,11):
 files=list((R/'results'/f'g{g}').glob('train-g*-n20000-dedup.npz')) or list((R/'results'/f'g{g}').glob('train-g*-n20000.npz'))
 assert len(files)==1,(g,files)
 c=np.load(files[0])['conn'];s=np.load(R.parent/f'curves/results/distance/sets/g{g}.npz')
 tr=set(map(bytes,np.packbits(c[:16000],axis=1)));va=set(map(bytes,np.packbits(c[16000:],axis=1)))
 te=set(map(bytes,np.packbits(s['conn_a'],axis=1)))|set(map(bytes,np.packbits(s['conn_b'],axis=1)))
 assert not tr&va and not (tr|va)&te,g
 validation['splits'][g]={'unique_train':len(tr),'unique_validation':len(va),'train_validation_overlap':0,'fit_eval_overlap':0,'source':str(files[0].relative_to(R))}
 for seed in [0,1]:
  for cond in ['fixed','trainable']:
   tag=f'g{g}-s{seed}-{cond}';d=R/'results'/f'g{g}'/tag;assert (d/'DONE').exists(),tag
   reference=None;methods=0
   for site in ['injection','post0']:
    for method in (['graft','unedited'] if site=='injection' else [])+M:
     p=d/f'{site}-{method}.json';res=json.loads(p.read_text());assert np.isfinite(res['test_solved']);a=np.load(d/f'{site}-{method}-eval.npz');ids=a['ids']
     assert len(ids)==len(np.unique(ids));assert np.isclose(a['solved'].mean(),res['test_solved'])
     if reference is None:reference=ids
     else:assert np.array_equal(reference,ids),(tag,method,'unpaired')
     b=np.random.default_rng(91001).permutation(3000);assert not np.isin(a['maze'],b[:20]).any()
     assert np.isin(a['maze'],b[20:80]).all()
     if method=='generative_ols' and site=='injection':assert res['identifiable']['label_selected_mean']>.9999
     methods+=1
   validation['models'][tag]={'evaluated_methods_and_sites':methods,'paired_test_items':len(reference)}
kv=json.loads((R/'results/g6/kv_validation.json').read_text());assert len(kv)==3 and all(v['exact_reference_match'] for v in kv.values());validation['kv_reference']=kv
for seed in [0,1]:
 d=R/'results/g6'/f'g6-s{seed}-baseline';assert (d/'DONE').exists()
 for m in ['unedited']+M:assert (d/f'post0-{m}-eval.npz').exists()
 d=R/'results/g11'/f'legacy-s{seed}';assert (d/'DONE').exists()
 assert len(list(d.glob('L*-*.json')))==18
 zeros=[np.load(p) for p in d.glob('L*-unedited.npz')]
 assert len(zeros)==2
 for k in ['ids','solved','crossed_block','used_open','reached']:assert np.array_equal(zeros[0][k],zeros[1][k]),(seed,k)
assert len(list((R/'results/g12/robustness').glob('*.json')))==16
for p in R.glob('connection-g*.json'):
 g=json.loads(p.read_text())['grid'];d=R/'results'/f'g{g}';term=json.loads((d/'TERMINATED.json').read_text());assert term['id']==json.loads(p.read_text())['id']
 manifest=json.loads((d/'verified-manifest.json').read_text())
 for name,sha in manifest.items():assert hashlib.sha256((d/name).read_bytes()).hexdigest()==sha,(g,name)
 validation['pods'][g]={'id':term['id'],'verified_files':len(manifest),'terminated':True}
assert len(validation['pods'])==11
for name in ['direction_recovery','nearest_atom_recovery','reconstruction_and_sparsity','maze_diff_accuracy','maze_diff_accuracy_post0','maze_diff_probability','legacy_steering','sensitivity']:
 for ext in ['png','svg']:
  p=R/f'figures/{name}.{ext}';assert p.exists() and p.stat().st_size>1000;validation['figures'].append(str(p.relative_to(R)))
validation['status']='passed';(R/'validation.json').write_text(json.dumps(validation,indent=2));print('PASS:',len(validation['models']),'grafted models,',len(validation['pods']),'verified/terminated pods')
