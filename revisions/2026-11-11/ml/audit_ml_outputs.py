"""Check leakage, source-feature identity, vote reconstruction and SHAP additivity."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--feature-root',type=Path,required=True);args=p.parse_args()
    out={}
    features=pd.read_csv(args.results/'sampling_features.csv')
    baseline=pd.read_csv(args.feature_root/'window_1500_features.csv')
    columns=[x for x in baseline.columns if x.endswith('_entropy')]
    assert len(columns)==18
    merged=features[features.scheme=='prefix'].merge(baseline,on=['edition','chapter','window_id'],suffixes=('_new','_base'),validate='one_to_one')
    diff=max(np.abs(merged[x+'_new']-merged[x+'_base']).max() for x in columns)
    assert len(merged)==722 and diff<1e-12
    out['prefix_feature_max_abs_difference']=float(diff)
    manifests=pd.read_csv(args.results/'sampling_window_manifest.csv')
    assert len(manifests)==3610
    for key, group in manifests.groupby(['scheme','edition','chapter']):
        group=group.sort_values('regex_start_0based')
        assert (group.regex_end_exclusive-group.regex_start_0based==1500).all()
        assert (group.regex_end_exclusive<=group.chapter_regex_words).all()
        assert np.all(group.regex_start_0based.to_numpy()[1:]>=group.regex_end_exclusive.to_numpy()[:-1])
    out['all_selected_windows_exactly_1500_regex_words_and_nonoverlapping']=True
    assert (manifests.groupby(['scheme','chapter','edition']).size().unstack().nunique(axis=1)==1).all()
    out['matched_counts_all_chapters']=True
    oof=pd.read_csv(args.results/'sampling_oof_predictions.csv')
    assert len(oof)==32490
    assert (oof.groupby(['scheme','algorithm','sample_id']).size()==1).all()
    assert (oof.groupby(['scheme','algorithm','chapter']).fold.nunique()==1).all()
    assert (oof.groupby(['scheme','algorithm']).size()==722).all()
    pv=oof.pivot(index=['scheme','sample_id'],columns='algorithm',values='pred_yang')
    assert ((pv.drop(columns='Voting Ensemble').sum(axis=1)>4).astype(int)==pv['Voting Ensemble']).all()
    out['all_oof_predictions_one_per_sample_and_chapter_grouped']=True
    out['hard_vote_exact_majority_with_class0_tie_break']=True
    held=pd.read_csv(args.results/'heldout_predictions.csv')
    hm=json.loads((args.results/'heldout_manifest.json').read_text())
    for dataset, spec in hm.items():
        assert set(spec['test_chapters']).isdisjoint(spec['train_chapters'])
        assert len(spec['test_chapters'])==24 and len(spec['train_chapters'])==96
    out['heldout_source_chapters_disjoint']=True
    shap=pd.read_csv(args.results/'heldout_shap_values.csv')
    agg=shap.groupby(['dataset','sample_id']).agg(total_shap=('shap_yang','sum'),base=('base_yang','first'))
    agg['explained_probability']=agg.total_shap+agg.base
    compare=agg.merge(held[held.algorithm=='Random Forest'],on=['dataset','sample_id'],validate='one_to_one')
    error=float(np.abs(compare.explained_probability-compare.score_yang).max())
    assert error<1e-6
    out['shap_probability_additivity_max_error']=error
    meta=json.loads((args.results/'ml_metadata.json').read_text())
    out['input_sha256']={x:hashlib.sha256((args.feature_root/(x+'_features.csv')).read_bytes()).hexdigest() for x in ['chapter_level','window_1500']}
    assert out['input_sha256']==meta['input_sha256']
    out['status']='PASS'
    (args.results/'ml_qa_summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
