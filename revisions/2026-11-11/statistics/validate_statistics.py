"""Independent numerical checks of released statistical outputs."""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

p=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--feature-dir',type=Path,default=p.parent/'data',
                    help='Corrected feature directory (default: package-root/data)')
parser.add_argument('--result-dir',type=Path,default=p/'results',
                    help='Statistical result directory (default: package-root/statistics/results)')
args=parser.parse_args()
d=pd.read_csv(args.result_dir/'paired_tests_36.csv')
c=pd.read_csv(args.feature_dir/'chapter_level_features.csv')
s=pd.read_csv(args.result_dir/'segment_chapter_means.csv')
maxerr={}
for method,col in [('fdr_bh','q_BH36'),('holm','p_Holm36')]:
    maxerr[method]=float(np.max(np.abs(d[col]-multipletests(d.p_raw,method=method)[1])))
for level,frame,file in [('chapter',c,'length_adjusted_paired_regression.csv'),('segment_means',s,'length_adjusted_segment_means_regression.csv')]:
    out=pd.read_csv(args.result_dir/file)
    h=frame[frame.edition.eq('hawkes')].sort_values('chapter')
    y=frame[frame.edition.eq('yang')].sort_values('chapter')
    X=sm.add_constant(np.log(h.word_count.to_numpy())-np.log(y.word_count.to_numpy()))
    for metric in ['coefficient','se','p']:maxerr[f'{level}_HC3_{metric}']=0.
    for row in out.itertuples(index=False):
        diff=h[row.feature].to_numpy()-y[row.feature].to_numpy()
        fit=sm.OLS(diff,X).fit(cov_type='HC3',use_t=True)
        for name,actual,stored in [('coefficient',fit.params[0],row.adjusted_edition_difference),('se',fit.bse[0],row.hc3_se),('p',fit.pvalues[0],row.p_raw)]:
            key=f'{level}_HC3_{name}';maxerr[key]=max(maxerr[key],float(abs(actual-stored)))
assert all(v<1e-10 for v in maxerr.values()),maxerr
assert len(d)==36 and set(d.n_pairs)=={120} and set(d.df)=={119}
assert ((d.difference_ci_low<d.mean_difference)&(d.mean_difference<d.difference_ci_high)).all()
assert (d.q_BH36>=d.p_raw-1e-15).all()
meta=json.loads((args.result_dir/'analysis_metadata.json').read_text(encoding='utf-8'))
raw_check=meta.get('raw_text_punctuation_verification')
if raw_check is not None:
    assert raw_check['mismatches_above_1e_12']==0
report=dict(passed=True,independent_reference='statsmodels 0.14.4',max_absolute_errors=maxerr,
    chapter_pairs=120,segment_chapter_pairs=120,
    raw_text_punctuation_rows_verified=raw_check['n_samples'] if raw_check is not None else None,
    input_hashes={k:v for k,v in meta.items() if k.endswith('_sha256')})
(args.result_dir/'statistical_numerical_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
