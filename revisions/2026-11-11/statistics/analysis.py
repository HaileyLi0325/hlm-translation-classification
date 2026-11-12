"""Paired inference and length sensitivity for Hongloumeng edition attribution.

Run with --chapter-csv and --segment-csv pointing to the final corrected
feature matrices. --text-root may point to hlm/data to independently validate
punctuation entropy directly from all retained chapter/window text files.
All contrasts are Hawkes-Minford minus Yang-Yang. Legacy edition identifiers
`hawkes` and `yang` are retained solely for compatibility with released inputs.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import sys

import numpy as np
import pandas as pd
import scipy
from scipy import stats

FEATURES = [
    'wordform_unigram_entropy', 'lemma_unigram_entropy', 'function_word_entropy',
    'word_length_entropy', 'lexical_frequency_band_entropy', 'punctuation_entropy',
    'pos_unigram_entropy', 'pos_bigram_entropy', 'pos_trigram_entropy',
    'clause_marker_entropy', 'dependency_relation_entropy', 'coordination_subordination_entropy',
    'character_entropy', 'character_2gram_entropy', 'character_3gram_entropy',
    'character_4gram_entropy', 'word_bigram_entropy', 'word_trigram_entropy']
CODES = ['WFU', 'LU', 'FW', 'WL', 'FB', 'PUNC', 'POSU', 'POSB', 'POST', 'CM',
         'DR', 'CS', 'CHAR', 'C2G', 'C3G', 'C4G', 'WB', 'WT']
SEED = 20261111
BOOTSTRAPS = 10000


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def holm(p):
    p = np.asarray(p); order = np.argsort(p); m = len(p)
    out = np.empty(m)
    out[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(m, 0, -1)))
    return out


def pair_arrays(frame):
    h = frame.loc[frame.edition.eq('hawkes')].set_index('chapter').sort_index()
    y = frame.loc[frame.edition.eq('yang')].set_index('chapter').sort_index()
    assert h.index.is_unique and y.index.is_unique and h.index.equals(y.index)
    return h, y


def paired_results(frame, level, seed=SEED):
    h, y = pair_arrays(frame); n = len(h)
    # Resample whole chapter pairs, retaining feature covariance in each draw.
    idx = np.random.default_rng(seed).integers(0, n, (BOOTSTRAPS, n))
    rows = []
    for feature, code in zip(FEATURES, CODES):
        a, b = h[feature].to_numpy(), y[feature].to_numpy(); d = a - b
        result = stats.ttest_rel(a, b); ci = result.confidence_interval()
        dz = d.mean() / d.std(ddof=1)
        boot = d[idx]; boot_dz = boot.mean(axis=1) / boot.std(axis=1, ddof=1)
        lo, hi = np.quantile(boot_dz, [.025, .975])
        # Nonparametric signed-rank sensitivity is descriptive, not primary.
        w = stats.wilcoxon(d, alternative='two-sided', method='auto')
        rows.append(dict(level=level, feature=feature, code=code, n_pairs=n,
            hm_mean=a.mean(), hm_sd=a.std(ddof=1), hm_median=np.median(a),
            yy_mean=b.mean(), yy_sd=b.std(ddof=1), yy_median=np.median(b),
            mean_difference=d.mean(), difference_ci_low=ci.low, difference_ci_high=ci.high,
            t=result.statistic, df=result.df, p_raw=result.pvalue, cohen_dz=dz,
            dz_ci_low=lo, dz_ci_high=hi, wilcoxon_W=w.statistic, wilcoxon_p=w.pvalue))
    out = pd.DataFrame(rows)
    out['q_BH_within18'] = stats.false_discovery_control(out.p_raw, method='bh')
    return out


def hc3_regression(X, y):
    """OLS coefficients and HC3 covariance; t inference with residual df."""
    n, k = X.shape
    inv = np.linalg.inv(X.T @ X)
    beta = inv @ X.T @ y
    resid = y - X @ beta
    leverage = np.sum((X @ inv) * X, axis=1)
    robust_cov = inv @ (X.T @ (X * (resid / (1-leverage))[:, None]**2)) @ inv
    se = np.sqrt(np.diag(robust_cov)); t = beta / se
    p = 2 * stats.t.sf(abs(t), n-k); cv = stats.t.ppf(.975, n-k)
    return beta, se, t, p, beta-cv*se, beta+cv*se, n-k, leverage.max()


def length_analysis(chapter):
    h, y = pair_arrays(chapter)
    x = np.log(h.word_count.to_numpy()) - np.log(y.word_count.to_numpy())
    X = np.column_stack([np.ones(len(x)), x])
    rows = []
    for feature, code in zip(FEATURES, CODES):
        diff = h[feature].to_numpy() - y[feature].to_numpy()
        beta, se, t, p, lo, hi, df, leverage = hc3_regression(X, diff)
        rows.append(dict(feature=feature, code=code, n_pairs=len(x),
            adjusted_edition_difference=beta[0], hc3_se=se[0], t=t[0], df=df,
            p_raw=p[0], ci_low=lo[0], ci_high=hi[0],
            log_length_slope=beta[1], log_length_slope_p=p[1],
            max_leverage=leverage))
    out = pd.DataFrame(rows)
    out['q_BH18'] = stats.false_discovery_control(out.p_raw, method='bh')
    out['p_Holm18'] = holm(out.p_raw)
    corrs = []
    for edition, frame in chapter.groupby('edition', sort=True):
        for feature, code in zip(FEATURES, CODES):
            r = stats.spearmanr(frame.word_count, frame[feature])
            corrs.append(dict(edition=edition, feature=feature, code=code,
                n=len(frame), spearman_rho=r.statistic, p_raw=r.pvalue))
    corrs = pd.DataFrame(corrs)
    corrs['q_BH36'] = stats.false_discovery_control(corrs.p_raw, method='bh')
    diagnostics = dict(log_length_ratio_mean=float(x.mean()), log_length_ratio_min=float(x.min()),
        log_length_ratio_max=float(x.max()), n_hm_shorter=int((x<0).sum()),
        n_hm_longer=int((x>0).sum()), max_leverage=float(leverage),
        model='H_HM,c - H_YY,c = beta_edition + beta_loglength * (ln(N_HM,c) - ln(N_YY,c)) + error_c',
        note='Equivalent to a chapter fixed-effect model with a common log-token-count slope. HC3 standard errors; two-sided t inference with 118 df. The edition coefficient estimates the contrast at equal token counts. Observational adjustment cannot isolate causal edition or translator effects.')
    return out, corrs, diagnostics


def direct_punctuation_audit(chapter, segments, text_root):
    rows = []; pattern = re.compile(r'\.\.\.|--|[.,;:?!()"\']')
    for level, frame in [('chapter', chapter), ('segment', segments)]:
        for row in frame.itertuples(index=False):
            if level == 'chapter':
                path = text_root / 'chapters' / row.edition / f'chapter_{row.chapter:03d}.txt'
            else:
                path = text_root / 'windows_1500' / row.edition / f'chapter_{row.chapter:03d}' / f'window_{row.window_id:02d}.txt'
            counts = Counter(pattern.findall(path.read_text(encoding='utf-8')))
            total = sum(counts.values())
            value = -sum((v/total)*math.log2(v/total) for v in counts.values()) if total else 0.
            rows.append(dict(sample_id=row.sample_id, level=level, edition=row.edition,
                chapter=row.chapter, window_id=row.window_id, stored=row.punctuation_entropy,
                recomputed=value, absolute_difference=abs(value-row.punctuation_entropy)))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feature-dir', type=Path, help='Directory containing chapter_level_features.csv and window_1500_features.csv')
    parser.add_argument('--chapter-csv', type=Path)
    parser.add_argument('--segment-csv', type=Path)
    parser.add_argument('--original-punctuation-csv', type=Path, help='Prior submission chapter punctuation values for the Table 3 discrepancy audit')
    parser.add_argument('--text-root', type=Path)
    parser.add_argument('--out', type=Path, default=Path('statistical_results'))
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    if args.feature_dir:
        args.chapter_csv = args.feature_dir / 'chapter_level_features.csv'
        args.segment_csv = args.feature_dir / 'window_1500_features.csv'
    if args.chapter_csv is None or args.segment_csv is None:
        parser.error('Provide --feature-dir or both input CSV paths.')
    chapter, segments = pd.read_csv(args.chapter_csv), pd.read_csv(args.segment_csv)
    assert set(chapter.edition) == set(segments.edition) == {'hawkes', 'yang'}
    assert not chapter[FEATURES+['word_count']].isna().any().any()
    assert not segments[FEATURES+['word_count']].isna().any().any()
    assert len(chapter) == 240
    assert segments.groupby('edition').size().nunique() == 1
    assert not segments.duplicated(['edition','chapter','window_id']).any()
    nwin = segments.groupby(['edition','chapter']).size().unstack(0)
    assert (nwin.hawkes == nwin.yang).all()
    # Every chapter has equal weight, regardless of how many segments it contributes.
    means = segments.groupby(['edition','chapter'])[FEATURES+['word_count']].mean().reset_index()
    means.to_csv(args.out/'segment_chapter_means.csv', index=False)
    results = pd.concat([paired_results(chapter, 'chapter'),
                         paired_results(means, 'segment_chapter_mean', SEED+1)], ignore_index=True)
    results['q_BH36'] = stats.false_discovery_control(results.p_raw, method='bh')
    results['p_Holm36'] = holm(results.p_raw)
    results['wilcoxon_q_BH36'] = stats.false_discovery_control(results.wilcoxon_p, method='bh')
    results.to_csv(args.out/'paired_tests_36.csv', index=False)
    # Individual-segment descriptives retain all rows; inference uses chapter means.
    desc = []
    for level, frame in [('chapter',chapter),('segment_individual',segments),('segment_chapter_mean',means)]:
        for edition, group in frame.groupby('edition'):
            for feature, code in zip(FEATURES,CODES):
                a=group[feature].to_numpy()
                desc.append(dict(level=level,edition=edition,feature=feature,code=code,n=len(a),
                    mean=a.mean(),sd=a.std(ddof=1),median=np.median(a),minimum=a.min(),maximum=a.max()))
    pd.DataFrame(desc).to_csv(args.out/'descriptive_statistics.csv',index=False)
    length, correlations, diagnostics=length_analysis(chapter)
    length.to_csv(args.out/'length_adjusted_paired_regression.csv',index=False)
    correlations.to_csv(args.out/'token_count_entropy_correlations.csv',index=False)
    segment_length, _, segment_length_diagnostics = length_analysis(means)
    segment_length.to_csv(args.out/'length_adjusted_segment_means_regression.csv',index=False)
    # Exploratory historical strata: still edition contrasts, not isolated individual authorship.
    strat=[]
    for block,lo,hi in [('chapters_1_80',1,80),('chapters_81_120',81,120)]:
        for level,frame in [('chapter',chapter),('segment_chapter_mean',means)]:
            result=paired_results(frame[frame.chapter.between(lo,hi)],level,SEED+lo)
            result.insert(0,'block',block);strat.append(result)
    strat=pd.concat(strat,ignore_index=True)
    strat['q_BH72']=stats.false_discovery_control(strat.p_raw,method='bh')
    strat.to_csv(args.out/'exploratory_paired_tests_by_historical_block.csv',index=False)
    h,y=pair_arrays(chapter); a=h.punctuation_entropy.to_numpy();b=y.punctuation_entropy.to_numpy()
    rounded_equal=stats.ttest_ind_from_stats(2.17,.10,120,2.22,.20,120,equal_var=True)
    rounded_welch=stats.ttest_ind_from_stats(2.17,.10,120,2.22,.20,120,equal_var=False)
    raw_equal=stats.ttest_ind(a,b,equal_var=True);raw_welch=stats.ttest_ind(a,b,equal_var=False)
    original_summary=None
    if args.original_punctuation_csv:
        old=pd.read_csv(args.original_punctuation_csv)
        oa=old.loc[old.edition.eq('hawkes'),'punctuation_entropy'].to_numpy()
        ob=old.loc[old.edition.eq('yang'),'punctuation_entropy'].to_numpy()
        oe=stats.ttest_ind(oa,ob,equal_var=True);ow=stats.ttest_ind(oa,ob,equal_var=False)
        original_summary=dict(hm_mean=float(oa.mean()),hm_sd=float(oa.std(ddof=1)),yy_mean=float(ob.mean()),yy_sd=float(ob.std(ddof=1)),
            student_t=float(oe.statistic),student_df=float(oe.df),student_p=float(oe.pvalue),
            welch_t=float(ow.statistic),welch_df=float(ow.df),welch_p=float(ow.pvalue),
            input_sha256=sha256(args.original_punctuation_csv))
    verification=dict(rounded_summary_student_t=float(rounded_equal.statistic),
        rounded_summary_student_df=238,rounded_summary_student_p=float(rounded_equal.pvalue),
        rounded_summary_welch_t=float(rounded_welch.statistic),rounded_summary_welch_p=float(rounded_welch.pvalue),
        raw_student_t=float(raw_equal.statistic),raw_student_df=float(raw_equal.df),raw_student_p=float(raw_equal.pvalue),
        raw_welch_t=float(raw_welch.statistic),raw_welch_df=float(raw_welch.df),raw_welch_p=float(raw_welch.pvalue),
        original_submission_unrounded_values=original_summary,
        corrected_paired=results.loc[results.level.eq('chapter') & results.code.eq('PUNC')].iloc[0].to_dict(),
        conclusion='The previous p<0.001 marker attributed to an independent-samples test is not supported by the rounded summaries or the unrounded feature values. The newly appropriate paired test can legitimately yield a different p-value. Replace the previous inference with paired testing and multiplicity adjustment; do not describe the discrepancy as rounding alone.')
    (args.out/'table3_punctuation_verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    direct = None
    if args.text_root:
        audit=direct_punctuation_audit(chapter,segments,args.text_root)
        audit.to_csv(args.out/'punctuation_raw_text_audit.csv',index=False)
        direct=dict(n_samples=len(audit),maximum_absolute_difference=float(audit.absolute_difference.max()),
            mismatches_above_1e_12=int((audit.absolute_difference>1e-12).sum()))
        assert direct['mismatches_above_1e_12'] == 0
    metadata=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,
        chapter_input_sha256=sha256(args.chapter_csv),segment_input_sha256=sha256(args.segment_csv),
        chapter_pairs=len(chapter)//2,segment_pairs=len(means)//2,segment_rows_per_edition=len(segments)//2,
        missing_segment_chapters=sorted(set(range(1,121))-set(means.chapter)),
        effect_direction='Hawkes-Minford minus Yang-Yang',effect_size='Cohen dz: mean paired difference / sample SD of paired differences',
        bootstrap='10000 chapter-pair percentile bootstrap replicates, two-sided pointwise 95% intervals',
        bootstrap_seed_chapter=SEED,bootstrap_seed_segment=SEED+1,
        multiplicity='Primary Benjamini-Hochberg adjustment over all 36 paired tests. Separate within-level BH18 and familywise Holm36 are sensitivity outputs. Chapter and segment-mean length-adjusted edition coefficients each use separate BH18; 36 chapter length correlations use separate BH36; historical block exploratory tests use BH72.',
        length_diagnostics=diagnostics,raw_text_punctuation_verification=direct,
        segment_mean_length_diagnostics=segment_length_diagnostics,
        significant_BH36=results.groupby('level').q_BH36.apply(lambda x:int((x<.05).sum())).to_dict(),
        significant_Holm36=results.groupby('level').p_Holm36.apply(lambda x:int((x<.05).sum())).to_dict(),
        significant_length_adjusted_BH18=int((length.q_BH18<.05).sum()),
        significant_segment_mean_length_adjusted_BH18=int((segment_length.q_BH18<.05).sum()),
        inference_scope='Chapter-level model-based comparisons conditional on these two editions of one literary work. Chapter independence is an approximation; inferential intervals do not establish generalization to other works or individual translator causality.')
    (args.out/'analysis_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps(metadata,indent=2))
    print(results[['level','code','n_pairs','mean_difference','t','p_raw','q_BH36','cohen_dz']].to_string(index=False))


if __name__=='__main__':
    main()
