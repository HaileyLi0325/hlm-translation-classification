"""Recompute corrected chapter records and affected chapter-matched windows.

This incremental rebuild is an audit convenience. For a full reconstruction,
run the corrected run_hlm_pipeline.py on legally obtained source PDFs.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--previous-feature-dir',type=Path,required=True)
    p.add_argument('--previous-window-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('corrected_hlm_pipeline',a.source_root/'run_hlm_pipeline.py')
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    c=pd.read_csv(a.previous_feature_dir/'chapter_level_features.csv')
    w=pd.read_csv(a.previous_feature_dir/'window_1500_features.csv')
    chapter_root=a.source_root/'data'/'chapters'; window_root=a.source_root/'data'/'windows_1500'
    # Copy existing window texts into the work tree; original inputs remain untouched.
    shutil.copytree(a.previous_window_dir,window_root,dirs_exist_ok=True)
    samples=[]
    changed={'hawkes':[101,102],'yang':[38,61,102,109]}
    for ed,chs in {'hawkes':[100,101,102],'yang':[38,61,102,109]}.items():
        for ch in chs:
            text=(chapter_root/ed/f'chapter_{ch:03d}.txt').read_text(encoding='utf-8')
            samples.append(dict(edition=ed,sample_id=f'{ed}_chapter_{ch:03d}',
                sample_type='chapter',chapter=ch,title=m.chapter_title(text),window_id=None,text=text))
    nlp=m.build_nlp()
    computed=m.build_feature_frame(samples,nlp)
    original100=c[(c.edition=='hawkes')&(c.chapter==100)][m.FEATURE_COLUMNS].iloc[0].to_numpy(float)
    repeated100=computed[(computed.edition=='hawkes')&(computed.chapter==100)][m.FEATURE_COLUMNS].iloc[0].to_numpy(float)
    maxdiff=float(np.abs(original100-repeated100).max())
    assert maxdiff<1e-12, f'Feature-extraction environment drift: sentinel maxdiff {maxdiff}'
    affected=((c.edition=='hawkes')&c.chapter.isin(changed['hawkes']))|((c.edition=='yang')&c.chapter.isin(changed['yang']))
    replacements=computed[~((computed.edition=='hawkes')&computed.chapter.eq(100))]
    newc=pd.concat([c[~affected],replacements],ignore_index=True).sort_values(['edition','chapter'])
    samples=[];counts=[]
    for ch in [101,102]:
        texts={ed:(chapter_root/ed/f'chapter_{ch:03d}.txt').read_text(encoding='utf-8') for ed in ['hawkes','yang']}
        rawcounts={ed:len(m.token_spans(text)) for ed,text in texts.items()}
        keep=min(n//m.WINDOW_WORDS for n in rawcounts.values())
        for ed,text in texts.items():
            dest=window_root/ed/f'chapter_{ch:03d}';dest.mkdir(parents=True,exist_ok=True)
            # Only this explicitly created work-tree directory is modified.
            for old in dest.glob('window_*.txt'):old.unlink()
            windows=m.aligned_windows(text,m.WINDOW_WORDS,keep)
            for wi,window in enumerate(windows,1):
                (dest/f'window_{wi:02d}.txt').write_text(window,encoding='utf-8')
                samples.append(dict(edition=ed,sample_id=f'{ed}_chapter_{ch:03d}_window_{wi:02d}',
                    sample_type='window_1500',chapter=ch,title=m.chapter_title(text),window_id=wi,text=window))
            counts.append(dict(edition=ed,chapter=ch,regex_words=rawcounts[ed],retained_windows=keep))
    computedw=m.build_feature_frame(samples,nlp)
    neww=pd.concat([w[~w.chapter.isin([101,102])],computedw],ignore_index=True).sort_values(['edition','chapter','window_id'])
    newc.to_csv(a.out/'chapter_level_features.csv',index=False,encoding='utf-8-sig')
    neww.to_csv(a.out/'window_1500_features.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(counts).to_csv(a.out/'corrected_chapter_window_counts.csv',index=False)
    # Keep a compact record of the exact prior observations behind the disputed Table 3.
    c[['edition','sample_id','chapter','punctuation_entropy']].to_csv(a.out/'original_table3_punctuation_values.csv',index=False)
    rows=[]
    for frame,old,level in [(newc,c,'chapter'),(neww,w,'segment')]:
        merged=frame.merge(old,on='sample_id',suffixes=('_new','_old'),how='outer',indicator=True)
        for row in merged.to_dict('records'):
            for f in ['word_count']+m.FEATURE_COLUMNS:
                nv,ov=row.get(f+'_new'),row.get(f+'_old')
                if pd.isna(nv) or pd.isna(ov) or abs(nv-ov)>1e-12:
                    rows.append(dict(level=level,sample_id=row['sample_id'],variable=f,previous_value=ov,corrected_value=nv,status=str(row['_merge'])))
    pd.DataFrame(rows).to_csv(a.out/'feature_correction_audit.csv',index=False)
    meta=dict(chapter_rows=len(newc),segment_rows=len(neww),chapter_pairs=newc.chapter.nunique(),
        segment_chapter_pairs=neww.chapter.nunique(),changed_hm_chapters=[101,102],
        changed_yy_chapters=[38,61,102,109],
        recomputed_windows_both_editions=[101,102],unchanged_sentinel_chapter=100,
        sentinel_max_absolute_feature_difference=maxdiff,
        versions={x:importlib.metadata.version(x) for x in ['numpy','pandas','scipy','spacy','en_core_web_sm','nltk','wordfreq','scikit-learn','shap','pypdf']},
        counts=counts)
    (a.out/'correction_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(meta,indent=2),flush=True)

if __name__=='__main__':main()
