"""Position sensitivity using the unchanged original entropy implementation.

Usage: python extract_sampling_features.py --source-root PATH_TO_HLM --output DIR
The source editions must have been acquired legally; no text is exported here.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def load_source(path):
    spec = importlib.util.spec_from_file_location('original_entropy_pipeline', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--processes', type=int, default=4)
    parser.add_argument('--chapter-root', type=Path)
    parser.add_argument('--hm-pdf', type=Path, help='Legally obtained Hawkes-Minford PDF; requires --yy-pdf')
    parser.add_argument('--yy-pdf', type=Path, help='Legally obtained Yang-Yang PDF; requires --hm-pdf')
    parser.add_argument('--cache-features', type=Path)
    parser.add_argument('--invalidate-chapters', default='')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = load_source(args.source_root / 'run_hlm_pipeline.py')
    chapter_root = args.chapter_root or args.source_root / 'data' / 'chapters'
    if bool(args.hm_pdf) != bool(args.yy_pdf):
        parser.error('--hm-pdf and --yy-pdf must be provided together')
    if args.hm_pdf and args.chapter_root:
        parser.error('Use either PDF inputs or --chapter-root, not both')
    extracted = {}
    if args.hm_pdf:
        for edition, pdf_path in [('hawkes',args.hm_pdf),('yang',args.yy_pdf)]:
            extracted[edition] = {r.chapter:r.text for r in source.extract_chapters(source.read_pdf_text(pdf_path),edition)}
        print('Extracted chapters in memory from the two supplied PDF paths.',flush=True)
    samples = []
    starts = []
    # Early reproduces original prefix selection. Position-stratified spreads
    # the same k disjoint windows from chapter start to chapter end. Three
    # random replicates allocate the available unused words as random gaps.
    schemes = ['prefix', 'position_stratified', 'random_101', 'random_202', 'random_303']
    for chapter in range(1, 121):
        data = {}
        for edition in ['hawkes', 'yang']:
            text = extracted[edition][chapter] if extracted else (chapter_root / edition / f'chapter_{chapter:03d}.txt').read_text(encoding='utf-8')
            spans = source.token_spans(text)
            data[edition] = (text, spans)
        keep = min(len(z[1]) // 1500 for z in data.values())
        if keep == 0:
            continue
        for edition, (text, spans) in data.items():
            slack = len(spans) - 1500 * keep
            for scheme in schemes:
                if scheme == 'prefix':
                    offsets = np.zeros(keep, dtype=int)
                elif scheme == 'position_stratified':
                    offsets = np.rint(np.linspace(0, slack, keep)).astype(int) if keep > 1 else np.array([slack // 2])
                else:
                    seed = int(scheme.split('_')[1])
                    # Independent stream for each edition/chapter/replicate.
                    rng = np.random.default_rng(np.random.SeedSequence([seed, chapter, 0 if edition == 'hawkes' else 1]))
                    offsets = np.sort(rng.integers(0, slack + 1, size=keep))
                for j, offset in enumerate(offsets):
                    start = 1500 * j + int(offset)
                    end = start + 1499
                    sample_id = f'{scheme}_{edition}_chapter_{chapter:03d}_window_{j+1:02d}'
                    segment = text[spans[start].start():spans[end].end()].strip()
                    record = {'scheme': scheme, 'edition': edition, 'chapter': chapter,
                              'window_id': j+1, 'sample_id': sample_id, 'regex_start_0based': start,
                              'regex_end_exclusive': end+1, 'chapter_regex_words': len(spans),
                              'windows_retained': keep, 'regex_words': 1500,
                              'chapter_text_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest()}
                    starts.append(record)
                    samples.append((record, segment))
    manifest = pd.DataFrame(starts)
    manifest.to_csv(args.output / 'sampling_window_manifest.csv', index=False)
    coverage = manifest[['edition','chapter','chapter_regex_words','windows_retained']].drop_duplicates().copy()
    coverage['retained_regex_words'] = 1500 * coverage['windows_retained']
    coverage['retained_fraction'] = coverage['retained_regex_words'] / coverage['chapter_regex_words']
    coverage.to_csv(args.output / 'sampling_coverage_by_chapter.csv',index=False)
    # Deduplicate exactly coincident windows across strategies for computation.
    unique = {}
    for r, txt in samples:
        key = (r['edition'], r['chapter'], r['regex_start_0based'])
        unique.setdefault(key, (r, txt))
    print(f'{len(samples)} selected windows; {len(unique)} unique windows', flush=True)
    nlp = source.build_nlp()
    rows = []
    if args.cache_features:
        cache = pd.read_csv(args.cache_features)
        invalid = {int(x) for x in args.invalidate_chapters.split(',') if x}
        cache = cache[~cache.chapter.isin(invalid)].drop_duplicates(['edition','chapter','regex_start_0based'])
        if 'chapter_text_sha256' in cache:
            fingerprints=manifest[['edition','chapter','chapter_text_sha256']].drop_duplicates()
            cache=cache.merge(fingerprints,on=['edition','chapter','chapter_text_sha256'],validate='many_to_one')
        needed = ['edition','chapter','regex_start_0based','word_count','sample_type',*source.FEATURE_COLUMNS]
        for row in cache[needed].to_dict('records'):
            key = (row['edition'],row['chapter'],row['regex_start_0based'])
            if key in unique:
                rows.append(row)
                del unique[key]
        print(f'Reused {len(rows)} unaffected cached windows; computing {len(unique)}',flush=True)
    t0 = time.time()
    items = list(unique.items())
    for i, (doc, (key, (r, txt))) in enumerate(zip(nlp.pipe((x[1][1] for x in items), batch_size=16, n_process=args.processes), items)):
        row = source.compute_feature_row(doc, r['edition'], r['sample_id'], 'window_1500', r['chapter'], '', r['window_id'])
        row['regex_start_0based'] = r['regex_start_0based']
        rows.append(row)
        if (i + 1) % 100 == 0:
            print(f'{i+1}/{len(items)} feature rows; {time.time()-t0:.1f} s', flush=True)
            pd.DataFrame(rows).to_csv(args.output / 'unique_sampling_features.partial.csv', index=False)
    features = pd.DataFrame(rows).drop(columns=['sample_id', 'window_id', 'title'],errors='ignore')
    result = manifest.merge(features, on=['edition', 'chapter', 'regex_start_0based'], validate='many_to_one')
    if len(result) != len(samples) or result[source.FEATURE_COLUMNS].isna().any().any():
        raise AssertionError('Unexpected count or missing feature value')
    result.to_csv(args.output / 'sampling_features.csv', index=False)
    metadata = {'python': sys.version, 'schemes': schemes, 'seed_definition': '[replicate_seed, chapter, edition_index]',
                'each_scheme_samples': result.groupby('scheme').size().to_dict(),
                'features': source.FEATURE_COLUMNS,
                'sampling': 'Nonoverlapping 1500-regex-word windows within each strategy and chapter; equal k per corresponding chapter; no textual alignment assertion.',
                'position_stratified': 'Start offsets j*1500 + round(linspace(0, chapter_words-1500*k,k)); k=1 uses center.',
                'random': 'Start offsets j*1500 + sorted independent discrete Uniform(0, chapter_words-1500*k) offsets.',
                'packages': {x: importlib.metadata.version(x) for x in ['numpy','pandas','spacy','en-core-web-sm','nltk','wordfreq','scikit-learn','shap']}}
    (args.output / 'sampling_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
