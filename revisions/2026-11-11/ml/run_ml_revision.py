"""Reproducible chapter-grouped evaluation, sampling sensitivity and held-out SHAP.

All predictors and estimators are imported from the preserved source program.
Run with --source-root HLM --feature-root FEATURES --output RESULTS.
The --stage option allows heldout, sampling, stability or all.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold


def module_from(path):
    spec = importlib.util.spec_from_file_location('original_classification', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def metric_values(y, pred, score=None):
    tn = np.sum((y == 0) & (pred == 0)); fp = np.sum((y == 0) & (pred == 1))
    fn = np.sum((y == 1) & (pred == 0)); tp = np.sum((y == 1) & (pred == 1))
    p1 = tp / (tp + fp) if tp + fp else 0.
    p0 = tn / (tn + fn) if tn + fn else 0.
    r1 = tp / (tp + fn); r0 = tn / (tn + fp)
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.
    f0 = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.
    return {'accuracy': (tp + tn) / len(y), 'macro_precision': (p1 + p0) / 2,
            'macro_recall': (r1 + r0) / 2, 'macro_f1': (f1 + f0) / 2,
            'yang_precision': p1, 'yang_recall': r1, 'yang_f1': f1,
            'specificity': r0, 'roc_auc': roc_auc_score(y, score) if score is not None else np.nan}


def bootstrap(y, pred, score, chapter, n=2000):
    rng = np.random.default_rng(20261111)
    groups = sorted(np.unique(chapter)); indices = [np.flatnonzero(chapter == g) for g in groups]
    rows = []
    for selected in rng.integers(0, len(groups), (n, len(groups))):
        ix = np.concatenate([indices[j] for j in selected])
        rows.append(metric_values(y[ix], pred[ix], None if score is None else score[ix]))
    frame = pd.DataFrame(rows)
    result = {}
    for key in frame.columns:
        result[key+'_ci_low'], result[key+'_ci_high'] = frame[key].quantile([.025, .975]).to_numpy()
    return result


def fit_models(source, train, test):
    fitted = {}; predicted = {}; scores = {}; trainpred = {}; warning_records = []
    xtrain, xtest = train[source.ALL_FEATURES], test[source.ALL_FEATURES]
    ytrain = (train.edition == 'yang').astype(int)
    for name, builder in source.MODEL_BUILDERS.items():
        model = builder()
        # A single estimator thread avoids nested process oversubscription.
        if 'n_jobs' in model.get_params(deep=False):
            model.set_params(n_jobs=1)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            model.fit(xtrain, ytrain)
        warning_records += [{'algorithm': name, 'warning': str(w.message)} for w in ws]
        fitted[name] = model
        predicted[name] = model.predict(xtest).astype(int)
        trainpred[name] = model.predict(xtrain).astype(int)
        if hasattr(model, 'predict_proba'):
            scores[name] = model.predict_proba(xtest)[:, 1]
        elif hasattr(model, 'decision_function'):
            scores[name] = model.decision_function(xtest)
        else:
            scores[name] = None
    # This reproduces sklearn hard voting with sorted class labels: a tie is 0.
    predicted['Voting Ensemble'] = (np.column_stack(list(predicted.values())).sum(axis=1) > 4).astype(int)
    trainpred['Voting Ensemble'] = (np.column_stack(list(trainpred.values())).sum(axis=1) > 4).astype(int)
    scores['Voting Ensemble'] = None  # hard voting does not produce a probability score
    return fitted, predicted, scores, trainpred, warning_records


def heldout(args, source):
    rows = []; predictions = []; split = {}; shap_rows = []; shap_summaries = []; warns = []; corrs = []
    import shap
    datasets=args.datasets.split(',')
    for dataset in datasets:
        df = pd.read_csv(args.feature_root / (dataset+'_features.csv'))
        chapters = np.array(sorted(df.chapter.unique()))
        test_chapters = np.sort(np.random.default_rng(42).choice(chapters, size=24, replace=False))
        train, test = df[~df.chapter.isin(test_chapters)].copy(), df[df.chapter.isin(test_chapters)].copy()
        split[dataset] = {'test_chapters': test_chapters.tolist(), 'n_train':len(train), 'n_test':len(test),
                          'train_chapters': sorted(map(int, train.chapter.unique())), 'seed':42}
        fitted, preds, scores, trains, ws = fit_models(source, train, test)
        warns += [{'dataset':dataset, **w} for w in ws]
        y = (test.edition == 'yang').astype(int).to_numpy()
        ytrain = (train.edition == 'yang').astype(int).to_numpy()
        for name in preds:
            values = metric_values(y, preds[name], scores[name])
            row = {'dataset': dataset, 'algorithm':name, 'n_train':len(train), 'n_test':len(test),
                   'n_test_chapters':len(test_chapters), **values,
                   'train_accuracy':accuracy_score(ytrain, trains[name]),
                   'generalization_gap':accuracy_score(ytrain, trains[name])-values['accuracy'],
                   **bootstrap(y, preds[name], scores[name], test.chapter.to_numpy())}
            rows.append(row)
            for i, r in enumerate(test.itertuples()):
                predictions.append({'dataset':dataset,'algorithm':name,'sample_id':r.sample_id,'chapter':r.chapter,
                                    'edition':r.edition,'true_yang':int(y[i]),'pred_yang':int(preds[name][i]),
                                    'score_yang': None if scores[name] is None else float(scores[name][i])})
        explainer = shap.TreeExplainer(fitted['Random Forest'], feature_perturbation='tree_path_dependent', model_output='raw')
        sv = explainer.shap_values(test[source.ALL_FEATURES], check_additivity=True)
        sv = sv[:, :, 1] if isinstance(sv, np.ndarray) and sv.ndim == 3 else sv[1]
        base = float(np.asarray(explainer.expected_value)[1])
        error = float(np.max(np.abs(base+sv.sum(axis=1)-scores['Random Forest'])))
        assert error < 1e-6, error
        split[dataset]['shap'] = {'explained_partition':'held-out only', 'n_explained':len(test),
                                 'training_n':len(train), 'class':'Yang–Yang edition', 'expected_value':base,
                                 'maximum_additivity_error':error,
                                 'perturbation':'tree_path_dependent', 'background':'training path counts stored in fitted trees'}
        for j, feature in enumerate(source.ALL_FEATURES):
            shap_summaries.append({'dataset':dataset,'feature':feature,'mean_absolute_shap':float(np.abs(sv[:,j]).mean())})
            for i, r in enumerate(test.itertuples()):
                shap_rows.append({'dataset':dataset,'sample_id':r.sample_id,'chapter':r.chapter,'edition':r.edition,
                                  'feature':feature,'feature_value':float(test.iloc[i][feature]),'shap_yang':float(sv[i,j]),'base_yang':base})
        for edition, part in [('pooled',df), *list(df.groupby('edition'))]:
            cm = part[source.ALL_FEATURES].corr(method='spearman')
            for i, f in enumerate(source.ALL_FEATURES):
                for g in source.ALL_FEATURES[i+1:]:
                    corrs.append({'dataset':dataset,'edition':edition,'feature_1':f,'feature_2':g,'spearman_r':cm.loc[f,g]})
        print(f'{dataset}: heldout+SHAP finished',flush=True)
    def save_dataset_csv(records,filename):
        new=pd.DataFrame(records);path=args.output/filename
        if path.exists() and len(datasets)<2:
            old=pd.read_csv(path);new=pd.concat([old[~old.dataset.isin(datasets)],new],ignore_index=True)
        new.to_csv(path,index=False)
    save_dataset_csv(rows,'heldout_metrics.csv')
    save_dataset_csv(predictions,'heldout_predictions.csv')
    save_dataset_csv(shap_rows,'heldout_shap_values.csv')
    save_dataset_csv(shap_summaries,'heldout_shap_summary.csv')
    save_dataset_csv(corrs,'feature_correlations.csv')
    if (args.output/'heldout_manifest.json').exists() and len(datasets)<2:
        old=json.loads((args.output/'heldout_manifest.json').read_text(encoding='utf-8'))
        old.update(split);split=old
    (args.output/'heldout_manifest.json').write_text(json.dumps(split,indent=2),encoding='utf-8')
    (args.output/'fit_warnings.json').write_text(json.dumps(warns,indent=2),encoding='utf-8')


def sampling(args, source):
    df = pd.read_csv(args.output/'sampling_features.csv')
    groups = np.array(sorted(df.chapter.unique()))
    folds = list(KFold(5,shuffle=True,random_state=42).split(groups))
    rows = []; predictions = []; manifests = []
    for scheme, part in df.groupby('scheme',sort=True):
        for fold, (tr,te) in enumerate(folds,1):
            train, test = part[part.chapter.isin(groups[tr])], part[part.chapter.isin(groups[te])]
            assert set(train.chapter).isdisjoint(set(test.chapter))
            fitted, preds, scores, trains, ws = fit_models(source,train,test)
            y=(test.edition=='yang').astype(int).to_numpy(); yt=(train.edition=='yang').astype(int).to_numpy()
            for name in preds:
                vals=metric_values(y,preds[name],scores[name])
                rows.append({'scheme':scheme,'fold':fold,'algorithm':name,'n_train':len(train),'n_test':len(test),
                             **vals,'train_accuracy':accuracy_score(yt,trains[name]),
                             'generalization_gap':accuracy_score(yt,trains[name])-vals['accuracy']})
                for i,r in enumerate(test.itertuples()):
                    predictions.append({'scheme':scheme,'fold':fold,'algorithm':name,'sample_id':r.sample_id,'chapter':r.chapter,
                                        'edition':r.edition,'true_yang':int(y[i]),'pred_yang':int(preds[name][i])})
            manifests.append({'scheme':scheme,'fold':fold,'test_chapters':groups[te].tolist(),'warnings':ws})
            pd.DataFrame(rows).to_csv(args.output/'sampling_fold_metrics.csv',index=False)
            print(f'{scheme} fold {fold} finished',flush=True)
    pd.DataFrame(predictions).to_csv(args.output/'sampling_oof_predictions.csv',index=False)
    result=pd.DataFrame(rows)
    summary=result.groupby(['scheme','algorithm']).agg(mean_accuracy=('accuracy','mean'),sd_accuracy=('accuracy','std'),
              mean_macro_f1=('macro_f1','mean'),mean_gap=('generalization_gap','mean')).reset_index()
    summary.to_csv(args.output/'sampling_summary.csv',index=False)
    stability_rows=[]
    for scheme,part in summary.groupby('scheme'):
        vote=part[part.algorithm=='Voting Ensemble'].iloc[0]
        for r in part[part.algorithm!='Voting Ensemble'].itertuples():
            stability_rows.append({'scheme':scheme,'algorithm':r.algorithm,
                'base_sd':r.sd_accuracy,'vote_sd':vote.sd_accuracy,
                'vote_lower_sd':bool(vote.sd_accuracy<r.sd_accuracy),
                'base_gap':r.mean_gap,'vote_gap':vote.mean_gap,
                'vote_lower_gap':bool(vote.mean_gap<r.mean_gap)})
    pd.DataFrame(stability_rows).to_csv(args.output/'grouped_stability_comparison.csv',index=False)
    (args.output/'sampling_cv_manifest.json').write_text(json.dumps(manifests,indent=2),encoding='utf-8')


def stability(args,source):
    old=args.previous_root/'03_experiments'/'review 2-comment 4'/'results'/'training_size_summary.csv'
    if not old.exists(): return
    df=pd.read_csv(old)
    rows=[]
    for dataset,part in df.groupby('dataset'):
        votes=part[part.algorithm=='Voting Ensemble'].set_index('train_fraction')
        for r in part.itertuples():
            if r.algorithm=='Voting Ensemble':continue
            v=votes.loc[r.train_fraction]
            rows.append({'dataset':dataset,'algorithm':r.algorithm,'train_fraction':r.train_fraction,
                         'base_sd_percentage_points':r.sd_test_accuracy,'vote_sd_percentage_points':v.sd_test_accuracy,
                         'vote_lower_sd':bool(v.sd_test_accuracy<r.sd_test_accuracy),
                         'base_gap_percentage_points':r.mean_generalization_gap,'vote_gap_percentage_points':v.mean_generalization_gap,
                         'historical_note':'Prior sample-stratified splits; descriptive sensitivity, not independent inferential replicates.'})
    pd.DataFrame(rows).to_csv(args.output/'prior_stability_comparison.csv',index=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--feature-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--previous-root',type=Path,default=Path('Revision_20260824'))
    p.add_argument('--datasets',default='chapter_level,window_1500')
    p.add_argument('--stage',choices=['heldout','sampling','stability','all'],default='all');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True);source=module_from(args.source_root/'run_hlm_classification.py')
    if args.stage in ['heldout','all']:heldout(args,source)
    if args.stage in ['sampling','all']:sampling(args,source)
    if args.stage in ['stability','all']:stability(args,source)
    meta={'python':sys.version,'packages':{x:importlib.metadata.version(x) for x in ['numpy','pandas','scipy','scikit-learn','spacy','en-core-web-sm','nltk','wordfreq','shap']},
          'models':'Original eight source estimators at recorded defaults; hard vote reconstructed from their predictions; ties favor class0 (Hawkes–Minford).',
          'bootstrap':'2000 percentile bootstrap draws of complete source-chapter groups; seed20261111; uncertainty conditional on fitted model and fixed test set.',
          'positive_class':'Yang–Yang edition', 'negative_class':'Hawkes–Minford edition', 'hard_vote_auc':'Not applicable: no probability or decision score.',
          'stability':'SD of fold accuracy plus train-test accuracy gap; descriptive, not independent replicate inference.',
          'features':source.ALL_FEATURES,
          'input_sha256':{x:hashlib.sha256((args.feature_root/(x+'_features.csv')).read_bytes()).hexdigest() for x in ['chapter_level','window_1500']},
          'classifier_source_sha256':hashlib.sha256((args.source_root/'run_hlm_classification.py').read_bytes()).hexdigest()}
    (args.output/'ml_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')


if __name__=='__main__':main()
