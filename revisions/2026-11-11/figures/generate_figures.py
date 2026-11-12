"""Regenerate the main figures from the final corrected numerical results.

Python / matplotlib is the exclusive graphics backend. No source raster is edited.
Run with --features, --baseline, --ml-results and --output directory arguments.
"""
from pathlib import Path
import argparse
import json
import string
import io
import hashlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from PIL import Image

FEATURES = {
    "wordform_unigram_entropy": ("WFU", "Wordform unigram"),
    "lemma_unigram_entropy": ("LU", "Lemma unigram"),
    "function_word_entropy": ("FW", "Function word"),
    "word_length_entropy": ("WL", "Word length"),
    "lexical_frequency_band_entropy": ("FB", "Frequency band"),
    "punctuation_entropy": ("PUNC", "Punctuation"),
    "pos_unigram_entropy": ("POSU", "POS unigram"),
    "pos_bigram_entropy": ("POSB", "POS bigram"),
    "pos_trigram_entropy": ("POST", "POS trigram"),
    "clause_marker_entropy": ("CM", "Clause marker"),
    "dependency_relation_entropy": ("DR", "Dependency relation"),
    "coordination_subordination_entropy": ("CS", "Coordination/subordination"),
    "character_entropy": ("CHAR", "Character"),
    "character_2gram_entropy": ("C2G", "Character 2-gram"),
    "character_3gram_entropy": ("C3G", "Character 3-gram"),
    "character_4gram_entropy": ("C4G", "Character 4-gram"),
    "word_bigram_entropy": ("WB", "Word bigram"),
    "word_trigram_entropy": ("WT", "Word trigram"),
}
COLORS = ["#73618A", "#D18D76"]
LABELS = ["Hawkes–Minford edition", "Yang–Yang edition"]
ALGORITHMS = ["Logistic Regression", "Naive Bayes", "SVM", "K-NN", "Random Forest",
              "Ridge Classifier", "Gradient Boosting", "Neural Network", "Voting Ensemble"]
SHORT = ["LR", "NB", "SVM", "K-NN", "RF", "Ridge", "GB", "NN", "Vote"]
MODEL_SHORT = dict(zip(ALGORITHMS, SHORT))
CMAP = LinearSegmentedColormap.from_list("edition_accuracy", ["#F6F2F0", "#CAB9CA", "#72577E"])
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
    "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42,
    "savefig.facecolor": "white",
})


def save(fig, num, output):
    output.mkdir(parents=True, exist_ok=True)
    # Keep every printed label within the journal's 8–12 pt range.
    for text in fig.findobj(match=matplotlib.text.Text):
        text.set_fontfamily('Arial')
        text.set_fontsize(min(12, max(8, text.get_fontsize())))
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    width,height=fig.bbox.width,fig.bbox.height
    clipped=[]
    # Matplotlib retains labels for ticks outside the view interval but does
    # not draw them. Exclude those dormant labels from the canvas audit.
    dormant=set()
    for ax in fig.axes:
        for axis in [ax.xaxis,ax.yaxis]:
            lo,hi=sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                if not lo <= tick.get_loc() <= hi:
                    dormant.update([id(tick.label1),id(tick.label2)])
    for text in fig.findobj(match=matplotlib.text.Text):
        if text.get_visible() and text.get_text() and id(text) not in dormant:
            box=text.get_window_extent(renderer)
            if box.x0 < -1 or box.y0 < -1 or box.x1 > width+1 or box.y1 > height+1:
                clipped.append({'text':text.get_text(),'bounds':list(box.extents)})
    sizes=[text.get_fontsize() for text in fig.findobj(match=matplotlib.text.Text)
           if text.get_visible() and text.get_text() and id(text) not in dormant]
    (output/f'Fig {num}_layout_qa.json').write_text(json.dumps({'text_outside_canvas':clipped,
        'minimum_font_size_pt':min(sizes),'maximum_font_size_pt':max(sizes),'font_family':'Arial'},indent=2),encoding='utf-8')
    # Fixed physical canvas: no tight-bbox enlargement beyond journal width.
    raster=io.BytesIO()
    fig.savefig(raster,format='tiff',dpi=600)
    raster.seek(0)
    with Image.open(raster) as im:
        im.convert('RGB').save(output / f"Fig {num}.tif",dpi=(600,600),compression='tiff_lzw')
    fig.savefig(output / f"Fig {num}.png", dpi=180)
    fig.savefig(output / f"Fig {num}.svg")
    plt.close(fig)


def panel(ax, letter):
    ax.text(-0.16, 1.08, letter, transform=ax.transAxes, fontsize=10, fontweight="bold")


def distribution_figures(features_root, output, data_out):
    chapter = pd.read_csv(features_root / "chapter_level_features.csv")
    segment = pd.read_csv(features_root / "window_1500_features.csv")
    segment = segment.groupby(["edition", "chapter"], as_index=False)[list(FEATURES)].mean()
    long = []
    for dataset, frame in [("chapter_level", chapter), ("segment_chapter_mean", segment)]:
        p = frame.melt(id_vars=["edition", "chapter"], value_vars=list(FEATURES),
                       var_name="feature", value_name="entropy_bits")
        p["dataset"] = dataset
        long.append(p)
    pd.concat(long).to_csv(data_out / "Figs1_3_entropy_distributions.csv", index=False)
    captions = {}
    for group, num in [(list(FEATURES)[:6], 1), (list(FEATURES)[6:12], 2), (list(FEATURES)[12:], 3)]:
        fig, axes = plt.subplots(4, 3, figsize=(7.4, 8.3))
        fig.subplots_adjust(left=.095, right=.98, bottom=.055, top=.90, hspace=.76, wspace=.42)
        fig.legend(handles=[Patch(facecolor=c, label=l) for c,l in zip(COLORS,LABELS)],
                   loc="upper center", ncol=2, bbox_to_anchor=(.5, .985), fontsize=8)
        for level, frame in enumerate([chapter, segment]):
            startrow = level * 2
            ns = frame.groupby("edition").size()
            title = f"Chapter observations (n = {ns.iloc[0]} per edition)" if level == 0 else f"Means of chapter-matched segments (n = {ns.iloc[0]} chapters per edition)"
            fig.text(.5, .94 if level == 0 else .485, title, ha="center", fontsize=9, fontweight="bold")
            for j, feature in enumerate(group):
                ax = axes[startrow+j//3,j%3]
                vals = [frame.loc[frame.edition.eq(e),feature].to_numpy() for e in ["hawkes","yang"]]
                violins = ax.violinplot(vals, positions=[0,1], widths=.8, showextrema=False)
                for body,col in zip(violins["bodies"], COLORS):
                    body.set_facecolor(col); body.set_edgecolor("#444444"); body.set_alpha(.85); body.set_linewidth(.6)
                ax.boxplot(vals, positions=[0,1], widths=.10, showfliers=False,
                           medianprops={"color":"white","linewidth":1.1},
                           boxprops={"color":"#333333","linewidth":.9},
                           whiskerprops={"color":"#333333","linewidth":.9},
                           capprops={"color":"#333333","linewidth":.9})
                ax.set_xticks([0,1], ["HM", "YY"])
                ax.set_title(f"{FEATURES[feature][1]} ({FEATURES[feature][0]})", pad=5)
                ax.set_ylabel("Entropy (bits)")
                ax.grid(axis="y", color="#E7E7E7", linewidth=.5); ax.set_axisbelow(True)
                panel(ax,string.ascii_lowercase[level*6+j])
        save(fig,num,output)
        captions[f"Fig {num}"] = (f"Distribution of {'lexical' if num == 1 else 'syntactic' if num == 2 else 'character and word n-gram'} entropy by translation edition. "
            "Panels a–f show complete-chapter observations; panels g–l show the arithmetic mean of retained 1,500-regex-word segment entropies within each chapter. "
            "Each chapter receives equal weight in the lower panels. HM denotes the Hawkes–Minford edition and YY the Yang–Yang edition. "
            "Violin width represents kernel-density estimates, the central line is the median, boxes show the interquartile range, and whiskers extend to observations within 1.5 interquartile ranges. "
            "Entropy uses base-2 logarithms. Exact paired tests, adjusted p-values and effect sizes are reported in the corresponding table and supporting data.")
    return captions


def top_features(baseline, output, data_out):
    frames=[]
    for dataset in ["chapter_level", "window_1500"]:
        mean = pd.read_csv(baseline/dataset/"individual_feature_accuracy_mean.csv").set_index("feature")
        std = pd.read_csv(baseline/dataset/"individual_feature_accuracy_std.csv").set_index("feature")
        rows=[]
        for feat,row in mean.iterrows():
            best=row.astype(float).idxmax()
            rows.append({"dataset":dataset,"feature":feat,"algorithm":best,"accuracy_percent":float(row[best]),"fold_sd_percent":float(std.loc[feat,best])})
        frames.append(pd.DataFrame(rows).sort_values(["accuracy_percent","feature"],ascending=[False,True]).head(5))
    source=pd.concat(frames);source.to_csv(data_out/"Fig4_top_single_features.csv",index=False)
    fig,axes=plt.subplots(2,1,figsize=(7.4,5.7))
    fig.subplots_adjust(left=.30,right=.96,bottom=.10,top=.93,hspace=.5)
    for ax,(dataset,part),letter in zip(axes,source.groupby("dataset",sort=False),"ab"):
        ax.barh(np.arange(5),part.accuracy_percent,color=[COLORS[1]]+[COLORS[0]]*4,height=.65)
        ax.set_yticks(np.arange(5),[FEATURES[x][1] for x in part.feature]);ax.invert_yaxis()
        for y,row in enumerate(part.itertuples()):ax.text(row.accuracy_percent+.8,y,f"{row.accuracy_percent:.2f}  {MODEL_SHORT[row.algorithm]}",va="center",fontsize=8)
        ax.set_xlim(0,103);ax.set_xticks([0,20,40,60,80,100]);ax.set_xlabel("Mean cross-validation accuracy (%)")
        ax.set_title("Chapter observations" if dataset=="chapter_level" else "Chapter-matched 1,500-word segments",fontsize=10)
        ax.grid(axis="x",color="#E7E7E7",linewidth=.6);ax.set_axisbelow(True);panel(ax,letter)
    save(fig,4,output)
    return {"Fig 4":"Five highest-performing individual entropy features at each sampling level. Bars show the highest mean accuracy among the eight prespecified base classifiers under 10-fold sample-stratified cross-validation (shuffle=True, random_state=42). The classifier achieving each displayed value is identified next to the bar. This is an exploratory ranking over classifiers, not a separate held-out estimate. Fold-level variability is supplied in the machine-readable results. Chapter-grouped held-out evaluation is reported separately. LR, logistic regression; NB, Gaussian naive Bayes; K-NN, k-nearest neighbors; SVM, support vector machine; RF, Random Forest; Ridge, ridge classifier; GB, gradient boosting; NN, neural network."}


def heatmap(baseline, output, data_out):
    rows=[];arrays=[]
    groups=["lexical","syntactic","ngram","all_features"]
    for dataset in ["chapter_level","window_1500"]:
        frame=pd.read_csv(baseline/dataset/"feature_group_accuracy_mean.csv").set_index("feature_group").loc[groups,ALGORITHMS]
        arrays.append(frame.to_numpy(float))
        a=frame.reset_index().melt(id_vars="feature_group",var_name="algorithm",value_name="accuracy_percent");a["dataset"]=dataset;rows.append(a)
    pd.concat(rows).to_csv(data_out/"Fig6_feature_group_accuracy.csv",index=False)
    fig,axes=plt.subplots(2,1,figsize=(7.4,5.3))
    fig.subplots_adjust(left=.14,right=.87,bottom=.10,top=.92,hspace=.50)
    for i,(ax,arr) in enumerate(zip(axes,arrays)):
        im=ax.imshow(arr,vmin=50,vmax=100,cmap=CMAP,aspect="auto")
        for r in range(4):
            for c in range(9):ax.text(c,r,f"{arr[r,c]:.1f}",ha="center",va="center",fontsize=8,color="white" if arr[r,c]>87 else "#282228")
        ax.set_xticks(np.arange(9),SHORT);ax.set_yticks(np.arange(4),["Lexical","Syntactic","N-gram","All 18"])
        ax.tick_params(length=0);ax.set_title("Chapter observations" if i==0 else "Chapter-matched 1,500-word segments",fontsize=10,pad=10);panel(ax,"ab"[i])
    cax=fig.add_axes([.895,.18,.018,.64]);fig.colorbar(im,cax=cax,label="Mean accuracy (%)")
    save(fig,6,output)
    return {"Fig 6":"Feature-group classification performance at the chapter and segment levels. Cells report mean accuracy (%) under 10-fold sample-stratified cross-validation (shuffle=True, random_state=42), using the corrected source-chapter extraction. Each row represents six lexical, six syntactic, six n-gram, or all 18 entropy features. Columns represent the eight prespecified base classifiers and the hard-voting ensemble. Colors share a 50–100% scale across panels. These sample-stratified results are descriptive baselines; the chapter-grouped evaluations assess generalization to unseen chapters. Algorithm abbreviations are defined in Fig 4."}


def heldout_figures(results, output, data_out):
    pred=pd.read_csv(results/'heldout_predictions.csv')
    shap_rows=pd.read_csv(results/'heldout_shap_values.csv')
    summary=pd.read_csv(results/'heldout_shap_summary.csv')
    manifest=json.loads((results/'heldout_manifest.json').read_text(encoding='utf-8'))
    captions={}
    models=['Random Forest','Voting Ensemble']
    source=pred[pred.algorithm.isin(models)].copy();source.to_csv(data_out/'Fig5_heldout_predictions.csv',index=False)
    cm_rows=[]
    fig,axes=plt.subplots(2,2,figsize=(7.4,6.1))
    fig.subplots_adjust(left=.10,right=.96,bottom=.12,top=.88,wspace=.42,hspace=.72)
    fig.text(.5,.97,'HM = Hawkes–Minford edition     YY = Yang–Yang edition',ha='center',fontsize=8)
    for i,dataset in enumerate(['chapter_level','window_1500']):
        for j,model in enumerate(models):
            ax=axes[i,j];part=source[(source.dataset==dataset)&(source.algorithm==model)]
            cm=np.zeros((2,2),dtype=int)
            for r in part.itertuples():cm[int(r.true_yang),int(r.pred_yang)]+=1
            pct=cm/cm.sum(axis=1,keepdims=True)*100
            ax.imshow(pct,vmin=0,vmax=100,cmap=CMAP)
            for r in range(2):
                for c in range(2):
                    ax.text(c,r,f'{cm[r,c]}\n({pct[r,c]:.1f}%)',ha='center',va='center',fontsize=10,color='white' if pct[r,c]>65 else '#272727')
                    cm_rows.append({'dataset':dataset,'algorithm':model,'true_edition':['hawkes','yang'][r],'predicted_edition':['hawkes','yang'][c],'count':int(cm[r,c]),'row_percent':float(pct[r,c])})
            ax.set_xticks([0,1],['HM','YY']);ax.set_yticks([0,1],['HM','YY']);ax.set_xlabel('Predicted edition');ax.set_ylabel('True edition')
            ax.set_title(f'{model}\n'+('Chapters' if i==0 else '1,500-word segments')+f' (n = {len(part)})',pad=9)
            ax.tick_params(length=0);panel(ax,'abcd'[i*2+j])
    pd.DataFrame(cm_rows).to_csv(data_out/'Fig5_heldout_confusion_matrices.csv',index=False);save(fig,5,output)
    captions['Fig 5']='Confusion matrices for Random Forest and hard voting on the chapter-grouped held-out set. The same 24 source chapters were selected using random seed 42 at both sampling levels; all observations from those chapters in both editions were excluded from training. Models used all 18 entropy features and were fitted on the remaining 96 chapters. Panels a–b contain 48 chapter observations and panels c–d contain 142 segments. Each cell shows the observation count and percentage within its true-edition row; shading uses row percentages. HM denotes the Hawkes–Minford edition and YY the Yang–Yang edition. Rows represent true editions and columns predicted editions.'
    import shap
    color=LinearSegmentedColormap.from_list('entropy_value',[COLORS[0],COLORS[1]])
    shap_rows.to_csv(data_out/'Figs7_10_heldout_shap_values.csv',index=False)
    summary.to_csv(data_out/'Fig9_heldout_shap_importance.csv',index=False)
    for dataset,num in [('chapter_level',7),('window_1500',8)]:
        part=shap_rows[shap_rows.dataset==dataset]
        x=part.pivot(index='sample_id',columns='feature',values='feature_value').reindex(columns=list(FEATURES))
        s=part.pivot(index='sample_id',columns='feature',values='shap_yang').reindex(index=x.index,columns=x.columns)
        ex=shap.Explanation(values=s.to_numpy(),data=x.to_numpy(),feature_names=[FEATURES[f][1] for f in x.columns])
        np.random.seed(42)  # Deterministic display-only beeswarm jitter.
        ax=shap.plots.beeswarm(ex,max_display=18,show=False,plot_size=(7.4,6.5),color=color)
        fig=plt.gcf();fig.subplots_adjust(left=.36,right=.90,bottom=.11,top=.92)
        ax.set_title(('Held-out chapters' if num==7 else 'Held-out chapter-matched segments')+f' (n = {len(x)})',fontsize=10,pad=15)
        ax.set_xlabel('SHAP contribution to Yang–Yang probability',fontsize=8);ax.tick_params(axis='y',labelsize=8);ax.tick_params(axis='x',labelsize=8)
        save(fig,num,output)
        captions[f'Fig {num}']=f'SHAP beeswarm plot for the Random Forest model evaluated on {len(x)} held-out '+('chapter observations' if num==7 else 'chapter-matched 1,500-word segments')+'. Each point is a held-out observation, ordered vertically by mean absolute SHAP contribution. Positive contributions increase the fitted probability of the Yang–Yang edition; negative contributions decrease that probability toward the Hawkes–Minford edition. Color indicates the feature value from low (purple) to high (orange). Tree SHAP used tree-path-dependent perturbation and training path counts stored in the fitted trees. The explained observations were excluded from model fitting. Contributions describe this fitted model and do not identify causal linguistic determinants; correlated entropy predictors can share or redistribute attribution.'
    fig,axes=plt.subplots(2,1,figsize=(7.4,8.6));fig.subplots_adjust(left=.35,right=.96,bottom=.065,top=.94,hspace=.30)
    for i,dataset in enumerate(['chapter_level','window_1500']):
        ax=axes[i];p=summary[summary.dataset==dataset].sort_values('mean_absolute_shap',ascending=False)
        ax.barh(np.arange(len(p)),p.mean_absolute_shap,color=[COLORS[1]]*3+[COLORS[0]]*(len(p)-3),height=.7)
        ax.set_yticks(np.arange(len(p)),[FEATURES[f][1] for f in p.feature]);ax.invert_yaxis()
        ax.set_xlabel('Mean absolute SHAP contribution');ax.set_title('Held-out chapters' if i==0 else 'Held-out chapter-matched segments',fontsize=10,pad=8)
        ax.grid(axis='x',color='#E7E7E7',linewidth=.6);ax.set_axisbelow(True);panel(ax,'ab'[i])
    save(fig,9,output)
    captions['Fig 9']='Random Forest feature importance calculated as the mean absolute SHAP contribution across the held-out observations shown in Figs 7 and 8. The three largest contributions at each sampling level are highlighted in orange. The rankings describe prediction attribution for each fitted model on its held-out observations. They are not effect sizes, hypothesis-test results, or evidence that the highlighted features causally determine edition differences.'
    fig,axes=plt.subplots(2,3,figsize=(7.4,5.6));fig.subplots_adjust(left=.09,right=.98,bottom=.11,top=.88,wspace=.40,hspace=.70)
    fig.text(.5,.96,'Held-out chapter observations',ha='center',fontsize=10,fontweight='bold')
    fig.text(.5,.48,'Held-out chapter-matched segments',ha='center',fontsize=10,fontweight='bold')
    selection=[]
    for i,dataset in enumerate(['chapter_level','window_1500']):
        best=summary[summary.dataset==dataset].nlargest(3,'mean_absolute_shap').feature.tolist()
        for j,feature in enumerate(best):
            p=shap_rows[(shap_rows.dataset==dataset)&(shap_rows.feature==feature)];ax=axes[i,j]
            ax.scatter(p.feature_value,p.shap_yang,color=COLORS[0],s=12,alpha=.7,edgecolors='none')
            ax.axhline(0,color='#999999',linewidth=.7,linestyle='--')
            ax.set_xlabel('Observed entropy (bits)');ax.set_ylabel('SHAP contribution')
            ax.set_title(f'{FEATURES[feature][1]} ({FEATURES[feature][0]})',fontsize=8,pad=6);panel(ax,'abcdef'[i*3+j]);selection.append(p)
    pd.concat(selection).to_csv(data_out/'Fig10_top_feature_dependence.csv',index=False);save(fig,10,output)
    captions['Fig 10']='SHAP dependence plots for the three features with the largest mean absolute held-out contribution at each sampling level. Panels a–c show chapter observations and d–f show chapter-matched 1,500-word segments. Each point pairs one observation’s entropy value with its contribution to the fitted Random Forest probability of the Yang–Yang edition; the dashed horizontal line denotes zero contribution. Features were selected from the held-out importance ranking for descriptive presentation. No causal threshold or breakpoint is inferred, and no trend curve is fitted. The same model, split and explanation settings as Figs 7–9 apply.'
    return captions


def main():
    p=argparse.ArgumentParser();p.add_argument('--features',type=Path);p.add_argument('--baseline',type=Path);p.add_argument('--ml-results',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();data=a.output/'source_data';data.mkdir(parents=True,exist_ok=True)
    captions={}
    if a.features:captions.update(distribution_figures(a.features,a.output,data))
    if a.baseline:captions.update(top_features(a.baseline,a.output,data));captions.update(heatmap(a.baseline,a.output,data))
    if a.ml_results:captions.update(heldout_figures(a.ml_results,a.output,data))
    existing=a.output/'captions.json'
    old=json.loads(existing.read_text(encoding='utf-8')) if existing.exists() else {}
    old.update(captions);existing.write_text(json.dumps(old,ensure_ascii=False,indent=2),encoding='utf-8')
    qa=[]
    for fn in sorted(a.output.glob('Fig *.tif')):
        with Image.open(fn) as im:qa.append({'file':fn.name,'pixels':list(im.size),'mode':im.mode,'dpi':[float(x) for x in im.info.get('dpi')],'compression':im.info.get('compression'),'width_inches':float(im.width/im.info['dpi'][0])})
    (a.output/'figure_export_qa.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
    provenance_file=a.output/'figure_input_manifest.json'
    provenance=json.loads(provenance_file.read_text(encoding='utf-8')) if provenance_file.exists() else {}
    for label,root in [('features',a.features),('baseline',a.baseline),('ml_results',a.ml_results)]:
        if root:
            provenance[label]={str(f.relative_to(root)).replace('\\','/'):hashlib.sha256(f.read_bytes()).hexdigest()
                               for f in sorted(root.rglob('*.csv'))}
    provenance['plot_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    provenance_file.write_text(json.dumps(provenance,indent=2),encoding='utf-8')


if __name__=='__main__':main()
