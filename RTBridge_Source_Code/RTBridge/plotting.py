from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from .core import DEFAULT_MODEL_NAMES, fit_predict_model

plt.rcParams.update({
    'font.family':'DejaVu Sans', 'font.sans-serif':['DejaVu Sans'], 'font.size':11,
    'axes.labelsize':12, 'axes.titlesize':13, 'xtick.labelsize':10, 'ytick.labelsize':10,
    'legend.fontsize':10, 'axes.spines.top':False, 'axes.spines.right':False,
    'pdf.fonttype':42, 'ps.fonttype':42,
})
BLUE='#2C7FB8'; ORANGE='#F28E2B'; RED='#D62728'; PURPLE='#9467BD'; GRAY='#6B7280'; EDGE='#17365D'; LIGHT_BLUE='#9ECAE1'; LIGHT_ORANGE='#FDBF6F'
PAPER_MODEL_ORDER=['Linear regression','Piecewise linear','LOESS','PCHIP','Smoothing spline/GAM','Monotone GAM']
PAPER_DATASET_ORDER=['PL_NEG','PL_POS','WB_NEG','WB_POS']

_ACTIVE_OUTPUT_FORMATS = ('png', 'tif', 'pdf')


def format_three_significant(value) -> str:
    """Display a measurement with exactly three significant figures.

    The alternate general format preserves scientifically meaningful trailing
    zeroes: 0.033 becomes 0.0330, while 0.86926 becomes 0.869.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return '' if value is None else str(value)
    if not np.isfinite(number):
        return ''
    return format(number, '#.3g')


def _is_precision_metric_column(column: str) -> bool:
    label = str(column).lower().replace('_', ' ').replace('-', ' ')
    return ('acs' in label or 'rt error' in label or 'residual' in label)


def format_publication_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Format display/export metrics without modifying calculation inputs."""
    formatted = frame.copy()
    for column in formatted.columns:
        if not _is_precision_metric_column(column):
            continue
        if pd.api.types.is_bool_dtype(formatted[column]):
            continue
        numeric = pd.to_numeric(formatted[column], errors='coerce')
        present = formatted[column].notna()
        if present.any() and numeric[present].notna().all():
            formatted[column] = numeric.map(format_three_significant)
    return formatted


def _write_publication_csv(frame: pd.DataFrame, path: Path) -> None:
    format_publication_table(frame).to_csv(path, index=False)


def _significant_tick(value, _position=None) -> str:
    return format_three_significant(value)


def _save(fig, out: Path, dpi:int=600):
    out.parent.mkdir(parents=True, exist_ok=True)
    formats = set(_ACTIVE_OUTPUT_FORMATS)
    if 'png' in formats:
        fig.savefig(out.with_suffix('.png'), dpi=dpi, bbox_inches='tight', facecolor='white')
    if 'tif' in formats or 'tiff' in formats:
        fig.savefig(out.with_suffix('.tif'), dpi=dpi, bbox_inches='tight', facecolor='white')
    if 'pdf' in formats:
        fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)

def short_model_name(m:str)->str:
    return {'Linear regression':'Linear','Piecewise linear':'Piecewise','Smoothing spline/GAM':'Spline/GAM','Monotone GAM':'Monotone GAM','Quadratic regression':'Quadratic','Cubic regression':'Cubic','Ridge cubic regression':'Ridge cubic','RANSAC linear':'RANSAC','Huber linear':'Huber','Theil-Sen linear':'Theil-Sen','Cubic spline':'Cubic spline','Akima spline':'Akima','Isotonic regression':'Isotonic','SVR-RBF':'SVR','Gaussian process':'GPR','Random forest':'RF','Extra trees':'ExtraTrees','Gradient boosting':'Boosting','KNN regression':'KNN'}.get(m,m)

def _order_datasets(datasets):
    ds=list(dict.fromkeys(map(str,datasets)))
    out=[d for d in PAPER_DATASET_ORDER if d in ds]
    out += [d for d in ds if d not in out]
    return out

def _mode_from_dataset(ds):
    s=str(ds).upper()
    if s.endswith('NEG'): return 'NEG'
    if s.endswith('POS'): return 'POS'
    return 'MODE-SPECIFIC'

def _mode_specific_config(results, ds):
    configs=set(results.loc[results['Dataset']==ds,'Anchor_configuration'].astype(str)) if not results.empty else set()
    mode=_mode_from_dataset(ds)
    if mode in configs: return mode
    for c in configs:
        if c!='COMBINED': return c
    return None

def select_primary_configs(results:pd.DataFrame, model:str='PCHIP')->dict:
    fixed={'PL_NEG':'NEG','PL_POS':'POS','WB_NEG':'NEG','WB_POS':'POS'}
    out={}
    if results.empty: return out
    for ds in _order_datasets(results['Dataset'].unique()):
        configs=set(results.loc[results['Dataset']==ds,'Anchor_configuration'].astype(str))
        if ds in fixed and fixed[ds] in configs:
            out[ds]=fixed[ds]; continue
        sub=results[(results['Dataset']==ds)&(results['Workflow']=='All selected anchors')&(results['Model']==model)].copy()
        if sub.empty: sub=results[(results['Dataset']==ds)&(results['Workflow']=='All selected anchors')].copy()
        if sub.empty: continue
        sub=sub.sort_values(['Total_verification','Primary_pairs','Median_RT_error_min','Median_ACS'], ascending=[False,False,True,False])
        out[ds]=str(sub.iloc[0]['Anchor_configuration'])
    return out

# ---------------- main figures ----------------
def _paper_heatmap_data(results, workflow, models=None):
    df=results[results['Workflow']==workflow].copy()
    models=models or [m for m in PAPER_MODEL_ORDER if m in set(df['Model'])]
    if not models: models=list(dict.fromkeys(df['Model'].tolist()))[:6]
    datasets=_order_datasets(df['Dataset'].unique())
    col_keys=[(cfg,m) for cfg in ['MODE-SPECIFIC','COMBINED'] for m in models]
    matrices={}
    for metric in ['Primary_pairs','Median_RT_error_min','Median_ACS']:
        arr=np.full((len(datasets),len(col_keys)),np.nan)
        for i,ds in enumerate(datasets):
            mode_specific=_mode_specific_config(df,ds)
            for j,(label,m) in enumerate(col_keys):
                cfg=mode_specific if label=='MODE-SPECIFIC' else 'COMBINED'
                sub=df[(df['Dataset']==ds)&(df['Anchor_configuration']==cfg)&(df['Model']==m)]
                if len(sub): arr[i,j]=sub.iloc[0][metric]
        matrices[metric]=arr
    xt=[short_model_name(m) for _,m in col_keys]
    return datasets,xt,matrices

def _heatmap_scales(results):
    scales={}
    for metric in ['Primary_pairs','Median_RT_error_min','Median_ACS']:
        vals=[]
        for wf in ['All selected anchors','GAM-screened anchors']:
            _,_,mats=_paper_heatmap_data(results,wf); a=mats.get(metric)
            if a is not None: vals += a[np.isfinite(a)].tolist()
        if vals: scales[metric]=(float(np.nanmin(vals)),float(np.nanmax(vals)))
    return scales

def make_paper_heatmap(results, workflow, out:Path, dpi:int=600, scales=None):
    datasets,xt,mats=_paper_heatmap_data(results,workflow)
    if len(datasets)==0 or len(xt)==0: return
    fig,axes=plt.subplots(3,1,figsize=(17.2,10.8),constrained_layout=True)
    panels=[('Primary_pairs','A. Primary-pair recovery'),('Median_RT_error_min','B. Median RT error (min)'),('Median_ACS','C. Median ACS')]
    for ax,(metric,title) in zip(axes,panels):
        arr=mats[metric]; finite=arr[np.isfinite(arr)]
        if scales and metric in scales: vmin,vmax=scales[metric]
        elif finite.size: vmin,vmax=float(np.nanmin(finite)),float(np.nanmax(finite))
        else: vmin,vmax=0,1
        if vmax<=vmin: vmax=vmin+1
        im=ax.imshow(arr,aspect='auto',cmap='viridis',vmin=vmin,vmax=vmax)
        ax.set_title(title,fontweight='bold',fontsize=13,pad=31)
        ax.set_yticks(np.arange(len(datasets)),[d.replace('_',' ') for d in datasets],fontsize=11)
        ax.set_xticks(np.arange(len(xt)),xt,rotation=30,ha='right',rotation_mode='anchor',fontsize=10)
        model_count=len(xt)//2
        group_axis=ax.secondary_xaxis('top')
        group_axis.set_xticks(
            [(model_count-1)/2,model_count+(model_count-1)/2],
            ['Mode-specific anchors','COMBINED anchors'],
        )
        group_axis.tick_params(axis='x',length=0,pad=5,labelsize=10.5)
        for label in group_axis.get_xticklabels():
            label.set_fontweight('bold')
        group_axis.spines['top'].set_visible(False)
        ax.axvline(len(xt)/2-0.5,color='white',linewidth=2.5,alpha=0.98)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if np.isfinite(arr[i,j]):
                    label=(str(int(round(arr[i,j]))) if metric=='Primary_pairs'
                           else format_three_significant(arr[i,j]))
                    ax.text(j,i,label,ha='center',va='center',fontsize=10.2,color='#111111',fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.20',facecolor='white',edgecolor='#303030',linewidth=0.55,alpha=0.96))
        colorbar=fig.colorbar(im,ax=ax,fraction=.018,pad=.012)
        colorbar.ax.tick_params(labelsize=10)
        if metric != 'Primary_pairs':
            colorbar.ax.yaxis.set_major_formatter(FuncFormatter(_significant_tick))
    fig.suptitle(
        'All selected anchors: model and anchor-configuration outputs'
        if workflow=='All selected anchors'
        else 'GAM-screened anchors: model and anchor-configuration outputs',
        fontweight='bold',fontsize=15,
    )
    _save(fig,out,dpi)

def make_figure3_anchor_counts(anchors, out:Path, dpi:int=600):
    if anchors.empty: return
    sub=anchors[anchors['Anchor_configuration']!='COMBINED'].copy(); datasets=_order_datasets(sub['Dataset'].unique())
    selected=[]; screened=[]
    for ds in datasets:
        mode=_mode_specific_config(sub,ds) or _mode_from_dataset(ds)
        selected.append(len(sub[(sub['Dataset']==ds)&(sub['Workflow']=='All selected anchors')&(sub['Anchor_configuration']==mode)]))
        screened.append(len(sub[(sub['Dataset']==ds)&(sub['Workflow']=='GAM-screened anchors')&(sub['Anchor_configuration']==mode)]))
    x=np.arange(len(datasets)); w=.34; fig,ax=plt.subplots(figsize=(8.8,5.5),constrained_layout=True)
    b1=ax.bar(x-w/2,selected,w,color=BLUE,label='All selected anchors'); b2=ax.bar(x+w/2,screened,w,color=ORANGE,label='GAM-screened anchors')
    ax.bar_label(b1,padding=3,fontsize=11,fontweight='bold'); ax.bar_label(b2,padding=3,fontsize=11,fontweight='bold')
    ax.set_xticks(x,[d.replace('_',' ') for d in datasets],fontsize=11)
    ax.set_ylabel('Training anchors',fontsize=12)
    ax.set_title('Mode-specific anchors before and after GAM-based screening',fontweight='bold',fontsize=13,pad=10)
    ax.legend(frameon=False,fontsize=11); ax.grid(axis='y',alpha=.22)
    ax.margins(y=.12)
    _save(fig,out,dpi)

def make_figure5_pchip_verification(results, out:Path, dpi:int=600):
    """Figure 5: split authentic-standard and isotope-labeled verification counts.

    Counts are based only on the selected primary PCHIP configurations used in
    the manuscript-level comparison, not on pooled-polarity configurations.
    Each dataset has two stacked bars: all selected and GAM-screened anchors.
    """
    if results.empty or 'PCHIP' not in set(results['Model']):
        return
    cfg_map=select_primary_configs(results,'PCHIP')
    datasets=_order_datasets(cfg_map.keys())
    workflows=['All selected anchors','GAM-screened anchors']
    # collect authentic/isotope counts from model_metrics rows
    auth={wf:[] for wf in workflows}; iso={wf:[] for wf in workflows}; total={wf:[] for wf in workflows}
    for ds in datasets:
        cfg=cfg_map[ds]
        for wf in workflows:
            sub=results[(results['Dataset']==ds)&(results['Anchor_configuration']==cfg)&(results['Workflow']==wf)&(results['Model']=='PCHIP')]
            if len(sub):
                r=sub.iloc[0]
                a=int(r.get('Authentic_hits',0)); b=int(r.get('Isotope_hits',0))
            else:
                a=b=0
            auth[wf].append(a); iso[wf].append(b); total[wf].append(a+b)
    x=np.arange(len(datasets)); w=.34
    fig,ax=plt.subplots(figsize=(9.8,6.2),constrained_layout=True)
    # stacked bars: darker = authentic, lighter = isotope
    b1=ax.bar(x-w/2,auth['All selected anchors'],w,color=BLUE,label='All selected anchors: authentic')
    b2=ax.bar(x-w/2,iso['All selected anchors'],w,bottom=auth['All selected anchors'],color=LIGHT_BLUE,label='All selected anchors: isotope-labeled')
    b3=ax.bar(x+w/2,auth['GAM-screened anchors'],w,color=ORANGE,label='GAM-screened anchors: authentic')
    b4=ax.bar(x+w/2,iso['GAM-screened anchors'],w,bottom=auth['GAM-screened anchors'],color=LIGHT_ORANGE,label='GAM-screened anchors: isotope-labeled')
    # annotate components and totals
    for bars,vals in [(b1,auth['All selected anchors']),(b2,iso['All selected anchors']),(b3,auth['GAM-screened anchors']),(b4,iso['GAM-screened anchors'])]:
        for bar,val in zip(bars,vals):
            if val>0 and bar.get_height()>=3:
                ax.text(bar.get_x()+bar.get_width()/2,bar.get_y()+bar.get_height()/2,str(val),ha='center',va='center',fontsize=10,color='white' if bar.get_height()>8 else 'black',fontweight='bold')
    for xi,tc,ts in zip(x,total['All selected anchors'],total['GAM-screened anchors']):
        ax.text(xi-w/2,tc+1,str(tc),ha='center',va='bottom',fontsize=11,fontweight='bold')
        ax.text(xi+w/2,ts+1,str(ts),ha='center',va='bottom',fontsize=11,fontweight='bold')
    ax.set_xticks(x,[d.replace('_',' ') for d in datasets],fontsize=11)
    ax.set_ylabel('Verified primary pairs',fontsize=12)
    ax.set_title('Authentic-standard and isotope-labeled verification',fontweight='bold',fontsize=14,pad=10)
    ax.legend(frameon=False,ncol=2,fontsize=10,loc='upper center',bbox_to_anchor=(0.5,1.20))
    ax.grid(axis='y',alpha=.22)
    ax.margins(y=.16)
    _save(fig,out,dpi)

# ---------------- SI figures ----------------
def make_si_anchor_distribution_hist(anchors, out:Path, dpi:int=600):
    """Paper-style anchor RT distribution figure.

    This keeps the histogram layout used in the SI, but uses side-by-side bars
    within each RT bin so the all-selected and GAM-screened distributions
    can be visually separated by color. Bar heights are raw anchor counts.
    """
    if anchors.empty:
        return
    sub=anchors[anchors['Anchor_configuration']!='COMBINED'].copy()
    datasets=_order_datasets(sub['Dataset'].unique())
    if not datasets:
        return
    fig,axes=plt.subplots(2,len(datasets),figsize=(max(8.0,2.55*len(datasets)),5.15),constrained_layout=True)
    # Figure-level legends are not reserved automatically by constrained layout;
    # leave a dedicated top band so longer workflow labels cannot overlap titles.
    fig.get_layout_engine().set(rect=(0,0,1,0.90))
    if len(datasets)==1:
        axes=np.array([[axes[0]],[axes[1]]])
    for ci,ds in enumerate(datasets):
        mode=_mode_specific_config(sub,ds) or _mode_from_dataset(ds)
        for ri,(col,label) in enumerate([('Source_RT','RT5'),('Target_RT','RT25')]):
            ax=axes[ri,ci]
            a=sub[(sub['Dataset']==ds)&(sub['Anchor_configuration']==mode)&(sub['Workflow']=='All selected anchors')][col].dropna().to_numpy(float)
            b=sub[(sub['Dataset']==ds)&(sub['Anchor_configuration']==mode)&(sub['Workflow']=='GAM-screened anchors')][col].dropna().to_numpy(float)
            allv=np.concatenate([a,b]) if len(a)+len(b) else np.array([])
            if len(allv)==0:
                ax.set_axis_off(); continue
            lo=max(0,float(np.nanmin(allv))-0.05); hi=float(np.nanmax(allv))+0.05
            bins=np.linspace(lo,hi,16)
            centers=(bins[:-1]+bins[1:])/2
            width=(bins[1]-bins[0])*0.40
            ca,_=np.histogram(a,bins=bins)
            cb,_=np.histogram(b,bins=bins)
            ax.bar(centers-width/2,ca,width=width,color=BLUE,alpha=0.92,label='All selected anchors' if (ri==0 and ci==0) else None)
            ax.bar(centers+width/2,cb,width=width,color=ORANGE,alpha=0.92,label='GAM-screened anchors' if (ri==0 and ci==0) else None)
            ax.text(0.05,0.90,label,transform=ax.transAxes,fontsize=8,fontweight='bold')
            if ri==0:
                ax.set_title(ds.replace('_',' '),fontweight='bold')
            if ri==1:
                ax.set_xlabel('Retention time (min)')
            if ci==0:
                ax.set_ylabel('Anchors')
            ax.grid(axis='y',alpha=.22,linestyle=':')
            ax.margins(x=0.02)
    handles=[Line2D([0],[0],color=BLUE,lw=8),Line2D([0],[0],color=ORANGE,lw=8)]
    fig.legend(handles,['All selected anchors','GAM-screened anchors'],loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(0.5,0.995))
    _save(fig,out,dpi)

def make_si_pchip_distributions(pairs, results, out:Path, dpi:int=600):
    if pairs.empty or results.empty: return
    cfg_map=select_primary_configs(results,'PCHIP'); datasets=_order_datasets(cfg_map.keys())
    p=pairs[pairs['Model']=='PCHIP'].copy(); masks=[]
    for ds,cfg in cfg_map.items(): masks.append((p['Dataset']==ds)&(p['Anchor_configuration']==cfg))
    if not masks: return
    p=p[np.logical_or.reduce(masks)]
    if p.empty: return
    fig,axes=plt.subplots(1,2,figsize=(9,4.2),constrained_layout=True)
    for ax,metric,title,ylabel in [(axes[0],'RT_error_min','A. RT-error distributions','RT error (min)'),(axes[1],'ACS','B. ACS distributions','ACS')]:
        data=[]; positions=[]; colors=[]
        for i,ds in enumerate(datasets):
            for j,wf in enumerate(['All selected anchors','GAM-screened anchors']):
                data.append(p[(p['Dataset']==ds)&(p['Workflow']==wf)][metric].dropna().to_numpy(float)); positions.append(i*3+j); colors.append(BLUE if wf=='All selected anchors' else ORANGE)
        bp=ax.boxplot(data,positions=positions,widths=.75,showfliers=False,patch_artist=True)
        for box,c in zip(bp['boxes'],colors): box.set_facecolor(c); box.set_alpha(.5); box.set_edgecolor(c)
        for key in ['medians','whiskers','caps']:
            for obj in bp[key]: obj.set_color('#333333')
        ax.set_xticks([i*3+.5 for i in range(len(datasets))],[d.replace('_',' ') for d in datasets]); ax.set_title(title,fontweight='bold'); ax.set_ylabel(ylabel); ax.grid(axis='y',alpha=.2)
        ax.yaxis.set_major_formatter(FuncFormatter(_significant_tick))
    axes[0].legend([Line2D([0],[0],color=BLUE,lw=8,alpha=.55),Line2D([0],[0],color=ORANGE,lw=8,alpha=.55)],['All selected anchors','GAM-screened anchors'],frameon=False,loc='upper right')
    _save(fig,out,dpi)

def make_si_pchip_fit_diagnostics(diagnostics, results, out:Path, dpi:int=600):
    if diagnostics.empty or results.empty: return
    cfg_map=select_primary_configs(results,'PCHIP'); datasets=_order_datasets(cfg_map.keys())
    d=diagnostics[(diagnostics['Model']=='PCHIP')&(diagnostics['Workflow']=='All selected anchors')].copy()
    if d.empty: return
    fig,axes=plt.subplots(1,len(datasets),figsize=(max(8,2.3*len(datasets)),3.4),constrained_layout=True)
    if len(datasets)==1: axes=[axes]
    sc=None
    for ax,ds in zip(axes,datasets):
        cfg=cfg_map[ds]; sub=d[(d['Dataset']==ds)&(d['Anchor_configuration']==cfg)].copy()
        if sub.empty: ax.set_axis_off(); continue
        sc=ax.scatter(sub['Source_RT'],sub['Target_RT'],c=sub['ACS'],s=10,cmap='viridis',vmin=.5,vmax=1.0,alpha=.85,edgecolors='none')
        xs=np.linspace(sub['Source_RT'].min(),sub['Source_RT'].max(),400)
        try: yline,_=fit_predict_model('PCHIP',sub['Source_RT'].values,sub['Target_RT'].values,xs,1)
        except Exception: yline=np.interp(xs,np.sort(sub['Source_RT'].values),sub.sort_values('Source_RT')['Target_RT'].values)
        ax.plot(xs,yline,color=RED,lw=1.5); ax.set_title(f"{ds.replace('_',' ')} ({cfg})",fontweight='bold',fontsize=9); ax.set_xlabel('RT5 (min)'); ax.grid(alpha=.2)
        if ax is axes[0]: ax.set_ylabel('RT25 (min)')
    if sc is not None:
        cbar=fig.colorbar(sc,ax=axes,fraction=.025,pad=.01); cbar.set_label('ACS')
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(_significant_tick))
    _save(fig,out,dpi)


def make_si_mass_tolerance_sensitivity(summary: pd.DataFrame, out: Path, dpi: int = 600):
    """Generate Figure S4 with an explicit value above every sensitivity bar."""
    if summary.empty:
        return

    aliases = {
        'Mode_specific_anchors': [
            'Mode_specific_anchors', 'Mode-specific anchors',
            'Ionization_mode_specific_anchors', 'Ionization-mode-specific anchors',
        ],
        'Primary_pairs': ['Primary_pairs', 'Primary pairs'],
        'Mean_median_RT_error_min': [
            'Mean_median_RT_error_min', 'Mean median RT error (min)',
        ],
        'Mean_median_ACS': ['Mean_median_ACS', 'Mean median ACS'],
        'Selected_for_publication': [
            'Selected_for_publication', 'Selected for publication',
        ],
    }
    plot_data = summary.copy()
    for canonical, choices in aliases.items():
        source = next((column for column in choices if column in plot_data), None)
        if source is not None and source != canonical:
            plot_data[canonical] = plot_data[source]

    required = ['Setting', 'Mode_specific_anchors', 'Primary_pairs',
                'Mean_median_RT_error_min', 'Mean_median_ACS']
    if any(column not in plot_data for column in required):
        return
    if 'Selected_for_publication' not in plot_data:
        plot_data['Selected_for_publication'] = plot_data['Setting'].astype(str).eq('20/10/10')

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.5), constrained_layout=True)
    metrics = [
        ('Mode_specific_anchors', 'Mode-specific anchors', False),
        ('Primary_pairs', 'Primary pairs', False),
        ('Mean_median_RT_error_min', 'Mean median RT error (min)', True),
        ('Mean_median_ACS', 'Mean median ACS', True),
    ]
    x = np.arange(len(plot_data))
    for ax, (column, title, precise) in zip(axes.flat, metrics):
        values = pd.to_numeric(plot_data[column], errors='coerce').to_numpy(float)
        bars = ax.bar(x, values, color=BLUE, width=0.80)
        labels = [format_three_significant(value) if precise else str(int(round(value)))
                  if np.isfinite(value) else '' for value in values]
        ax.bar_label(bars, labels=labels, padding=4, fontsize=10.5,
                     fontweight='bold', color='#17202A')
        for index, selected in enumerate(plot_data['Selected_for_publication']):
            if str(selected).strip().lower() in {'true', '1', 'yes'}:
                bars[index].set_hatch('///')
                bars[index].set_edgecolor('#17365D')
                bars[index].set_linewidth(1.5)
        ax.set_xticks(x, plot_data['Setting'].astype(str).values,
                      rotation=30, ha='right', fontsize=9.5)
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel('U/A/C (ppm)', fontsize=11)
        ax.grid(axis='y', alpha=.20)
        ax.set_axisbelow(True)
        ax.margins(y=.18)
        if precise:
            ax.yaxis.set_major_formatter(FuncFormatter(_significant_tick))
    fig.suptitle(
        'Mass-tolerance sensitivity; hatched bar = selected 20/10/10 setting',
        fontweight='bold', fontsize=15,
    )
    _save(fig, out, dpi)

def make_model_summary(results,out:Path,dpi:int=600):
    """Aggregate model comparison using mode-specific configurations.

    This avoids double-counting separate and COMBINED configurations.
    each model is summarized across the same fixed primary configurations used for
    the paper-level comparison.
    """
    if results.empty:
        return
    cfg_map = select_primary_configs(results, 'PCHIP')
    rows=[]
    model_order=[m for m in PAPER_MODEL_ORDER if m in set(results['Model'])]
    model_order += [m for m in dict.fromkeys(results['Model'].tolist()) if m not in model_order]
    for wf in ['All selected anchors','GAM-screened anchors']:
        for model in model_order:
            total_pairs=0; total_ver=0; rts=[]
            for ds,cfg in cfg_map.items():
                sub=results[(results['Dataset']==ds)&(results['Anchor_configuration']==cfg)&(results['Workflow']==wf)&(results['Model']==model)]
                if len(sub):
                    rr=sub.iloc[0]
                    total_pairs += int(rr.get('Primary_pairs',0))
                    total_ver += int(rr.get('Total_verification',0))
                    if pd.notna(rr.get('Median_RT_error_min',np.nan)):
                        rts.append(float(rr['Median_RT_error_min']))
            rows.append({'Workflow':wf,'Model':model,'Total_primary_pairs':total_pairs,
                         'Mean_median_RT_error_min':float(np.nanmean(rts)) if rts else np.nan,
                         'Total_verification':total_ver})
    agg=pd.DataFrame(rows)
    order=agg[agg['Workflow']=='All selected anchors'].sort_values('Total_primary_pairs',ascending=False)['Model'].head(12).tolist()
    if not order:
        return
    x=np.arange(len(order)); w=.36
    fig,axes=plt.subplots(3,1,figsize=(11.2,8.4),constrained_layout=True)
    panels=[('Total_primary_pairs','Total primary pairs','{:.0f}'),('Mean_median_RT_error_min','Mean median RT error (min)','significant'),('Total_verification','Total verification hits','{:.0f}')]
    for ax,col,title,fmt in zip(axes,[p[0] for p in panels],[p[1] for p in panels],[p[2] for p in panels]):
        nr=agg[agg['Workflow']=='All selected anchors'].set_index('Model').reindex(order)[col].fillna(0).to_numpy(float)
        cg=agg[agg['Workflow']=='GAM-screened anchors'].set_index('Model').reindex(order)[col].fillna(0).to_numpy(float)
        b1=ax.bar(x-w/2,nr,w,color=BLUE,label='All selected anchors')
        b2=ax.bar(x+w/2,cg,w,color=ORANGE,label='GAM-screened anchors')
        label_value=lambda value: (format_three_significant(value) if fmt=='significant' else fmt.format(value)) if np.isfinite(value) and value else ''
        ax.bar_label(b1,labels=[label_value(v) for v in nr],fontsize=7,padding=2)
        ax.bar_label(b2,labels=[label_value(v) for v in cg],fontsize=7,padding=2)
        ax.set_title(title,fontweight='bold')
        ax.set_xticks(x,[short_model_name(m) for m in order],rotation=35,ha='right')
        ax.grid(axis='y',alpha=.20)
        ax.margins(y=0.12)
        if fmt=='significant':
            ax.yaxis.set_major_formatter(FuncFormatter(_significant_tick))
    axes[0].legend(frameon=False,ncol=2,loc='upper right')
    fig.suptitle('Aggregate model comparison using mode-specific anchors',fontweight='bold')
    _save(fig,out,dpi)


def make_model_summary_all_configs(results,out:Path,dpi:int=600):
    """Exploratory aggregate over all dataset/configuration rows."""
    if results.empty: return
    agg=results.groupby(['Workflow','Model'],as_index=False).agg(Total_primary_pairs=('Primary_pairs','sum'),Mean_median_RT_error_min=('Median_RT_error_min','mean'),Total_verification=('Total_verification','sum'))
    order=agg[agg['Workflow']=='All selected anchors'].sort_values('Total_primary_pairs',ascending=False)['Model'].head(15).tolist() or agg['Model'].unique().tolist()[:15]
    x=np.arange(len(order)); w=.36; fig,axes=plt.subplots(3,1,figsize=(11.2,8.4),constrained_layout=True)
    for ax,col,title,fmt in [(axes[0],'Total_primary_pairs','Total primary pairs - all configurations','{:.0f}'),(axes[1],'Mean_median_RT_error_min','Mean median RT error (min) - all configurations','significant'),(axes[2],'Total_verification','Total verification hits - all configurations','{:.0f}')]:
        nr=agg[agg['Workflow']=='All selected anchors'].set_index('Model').reindex(order)[col].fillna(0).to_numpy(float); cg=agg[agg['Workflow']=='GAM-screened anchors'].set_index('Model').reindex(order)[col].fillna(0).to_numpy(float)
        b1=ax.bar(x-w/2,nr,w,color=BLUE,label='All selected anchors'); b2=ax.bar(x+w/2,cg,w,color=ORANGE,label='GAM-screened anchors')
        label_value=lambda value: (format_three_significant(value) if fmt=='significant' else fmt.format(value)) if value else ''
        ax.bar_label(b1,labels=[label_value(v) for v in nr],fontsize=7,padding=2); ax.bar_label(b2,labels=[label_value(v) for v in cg],fontsize=7,padding=2)
        ax.set_title(title,fontweight='bold'); ax.set_xticks(x,[short_model_name(m) for m in order],rotation=35,ha='right'); ax.grid(axis='y',alpha=.2); ax.margins(y=.12)
        if fmt=='significant':
            ax.yaxis.set_major_formatter(FuncFormatter(_significant_tick))
    axes[0].legend(frameon=False,ncol=2); fig.suptitle('Exploratory aggregate over all anchor configurations',fontweight='bold'); _save(fig,out,dpi)


def _safe_filename(text: str) -> str:
    """Return a filesystem-safe short name for model/workflow labels."""
    keep = []
    for ch in str(text):
        if ch.isalnum():
            keep.append(ch)
        elif ch in [' ', '-', '_', '/', '+']:
            keep.append('_')
    out = ''.join(keep).strip('_')
    while '__' in out:
        out = out.replace('__','_')
    return out or 'item'


def make_model_fit_diagnostics_all(diagnostics: pd.DataFrame, results: pd.DataFrame, out_dir: Path, dpi: int = 600):
    """Export model-fit diagnostic plots for every model and workflow.

    Each figure follows the Figure S3 style: one panel per dataset, anchor
    points colored by ACS, and the fitted RT5-to-RT25 transfer curve. The
    selected primary anchor configuration is used for each dataset so that
    diagnostics are directly comparable with the manuscript-level summaries.
    """
    if diagnostics.empty or results.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_map = select_primary_configs(results, 'PCHIP')
    datasets = _order_datasets(cfg_map.keys())
    if not datasets:
        return
    workflows = [w for w in ['All selected anchors', 'GAM-screened anchors'] if w in set(diagnostics['Workflow'].astype(str))]
    other_workflows = [w for w in dict.fromkeys(diagnostics['Workflow'].astype(str).tolist()) if w not in workflows]
    workflows += other_workflows
    model_order = [m for m in PAPER_MODEL_ORDER if m in set(diagnostics['Model'].astype(str))]
    model_order += [m for m in dict.fromkeys(diagnostics['Model'].astype(str).tolist()) if m not in model_order]
    summary_rows = []
    for workflow in workflows:
        workflow_dir = out_dir / _safe_filename(workflow)
        workflow_dir.mkdir(parents=True, exist_ok=True)
        for model in model_order:
            d = diagnostics[(diagnostics['Model'] == model) & (diagnostics['Workflow'] == workflow)].copy()
            if d.empty:
                continue
            fig, axes = plt.subplots(1, len(datasets), figsize=(max(8, 2.35 * len(datasets)), 3.45), constrained_layout=True)
            if len(datasets) == 1:
                axes = [axes]
            sc = None
            has_data = False
            for ax, ds in zip(axes, datasets):
                cfg = cfg_map.get(ds)
                sub = d[(d['Dataset'] == ds) & (d['Anchor_configuration'] == cfg)].copy()
                if sub.empty:
                    ax.set_axis_off()
                    ax.set_title(ds.replace('_',' '), fontweight='bold', fontsize=9)
                    continue
                has_data = True
                sc = ax.scatter(sub['Source_RT'], sub['Target_RT'], c=sub['ACS'], s=12, cmap='viridis', vmin=0.5, vmax=1.0, alpha=0.88, edgecolors='none')
                x = sub['Source_RT'].to_numpy(float)
                y = sub['Target_RT'].to_numpy(float)
                if np.isfinite(x).sum() >= 3:
                    x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
                    xs = np.linspace(x_min, x_max, 400)
                    try:
                        yline, params = fit_predict_model(model, x, y, xs, 1)
                    except Exception as exc:
                        yline = np.full_like(xs, np.nan, dtype=float)
                        params = {'error': str(exc)}
                    if np.isfinite(yline).any():
                        order = np.argsort(xs)
                        ax.plot(xs[order], np.asarray(yline)[order], color=RED, lw=1.6, zorder=5)
                    else:
                        params = {'error': 'prediction failed'}
                else:
                    params = {'error': 'too few anchors'}
                med_res = float(np.nanmedian(sub['Abs_residual_min'])) if 'Abs_residual_min' in sub.columns else np.nan
                p90_res = float(np.nanquantile(sub['Abs_residual_min'], 0.90)) if 'Abs_residual_min' in sub.columns and len(sub) else np.nan
                ax.set_title(f"{ds.replace('_',' ')} ({cfg})", fontweight='bold', fontsize=9)
                ax.set_xlabel('RT5 (min)')
                ax.grid(alpha=0.18)
                if ax is axes[0]:
                    ax.set_ylabel('RT25 (min)')
                ax.text(0.03, 0.97, f"n={len(sub)}\nmed |res|={format_three_significant(med_res)}", transform=ax.transAxes, va='top', ha='left', fontsize=6.5,
                        bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='none', alpha=0.75))
                summary_rows.append({
                    'Workflow': workflow,
                    'Model': model,
                    'Dataset': ds,
                    'Anchor_configuration': cfg,
                    'Anchors': int(len(sub)),
                    'Median_abs_residual_min': med_res,
                    'P90_abs_residual_min': p90_res,
                    'Figure_file_base': f"{_safe_filename(workflow)}__{_safe_filename(model)}"
                })
            if not has_data:
                plt.close(fig)
                continue
            if sc is not None:
                cbar = fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.012)
                cbar.set_label('ACS')
                cbar.ax.yaxis.set_major_formatter(FuncFormatter(_significant_tick))
            fig.suptitle(f"{model}: {workflow} model-fit diagnostics", fontweight='bold', y=1.03)
            _save(fig, workflow_dir / f"{_safe_filename(workflow)}__{_safe_filename(model)}", dpi)
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(out_dir / 'model_fit_diagnostics_figure_index.csv', index=False)


def make_model_residual_figure(diag,out:Path,dpi:int=600):
    if diag.empty: return
    models=diag['Model'].value_counts().head(12).index.tolist(); d=diag[diag['Model'].isin(models)]
    fig,ax=plt.subplots(figsize=(max(9,.75*len(models)),4.8)); data=[d[(d['Model']==m)&(d['Workflow']=='All selected anchors')]['Abs_residual_min'].dropna().values for m in models]
    model_labels=[short_model_name(m) for m in models]
    # Matplotlib renamed the Axes.boxplot keyword from ``labels`` to
    # ``tick_labels`` and newer releases removed ``labels`` entirely.
    # Inspect the installed API directly. If
    # neither keyword is exposed, draw the boxplot first and set tick labels
    # explicitly. This changes presentation compatibility only; the data and
    # all RTBridge numerical results are unchanged.
    import inspect
    boxplot_parameters=inspect.signature(ax.boxplot).parameters
    boxplot_kwargs=dict(showfliers=False,patch_artist=True)
    if 'tick_labels' in boxplot_parameters:
        boxplot_kwargs['tick_labels']=model_labels
        bp=ax.boxplot(data,**boxplot_kwargs)
    elif 'labels' in boxplot_parameters:
        boxplot_kwargs['labels']=model_labels
        bp=ax.boxplot(data,**boxplot_kwargs)
    else:
        bp=ax.boxplot(data,**boxplot_kwargs)
        ax.set_xticks(range(1,len(model_labels)+1))
        ax.set_xticklabels(model_labels)
    for p in bp['boxes']: p.set_facecolor(BLUE); p.set_alpha(.55)
    ax.set_ylabel('Anchor absolute residual (min)'); ax.set_title('Residual distributions using all selected anchors',fontweight='bold'); ax.tick_params(axis='x',rotation=40); ax.grid(axis='y',alpha=.2); _save(fig,out,dpi)

# ---------------- tables ----------------
def make_tables(results, anchors, verification, audit, feature_summary, diagnostics, pairs, result_dir:Path, output_dir:Path, candidate_pairs:pd.DataFrame=None):
    """Create manuscript and Supporting Information tables.

    The publication-level primary configuration is mode-specific
    for all four datasets. COMBINED summaries are retained separately as
    a sensitivity analysis.
    """
    td=output_dir/'tables'; td.mkdir(parents=True,exist_ok=True)

    # Main Table 1: feature and mode-specific anchor summary.
    colmap={'Five_min_features':'5 min features','Twentyfive_min_features':'25 min features',
            'Unique_5min':'Unique 5 min','Unique_25min':'Unique 25 min',
            'Unique_anchor_pairs':'Unique anchor pairs',
            'High_RT5_anchors_GE_2_5':'RT5 >= 2.5 min',
            'High_RT25_anchors_GE_7':'RT25 >= 7 min',
            'Median_anchor_ACS':'Median ACS'}
    t1=feature_summary.rename(columns=colmap).copy()
    if 'Anchor_configuration' in t1.columns:
        mode_specific_mask=[]
        for _,r in t1.iterrows():
            mode_specific_mask.append(str(r.get('Anchor_configuration','')).upper()==_mode_from_dataset(r.get('Dataset','')))
        t1=t1.loc[mode_specific_mask].copy()
    _write_publication_csv(t1,td/'Table_1_unique_anchor_counts_high_RT_coverage.csv')

    model_order=[m for m in PAPER_MODEL_ORDER if m in set(results['Model'])]
    model_order += [m for m in dict.fromkeys(results['Model'].tolist()) if m not in model_order]
    datasets=_order_datasets(results['Dataset'].unique())

    def _aggregate_for_config(config_kind:str) -> pd.DataFrame:
        rows=[]
        for model in model_order:
            rec={'Model':model}
            for wf,qualifier in [('All selected anchors','all selected anchors'),('GAM-screened anchors','GAM-screened anchors')]:
                pair_total=0; rt_values=[]; acs_values=[]
                for ds in datasets:
                    cfg=_mode_from_dataset(ds) if config_kind=='mode-specific' else 'COMBINED'
                    sub=results[(results['Dataset']==ds)&
                                (results['Anchor_configuration']==cfg)&
                                (results['Workflow']==wf)&
                                (results['Model']==model)]
                    if len(sub):
                        rr=sub.iloc[0]
                        pair_total += int(rr.get('Primary_pairs',0))
                        if pd.notna(rr.get('Median_RT_error_min',np.nan)):
                            rt_values.append(float(rr['Median_RT_error_min']))
                        if pd.notna(rr.get('Median_ACS',np.nan)):
                            acs_values.append(float(rr['Median_ACS']))
                rec[f'Primary pairs ({qualifier})']=pair_total
                rec[f'Mean median RT error (min; {qualifier})']=float(np.nanmean(rt_values)) if rt_values else np.nan
                rec[f'Mean median ACS ({qualifier})']=float(np.nanmean(acs_values)) if acs_values else np.nan
            rows.append(rec)
        return pd.DataFrame(rows)

    # Main Table 2: aggregate mode-specific comparison.
    t2=_aggregate_for_config('mode-specific')
    _write_publication_csv(t2,td/'Table_2_aggregate_mode_specific_all_selected_vs_GAM_screened.csv')

    # Table S1: isotope-labeled standards.
    iso_path=result_dir/'isotope_standards_parsed_all.csv'
    if iso_path.exists():
        iso_df=pd.read_csv(iso_path)
        if not iso_df.empty:
            s1=pd.DataFrame()
            s1['Metabolite']=iso_df['Standard_name'] if 'Standard_name' in iso_df.columns else [f'isotope_{i+1}' for i in range(len(iso_df))]
            mz_col=next((c for c in ['Reference_mz','Target_ref_mz','Source_ref_mz'] if c in iso_df.columns),None)
            s1['MW']=pd.to_numeric(iso_df[mz_col],errors='coerce') if mz_col else np.nan
            s1['RT_5min']=pd.to_numeric(iso_df['RT5_ref'],errors='coerce') if 'RT5_ref' in iso_df.columns else np.nan
            s1['RT_25min']=pd.to_numeric(iso_df['RT25_ref'],errors='coerce') if 'RT25_ref' in iso_df.columns else np.nan
            s1=s1.drop_duplicates().sort_values('Metabolite').reset_index(drop=True)
            s1.to_csv(td/'Table_S1_stable_isotope_labeled_standards.csv',index=False)

    # Table S2: GAM-based residual-screen audit for mode-specific configurations.
    if not audit.empty:
        s2=audit[audit['Anchor_configuration']!='COMBINED'].copy()
        s2=s2[[c for c in ['Dataset','Anchor_configuration','Iteration','Anchors_before','Anchors_removed','Anchors_after','Residual_cutoff_min'] if c in s2.columns]]
        _write_publication_csv(s2,td/'Table_S2_GAM_based_screen_audit_mode_specific.csv')

    # Table S3: aggregate COMBINED-anchor sensitivity.
    s3=_aggregate_for_config('combined')
    _write_publication_csv(s3,td/'Table_S3_aggregate_COMBINED_all_selected_vs_GAM_screened.csv')

    # Table S4: aggregate standard verification for mode-specific configurations.
    s4_rows=[]
    for model in model_order:
        rec={'Model':model}
        for wf,qualifier in [('All selected anchors','all selected anchors'),('GAM-screened anchors','GAM-screened anchors')]:
            auth_total=0; iso_total=0
            for ds in datasets:
                cfg=_mode_from_dataset(ds)
                sub=results[(results['Dataset']==ds)&
                            (results['Anchor_configuration']==cfg)&
                            (results['Workflow']==wf)&
                            (results['Model']==model)]
                if len(sub):
                    rr=sub.iloc[0]
                    auth_total += int(rr.get('Authentic_hits',0))
                    iso_total += int(rr.get('Isotope_hits',0))
            rec[f'Authentic-standard verified pairs ({qualifier})']=auth_total
            rec[f'Isotope-labeled verified pairs ({qualifier})']=iso_total
            rec[f'Total verified pairs ({qualifier})']=auth_total+iso_total
        s4_rows.append(rec)
    s4=pd.DataFrame(s4_rows)
    _write_publication_csv(s4,td/'Table_S4_verification_composition_mode_specific.csv')

    # Sensitivity tables may have been computed in an earlier publication run.
    # Normalize their presentation without rounding the underlying calculations.
    for sensitivity_name in [
        'Table_S5_mass_tolerance_sensitivity_mode_specific.csv',
        'Table_S6_ACS_weighting_end_to_end_sensitivity_mode_specific.csv',
        'Table_S7_signed_mass_offset.csv',
    ]:
        sensitivity_path=td/sensitivity_name
        if sensitivity_path.exists():
            sensitivity_table=pd.read_csv(sensitivity_path).rename(columns={
                'Ionization_mode_specific_anchors':'Mode_specific_anchors',
                'Ionization_mode_specific_anchor_pairs':'Mode_specific_anchor_pairs',
            })
            _write_publication_csv(sensitivity_table,sensitivity_path)

    # Table S8: donor aggregation and QC columns used.
    donor_path=result_dir/'biological_sample_aggregation_report.csv'
    if donor_path.exists():
        s8=pd.read_csv(donor_path)
        s8.to_csv(td/'Table_S8_donor_aggregation_and_QC_columns.csv',index=False)

    # Expanded audit outputs.
    feature_summary.to_csv(td/'All_feature_anchor_summary_raw.csv',index=False)
    results.to_csv(td/'All_full_model_metrics_by_dataset_config.csv',index=False)
    if not verification.empty:
        verification.groupby(['Workflow','Model','Anchor_configuration','Standard_type'],as_index=False).size().rename(columns={'size':'Verified pairs'}).to_csv(td/'All_verification_hit_rows_by_configuration.csv',index=False)
    if candidate_pairs is not None and not candidate_pairs.empty:
        cand_sum = candidate_pairs.groupby(['Workflow','Dataset','Anchor_configuration','Model'], as_index=False).agg(
            Candidate_pairs=('Target_row','size'),
            Source_features_with_candidates=('Source_row','nunique'),
            Median_candidate_RT_error_min=('RT_error_min','median'),
            Median_candidate_ACS=('ACS','median'),
            Primary_pairs=('Is_primary_pair', lambda x: int(np.nansum(x.astype(bool))))
        )
        _write_publication_csv(cand_sum,td/'All_candidate_pair_counts_by_model.csv')
    diag_sum=pd.DataFrame()
    if not diagnostics.empty:
        diag_sum=diagnostics.groupby(['Workflow','Dataset','Anchor_configuration','Model'],as_index=False).agg(
            Median_abs_anchor_residual_min=('Abs_residual_min','median'),
            P90_abs_anchor_residual_min=('Abs_residual_min',lambda x: np.nanquantile(x,.9)),
            N_anchors=('Abs_residual_min','size'))
        _write_publication_csv(diag_sum,td/'All_anchor_residual_diagnostics_summary.csv')

    # Publication workbook.
    xlsx=output_dir/'RTBridge_paper_and_all_tables.xlsx'
    with pd.ExcelWriter(xlsx,engine='xlsxwriter') as writer:
        workbook=writer.book
        header_fmt=workbook.add_format({'bold':True,'font_color':'white','bg_color':'#17365D','align':'center','valign':'vcenter','border':1,'text_wrap':True})
        cell_fmt=workbook.add_format({'align':'center','valign':'vcenter','border':1})
        number_formats={}
        general_num_fmt=workbook.add_format({'align':'center','valign':'vcenter','border':1,'num_format':'0.########'})
        integer_fmt=workbook.add_format({'align':'center','valign':'vcenter','border':1,'num_format':'#,##0'})

        def _metric_format(value):
            number=float(value)
            decimals=2 if number==0 else max(0,2-int(np.floor(np.log10(abs(number)))))
            decimals=min(decimals,12)
            if decimals not in number_formats:
                picture='0' if decimals==0 else '0.'+'0'*decimals
                number_formats[decimals]=workbook.add_format({
                    'align':'center','valign':'vcenter','border':1,
                    'num_format':picture,
                })
            return number_formats[decimals]

        def _write_sheet(df, sheet):
            df.to_excel(writer,sheet_name=sheet,index=False)
            ws=writer.sheets[sheet]
            ws.freeze_panes(1,0)
            ws.set_row(0,35)
            for j,col in enumerate(df.columns):
                ws.write(0,j,col,header_fmt)
                vals=df[col].head(100).fillna('').astype(str).tolist()
                width=max(10,min(34,max([len(str(col))]+[len(v) for v in vals])+2))
                if pd.api.types.is_integer_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
                    column_fmt=integer_fmt
                elif pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
                    column_fmt=general_num_fmt
                else:
                    column_fmt=cell_fmt
                ws.set_column(j,j,width,column_fmt)
                if _is_precision_metric_column(col):
                    for row_index,value in enumerate(df[col],start=1):
                        try:
                            number=float(value)
                        except (TypeError,ValueError):
                            continue
                        if np.isfinite(number):
                            ws.write_number(row_index,j,number,_metric_format(number))
        _write_sheet(t1,'Table_1')
        _write_sheet(t2,'Table_2')
        for sheet,file in [
            ('Table_S1','Table_S1_stable_isotope_labeled_standards.csv'),
            ('Table_S2','Table_S2_GAM_based_screen_audit_mode_specific.csv'),
            ('Table_S3','Table_S3_aggregate_COMBINED_all_selected_vs_GAM_screened.csv'),
            ('Table_S4','Table_S4_verification_composition_mode_specific.csv'),
            ('Table_S5','Table_S5_mass_tolerance_sensitivity_mode_specific.csv'),
            ('Table_S6','Table_S6_ACS_weighting_end_to_end_sensitivity_mode_specific.csv'),
            ('Table_S7','Table_S7_signed_mass_offset.csv'),
            ('Table_S8','Table_S8_donor_aggregation_and_QC_columns.csv')]:
            fp=td/file
            if fp.exists(): _write_sheet(pd.read_csv(fp),sheet)
        _write_sheet(results,'All_model_metrics')
        cand_path=td/'All_candidate_pair_counts_by_model.csv'
        if cand_path.exists(): _write_sheet(pd.read_csv(cand_path),'All_candidate_pair_counts')
        if not diag_sum.empty: _write_sheet(diag_sum,'Anchor_residuals')

def write_caption_file(output_dir:Path):
    text={
        'Figure_2_all_selected_anchor_heatmaps':'Figure 2. Dataset-by-model heatmaps using all selected anchors. For each dataset and model, mode-specific and COMBINED anchor configurations are shown. Three panels summarize primary pairs, median RT error, and median abundance-concordance score.',
        'Figure_3_GAM_based_anchor_counts':'Figure 3. Mode-specific training-anchor counts for all selected anchors and GAM-screened anchors.',
        'Figure_4_GAM_based_heatmaps':'Figure 4. Dataset-by-model heatmaps using GAM-screened anchors. Mode-specific and COMBINED anchor configurations are shown. Panels A-C summarize primary pairs, median RT error, and median abundance-concordance score.',
        'Figure_5_PCHIP_verification':'Figure 5. Authentic-standard and stable-isotope-labeled-standard verification using PCHIP with all selected and GAM-screened mode-specific anchors.',
        'Figure_S1_anchor_RT_distributions':'Figure S1. RT distributions for all selected and GAM-screened mode-specific anchors. Histograms show RT5 and RT25 coverage for each dataset.',
        'Figure_S2_PCHIP_RT_error_ACS_distributions':'Figure S2. RT-error and ACS distributions for PCHIP primary pairs obtained using all selected and GAM-screened mode-specific anchors.',
        'Figure_S3_all_selected_anchor_PCHIP_diagnostics':'Figure S3. PCHIP model-fit diagnostics using all selected mode-specific anchors. Points are colored by ACS, and red curves show the fitted RT5-to-RT25 mapping.',
        'Figure_S4_mass_tolerance_sensitivity_mode_specific':'Figure S4. Mass-tolerance sensitivity across six U/A/C settings using mode-specific anchors. Every bar is labeled; ACS and RT-error measurements are displayed with three significant figures, and the selected 20/10/10 ppm setting is hatched.',
        'All_model_fit_diagnostics':'All diagnostic figures. Per-model RT-transfer fit diagnostics are saved in figures/all/model_fit_diagnostics/. Each file shows anchor points colored by ACS and the fitted RT5-to-RT25 curve for one selected model and workflow.',
    }
    with open(output_dir/'figure_table_captions.txt','w',encoding='utf-8') as f:
        for k,v in text.items(): f.write(f'{k}: {v}\n')


def _read_csv_optional(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path) if Path(path).exists() else pd.DataFrame()
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _standardize_legacy_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep older saved RTBridge runs readable after terminology improvements."""
    if frame.empty:
        return frame
    updated=frame.copy()
    if 'Workflow' in updated.columns:
        updated['Workflow']=updated['Workflow'].replace({
            'Complete anchors':'All selected anchors',
            'Full anchor set':'All selected anchors',
            'GAM-based screen':'GAM-screened anchors',
            'GAM-screened anchor set':'GAM-screened anchors',
        })
    if 'Anchor_configuration' in updated.columns:
        updated['Anchor_configuration']=updated['Anchor_configuration'].replace({
            'COMBINED-POLARITY':'COMBINED',
            'combined-polarity':'COMBINED',
        })
    return updated

def make_all_figures_tables(output_dir:Path, dpi:int=600, output_formats=('png','tif','pdf'), include_exploratory:bool=True):
    global _ACTIVE_OUTPUT_FORMATS
    _ACTIVE_OUTPUT_FORMATS = tuple(output_formats)
    result_dir=output_dir/'results'; fig_main=output_dir/'figures'/'main'; fig_si=output_dir/'figures'/'SI'; fig_extra=output_dir/'figures'/'all'
    results=_standardize_legacy_labels(pd.read_csv(result_dir/'model_metrics_all.csv'))
    pairs=_standardize_legacy_labels(_read_csv_optional(result_dir/'primary_pairs_all.csv'))
    candidate_pairs=_standardize_legacy_labels(_read_csv_optional(result_dir/'candidate_pairs_all.csv'))
    anchors=_standardize_legacy_labels(pd.read_csv(result_dir/'anchors_all.csv'))
    verification=_standardize_legacy_labels(_read_csv_optional(result_dir/'verification_hits_all.csv'))
    audit=_standardize_legacy_labels(_read_csv_optional(result_dir/'GAM_based_screen_audit.csv'))
    feature_summary=_standardize_legacy_labels(pd.read_csv(result_dir/'feature_anchor_summary.csv'))
    diagnostics=_standardize_legacy_labels(_read_csv_optional(result_dir/'model_anchor_residual_diagnostics_all.csv'))
    # Main paper figures
    scales=_heatmap_scales(results)
    make_paper_heatmap(results,'All selected anchors',fig_main/'Figure_2_all_selected_anchor_heatmaps',dpi,scales)
    make_figure3_anchor_counts(anchors,fig_main/'Figure_3_GAM_based_anchor_counts',dpi)
    make_paper_heatmap(results,'GAM-screened anchors',fig_main/'Figure_4_GAM_based_heatmaps',dpi,scales)
    make_figure5_pchip_verification(results,fig_main/'Figure_5_PCHIP_verification',dpi)
    # SI figures required by paper
    make_si_anchor_distribution_hist(anchors,fig_si/'Figure_S1_anchor_RT_distributions',dpi)
    make_si_pchip_distributions(pairs,results,fig_si/'Figure_S2_PCHIP_RT_error_ACS_distributions',dpi)
    make_si_pchip_fit_diagnostics(diagnostics,results,fig_si/'Figure_S3_all_selected_anchor_PCHIP_diagnostics',dpi)
    mass_sensitivity=_read_csv_optional(output_dir/'tables'/'Table_S5_mass_tolerance_sensitivity_mode_specific.csv')
    if mass_sensitivity.empty:
        mass_sensitivity=_read_csv_optional(output_dir/'sensitivity_analysis'/'mass_tolerance'/'Table_S5_mass_tolerance_sensitivity_mode_specific.csv')
    if not mass_sensitivity.empty:
        make_si_mass_tolerance_sensitivity(mass_sensitivity,fig_si/'Figure_S4_mass_tolerance_sensitivity_mode_specific',dpi)
    if include_exploratory:
        # Per-model and exploratory figures are optional in Fast mode.
        make_model_fit_diagnostics_all(diagnostics, results, fig_extra/'model_fit_diagnostics', dpi)
        make_model_summary(results,fig_extra/'All_Figure_model_summary_selected_primary_configs',dpi)
        make_model_summary_all_configs(results,fig_extra/'All_Figure_model_summary_all_configurations',dpi)
        make_model_residual_figure(diagnostics,fig_extra/'All_Figure_anchor_residuals_by_model',dpi)
        make_paper_heatmap(results,'All selected anchors',fig_extra/'All_Figure_all_selected_anchor_heatmaps',dpi,scales)
        make_paper_heatmap(results,'GAM-screened anchors',fig_extra/'All_Figure_GAM_screened_anchor_heatmaps',dpi,scales)
    make_tables(results,anchors,verification,audit,feature_summary,diagnostics,pairs,result_dir,output_dir,candidate_pairs=candidate_pairs)
    write_caption_file(output_dir)
