from pathlib import Path
import json
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/eeg-report-matplotlib")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from eeg_apnea import predict_features
OUT=ROOT/'reports/report_assets';OUT.mkdir(parents=True,exist_ok=True)
m=json.loads((ROOT/'models/eeg_svm.json').read_text());r=m['report']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':220})
BLUE='#2864C5';TEAL='#087F83';NAVY='#182B49';GRAY='#596779'

def save(fig,name):
 fig.savefig(OUT/name,bbox_inches='tight',facecolor='white');plt.close(fig)

def node(ax,x,y,w,h,label,color=BLUE):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=0.04',facecolor=color,edgecolor=color))
 ax.text(x+w/2,y+h/2,label,ha='center',va='center',color='white',fontsize=10)

def diagram(name,rows):
 fig,ax=plt.subplots(figsize=(7.4,3.0));ax.set(xlim=(0,10),ylim=(0,4.1));ax.axis('off')
 for yy,label,nodes in rows:
  ax.text(.08,yy+1.02,label,color=NAVY,fontweight='bold',fontsize=10)
  width=(9.6-.35*(len(nodes)-1))/len(nodes)
  for i,txt in enumerate(nodes):
   x=.15+i*(width+.35);node(ax,x,yy,width,.79,txt,TEAL if yy<1 else BLUE)
   if i<len(nodes)-1:ax.add_patch(FancyArrowPatch((x+width+.03,yy+.395),(x+width+.30,yy+.395),arrowstyle='-|>',mutation_scale=12,color=GRAY))
 save(fig,name)

diagram('system_flow.png',[(2.23,'Upstream signal processing (Group 1)', ['Recorded EEG','IIR filter +\nHaar DWT','Feature vector']),(.35,'Classification software (Group 2)', ['Validate\n15 parameters','Transform, scale\nand select','Trained\nRBF SVM','Normal /\napnea label'])])
diagram('training_inference.png',[(2.23,'Offline development: fit parameters using training subjects', ['Labeled feature\nrows','Grouped CV\nselection','Save model\nand feature contract']),(.35,'Runtime: reuse fitted parameters for each supplied window', ['FPGA / CSV /\nmanual parameters','Log transform\nand scaling','Selected\n10 features','SVM decision\nmargin and label'])])
diagram('module_flow.png',[(2.23,'Data and training modules', ['prepare_eeg_dataset.py','eeg_apnea.py\ntraining','eeg_svm.json']),(.35,'Application modules', ['fpga_serial.py /\nCSV / demo','ui.py\ninput validation','eeg_apnea.py\ninference','Session results\nand CSV export'])])
records=json.loads((ROOT/'models/eeg_dataset_provenance.json').read_text())['records']
fig,ax=plt.subplots(figsize=(7,3.2))
names=[v['record'] for v in records];normal=[v['counts']['normal'] for v in records];apnea=[v['counts']['apnea'] for v in records]
x=np.arange(len(names));ax.bar(x,normal,color=BLUE,label='Normal');ax.bar(x,apnea,bottom=normal,color=TEAL,label='Apnea')
ax.set_xticks(x,names,rotation=40,ha='right');ax.set_ylabel('Included windows');ax.legend(frameon=False,ncol=2)
for i,name in enumerate(names):
 if name in r['test_subjects']:ax.text(i,normal[i]+apnea[i]+12,'test',ha='center',fontsize=8,color=GRAY)
fig.tight_layout();save(fig,'subject_windows.png')
fig,ax=plt.subplots(figsize=(5.0,3.8));cm=np.array(r['confusion_matrix']);ax.imshow(cm,cmap='Blues',vmin=0,vmax=cm.max())
for i in range(2):
 for j in range(2):ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=24,color='white' if cm[i,j]>200 else NAVY)
ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Normal','Apnea'],yticklabels=['Normal','Apnea'],xlabel='Predicted class',ylabel='Actual class')
for spine in ax.spines.values():spine.set_visible(False)
fig.tight_layout();save(fig,'confusion_matrix.png')
fig,ax=plt.subplots(figsize=(7,3.4));keys=['accuracy','balanced_accuracy','sensitivity','specificity','precision','f1','roc_auc']
labels=['Accuracy','Balanced acc.','Sensitivity','Specificity','Precision','F1','ROC AUC']
values=[r[k]*100 for k in keys];bars=ax.barh(labels[::-1],values[::-1],color=BLUE)
for bar,value in zip(bars,values[::-1]):ax.text(value+1.2,bar.get_y()+bar.get_height()/2,f'{value:.1f}%',va='center',fontsize=10)
ax.set_xlim(0,100);ax.set_xlabel('Percentage / AUC × 100');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
fig.tight_layout();save(fig,'measured_metrics.png')
fig,ax=plt.subplots(figsize=(7,3.2));subs=r['holdout_subject_metrics'];x=np.arange(len(subs));w=.25
for off,key,label,color in [(-w,'accuracy','Accuracy',BLUE),(0,'sensitivity','Sensitivity',TEAL),(w,'specificity','Specificity','#9A6B25')]:
 vals=[v[key]*100 for v in subs];bars=ax.bar(x+off,vals,w,color=color,label=label)
 for b,v in zip(bars,vals):ax.text(b.get_x()+w/2,v+1,f'{v:.1f}',ha='center',fontsize=9)
ax.set_xticks(x,[v['subject'] for v in subs]);ax.set_ylim(0,110);ax.set_ylabel('Percent');ax.legend(frameon=False,ncol=3,loc='upper center');fig.tight_layout();save(fig,'subject_performance.png')
frame=pd.read_csv(ROOT/'data/processed/slpdb/eeg_features.csv')
test=frame.loc[frame.subject_id.isin(r['test_subjects'])].reset_index(drop=True)
scores=predict_features(m,test).decision_margin.to_numpy();y=test.label.to_numpy()
assert np.isclose(roc_auc_score(y,scores),r['roc_auc'])
assert np.isclose(average_precision_score(y,scores),r['average_precision'])
fpr,tpr,_=roc_curve(y,scores);precision,recall,_=precision_recall_curve(y,scores)
fig,axes=plt.subplots(1,2,figsize=(7.2,3.3))
axes[0].plot(fpr,tpr,color=BLUE,label=f'ROC AUC = {r["roc_auc"]:.3f}')
axes[0].plot([0,1],[0,1],'--',color='#999999',lw=1);axes[0].set(xlabel='False-positive rate',ylabel='True-positive rate',xlim=(0,1),ylim=(0,1.02));axes[0].legend(frameon=False,loc='lower right',fontsize=8)
axes[1].plot(recall,precision,color=TEAL,label=f'AP = {r["average_precision"]:.3f}')
axes[1].axhline(y.mean(),ls='--',color='#999999',lw=1);axes[1].set(xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1.02));axes[1].legend(frameon=False,fontsize=8)
fig.tight_layout();save(fig,'roc_precision_recall.png')
print('Saved report figures; ROC/Average Precision verified against the recorded model report.')
