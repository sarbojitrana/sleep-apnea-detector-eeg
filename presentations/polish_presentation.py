"""Editorial cleanup of an existing deck, preserving its current cover and results."""
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile
import argparse
import json
import re

from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_DATA_LABEL_POSITION, XL_TICK_MARK
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt
from build_presentation import strip_annotations

HERE = Path(__file__).resolve().parent
DEFAULT = HERE / 'Sleep_Apnea_SVM_Software_Presentation.pptx'
NAVY, BLUE, GRAY, PALE, TEAL = '182B49', '2864C5', '596779', 'F1F5FA', '087F83'


def delete(shape):
    element=shape._element
    element.getparent().remove(element)


def replace_text(shape, value):
    """Keep the current typography when editing an existing text box."""
    frame=shape.text_frame
    first=next((run for p in frame.paragraphs for run in p.runs),None)
    font=first.font if first is not None else None
    attrs=(font.name,font.size,font.bold,font.italic) if font else ('Arial',Pt(18),False,False)
    color=font.color.rgb if font is not None and font.color.type is not None and font.color.type==1 else None
    align=frame.paragraphs[0].alignment
    spacing=frame.paragraphs[0].line_spacing
    frame.clear()
    for i,line in enumerate(value.split('\n')):
        p=frame.paragraphs[0] if i==0 else frame.add_paragraph()
        p.alignment=align
        if spacing is not None:p.line_spacing=spacing
        p.space_after=Pt(5)
        run=p.add_run();run.text=line
        run.font.name,run.font.size,run.font.bold,run.font.italic=attrs
        if color is not None:run.font.color.rgb=color


def textbox(slide,x,y,w,h,value,size=18,color=NAVY,bold=False):
    shape=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h))
    tf=shape.text_frame;tf.word_wrap=True
    tf.margin_left=tf.margin_right=Inches(.02)
    tf.margin_top=tf.margin_bottom=Inches(.01)
    for i,line in enumerate(value.split('\n')):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph()
        p.space_after=Pt(7);p.line_spacing=1.12
        run=p.add_run();run.text=line
        run.font.name='Arial';run.font.size=Pt(size);run.font.bold=bold
        run.font.color.rgb=RGBColor.from_string(color)
    return shape


def cell_style(cell,value,fill='FFFFFF',color=NAVY,bold=False,size=16):
    cell.text=value;cell.fill.solid();cell.fill.fore_color.rgb=RGBColor.from_string(fill)
    cell.margin_left=cell.margin_right=Inches(.08)
    cell.margin_top=cell.margin_bottom=Inches(.03)
    cell.vertical_anchor=MSO_ANCHOR.MIDDLE
    for p in cell.text_frame.paragraphs:
        p.alignment=PP_ALIGN.CENTER
        for run in p.runs:
            run.font.name='Arial';run.font.size=Pt(size);run.font.bold=bold
            run.font.color.rgb=RGBColor.from_string(color)


def remove_body(slide):
    for sh in list(slide.shapes):
        if Inches(2.3)<=sh.top<Inches(6.95):delete(sh)


def feature_table(slide):
    for sh in list(slide.shapes):
        if sh.left<Inches(8) and Inches(2.4)<=sh.top<Inches(6.5):delete(sh)
    table=slide.shapes.add_table(6,4,Inches(.82),Inches(2.53),Inches(6.94),Inches(3.25)).table
    table.columns[0].width=Inches(1.57)
    for index in (1,2,3):table.columns[index].width=Inches(1.79)
    for c,label in enumerate(('Band','Energy','Mobility','Kurtosis')):
        cell_style(table.cell(0,c),label,BLUE,'FFFFFF',True,17)
    for row,band in enumerate(('A5','D5','D4','D3','D2'),1):
        for col,label in enumerate((band,f'E_{band}',f'M_{band}',f'K_{band}')):
            cell_style(table.cell(row,col),label,PALE if row%2 else 'FFFFFF',bold=(col==0),size=17)


def results_slide(slide,report):
    remove_body(slide)
    data=CategoryChartData()
    # Reversed order puts accuracy at the top in a horizontal bar chart.
    fields=[('F1','f1'),('Precision','precision'),('Specificity','specificity'),
            ('Sensitivity','sensitivity'),('Accuracy','accuracy')]
    data.categories=[name for name,key in fields]
    data.add_series('Held-out result',[report[key] for name,key in fields])
    chart=slide.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED,Inches(.78),Inches(2.48),
                                 Inches(6.88),Inches(3.49),data).chart
    chart.has_legend=False
    plot=chart.plots[0];plot.gap_width=75;plot.has_data_labels=True
    labels=plot.data_labels;labels.position=XL_DATA_LABEL_POSITION.OUTSIDE_END
    labels.number_format='0.0%';labels.font.name='Arial';labels.font.size=Pt(13)
    labels.font.color.rgb=RGBColor.from_string(NAVY)
    series=chart.series[0];series.format.fill.solid()
    series.format.fill.fore_color.rgb=RGBColor.from_string(BLUE)
    series.format.line.fill.background()
    axis=chart.value_axis;axis.minimum_scale=0;axis.maximum_scale=1;axis.major_unit=.25
    axis.tick_labels.number_format='0%';axis.tick_labels.font.size=Pt(11)
    axis.tick_labels.font.name='Arial';axis.tick_labels.font.color.rgb=RGBColor.from_string(GRAY)
    axis.major_tick_mark=XL_TICK_MARK.NONE;axis.has_major_gridlines=True
    axis.major_gridlines.format.line.color.rgb=RGBColor.from_string('E3E8EF')
    axis.format.line.fill.background()
    chart.category_axis.tick_labels.font.name='Arial'
    chart.category_axis.tick_labels.font.size=Pt(14)
    chart.category_axis.tick_labels.font.color.rgb=RGBColor.from_string(NAVY)
    chart.category_axis.major_tick_mark=XL_TICK_MARK.NONE
    chart.category_axis.format.line.fill.background()
    textbox(slide,8.02,2.50,4.27,.43,'Confusion matrix',21,bold=True)
    table=slide.shapes.add_table(3,3,Inches(8.02),Inches(3.12),Inches(4.28),Inches(1.74)).table
    table.columns[0].width=Inches(1.58)
    table.columns[1].width=table.columns[2].width=Inches(1.35)
    matrix=report['confusion_matrix']
    contents=[['Actual / Pred.','Normal','Apnea'],['Normal',str(matrix[0][0]),str(matrix[0][1])],
              ['Apnea',str(matrix[1][0]),str(matrix[1][1])]]
    for row in range(3):
        for col in range(3):
            header=row==0 or col==0
            fill=BLUE if header else PALE if row==col else 'FFF3DF'
            cell_style(table.cell(row,col),contents[row][col],fill,'FFFFFF' if header else NAVY,header,13 if header else 20)
    textbox(slide,8.04,5.03,4.28,.64,'Rows: actual class\nColumns: predicted class',13,GRAY)
    textbox(slide,.84,6.13,11.65,.37,'49 apnea windows missed; 141 normal windows flagged.',18,bold=True)
    textbox(slide,.84,6.61,11.65,.30,
            f'ROC AUC: {report["roc_auc"]:.1%}  ·  Prediction time: about {report["single_window_latency_ms_median"]:.0f} ms per window on the test host.',15,GRAY)


def limitations(slide):
    remove_body(slide)
    for y,title,body in [
        (2.48,'Data and consistency','The dataset has ten subjects. Accuracy is 74.0%, with substantial variation between subjects.'),
        (3.84,'Hardware integration','The FPGA filter, wavelet scaling and feature definitions still need to match the software reference.'),
        (5.20,'Clinical scope','The current output is a window label. AHI, severity estimates and robust artifact handling remain future work.'),
    ]:
        textbox(slide,.84,y,11.4,.40,title,22,bold=True)
        textbox(slide,.84,y+.54,11.4,.71,body,20,GRAY)


def future_work(slide):
    remove_body(slide)
    textbox(slide,.84,2.45,5.45,.45,'Integration and validation',23,TEAL,True)
    textbox(slide,.84,3.17,5.45,2.82,
        'Agree on the feature definitions and serial format.\nCompare FPGA output with the Python reference.\nEvaluate a larger, more diverse subject group.',21)
    textbox(slide,7.03,2.45,5.45,.45,'Further development',23,TEAL,True)
    textbox(slide,7.03,3.17,5.45,2.82,
        'Explore SVM deployment on the FPGA.\nAdd signal-quality checks and temporal context.\nStudy combined EEG, ECG and SpO₂ models.',21)


def clean(path,output):
    prs=Presentation(path)
    cover='\n'.join(sh.text for sh in prs.slides[0].shapes if sh.has_text_frame)
    match=re.search(r'GROUP\s+(\d+)',cover,re.I)
    group=f'Group {match.group(1)}' if match else ''
    changes={
      'Illustrative airflow, not a patient recording.':'Schematic airflow',
      'Illustrative airflow — not a patient recording.':'Schematic airflow',
      'Uses statistical EEG features and an FPGA classifier.\nTakeaway: compact features can support hardware-oriented detection.':'Alam et al. use EEG statistics with a linear SVM on FPGA. This supports the use of compact features.',
      'Examines EEG classification with subject-wise evaluation.\nTakeaway: test whether a model works for unseen people.':'Barnes et al. study an explainable CNN for single-channel EEG and evaluate it on unseen subjects.',
      'Online processing on a wearable device.\nTakeaway: compact models and efficient preprocessing help deployment.':'Surrel et al. use ECG and a compact SVM for online detection on a wearable device.',
      'Tests how results transfer to external data.\nTakeaway: cross-validation alone does not establish real-world performance.':'Padovano et al. show that strong cross-validation scores may fall when a model is tested on another dataset.',
      'Research review: EEG approaches':'EEG-based approaches in the literature',
      'Research review: deployment and validation':'Wearable detection and validation',
      'The software workstream begins after FPGA feature values are available.':'Our software starts with the FPGA feature vector.',
      'UPSTREAM FPGA WORKSTREAM':'FPGA processing',
      'OUR CLASSIFICATION SOFTWARE':'Classification software',
      'Actual app output using clearly marked synthetic demonstration inputs.':'Example run with synthetic inputs.',
      'Actual performance table from the Model and performance tab.':'EEG-model evaluation in the application.',
      'Measured EEG performance is displayed alongside the dataset and validation method.':'The application shows the measured results and the validation method.',
      'Current shortcomings':'Current limitations',
      'Possible future additions':'Future work',
      'A working software classifier is ready for feature-interface validation.':'EEG window classification is implemented; hardware validation remains.',
      'INPUT → RESULT':'Inputs', 'REVIEW → EXPORT':'History', 'METHOD → METRICS':'Model details',
      'CURRENT SOFTWARE':'Implemented', 'REMAINING WORK':'Next stage',
      'CLINICAL REFERENCE':'Clinical reference','PORTABLE MONITORING':'Portable monitoring','OUR PROJECT':'Project approach',
    }
    filler={
      'Different signals offer different strengths and practical limitations.',
      'Two useful precedents for our feature-based classifier.',
      'Useful lessons beyond the choice of classifier.',
      'The existing layout is retained: Screening, History, and Model and performance.',
      'The present system is a research prototype with important practical limits.',
      'Proposed extensions to evaluate after the shared feature interface is validated.',
      'Verified research papers and project background sources.',
    }
    for number,s in enumerate(prs.slides,1):
        for sh in list(s.shapes):
            if not sh.has_text_frame:continue
            value=sh.text
            # Keep the academic cover line, remove repetitive section labels elsewhere.
            if number!=1 and .2<sh.top/Inches(1)<.7:
                delete(sh);continue
            if value in filler:
                delete(sh);continue
            if re.fullmatch(r'\d{2}\s*/\s*19',value.strip()):
                replace_text(sh,str(number));continue
            if value in changes:replace_text(sh,changes[value])
        if number==8:feature_table(s)
        if number==13:
            report=json.loads((HERE.parent/'models/eeg_svm.json').read_text())['report']
            results_slide(s,report)
        if number==15:limitations(s)
        if number==16:future_work(s)
    notes={
      1:'Classification of EEG windows using FPGA-derived features and a trained SVM. The current classifier runs in software.',
      2:'',
      3:'The waveform illustrates airflow. EEG provides information about sleep and arousal around breathing events.',
      4:'Polysomnography is the clinical reference. EEG, ECG and oxygen-saturation signals offer different information for automated screening.',
      5:'Alam et al. use conventional EEG bands and a linear SVM on Artix-7. Our design uses Haar subbands. Barnes et al. investigate an explainable CNN with subject-wise evaluation. Their datasets and validation procedures differ from this project.\n[1] https://doi.org/10.1109/ACCESS.2024.3374223\n[2] https://doi.org/10.1371/journal.pone.0272167',
      6:'Surrel et al. use patient-specific ECG classification on a wearable device. Padovano et al. compare cross-validation with external validation. George’s chapter is background reading; the research review covers papers [1]–[4].\n[3] https://doi.org/10.1109/TBCAS.2018.2824659\n[4] https://doi.org/10.22489/CinC.2022.086',
      7:'The IIR and DWT blocks are implemented upstream. Feature extraction and the shared numerical interface are still being finalized.',
      8:'Feature order: E_A5, M_A5, K_A5, E_D5, M_D5, K_D5, E_D4, M_D4, K_D4, E_D3, M_D3, K_D3, E_D2, M_D2, K_D2. Energy is total coefficient energy. Mobility uses population variances; kurtosis is ordinary Pearson kurtosis. The current model selects ten features.',
      9:'MIT-BIH C3–O1 EEG at 250 Hz. There are 2,644 development windows and 732 test windows. OA/X/CA/CAA sleep epochs are positive; hypopnea-only, wake, movement and unknown epochs are excluded. Grouped development CV fits log transformation, scaling, selection and SVM parameters within training folds.\n[6] https://doi.org/10.13026/C23K5S',
      10:'The three tabs cover input and prediction, session history, and model evidence. EEG is classified separately from the lifestyle model. ECG and SpO₂ are currently contextual inputs.',
      11:'Six synthetic feature windows are used in this demonstration. The resulting labels illustrate application behavior; the inputs have no clinical ground truth.',
      12:'The table shows the EEG model’s measured evaluation results. The majority-class result is included as a reference baseline.',
      13:'Confusion matrix: [[344,141],[49,198]], with actual rows and predicted columns [normal, apnea]. There were 49 missed apnea windows and 141 false positives. Sensitivity was 13.6% for slp03 and 94.6% for slp16. The holdout was inspected during development, so a new external cohort is needed for an independent performance estimate. Latency covers software feature-to-prediction processing and excludes FPGA computation, transport and UI rendering.',
      14:'The hardware design specifies Chebyshev Type-II filtering; the software reference uses Butterworth SOS. Hardware Haar divides by 2, while the software uses orthonormal 1/√2 scaling. The hardware proposal lists Mean/MAV/Energy/Variance, while this software accepts Energy/Mobility/Kurtosis. The two sides must also agree on whether normalization happens in hardware or software.',
      15:'Ten subjects form a small evaluation cohort. Performance varies between people. FPGA feature equivalence and robust signal-quality handling remain incomplete. A window classifier alone does not calculate AHI or clinical severity.',
      16:'These are proposed extensions. Temporal analysis must use only the context available during operation. A combined EEG/ECG/SpO₂ model requires aligned signals and labels. FPGA deployment requires timing, resource and fixed-point error measurements.',
      17:'The software classifier, demonstration interface and subject-wise evaluation are implemented. The next steps are hardware feature validation and testing on additional subjects.',
      18:'[1] https://doi.org/10.1109/ACCESS.2024.3374223\n[2] https://doi.org/10.1371/journal.pone.0272167\n[3] https://doi.org/10.1109/TBCAS.2018.2824659',
      19:'[4] https://doi.org/10.22489/CinC.2022.086\n[5] Background chapter listed in Advances in Wearables and Nearables for Healthcare Applications, CRC Press.\n[6] https://doi.org/10.13026/C23K5S',
    }
    for number,s in enumerate(prs.slides,1):s.notes_slide.notes_text_frame.text=notes.get(number,'')
    properties=prs.core_properties
    properties.title='Sleep Apnea Detection and Prediction'+(f' — {group}' if group else '')
    properties.author=group
    properties.subject='EEG feature classification using SVM'
    properties.comments='EEG classification project presentation, results and references.'
    properties.last_modified_by=''
    properties.modified=datetime.now(timezone.utc)
    properties.revision=properties.revision+1
    prs.save(output)
    # Remove invalid template dates and optional tool metadata, without inventing a
    # software application, original creation date, or manual editing history.
    with ZipFile(output) as z:parts=[(info,z.read(info.filename)) for info in z.infolist()]
    with ZipFile(output,'w') as z:
        for info,data in parts:
            if info.filename=='docProps/core.xml':
                root=etree.fromstring(data)
                for tag in ('{http://purl.org/dc/terms/}created','{http://schemas.openxmlformats.org/package/2006/metadata/core-properties}lastPrinted'):
                    for element in root.findall(tag):root.remove(element)
                data=etree.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
            if info.filename=='docProps/app.xml':
                root=etree.fromstring(data)
                namespace='{http://schemas.openxmlformats.org/officeDocument/2006/extended-properties}'
                for tag in ('Application','AppVersion','Template','TotalTime'):
                    for element in root.findall(namespace+tag):root.remove(element)
                data=etree.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
            z.writestr(info,data)
    strip_annotations(output)
    print(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=DEFAULT)
    parser.add_argument('--output',type=Path,default=DEFAULT)
    args=parser.parse_args();clean(args.input,args.output)
