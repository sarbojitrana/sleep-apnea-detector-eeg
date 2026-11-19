"""Build the Group 2 project report from recorded project evidence."""
from pathlib import Path
from datetime import datetime, timezone
import json
import re
from zipfile import ZipFile

from docx import Document
from docx.enum.section import WD_SECTION_START
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from lxml import etree

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports';ASSETS=OUT/'report_assets'
MODEL=json.loads((ROOT/'models/eeg_svm.json').read_text());R=MODEL['report']
PROVENANCE=json.loads((ROOT/'models/eeg_dataset_provenance.json').read_text())
MEMBERS=[('Satyam Mondal','23UEC034'),('Sarbojit Rana','23UEC038'),('Suraj Verma','23UEC046'),('Ramavath Rajesh','23UEC103')]
TITLE='SLEEP APNEA DETECTION AND PREDICTION'
SUBTITLE='Post-FPGA EEG Feature Classification Using Support Vector Machine'
DOC=Document();PAGES=[];FIGURES=[];TABLES=[]


def page(number,title,blocks,chapter=None):
    PAGES.append({'number':number,'title':title,'blocks':blocks,'chapter':chapter})


def p(text):return ('p',text)
def h(text):return ('h',text)
def bullets(*items):return ('bullets',items)
def table(caption,headers,rows,widths=None):
    TABLES.append((caption,len(PAGES)+1));return ('table',caption,headers,rows,widths)
def fig(caption,name,width=6.3):
    FIGURES.append((caption,len(PAGES)+1));return ('fig',caption,name,width)
def code(text):return ('code',text)


# Body pages 1–3: introduction.
page(1,'1 INTRODUCTION',[
 h('1.1 Background'),
 p('Sleep apnea involves repeated interruptions in breathing during sleep. Obstructive apnea is associated with upper-airway blockage, whereas central apnea involves a failure of breathing drive. Sleep disruption and reduced oxygen levels can accompany these events. Polysomnography combines several physiological measurements for assessment [1]. This project investigates a software classification stage for recorded EEG; it does not independently diagnose a patient.'),
 p('EEG describes brain electrical activity during sleep. Respiratory disturbances may be accompanied by changes in sleep or arousal, which makes EEG a useful research signal for event-related classification [3]. The input to the present classifier is a small numerical description of an EEG window rather than the complete recording.'),
 h('1.2 Project setting'),
 p('The overall project is divided between an FPGA signal-processing front end and a separately developed classifier. Group 1 reports the implementation and functional simulation of IIR filtering and five-level Haar decomposition. Its feature-processing and classifier hardware are proposed later stages [12]. Group 2 develops dataset preparation, feature validation, offline SVM training, software inference and a screening application.'),
 fig('Figure 1.1 Overall processing chain and Group 2 scope.','system_flow.png',6.35),
 p('The engineering handoff is the feature vector. Its order, definitions and scale must be agreed before the trained software can be connected to physical FPGA output.')
],chapter='Introduction')
page(2,'1.3 Problem statement and objectives',[
 p('The immediate problem is to convert one valid EEG feature vector into a reproducible normal/apnea window label. A useful implementation must do more than call a classifier: it must reject malformed inputs, preserve the training convention, report evidence honestly and support a practical demonstration before the hardware feature interface is available.'),
 h('1.3.1 Objectives'),
 bullets('Define a named, ordered interface for 15 energy, mobility and kurtosis values from A5, D5, D4, D3 and D2.',
         'Prepare real EEG windows with aligned apnea labels and keep subjects separate during model selection and evaluation.',
         'Fit preprocessing, feature selection and an SVM using development data only; save a portable numerical model.',
         'Accept CSV, manual, serial and synthetic-demonstration inputs in the existing application.',
         'Display EEG results, session history and measured performance without combining unrelated model scores.'),
 h('1.3.2 Deliverables'),
 table('Table 1.1 Workstream responsibilities.',['Area','Group 1 / upstream','Group 2 / this report'],[
 ['Signal processing','IIR and Haar RTL front end','Offline reference for training'],
 ['Feature interface','Planned feature hardware','15-value validation and input contract'],
 ['Classifier','Planned hardware inference','Trained software RBF SVM'],
 ['Verification','Vivado front-end simulation','Software tests and subject-wise evaluation'],
 ['Integration','Physical feature output pending','CSV/serial adapter and application'],
 ],[1.1,2.6,2.6]),
 p('The report records implemented software and measured software results. Board resource use, FPGA power and physical end-to-end latency are outside the measurements available for Group 2.')
])
page(3,'1.4 Scope and interpretation',[
 h('1.4.1 Meaning of prediction'),
 p('Prediction in this project means assigning a label to a supplied EEG window: 0 denotes normal sleep and 1 denotes an apnea-labeled window. It does not mean forecasting a future episode or estimating whether a person will develop sleep apnea. The software also exposes a signed SVM decision margin; that score is not a calibrated probability.'),
 h('1.4.2 EEG and the retained lifestyle model'),
 p('The application contains two independent classification paths. A lifestyle SVM uses demographic and lifestyle fields. A separate EEG SVM uses the 15 feature parameters. Changing EEG features can change the EEG window result, but it does not alter the lifestyle score. No combined EEG/lifestyle model has been trained.'),
 p('The EEG source is currently placed inside an optional-biosignals section because the application can still run the lifestyle model by itself. If EEG is left as Not supplied, no EEG classification runs. When valid EEG parameters are supplied, the EEG model produces separate window labels. ECG and SpO₂ uploads are contextual summaries and do not affect either classifier.'),
 h('1.4.3 Limits of the present study'),
 bullets('Training uses a floating-point Python reference; numerical agreement with real FPGA feature output remains unverified.',
         'The evaluated task excludes wake, unknown and hypopnea-only epochs; its label definition is narrower than all sleep-disordered breathing.',
         'The model does not compute AHI, event duration, clinical severity or a patient-level diagnosis.'),
 h('1.5 Report organisation'),
 p('Chapter 2 reviews related research. Chapter 3 explains the dataset, feature contract and learning method. Chapter 4 describes the software and application. Chapter 5 presents evaluation results, integration issues, limitations and future work. The appendices provide the input protocol and reproduction commands.')
])

# Literature: four application papers and the SVM foundation.
page(4,'2 LITERATURE REVIEW',[
 h('2.1 Related research'),
 p('The relevant literature addresses different parts of the problem: EEG feature design, learned signal representations, efficient deployment and the validity of performance estimates. These studies motivate the design choices below, but they use different cohorts, labels and evaluation procedures. Their numerical results are therefore not treated as directly comparable benchmarks.'),
 table('Table 2.1 Literature reviewed and relevance to this project.',['Reference','Approach','Relevance'],[
 ['Alam et al. [2]','EEG statistics and linear SVM on FPGA','Compact features and inference'],
 ['Barnes et al. [3]','Single-channel EEG with explainable CNN','Subject-wise evaluation'],
 ['Surrel et al. [4]','Online wearable ECG classification','Practical embedded deployment'],
 ['Padovano et al. [5]','Cross-validation versus external validation','Generalisation and evaluation limits'],
 ['Cortes and Vapnik [6]','Support-vector classification','SVM decision-function foundation'],
 ],[1.45,2.7,2.15]),
 h('2.2 Position of the present work'),
 p('Group 2 implements a modular feature-to-label software path. The 15-feature Haar design is a project interface choice rather than an exact reproduction of a published detector. Its suitability is assessed using the available labeled cohort and explicit subject separation.'),
 p('Wearable-sensor background reading also includes Sara Mohan George’s chapter, Sleep Apnea Detection Using Wearable Sensors: Principles, Methods, and Challenges [11]. It is listed separately from the four reviewed application research papers.')
],chapter='Literature review')
page(5,'2.3 EEG classification and the SVM choice',[
 h('2.3.1 Statistical EEG features'),
 p('Alam et al. describe conventional EEG-band statistics and a linear SVM implemented on an Artix-7 FPGA [2]. Their use of compact descriptors supports examining simple numerical features before selecting a larger model. Our implementation instead obtains descriptors from five Haar subbands. Different filtering, windows and wavelet scaling mean the two feature spaces are not interchangeable.'),
 h('2.3.2 Learned representations'),
 p('Barnes et al. examine an explainable CNN for single-channel EEG and use subject-wise evaluation [3]. Their work illustrates both the availability of apnea-related EEG information and the difficulty of generalisation across recording sources. A CNN is not used in this implementation; the project retains an explicit feature vector and an SVM for inspectable preprocessing and numerical export.'),
 h('2.3.3 Linear and kernel decision functions'),
 p('Support-vector classification constructs a decision boundary from labeled examples [6]. A linear model evaluates an inner product and bias. A kernel model evaluates similarity to support vectors. During development, the current software compared linear and RBF alternatives; the selected artifact uses an RBF kernel, so its runtime computation is not equivalent to a single linear multiply-accumulate chain.'),
 p('This distinction matters for hardware planning. The companion front end describes a proposed linear classifier, while the current software stores 2,181 support vectors. A later hardware classifier must implement the agreed RBF computation, or the project must train and validate a compatible linear model. Software export alone does not complete that hardware design.')
])
page(6,'2.4 Deployment and evaluation lessons',[
 h('2.4.1 Efficient wearable analysis'),
 p('Surrel et al. investigate an online, patient-specific wearable ECG method using compact analysis and SVM classification [4]. It provides a deployment example, but its ECG input and patient-specific setting differ from the present cross-subject EEG task. Its reported performance does not establish the accuracy or power use of our software.'),
 h('2.4.2 Generalisation beyond cross-validation'),
 p('Padovano et al. compare cross-validation with external-dataset evaluation for sleep-apnea classifiers [5]. Their results show that validation within a source can be more optimistic than transfer to other recordings. This motivates preserving subject boundaries and treating a viewed development holdout cautiously.'),
 h('2.4.3 Design implications'),
 bullets('The feature definition and normalization used during training must be reused during inference.',
         'Subject separation is more informative than randomly mixing a person’s windows across training and testing.',
         'A software score does not establish FPGA timing, power or fixed-point accuracy.',
         'Synthetic inputs demonstrate the application but provide no evidence of classification performance.'),
 p('The contribution of this project is a reproducible implementation and evaluation of one explicit post-FPGA interface. It does not claim a new clinical diagnostic method, a superior literature accuracy or a completed hardware classifier. The practical research question is whether the supplied features remain informative when the test person is absent from model fitting.')
])

# Methodology: pages 7–11.
page(7,'3 DATASET AND METHODOLOGY',[
 h('3.1 Dataset selection'),
 p('The EEG model uses MIT-BIH Polysomnographic Database version 1.0.0 [7,8]. The selected subset contains the ten C3–O1 recordings with usable apnea annotations: slp03, slp04, slp14, slp16, slp48, slp59, slp60, slp61, slp66 and slp67x. Keeping one montage avoids mixing electrode arrangements within this model.'),
 p('Record slp45 is omitted because apnea annotations are unavailable. Other excluded records either use another EEG montage or lack the required annotations. Inclusion was based on channel and annotation availability, not on which records produced better classification results. Source-file hashes and selection decisions are retained in the provenance manifest.'),
 fig('Figure 3.1 Included normal and apnea windows by subject.','subject_windows.png',6.35),
 p('The subset yields 3,376 non-overlapping 30-second sleep windows at 250 Hz. The underlying database describes a small male cohort. Results from this subset cannot establish performance across broader age groups, sexes, recording systems or electrode montages.')
],chapter='Dataset and methodology')
page(8,'3.2 Window alignment and ground truth',[
 p('Each sleep annotation in this database describes the following 30 seconds. The preparation script uses its exact sample index rather than assuming that every record begins on a fixed zero-based epoch grid. One window contains 7,500 EEG samples. The first 30 seconds are excluded where otherwise eligible to avoid the initial filter transient; incomplete windows are also omitted.'),
 table('Table 3.1 Label mapping and exclusion rules.',['Condition','Treatment'],[
 ['Sleep stage 1, 2, 3, 4 or R plus OA / X / CA / CAA','Apnea label = 1'],
 ['Known sleep stage without respiratory-event code','Normal label = 0'],
 ['Hypopnea-only H or HA','Excluded'],
 ['Wake W, movement MT, unknown annotation or no sleep stage','Excluded'],
 ['Incomplete or startup window; unusable constant signal','Excluded'],
 ],[3.95,2.35]),
 p('Repeated event codes within the same annotation do not create duplicate windows. An eligible window with an apnea code remains positive. Hypopnea-only windows are not reassigned to normal sleep. Consequently, this classifier has not been evaluated as a comprehensive apnea-plus-hypopnea detector.'),
 h('3.2.1 Subject-wise partition'),
 p('Eight subjects provide 2,644 development windows. The fixed test subjects are slp03 and slp16, contributing 732 windows: 485 normal and 247 apnea. Complete subjects remain together in every split. Stratification uses labels and groups to form viable folds, not the performance of competing models. Duplicate full feature vectors occurring across subject identifiers are rejected for investigation.')
])
page(9,'3.3 The 15-feature interface',[
 p('For each of A5, D5, D4, D3 and D2, the software expects total energy E, Hjorth mobility M and ordinary Pearson kurtosis K. The incoming order is band-major. Named CSV columns can be reordered safely; positional serial values must follow the interface exactly.'),
 table('Table 3.2 Ordered FPGA feature-vector layout.',['Positions','Band','Names'],[
 ['1–3','A5','E_A5, M_A5, K_A5'],['4–6','D5','E_D5, M_D5, K_D5'],
 ['7–9','D4','E_D4, M_D4, K_D4'],['10–12','D3','E_D3, M_D3, K_D3'],
 ['13–15','D2','E_D2, M_D2, K_D2'],
 ],[1.0,.8,4.5]),
 h('3.3.1 Definitions'),
 code('E = Σ c[n]²\nM = √( Var(c[n] − c[n−1]) / Var(c[n]) )\nK = mean((c − mean(c))⁴) / mean((c − mean(c))²)²'),
 p('The variances are population variances. Mobility is computed on each band’s coefficient sequence without an additional sample-rate multiplier. Kurtosis is Pearson kurtosis, for which an ideal Gaussian distribution has value 3; it is neither excess kurtosis nor a bias-corrected estimator. In a zero-variance band, mobility and kurtosis are set to zero. Entire constant training windows are rejected.'),
 p('The contract uses EEG amplitude in µV, so energy is expressed in squared coefficient-amplitude units. Runtime input must be decoded physical feature values with no prior software standardization. The model applies its own fitted preprocessing exactly once.')
])
page(10,'3.4 Offline reference preprocessing',[
 p('Raw EEG processing is an offline training reference. The application itself starts with feature values. The reference converts calibrated EEG to µV, applies a causal band-pass filter continuously over each recording, then decomposes eligible windows with a five-level Haar DWT. Continuous filtering preserves causal state between windows.'),
 table('Table 3.3 Reference feature-generation convention.',['Item','Current software convention'],[
 ['Sample rate and window','250 Hz; 30 seconds / 7,500 samples'],
 ['Band-pass filter','Butterworth prototype order 4; band-pass order 8; 0.5–45 Hz'],
 ['Filter implementation','Causal second-order sections; zero initial state per recording'],
 ['Wavelet and boundary','Orthonormal Haar, five levels, symmetric extension'],
 ['Included coefficients','A5=235, D5=235, D4=469, D3=938, D2=1,875'],
 ['Hardware equivalence','Not yet established for fixed-point RTL output'],
 ],[1.75,4.55]),
 p('Boundary handling affects coefficient counts because 7,500 is not divisible by 32. Symmetric extension is explicitly part of the software reference; a streaming FPGA that truncates or pads differently can produce different energy and higher-order moments. The library decomposition convention is documented in [10].'),
 p('The five DWT outputs are dyadic subbands, not exact conventional EEG rhythms. Their nominal ranges at 250 Hz are approximately 0–3.906, 3.906–7.813, 7.813–15.625, 15.625–31.25 and 31.25–62.5 Hz. The preceding filter attenuates the upper part of D2. D1 is excluded. Filter coefficients and boundary assumptions are saved in the feature contract.')
])
page(11,'3.5 Learning and inference',[
 fig('Figure 3.2 Separation of offline fitting and runtime prediction.','training_inference.png',6.35),
 p('The training pipeline compares identity/log transforms, 5/10/all features, linear/RBF kernels, C, gamma and class weights. Four-fold grouped development CV evaluates 144 candidates using mean balanced accuracy. Transformations, scaling and ANOVA feature selection are fitted inside training folds. Grouped validation is used because a person’s windows are correlated [9].'),
 p('The selected model applies log1p to all 15 values, StandardScaler, a ten-feature selector and an RBF SVM. Its parameters are C=0.1, gamma=0.01 and balanced class weighting. The final saved fit uses development subjects only; the test subjects are not included in an all-data refit.'),
 code('z_j = ( ln(1 + x_j) − μ_j ) / σ_j\nf(z) = Σ_i d_i exp(−γ ||z − s_i||²) + b\nprediction = 1 if f(z) > 0; otherwise 0'),
 p('Here μ and σ are development-derived scaling constants; s_i are the stored support vectors and d_i are their signed coefficients. The current model retains 2,181 support vectors. A positive margin denotes apnea. The score has no probability or clinical-severity interpretation. The artifact stores feature names, selected indices and the numerical parameters needed to reproduce this calculation.')
])
# Software implementation: pages 12–16.
page(12,'4 SOFTWARE IMPLEMENTATION',[
 h('4.1 Architecture'),
 p('The software separates offline data preparation from runtime prediction. prepare_eeg_dataset.py reads the selected recordings and labels, while eeg_apnea.py defines the feature contract, training pipeline and portable inference. The Streamlit interface orchestrates input, classification, result display and session history.'),
 fig('Figure 4.1 Software modules and the saved model artifact.','module_flow.png',6.35),
 table('Table 4.1 Implemented software components.',['Component','Role'],[
 ['prepare_eeg_dataset.py','Download/cache records; align windows; create real training features'],
 ['eeg_apnea.py','Validate features; fit/evaluate SVM; run numerical inference'],
 ['fpga_serial.py','Parse bounded newline-delimited 15-value feature frames'],
 ['app_inputs.py','Create supported lifestyle examples and synthetic sensor previews'],
 ['ui.py','Input forms, separate model results, session history and metrics'],
 ['models/eeg_svm.json','Portable numerical model, feature contract and evaluation'],
 ],[2.1,4.2]),
 p('Each module exposes a specific boundary for testing. This makes failures in a CSV, serial frame, feature definition or classifier distinguishable from failures in the application display.')
],chapter='Software implementation')
page(13,'4.2 Validation and portable inference',[
 p('The validator requires all 15 unique named columns and restores canonical order. It rejects empty or oversized feature sets, missing values, nonnumeric or nonfinite entries, negative descriptors and windows with zero energy in every selected band. Invalid rows are not silently dropped, because doing so would change the relationship between input windows and displayed predictions.'),
 p('The application limits EEG screening uploads to 10 MB and 1,000 windows. The general feature validator supports up to 100,000 rows for training and batch use. File-size and application limits serve different purposes and are not interchangeable.'),
 h('4.2.1 Model representation'),
 p('The JSON artifact stores normalization constants, selected feature indices, kernel parameters, support vectors, signed dual coefficients, bias and development feature bounds. Its schema and array dimensions are checked when loading. It is a numerical representation rather than an executable pickle or joblib object.'),
 table('Table 4.2 Important fields in the saved EEG artifact.',['Field','Meaning'],[
 ['features / selected_indices','15 input names and the retained subset'],
 ['input_transform / mean / scale','Fitted software preprocessing'],
 ['kernel / gamma / support_vectors / dual_coef / intercept','RBF decision function'],
 ['feature_min / feature_max','Development-range warning bounds'],
 ['contract / source / training_data_sha256','Feature convention and source traceability'],
 ['report','Split, metrics, parameters and recorded timings'],
 ],[3.3,3.0]),
 p('Inference computes the kernel matrix in chunks of 128 windows to bound intermediate memory. Before a training artifact is saved, its portable decision scores are compared with the evaluated scikit-learn estimator at tolerance 1e−7. Out-of-range features are flagged; the warning is not a validated detector of every possible recording shift.')
])
page(14,'4.3 Input modes and serial framing',[
 p('EEG parameters can be supplied using a named CSV, manual fields, serial capture or a synthetic demonstration. All four modes enter the same feature validator and predictor. The serial adapter expects one complete feature window per line, with exactly 15 comma-separated ASCII numbers in canonical order; an exact feature-name header is optional.'),
 p('Capture buffers fragmented reads and bounds frame size to 4,096 bytes and each capture to at most 1,000 windows. Nonfinite values, wrong field counts and incomplete final frames are rejected. The proposed adapter has been checked using simulated serial reads; an attached FPGA was not used for the reported tests.'),
 fig('Figure 4.2 Biosignal input controls with synthetic EEG feature preview.','app_eeg_inputs.png',6.3),
 p('The screenshot shows a demonstration configuration. ECG and SpO₂ controls remain available for context, while the EEG preview represents parameter rows rather than a raw waveform. Binary protocols, fixed-point decoding and hardware window identifiers require an agreed physical-interface specification.')
])
page(15,'4.4 Predictions and demonstration data',[
 p('The Add example data button selects a reproducible example, while Random valid values changes the demonstration seed. EEG demonstration rows are generated by perturbing unlabelled development feature profiles. This preserves plausible magnitudes while making the inputs clearly synthetic. No synthetic row contributes to the training cohort or the evaluation counts.'),
 fig('Figure 4.3 EEG window labels displayed for six synthetic demonstration inputs.','app_eeg_predictions.png',6.3),
 p('The results include a sequential window identifier, binary prediction, class name, signed decision margin, development-range warning, input source and synthetic flag. A downloadable CSV preserves source information. The demonstration labels illustrate the software path; they cannot be scored as correct or incorrect because the invented inputs have no ground truth.'),
 p('The displayed number of positive windows is not a count of independently identified breathing events. Windows can contain parts of longer events, and the application does not infer total sleep time or apnea-hypopnea burden. The retained lifestyle result is displayed separately below the EEG section.')
])
page(16,'4.5 History, performance evidence and tests',[
 fig('Figure 4.4 Application performance table for the evaluated EEG model.','app_eeg_performance.png',6.3),
 p('History retains up to 100 summary entries in the active browser session and supports CSV export. The latest screening can display its feature rows and window predictions, while historical summaries omit the large feature and prediction tables. Uploaded recordings are not persisted by the application.'),
 p('The Model and performance tab identifies the dataset, selected features, feature contract, evaluation scores and subject breakdown. Lifestyle-model evidence is shown in a separate section. Its scores refer to a different task and must not be substituted for the EEG model’s scores.'),
 table('Table 4.3 Software verification performed.',['Area','Verified behaviour'],[
 ['Feature contract','Canonical order; missing/duplicate/nonfinite/invalid input rejection'],
 ['Reference processing','Haar descriptors and causal filtering behaviour'],
 ['Training separation','Disjoint subject folds and training-only scaler fitting'],
 ['Portable inference','Agreement with scikit-learn and JSON round trip'],
 ['Serial adapter','Full frames, malformed data and fragmented reads'],
 ['Application','Tabs, demo flow, history, missing model and invalid manual input'],
 ],[1.55,4.75]),
 p('The current automated suite passed 38 tests on 19 November 2026. This verifies defined software behaviour; it does not substitute for classifier validation, electrical serial testing or FPGA numerical comparison.')
])

# Results and discussion: pages 17–22.
page(17,'5 RESULTS AND DISCUSSION',[
 h('5.1 Evaluation conditions'),
 p('All reported classification scores in this chapter describe the saved EEG software model. Model fitting uses eight development subjects. The two fixed test subjects, slp03 and slp16, supply 732 windows. Scaling, feature selection and hyperparameter tuning use grouped development folds. Test data are not included in the final fitted model.'),
 p('The fixed holdout was inspected during implementation. An initial model without logarithmic transformation was examined before the transform comparison; the latter was selected by development CV. The test set is therefore a provisional development benchmark, not an untouched final external cohort. No alternative splits were searched to improve the displayed test score.'),
 table('Table 5.1 Selected model and evaluation configuration.',['Parameter','Recorded value'],[
 ['Classifier','RBF SVM; C=0.1; gamma=0.01; balanced classes'],
 ['Software input','15 descriptors; log1p then StandardScaler'],
 ['Selected feature count','10'],
 ['Development / test windows','2,644 / 732'],
 ['Development / test subjects','8 / 2'],
 ['Grouped development CV','4 folds; 144 candidate configurations'],
 ['Development CV balanced accuracy',f'{R["development_cv_balanced_accuracy"]:.1%} (fold SD {R["development_cv_std"]:.1%})'],
 ['Support vectors',str(R['support_vectors'])],
 ],[2.55,3.75]),
 p('The selected inputs are E_A5, K_A5, E_D5, K_D5, K_D4, M_D3, K_D3, E_D2, M_D2 and K_D2. Selection is applied after preprocessing. Their inclusion does not prove a causal relationship between these statistics and disease; it records the subset preferred by this development procedure.')
],chapter='Results and discussion')
page(18,'5.2 Held-out classification performance',[
 table('Table 5.2 Measured EEG scores on the fixed holdout.',['Metric','Value'],[
 ['Accuracy',f'{R["accuracy"]:.1%}'],['Balanced accuracy',f'{R["balanced_accuracy"]:.1%}'],
 ['Sensitivity / recall',f'{R["sensitivity"]:.1%}'],['Specificity',f'{R["specificity"]:.1%}'],
 ['Precision',f'{R["precision"]:.1%}'],['F1',f'{R["f1"]:.1%}'],
 ['ROC AUC',f'{R["roc_auc"]:.1%}'],['Average precision',f'{R["average_precision"]:.1%}'],
 ['Majority-class baseline accuracy',f'{R["majority_baseline_accuracy"]:.1%}'],
 ],[4.4,1.9]),
 fig('Figure 5.1 Fixed-holdout classification metrics.','measured_metrics.png',6.3),
 p('Accuracy is 542 correct window labels out of 732. The majority-class reference would label all windows normal and obtain 66.3% accuracy, but would detect none of the apnea windows. Balanced accuracy also considers both classes, so it is more informative than accuracy alone in this imbalanced test set.'),
 p('Precision is 58.4%, which means many positive window labels correspond to normal ground-truth windows. Sensitivity is higher at 80.2%, but these pooled values conceal large differences between the two subjects. The model is a moderate research baseline rather than evidence of reliable patient-level screening.')
])
page(19,'5.3 Error analysis and metric definitions',[
 fig('Figure 5.2 Confusion matrix for 732 held-out windows.','confusion_matrix.png',4.3),
 p('Rows represent the annotated class and columns the predicted class. True negatives TN=344 and true positives TP=198 are correct predictions. False positives FP=141 are normal windows flagged as apnea. False negatives FN=49 are apnea windows labeled normal.'),
 code('Accuracy = (TP + TN) / (TP + TN + FP + FN)\nSensitivity = TP / (TP + FN)\nSpecificity = TN / (TN + FP)\nPrecision = TP / (TP + FP)\nF1 = 2TP / (2TP + FP + FN)\nBalanced accuracy = (Sensitivity + Specificity) / 2'),
 p('The model misses 49 of 247 apnea windows and flags 141 of 485 normal windows. Both errors matter: a higher recall can be obtained at the cost of more false alarms, while a more conservative threshold may miss more events. The current decision threshold remains zero; this operating point has not been clinically calibrated.')
])
page(20,'5.4 Variation between test subjects',[
 table('Table 5.3 Subject-level fixed-holdout results.',['Subject','Windows','Accuracy','Sensitivity','Specificity'],[
 [v['subject'],str(v['windows']),f'{v["accuracy"]:.1%}',f'{v["sensitivity"]:.1%}',f'{v["specificity"]:.1%}']
 for v in R['holdout_subject_metrics']
 ],[1.0,.85,1.2,1.55,1.7]),
 fig('Figure 5.3 Accuracy, sensitivity and specificity for each test subject.','subject_performance.png',6.3),
 p('For slp03, only 6 of 44 apnea windows are detected: sensitivity is 13.6%. Its apparent 78.0% accuracy is influenced by the 343 normal windows. For slp16, sensitivity is 94.6%, but specificity is only 33.8%; 94 of 142 normal windows are flagged. The two patterns are very different despite being pooled into the same overall score.'),
 p('This variation is a central limitation, not a detail that can be resolved by quoting a single accuracy. A larger cohort is needed to distinguish person-specific effects from recording differences and to estimate performance variability. Windows from one person are correlated; treating all 732 as independent patients would overstate the strength of the evidence.')
])
page(21,'5.5 Score behaviour and computation time',[
 fig('Figure 5.4 ROC and precision–recall curves for the recorded holdout scores.','roc_precision_recall.png',6.3),
 p('The ROC AUC of 0.828 summarizes score ranking across thresholds. Average precision is 0.657. The dashed horizontal line in the precision–recall plot indicates the test prevalence, 247/732, as a reference. These curves are reconstructed from the saved numerical model and the fixed test rows; their areas were checked against the recorded artifact.'),
 p('Threshold-independent scores do not guarantee useful precision and sensitivity at a selected operating point. The score is a signed decision margin and has not been calibrated to disease probability. The curves also inherit the small-cohort and inspected-holdout limitations.'),
 table('Table 5.4 Recorded software computation measurements.',['Measurement','Value / scope'],[
 ['Median single-window prediction',f'{R["single_window_latency_ms_median"]:.2f} ms'],
 ['95th-percentile prediction',f'{R["single_window_latency_ms_p95"]:.2f} ms'],
 ['Timing repetitions','30 single-window calls'],
 ['Training search time',f'{R["training_seconds"]:.1f} seconds on the training host'],
 ['Included in inference timing','Validation, transform, scaling, selection and numerical SVM'],
 ['Excluded','Raw EEG processing, FPGA computation, serial waiting, CSV parsing and UI rendering'],
 ],[2.45,3.85]),
 p('The host configuration was not stored with the original timing record. These values should be read as host-specific software measurements rather than a portable latency guarantee or a hardware throughput claim.')
])
page(22,'5.6 Further experiments, integration and conclusion',[
 p('Additional development-only experiments tested 144 preprocessing/SVM configurations and 108 subject/class-balanced configurations. The best mean grouped-CV balanced accuracy was 62.8%, compared with 62.5% for the retained model. A threshold experiment also traded recall against specificity and precision. These small development gains did not establish a reliable improvement, so the deployed artifact was retained.'),
 h('5.6.1 Hardware compatibility'),
 p('The Group 1 report [12] proposes 20 Mean/MAV/Energy/Variance descriptors and a linear SVM. The current Group 2 interface accepts 15 Energy/Mobility/Kurtosis values and the selected model is RBF. Mobility and kurtosis cannot generally be recovered from only the proposed simpler statistics. The teams must either agree on EMK hardware output or retrain the classifier on a shared feature set.'),
 p('There are also differences in the filter family, Haar normalization and the proposed normalization stage. Group 1 uses Chebyshev-II and division-by-two Haar arithmetic, while the training reference uses Butterworth SOS and orthonormal Haar. The software expects raw physical features and normalizes once. Fixed-point rounding, overflow and frame boundaries must be tested before a board-level classification claim.'),
 h('5.6.2 Future work and conclusion'),
 p('The next work is to settle the shared feature protocol, compare hardware values against the reference, expand the labeled cohort and evaluate new independent subjects. Signal-quality checks, temporal context and a jointly labeled multimodal dataset are possible later extensions. A hardware RBF implementation or a separately validated linear model is required for SVM deployment on FPGA.'),
 p('Group 2 has completed a working feature-to-label software path, a trained reference SVM, a demonstration interface and reproducible software evaluation. The measured results support further research; they do not establish clinical diagnosis or completed end-to-end FPGA validation.')
])

# References and appendices: body pages 23–26.
REFERENCES=[
 ('[1]','National Heart, Lung, and Blood Institute. “Sleep Apnea.” National Institutes of Health. Accessed 19 November 2026.','https://www.nhlbi.nih.gov/health/sleep-apnea'),
 ('[2]','M. S. Alam, Y. Siddiqui, M. Hasan, O. Farooq and Y. Himeur. “Energy-Efficient FPGA Based Sleep Apnea Detection Using EEG Signals.” IEEE Access, vol. 12, pp. 40182–40195, 2024.','https://doi.org/10.1109/ACCESS.2024.3374223'),
 ('[3]','L. D. Barnes, K. Lee, A. W. Kempa-Liehr and L. E. Hallum. “Detection of sleep apnea from single-channel electroencephalogram (EEG) using an explainable convolutional neural network (CNN).” PLOS ONE, vol. 17, no. 9, e0272167, 2022.','https://doi.org/10.1371/journal.pone.0272167'),
 ('[4]','G. Surrel, A. Aminifar, F. Rincon, S. Murali and D. Atienza. “Online Obstructive Sleep Apnea Detection on Medical Wearable Sensors.” IEEE Transactions on Biomedical Circuits and Systems, vol. 12, no. 4, pp. 762–773, 2018.','https://doi.org/10.1109/TBCAS.2018.2824659'),
 ('[5]','D. Padovano, A. Martinez-Rodrigo, J. M. Pastor, J. J. Rieta and R. Alcaraz. “Hidden Hazards Beneath Cross-Validation Methods in Machine Learning-Based Sleep Apnea Detection.” Computing in Cardiology, vol. 49, 2022.','https://doi.org/10.22489/CinC.2022.086'),
 ('[6]','C. Cortes and V. Vapnik. “Support-vector networks.” Machine Learning, vol. 20, pp. 273–297, 1995.','https://doi.org/10.1007/BF00994018'),
 ('[7]','G. Moody. MIT-BIH Polysomnographic Database, version 1.0.0. PhysioNet, 1999.','https://doi.org/10.13026/C23K5S'),
 ('[8]','Y. Ichimaru and G. B. Moody. “Development of the polysomnographic database on CD-ROM.” Psychiatry and Clinical Neurosciences, vol. 53, pp. 175–177, 1999.','https://physionet.org/content/slpdb/1.0.0/'),
 ('[9]','scikit-learn developers. “Cross-validation: evaluating estimator performance.” Official documentation, grouped-data section. Accessed 19 November 2026.','https://scikit-learn.org/stable/modules/cross_validation.html'),
 ('[10]','PyWavelets developers. “Discrete Wavelet Transform (DWT).” Official documentation. Accessed 19 November 2026.','https://pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-wavelet-transform.html'),
 ('[11]','S. M. George. “Sleep Apnea Detection Using Wearable Sensors: Principles, Methods, and Challenges.” Chapter in Advances in Wearables and Nearables for Healthcare Applications. CRC Press. Publisher chapter listing; background reading.','https://www.routledge.com/Advances-in-Wearables-and-Nearables-for-Healthcare-Applications/Naik/p/book/9781032796543'),
 ('[12]','Group 1. Hardware Accelerator for Support Vector Machine on FPGA for Sleep Apnea Detection. Internal final-year project report, Department of Electronics and Communication Engineering, National Institute of Technology Agartala, November 2026.',''),
]
page(23,'REFERENCES',[( 'ref',)+item for item in REFERENCES[:6]])
page(24,'REFERENCES (continued)',[( 'ref',)+item for item in REFERENCES[6:]]+[
 h('Dataset attribution'),
 p('The source database is distributed under the Open Data Commons Attribution License v1.0. Recorded example features, model support vectors and demonstration anchors are derived from MIT-BIH EEG. Dataset attribution is retained with the numerical model provenance and the project documentation. No source-paper figure is presented as an experimental result of Group 2.')
])
page(25,'APPENDIX A: INPUT AND FEATURE CONTRACT',[
 h('A.1 Runtime frame'),
 p('One serial line carries one window. The optional header uses the exact feature names below. A newline terminates every frame. Values are decoded physical features, with no added standardization or concealed fixed-point scale.'),
 code('E_A5, M_A5, K_A5, E_D5, M_D5, K_D5,\nE_D4, M_D4, K_D4, E_D3, M_D3, K_D3,\nE_D2, M_D2, K_D2'),
 p('The following frame is illustrative only and has no apnea label. It shows the transport order, not measured patient values:'),
 code('125.4,0.42,3.1,87.2,0.61,2.8,64.8,0.75,3.5,\n112.6,0.93,4.2,38.5,1.12,3.0'),
 p('In actual transport, all 15 numbers are on one line. The line breaks above are for page readability. A binary or indexed feature_valid / feature_index / feature_value hardware stream needs a matching decoder before this CSV-line parser can consume it.'),
 h('A.2 Training rows'),
 p('A labeled training CSV requires all 15 named numerical features, subject_id and label. Subject identifiers must consistently refer to people rather than individual files. The source column can retain provenance. Explicit synthetic/demo markers are rejected; arbitrary fabricated data cannot be identified automatically.'),
 h('A.3 Traceability'),
 p('The saved contract records units, coefficients, Haar boundary handling and feature definitions. The model includes a training-data digest and the dataset provenance stores hashes of header, annotation and waveform files. These support checking whether a regenerated feature table corresponds to the intended source records.')
])
page(26,'APPENDIX B: REPRODUCIBILITY',[
 ('link','Project repository: ','https://github.com/sarbojitrana/sleep-apnea-detector-eeg'),
 h('B.1 Run the application'),
 code('python -m venv .venv\nsource .venv/bin/activate\npython -m pip install -r requirements.txt\nstreamlit run ui.py'),
 h('B.2 Rebuild real EEG training features'),
 code('python -m pip install -r requirements-training.txt\npython prepare_eeg_dataset.py --download'),
 p('The downloader caches the selected source records under data/raw/slpdb. The preparation script writes data/processed/slpdb/eeg_features.csv, its feature contract and the provenance manifest. Raw and processed recordings are excluded from normal version-control tracking.'),
 h('B.3 Train and evaluate'),
 code('OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python eeg_apnea.py \\\n  --data data/processed/slpdb/eeg_features.csv \\\n  --contract data/processed/slpdb/feature_contract.json \\\n  --source https://physionet.org/content/slpdb/1.0.0/'),
 h('B.4 Verify and investigate alternatives'),
 code('python -m pytest -q\nOPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python optimize_eeg_svm.py'),
 p('The test suite checks software behaviour. The optimization script uses only the stored development subjects and saves a separate audit without replacing the deployed model. Retraining timings depend on the host. Runtime inference requires the bundled numerical artifact but does not require WFDB or a raw recording download.')
])

# Document formatting and front matter.
STYLE=DOC.styles['Normal'];STYLE.font.name='Times New Roman';STYLE.font.size=Pt(12)
STYLE.paragraph_format.line_spacing=1.25
STYLE.paragraph_format.space_after=Pt(7)
STYLE.paragraph_format.widow_control=True
for name,size in [('Title',18),('Heading 1',16),('Heading 2',13),('Heading 3',12)]:
    st=DOC.styles[name];st.font.name='Times New Roman';st.font.size=Pt(size);st.font.bold=True
    st.font.color.rgb=RGBColor(0,0,0)
    st.paragraph_format.space_before=Pt(8);st.paragraph_format.space_after=Pt(8)
    st.paragraph_format.keep_with_next=True
DOC.styles['Caption'].font.name='Times New Roman';DOC.styles['Caption'].font.size=Pt(10.5)
DOC.styles['Caption'].font.italic=True
DOC.styles['Caption'].paragraph_format.space_after=Pt(8)
DOC.styles['Caption'].paragraph_format.line_spacing=1.1
for s in DOC.sections:
    s.page_width=Inches(8.5);s.page_height=Inches(11)
    s.top_margin=Inches(.85);s.bottom_margin=Inches(.80)
    s.left_margin=s.right_margin=Inches(1)
    s.header_distance=Inches(.36);s.footer_distance=Inches(.36)


def normal(text,align=WD_ALIGN_PARAGRAPH.JUSTIFY,size=None,bold=False,after=None):
    para=DOC.add_paragraph();para.alignment=align
    run=para.add_run(text);run.bold=bold
    if size:run.font.size=Pt(size)
    if after is not None:para.paragraph_format.space_after=Pt(after)
    return para


def center(text,size=14,bold=False,after=10):
    return normal(text,WD_ALIGN_PARAGRAPH.CENTER,size,bold,after)


def heading(text,level=1,newpage=False):
    para=DOC.add_paragraph(text,style=f'Heading {level}')
    para.paragraph_format.page_break_before=newpage
    return para


def front_title(text,newpage=True):
    para=heading(text,1,newpage);para.alignment=WD_ALIGN_PARAGRAPH.CENTER
    para.paragraph_format.space_after=Pt(20)
    return para


def field(para,instruction):
    run=para.add_run();begin=OxmlElement('w:fldChar');begin.set(qn('w:fldCharType'),'begin')
    command=OxmlElement('w:instrText');command.set(qn('xml:space'),'preserve');command.text=' '+instruction+' '
    end=OxmlElement('w:fldChar');end.set(qn('w:fldCharType'),'end')
    run._r.append(begin);run._r.append(command);run._r.append(end)


def number_section(sec,fmt=None):
    sec.footer.is_linked_to_previous=False
    if fmt:
        num=OxmlElement('w:pgNumType');num.set(qn('w:fmt'),fmt);num.set(qn('w:start'),'1');sec._sectPr.append(num)
        para=sec.footer.paragraphs[0];para.alignment=WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.space_after=Pt(0)
        field(para,'PAGE')
        for run in para.runs:run.font.name='Times New Roman';run.font.size=Pt(10)


def format_table(t,widths=None):
    t.autofit=False;t.style='Table Grid'
    if widths:
        for col,width in zip(t.columns,widths):col.width=Inches(width)
        for row in t.rows:
            for c,width in zip(row.cells,widths):c.width=Inches(width)
    for ri,row in enumerate(t.rows):
        for cell in row.cells:
            tcpr=cell._tc.get_or_add_tcPr()
            margins=OxmlElement('w:tcMar')
            for key in ('top','bottom'):
                el=OxmlElement('w:'+key);el.set(qn('w:w'),'45');el.set(qn('w:type'),'dxa');margins.append(el)
            tcpr.append(margins)
            if ri==0:
                shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'E8EDF3');tcpr.append(shade)
            for para in cell.paragraphs:
                para.paragraph_format.line_spacing=1.06
                para.paragraph_format.space_after=Pt(2)
                para.paragraph_format.space_before=Pt(1)
                for run in para.runs:
                    run.font.name='Times New Roman';run.font.size=Pt(10.5);run.bold=(ri==0)
        trpr=row._tr.get_or_add_trPr()
        no_split=OxmlElement('w:cantSplit');trpr.append(no_split)
        if ri==0:
            repeat=OxmlElement('w:tblHeader');trpr.append(repeat)


def make_table(caption,headers,rows,widths):
    cap=DOC.add_paragraph(caption,style='Caption');cap.paragraph_format.keep_with_next=True
    tab=DOC.add_table(rows=1,cols=len(headers))
    for cell,value in zip(tab.rows[0].cells,headers):cell.text=str(value)
    for row in rows:
        cells=tab.add_row().cells
        for cell,value in zip(cells,row):cell.text=str(value)
    format_table(tab,widths)
    para=DOC.add_paragraph();para.paragraph_format.space_after=Pt(2);para.paragraph_format.line_spacing=.3
    return tab


def image(caption,name,width):
    path=ASSETS/name
    if not path.exists():path=ROOT/'presentations/assets'/name
    para=DOC.add_paragraph();para.alignment=WD_ALIGN_PARAGRAPH.CENTER
    para.paragraph_format.space_after=Pt(3)
    para.paragraph_format.keep_with_next=True
    para.add_run().add_picture(str(path),width=Inches(width))
    cap=DOC.add_paragraph(caption,style='Caption');cap.alignment=WD_ALIGN_PARAGRAPH.CENTER


def codeblock(value):
    para=DOC.add_paragraph();para.paragraph_format.line_spacing=1.05
    para.paragraph_format.space_before=Pt(4);para.paragraph_format.space_after=Pt(8)
    for i,line in enumerate(value.split('\n')):
        if i:para.add_run().add_break()
        run=para.add_run(line);run.font.name='DejaVu Sans Mono';run.font.size=Pt(9)
    shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'F3F3F3');para._p.get_or_add_pPr().append(shade)


def link(para,text,url):
    rel=para.part.relate_to(url,'http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink',is_external=True)
    hyperlink=OxmlElement('w:hyperlink');hyperlink.set(qn('r:id'),rel)
    run=OxmlElement('w:r');props=OxmlElement('w:rPr')
    font=OxmlElement('w:rFonts');font.set(qn('w:ascii'),'Times New Roman');font.set(qn('w:hAnsi'),'Times New Roman');props.append(font)
    size=OxmlElement('w:sz');size.set(qn('w:val'),'21');props.append(size)
    color=OxmlElement('w:color');color.set(qn('w:val'),'1F4E79');props.append(color)
    run.append(props);tx=OxmlElement('w:t');tx.text=text;run.append(tx);hyperlink.append(run);para._p.append(hyperlink)


def toc_line(label,number,bold=False):
    para=DOC.add_paragraph();para.paragraph_format.line_spacing=1.12
    para.paragraph_format.space_after=Pt(5)
    para.paragraph_format.tab_stops.add_tab_stop(Inches(6.28),WD_TAB_ALIGNMENT.RIGHT,WD_TAB_LEADER.DOTS)
    run=para.add_run(label+'\t'+str(number));run.font.size=Pt(11);run.bold=bold


# Cover (no number).
number_section(DOC.sections[0])
center(TITLE,20,True,12)
center(SUBTITLE,14,False,12)
center('Final Year Project Report · Group 2',14,True,12)
logo=ASSETS/'nit_agartala_logo.png'
par=DOC.add_paragraph();par.alignment=WD_ALIGN_PARAGRAPH.CENTER
par.add_run().add_picture(str(logo),height=Inches(1.45))
center('Submitted by',12,False,9)
for name,roll in MEMBERS:center(f'{name} ({roll})',14,True,7)
center('Under the supervision of',12,False,15)
center('Dr. Biman Debbarma',14,True,8)
center('DEPARTMENT OF ELECTRONICS AND\nCOMMUNICATION ENGINEERING',13,True,10)
center('NATIONAL INSTITUTE OF TECHNOLOGY AGARTALA\nAGARTALA – 799046, INDIA',13,True,12)
center('November 2026',13,False,0)

# Roman-numbered preliminary pages I–IX.
sec=DOC.add_section(WD_SECTION_START.NEW_PAGE);number_section(sec,'upperRoman')
front_title(TITLE,False)
center(SUBTITLE,13,False,18)
center('Project report submitted in partial fulfilment of the requirements\nfor the degree of',12,False,14)
center('Bachelor of Technology (B.Tech)\nin\nElectronics and Communication Engineering',14,True,14)
center('by',12,False,7)
for name,roll in MEMBERS:center(f'{name} ({roll})',13,True,5)
center('Group 2',13,True,12)
center('Under the supervision of\nDr. Biman Debbarma\nAssistant Professor',12,False,16)
center('Department of Electronics and Communication Engineering\nNational Institute of Technology Agartala',12,True,12)
center('November 2026',12,False,0)

front_title('APPROVAL SHEET')
normal(f'The project report entitled “Sleep Apnea Detection and Prediction: {SUBTITLE}” is submitted by the following Group 2 students for evaluation as part of the Bachelor of Technology programme in Electronics and Communication Engineering.')
for name,roll in MEMBERS:normal(f'{name} — {roll}',WD_ALIGN_PARAGRAPH.LEFT,12,True)
normal('Examiners',WD_ALIGN_PARAGRAPH.LEFT,13,True,18)
for _ in range(3):normal('Name: _________________________     Signature: _________________________',WD_ALIGN_PARAGRAPH.LEFT,11,False,17)
normal('Supervisor: Dr. Biman Debbarma',WD_ALIGN_PARAGRAPH.LEFT,12,True,16)
normal('Signature: ______________________________',WD_ALIGN_PARAGRAPH.LEFT,12,False,22)
normal('Chairperson: ___________________________     Signature: ___________________',WD_ALIGN_PARAGRAPH.LEFT,11,False,22)
normal('Decision: ___________________________________________________________',WD_ALIGN_PARAGRAPH.LEFT,11,False,18)
normal('Date: __________________     Place: ECE Department, NIT Agartala',WD_ALIGN_PARAGRAPH.LEFT,11)

front_title('DECLARATION')
normal('We submit this report as a record of the Group 2 work on EEG feature classification, SVM training, software inference and the associated application. The report identifies the source of the recorded EEG data and cites the research and documentation used in developing the method.')
normal('The experimental results describe the evaluated software reference model. Work implemented by the companion hardware group, planned extensions and unresolved integration requirements are distinguished from the software completed by Group 2. Synthetic demonstration inputs are identified separately from labeled training and test data.')
normal('The undersigned take responsibility for reviewing the accuracy of the submitted report and for retaining the stated sources and experimental limitations.')
for name,roll in MEMBERS:
    normal('_______________________________',WD_ALIGN_PARAGRAPH.LEFT,12,False,2)
    normal(f'{name}  |  Enrollment No.: {roll}',WD_ALIGN_PARAGRAPH.LEFT,12,True,14)
normal('Date: __________________     Place: NIT Agartala',WD_ALIGN_PARAGRAPH.LEFT,11)

front_title('CERTIFICATE')
normal(f'The project report “Sleep Apnea Detection and Prediction: {SUBTITLE}” is submitted by Satyam Mondal (23UEC034), Sarbojit Rana (23UEC038), Suraj Verma (23UEC046) and Ramavath Rajesh (23UEC103), Group 2, for evaluation in the final-year Bachelor of Technology programme at the National Institute of Technology Agartala.')
normal('The work documented in this report concerns the software stage following the proposed FPGA feature interface. It includes real EEG dataset preparation, feature validation, SVM model selection and numerical inference, application integration and software evaluation.')
normal('The report is forwarded for examination in the Department of Electronics and Communication Engineering.')
normal('\n\n',WD_ALIGN_PARAGRAPH.LEFT,12,False,10)
normal('_________________________________',WD_ALIGN_PARAGRAPH.RIGHT,12,False,5)
normal('Dr. Biman Debbarma',WD_ALIGN_PARAGRAPH.RIGHT,13,True,5)
normal('Assistant Professor\nDepartment of Electronics and Communication Engineering\nNational Institute of Technology Agartala',WD_ALIGN_PARAGRAPH.RIGHT,12,False,18)
normal('Date: __________________',WD_ALIGN_PARAGRAPH.RIGHT,11)

front_title('TABLE OF CONTENTS')
for label,num in [('Approval sheet','II'),('Declaration','III'),('Certificate','IV'),('List of figures','VI'),('List of tables','VII'),('List of abbreviations','VIII'),('Abstract','IX')]:toc_line(label,num)
for data in PAGES:
    if data['title']=='REFERENCES (continued)':
        continue
    bold=bool(data['chapter']) or data['title'].startswith(('REFERENCES','APPENDIX'))
    label=data['title'] if bold else '    '+data['title']
    toc_line(label,data['number'],bold)

front_title('LIST OF FIGURES')
for caption,num in FIGURES:toc_line(caption,num)
front_title('LIST OF TABLES')
for caption,num in TABLES:toc_line(caption,num)
front_title('LIST OF ABBREVIATIONS')
rows=[('AHI','Apnea–Hypopnea Index'),('AP','Average Precision'),('A5','Level-five approximation coefficients'),('CNN','Convolutional Neural Network'),('CSV','Comma-Separated Values'),('CV','Cross-Validation'),('D2–D5','Detail coefficient streams at levels two to five'),('DWT','Discrete Wavelet Transform'),('ECG','Electrocardiogram'),('EEG','Electroencephalogram'),('EMK','Energy, Mobility and Kurtosis'),('FPGA','Field-Programmable Gate Array'),('IIR','Infinite Impulse Response'),('JSON','JavaScript Object Notation'),('MAV','Mean Absolute Value'),('PSG','Polysomnography'),('RBF','Radial Basis Function'),('ROC AUC','Area Under the Receiver Operating Characteristic Curve'),('RTL','Register-Transfer Level'),('SOS','Second-Order Sections'),('SpO₂','Peripheral Oxygen Saturation'),('SVM','Support Vector Machine')]
make_table('', ['Abbreviation','Meaning'],rows,[1.35,4.95])
front_title('ABSTRACT')
normal('This report presents the Group 2 software work for sleep-apnea window classification following an FPGA EEG feature interface. The software accepts fifteen numerical values: energy, Hjorth mobility and Pearson kurtosis from Haar subbands A5, D5, D4, D3 and D2. It validates the input, applies training-derived preprocessing and feature selection, and evaluates a support vector machine. A Streamlit application supports CSV, manual and serial feature input, synthetic demonstrations, separate EEG results, session history and performance evidence.')
normal('The reference model was developed from 3,376 real EEG sleep windows from ten subjects in the MIT-BIH Polysomnographic Database. Eight subjects supply 2,644 development windows and two subjects supply 732 fixed-holdout windows. Grouped cross-validation selected a log transform, standard scaling, ten features and a class-balanced RBF SVM. The evaluated artifact achieves 74.0% accuracy, 80.2% sensitivity, 70.9% specificity, 58.4% precision and an F1 score of 67.6%. Median feature-to-prediction time was approximately 5 ms on the training host. The current software test suite passes 38 tests.')
normal('These results are provisional: the cohort is small, the holdout was viewed during development and performance varies sharply between the two test subjects. Hardware numerical equivalence is also incomplete. The companion front end proposes different feature statistics, filter and wavelet scaling, and a linear hardware classifier, whereas the present software uses EMK features and an RBF model. The completed contribution is a reproducible feature-to-label software system; common feature definitions, physical FPGA integration and independent evaluation remain future work.')
normal('Keywords: EEG, FPGA interface, sleep apnea, feature extraction, support vector machine, subject-wise validation.',WD_ALIGN_PARAGRAPH.LEFT,11,True)

# Arabic-numbered report body.
sec=DOC.add_section(WD_SECTION_START.NEW_PAGE);number_section(sec,'decimal')
for index,pagedata in enumerate(PAGES):
    heading(pagedata['title'],1,index>0)
    for block in pagedata['blocks']:
        kind=block[0]
        if kind=='p':normal(block[1])
        elif kind=='link':
            para=normal(block[1],size=11)
            link(para,block[2],block[2])
        elif kind=='h':heading(block[1],2)
        elif kind=='bullets':
            for item in block[1]:
                para=DOC.add_paragraph(item,style='List Bullet');para.paragraph_format.space_after=Pt(6);para.paragraph_format.line_spacing=1.2
        elif kind=='table':make_table(*block[1:])
        elif kind=='fig':image(*block[1:])
        elif kind=='code':codeblock(block[1])
        elif kind=='ref':
            para=normal(block[1]+' '+block[2],size=11,after=5)
            para.paragraph_format.line_spacing=1.15
            if block[3]:
                para=DOC.add_paragraph();para.paragraph_format.space_after=Pt(14);para.paragraph_format.line_spacing=1.0
                link(para,block[3],block[3])

properties=DOC.core_properties
properties.title='Sleep Apnea Detection and Prediction — Group 2 Project Report'
properties.subject=SUBTITLE
properties.author='Group 2'
properties.comments='Project report on EEG feature classification, software implementation and evaluation.'
properties.keywords='EEG, FPGA, SVM, sleep apnea, Group 2'
properties.created=properties.modified=datetime.now(timezone.utc)
properties.last_modified_by='';properties.revision=1
path=OUT/'Group_2_Project_Report.docx';DOC.save(path)
# Remove optional empty-template tool fields; leave real dates and report identity.
with ZipFile(path) as z:parts=[(info,z.read(info.filename)) for info in z.infolist()]
with ZipFile(path,'w') as z:
    for info,data in parts:
        if info.filename=='docProps/app.xml':
            root=etree.fromstring(data);ns='{http://schemas.openxmlformats.org/officeDocument/2006/extended-properties}'
            for tag in ('Application','AppVersion','Template','Company','TotalTime'):
                for el in root.findall(ns+tag):root.remove(el)
            data=etree.tostring(root,encoding='UTF-8',xml_declaration=True,standalone=True)
        z.writestr(info,data)
manifest={'front_pages':10,'body_pages':26,'expected_total_pages':36,
          'members':MEMBERS,'body_page_headings':{str(v['number']):v['title'] for v in PAGES},
          'figures':FIGURES,'tables':TABLES,'model_source':MODEL['source'],
          'model_created_at':MODEL['created_at'],'report_generated_at':datetime.now(timezone.utc).isoformat()}
(OUT/'report_assets/report_manifest.json').write_text(json.dumps(manifest,indent=2))
print(path)
print('Planned pages: 10 preliminary + 26 body = 36. Figures:',len(FIGURES),'Tables:',len(TABLES))
