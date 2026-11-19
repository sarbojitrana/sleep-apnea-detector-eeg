# Group 2 project report

The delivered files are:

- `Group_2_Project_Report.docx` — editable Word report.
- `Group_2_Project_Report.pdf` — checked 36-page PDF, with Roman-numbered front matter and Arabic-numbered body pages.

Running headers are omitted; chapter titles and page-number footers are retained.

Appendix B includes the clickable project repository link: https://github.com/sarbojitrana/sleep-apnea-detector-eeg.

The title, Group 2 label, supervisor, four names and enrollment numbers follow the current project presentation. Approval, certificate and declaration signatures/dates remain blank. No approval or signature has been fabricated.

The report contains five chapters, 11 figures, 12 numbered tables, references and two reproduction/input appendices. It documents the actual classifier, separate lifestyle/EEG paths, software tests, measured results and incomplete hardware agreement. It distinguishes the Group 1 hardware proposal from Group 2's current 15-feature RBF implementation. No arbitrary accuracy target is presented as a project requirement.

Metrics come from `models/eeg_svm.json`; data selection and counts come from `models/eeg_dataset_provenance.json`. ROC and precision–recall plots were reconstructed from the saved artifact and held-out feature rows, with numerical checks against the recorded AUC and average precision. The existing software test suite passed 38 tests on 19 November 2026.

The report's screenshots and logo are copied into `report_assets/` so rebuilding the report does not require keeping the presentation images in place. The formal front matter is unsigned, and the contents/list page numbers were checked against the delivered PDF. Review page references after making structural edits to the Word document.

## Regenerate the documents

From the project root:

```bash
python -m pip install -r reports/requirements.txt
python reports/build_group_2_report.py
libreoffice --headless --convert-to pdf --outdir reports reports/Group_2_Project_Report.docx
```

The builder uses the saved model, provenance and cached figure assets. It does not retrain or modify the EEG model, application or presentation.

To regenerate the scientific plots as well, first prepare the real training feature CSV using the application's documented dataset workflow, then run:

```bash
OPENBLAS_NUM_THREADS=1 python reports/make_report_figures.py
```

`report_assets/report_manifest.json` records the expected pagination and figure/table page mapping.
