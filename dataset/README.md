# Dataset

## Sepsis ICU Dataset

This project uses the `Sepsis_data.csv` dataset containing longitudinal ICU patient records and clinical measurements.

The dataset is used for exploratory clinical data analysis, dimensionality reduction, unsupervised patient phenotyping, anomaly detection, and cluster-specific regularized regression.

### Download the Dataset

The complete dataset is hosted externally because of its large file size.

**[Download Sepsis_data.csv from Google Drive]: https://drive.google.com/drive/folders/1r_4bn_bTI5HzIutuGJi-k-9n_oZtlBjt?usp=sharing**

### Instructions

1. Download `Sepsis_data.csv` using the link above.
2. Place the downloaded file in the project directory.
3. Run the analysis using the command below:

```bash
python Code.py --csv Sepsis_data.csv --id_col Patient_ID --target_col ICULOS
```

### Dataset Notes

- File name: `Sepsis_data.csv`
- Format: CSV
- Data type: Longitudinal ICU clinical records
- Main identifier: `Patient_ID`
- Clinical variables include physiological measurements, laboratory values, ICU length of stay, and sepsis labels.

Please refer to the main project README and the project report for details about preprocessing, modelling methodology, limitations, and reproducibility.

**Data usage:** Please follow the original dataset's licence, access conditions, and attribution requirements.
