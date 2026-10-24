# ml-based-drone-propeller-fault-detection

Drone Propeller Fault Detection Using Vibration Signal Processing, Feature Engineering, and Gaussian Support Vector Machines (SVM)

The project detects healthy and faulty drone propeller conditions using vibration measurements from the DronePropB Ground Testing Dataset.

Vibration signals from two accelerometer channels (C1 and C2) were segmented into non-overlapping 2-second windows. Eight features were extracted: Crest Factor, Kurtosis, Low Peak Frequency, and Autocorrelation Peak from each channel.

Feature engineering and exploratory analysis were conducted to evaluate healthy-recording consistency, fault separability, and feature redundancy. A compact feature set was selected based on signal characteristics and experimental model comparisons.

A Gaussian SVM classifier was developed using combined measurements from three operating speeds (SP1, SP2, and SP3). Model performance was evaluated using 3-fold recording-wise cross-validation, ensuring that windows from the same physical recording were not shared between training and testing.

### Key Results

- Processed 36 physical recording conditions across three operating speeds using two accelerometer channels.
- Extracted 1,080 non-overlapping 2-second windows and eight selected features.
- Analyzed feature correlations, healthy-recording variability, and healthy–faulty separability.
- Selected Crest Factor, Kurtosis, Low Peak Frequency, and Autocorrelation Peak from each vibration channel.
- Implemented a single Gaussian SVM combining SP1, SP2, and SP3 data.
- Applied recording-wise 3-fold cross-validation to prevent windows from the same recording appearing in both training and test sets.
- Achieved **100% recording-level accuracy (36/36)** with zero false positives and zero false negatives.
- Achieved **95.46% window-level accuracy (1,031/1,080)** and **95.12% balanced accuracy**.
- Achieved window-level accuracies of **98.06% for SP1**, **93.06% for SP2**, and **95.28% for SP3**.
- Saved the trained Gaussian SVM model and exported recording-level predictions and performance metrics.

### Result Snapshots

[Confusion Matrix](docs/results/confusion_matrix.png): Recording-level classification results across SP1, SP2, and SP3, showing 9 correctly classified healthy recordings and 27 correctly classified faulty recordings.

[Feature Quality](docs/results/feature_quality.png): Comparison of selected and rejected features based on healthy-recording variability and the proportion of faulty recordings outside the healthy range.

[Feature Correlation](docs/results/feature_correlation.png): Correlation analysis using within-speed standardized recording-level feature medians.

[RMS vs Low Peak Frequency](docs/results/rms_vs_low_peak.png): Comparison of healthy and faulty feature distributions across all three operating speeds and both accelerometer channels.

### Project Files

`extract_features.py`: Extracts vibration features from the raw MATLAB recordings.
- `visualize_features.py`: Generates feature analysis and justification figures.
- `train_evaluate.py`: Trains and evaluates the Gaussian SVM using recording-wise cross-validation.
- `model_training/features.csv`: Generated feature dataset used for model training.
- `model_training/gaussian_svm.joblib`: Trained model saved for future use.
- `results/all_window_features.csv`: Generated feature dataset for exploratory analysis.
- `docs/results/`: Confusion matrix, feature analysis figures, and evaluation tables.
- `requirements.txt`: Python dependencies.

**Keywords:** Drone Propeller Fault Detection, Vibration Analysis, Signal Processing, Feature Engineering, Machine Learning, Gaussian SVM, Support Vector Machine, Fault Classification, Condition Monitoring, Autocorrelation, Spectral Analysis, Python.

**Development Environment:** Python 3.11.7, Visual Studio Code, NumPy, Pandas, SciPy, Scikit-learn, Matplotlib.

**Data Source:** DronePropB Ground Testing Dataset, C1 and C2 vibration measurements across SP1, SP2, and SP3 operating conditions.
