# AI-Enabled Visa Status Prediction and Processing Time Estimator

## 📋 Project Overview

This project aims to build an AI-powered system that predicts US PERM visa application status (Certified/Denied) and estimates processing time. By leveraging machine learning and deep learning techniques, this system provides valuable insights into visa application outcomes and processing timelines, assisting immigration professionals, employers, and visa applicants in understanding application trajectories and timelines.

## 🎯 Objectives

- **Visa Status Prediction**: Predict whether a visa application will be Certified or Denied
- **Processing Time Estimation**: Estimate the number of days required for visa application processing
- **Data-Driven Insights**: Provide actionable analytics on factors affecting visa approvals
- **Model Comparison**: Evaluate multiple machine learning and deep learning approaches

## 📊 Dataset

**Source**: US PERM Visa Applications Dataset

**Original Dataset Specifications**:
- **Total Records**: ~150,000+ visa applications
- **Total Features**: 154 attributes
- **Data Format**: CSV file (us_perm_visas.csv)

**Processed Dataset** (After Milestone 1 Preprocessing):
- **Total Records**: 14,622 cleaned records
- **Final Features**: 14 engineered features
- **Target Variables**: 
  - `case_status`: Binary classification (Certified/Denied)
  - `processing_days`: Regression target

## 📁 Project Structure

```
.
├── README.md                              # Project documentation
├── dataset/
│   ├── us_perm_visas.csv                 # Original raw dataset
│   └── cleaned_us_visa_dataset.csv       # Cleaned dataset (Milestone 1 output)
├── notebooks/
│   ├── VISA_DATASET_PREPROCESSING.ipynb  # Milestone 1: Data preprocessing & EDA
│   ├── milestone_2_eda_analysis.ipynb    # Milestone 2: Exploratory Data Analysis
│   ├── milestone_3_modeling.ipynb        # Milestone 3: Model development
│   └── milestone_4_deployment.ipynb      # Milestone 4: Model deployment
├── src/
│   ├── preprocessing.py                  # Data preprocessing utilities
│   ├── feature_engineering.py            # Feature engineering functions
│   ├── model_training.py                 # Model training pipelines
│   └── evaluation.py                     # Evaluation metrics and utilities
├── models/                               # Trained model artifacts
├── results/                              # Analysis results and visualizations
└── requirements.txt                      # Project dependencies
```

## 🔄 Project Milestones

### **Milestone 1: Data Preprocessing** ✅ (Completed)

**Objective**: Clean, validate, and prepare data for analysis

**Key Activities**:
- Data loading and initial exploration
- Handling missing values and data quality issues
- Feature selection (154 → 14 features)
- Text normalization (uppercase, trimming)
- Temporal feature engineering
- Status filtering (keeping only Certified/Denied cases)
- Duplicate removal
- Data validation

**Output Dataset Features**:
1. `country_of_citizenship` - Applicant's country of origin
2. `class_of_admission` - Visa class category
3. `employer_state` - Employer location state
4. `us_economic_sector` - Industry/economic sector
5. `case_status` - Target: Certified or Denied
6. `foreign_worker_info_education` - Education level
7. `foreign_worker_info_major` - Field of study/expertise
8. `case_received_date` - Application submission date
9. `decision_date` - Decision date
10. `processing_days` - Days between submission and decision
11. `application_year` - Year of application
12. `application_month` - Month of application
13. `application_quarter` - Quarter of application
14. `application_dayofweek` - Day of week submitted

**Data Quality Metrics**:
- Final records: 14,622
- Missing values: 0 (Zero null values in final dataset)
- Duplicates removed: Yes
- Data completeness: 100%



## 🛠️ Technologies & Libraries

**Programming Language**: Python 3.11.9

**Core Libraries**:
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computations
- `matplotlib` & `seaborn` - Data visualization
- `jupyter` - Interactive notebooks
- `flask`/`fastapi` - API development (for deployment)

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Installation Steps

```bash
# Clone the repository
git clone "https://github.com/ManideepPulla/AI-Enabled-Visa-Status-Prediction-and-Processing-Time-Estimator.git"
cd visa-prediction-project

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook
```

## 📋 Dependencies

```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
```

See `requirements.txt` for complete list with specific versions.

## 📈 Expected Outcomes

- **Classification Accuracy**: Target > 85%
- **Processing Time RMSE**: Within ±15 days
- **Model Interpretability**: Feature importance analysis
- **Deployment Ready**: REST API for predictions
- **Documentation**: Complete project documentation

## 📝 Data Dictionary

| Feature | Type | Description |
|---------|------|-------------|
| country_of_citizenship | Categorical | Applicant's country of birth/citizenship |
| class_of_admission | Categorical | Visa category (H-1B, L-1, PERM, etc.) |
| employer_state | Categorical | US state where employer is located |
| us_economic_sector | Categorical | Industry/sector (IT, Healthcare, Finance, etc.) |
| case_status | Categorical | Target variable (Certified/Denied) |
| foreign_worker_info_education | Categorical | Education level (Bachelor's, Master's, PhD, etc.) |
| foreign_worker_info_major | Categorical | Field of study or expertise |
| case_received_date | DateTime | Date application was received |
| decision_date | DateTime | Date decision was made |
| processing_days | Numeric | Target variable for regression (days to process) |
| application_year | Numeric | Year application was submitted |
| application_month | Numeric | Month of submission (1-12) |
| application_quarter | Numeric | Quarter of submission (1-4) |
| application_dayofweek | Numeric | Day of week submitted (0-6) |

## 🔍 Key Features of This Project

✅ **Comprehensive Data Preprocessing**: Rigorous cleaning and validation pipeline <br>
✅ **Feature Engineering**: Temporal and categorical feature creation<br>

## 📊 Preprocessing Workflow

```
Raw Dataset (154 features)
    ↓
Load & Explore Data
    ↓
Handle Missing Values
    ↓
Feature Selection (154 → 14)
    ↓
Text Normalization
    ↓
Temporal Feature Engineering
    ↓
Status Filtering (Certified/Denied only)
    ↓
Duplicate Removal
    ↓
Data Validation
    ↓
Cleaned Dataset (14,622 records, 14 features)
```

## 📖 How to Use This Repository

1. **Start with Milestone 1 Notebook**: Review `VISA_DATASET_PREPROCESSING.ipynb` to understand data preprocessing
2. **Run EDA**: Execute Milestone 2 for exploratory analysis
3. **Model Development**: Follow Milestone 3 notebooks for model training
4. **Deployment**: Reference Milestone 4 for API and interface setup
5. **Customize**: Modify hyperparameters and features based on your needs

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:
- Create a new branch for features
- Write clear commit messages
- Update documentation
- Test thoroughly before submitting

## 📞 Support & Contact

For questions, issues, or suggestions:
- Open an issue in the repository
- Contact the development team
- Review documentation for FAQs

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- US Department of Labor for providing visa application data
- Open-source community for excellent ML libraries
- Contributors and team members

## 📚 References & Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [TensorFlow/Keras Guide](https://www.tensorflow.org/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [US Visa Classifications](https://www.uscis.gov/)

## 🔄 Version History

**v1.0.0** - Milestone 1 Complete
- Initial data preprocessing and cleaning
- Feature engineering completed
- Dataset ready for analysis

**v2.0.0** - Milestone 2 (In Development)
- Exploratory data analysis
- Statistical analysis and visualizations

**v3.0.0** - Milestone 3 (Planned)
- Model development and training
- Performance evaluation and comparison

**v4.0.0** - Milestone 4 (Planned)
- API development
- Web interface and deployment

---

**Last Updated**: February 2026
**Status**: Active Development
**Milestone**: 1 of 4 Completed ✅