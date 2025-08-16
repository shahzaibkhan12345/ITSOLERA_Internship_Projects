# 🛌 AI-Powered Sleep Quality Predictor

A comprehensive machine learning system that analyzes smartphone sensor data to predict sleep quality and provide personalized AI-powered sleep recommendations.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data Sources](#data-sources)
- [Technical Architecture](#technical-architecture)
- [API Documentation](#api-documentation)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project combines smartphone sensor data analysis with machine learning to predict sleep quality and provide personalized sleep improvement recommendations. The system uses accelerometer, gyroscope, light sensor, and other smartphone data to create sleep-related features and predict sleep quality using an XGBoost classifier.

### Key Capabilities

- **Sensor Data Analysis**: Processes raw smartphone sensor data to extract sleep-related features
- **Sleep Quality Prediction**: Uses machine learning to predict sleep quality (Good vs. Not Good)
- **AI-Powered Recommendations**: Integrates with Google Gemini AI to provide personalized sleep advice
- **Interactive Web Interface**: Streamlit-based dashboard for easy interaction
- **Comprehensive Reporting**: Generates detailed sleep analysis reports

## ✨ Features

### 🔬 Data Processing
- Automated sensor data preprocessing and cleaning
- Feature engineering from raw accelerometer, gyroscope, and light sensor data
- Missing value handling and data type optimization
- Multi-dataset integration and merging

### 🤖 Machine Learning
- XGBoost-based binary classification for sleep quality prediction
- 94% accuracy on test data
- Model persistence and loading capabilities
- Feature importance analysis

### 🧠 AI Integration
- Google Gemini AI integration for personalized recommendations
- Rule-based fallback when AI is unavailable
- Context-aware suggestions based on user data
- Scientific backing for all recommendations

### 📊 Interactive Dashboard
- Real-time sleep metrics calculation
- Interactive input sliders and controls
- Live prediction updates
- Beautiful visualizations with Plotly
- Exportable sleep reports

## 📁 Project Structure

```
Team_zeta_3_task/
├── app.py                              # Main Streamlit application
├── Task_3.ipynb                        # Jupyter notebook with data processing and model training
├── requirements.txt                    # Python dependencies
├── sleep_quality_binary_model.pkl      # Trained XGBoost model
├── sleep_analysis_for_powerbi.csv      # Processed data for Power BI
├── README.md                           # This file
├── env/                                # Virtual environment (if used)
└── Datasets/                           # Local dataset storage
    ├── B.HEALTH classification.csv
    ├── B.Health components.csv
    ├── SENSORDATA.csv
    ├── Sensors.csv
    ├── sleep_analysis_for_powerbi.csv
    ├── SleepQual classification.csv
    ├── SleepQual components.csv
    └── UserInfo.csv
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Google Gemini API key (optional, for AI recommendations)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Team_zeta_3_task
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv env
   
   # On Windows
   env\Scripts\activate
   
   # On macOS/Linux
   source env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up API key (optional)**
   ```bash
   # Set your Google Gemini API key as an environment variable
   export GOOGLE_API_KEY="your-api-key-here"
   
   # Or modify the API key in app.py (not recommended for production)
   ```

## 📖 Usage

### Running the Web Application

1. **Start the Streamlit app**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser**
   - Navigate to `http://localhost:8501`
   - The application will load with the sleep quality predictor interface

### Using the Application

1. **Input Sleep Metrics**
   - Adjust sliders for sleep timing (time in bed, time to fall asleep)
   - Set sleep quality metrics (awake time during night)
   - Configure technology usage (phone usage, unlock count)

2. **Get Predictions**
   - Click "🔮 Analyze Sleep Quality" to get ML predictions
   - View real-time metrics and sleep efficiency gauge
   - Check prediction confidence scores

3. **AI Recommendations**
   - Click "🤖 Get Personalized AI Suggestions" for expert advice
   - Enable "Expert Mode" for additional clinical factors
   - Review priority-based recommendations with scientific backing

4. **Export Reports**
   - Download comprehensive sleep analysis reports
   - View sleep pattern trends and visualizations
   - Access sleep medicine insights and tips

### Running the Jupyter Notebook

1. **Open the notebook**
   ```bash
   jupyter notebook Task_3.ipynb
   ```

2. **Execute cells sequentially**
   - Data loading and preprocessing
   - Feature engineering
   - Model training and evaluation
   - Prediction generation

## 📊 Data Sources

The project uses multiple datasets from Kaggle:

### Primary Datasets
- **Embedded Smartphone Sensor Data**: Raw sensor readings from smartphones
  - Accelerometer data (X, Y, Z axes)
  - Gyroscope data (X, Y, Z axes)
  - Light sensor readings
  - Battery level and orientation data
  - GPS coordinates (filtered out for privacy)

- **Sleep Quality and B.Health Dataset**: Ground truth sleep quality labels
  - Sleep quality classifications
  - Sleep duration and latency metrics
  - Phone usage patterns during sleep

### Data Processing Pipeline

1. **Data Loading**: Automated download from Kaggle using `kagglehub`
2. **Cleaning**: Handle missing values, convert data types, clean column names
3. **Feature Engineering**: Create sleep-related features from sensor data
4. **Merging**: Combine sensor data with sleep quality labels
5. **Validation**: Ensure data quality and consistency

## 🏗️ Technical Architecture

### Data Processing Flow

```
Raw Sensor Data → Preprocessing → Feature Engineering → Model Training → Prediction
```

### Key Components

#### 1. Data Preprocessing (`Task_3.ipynb`)
- **Sensor Data Processing**: Combines multiple sensor datasets
- **Feature Engineering**: Creates sleep-related features from raw sensor data
- **Data Cleaning**: Handles missing values and data type conversions

#### 2. Machine Learning Model
- **Algorithm**: XGBoost Classifier
- **Problem Type**: Binary Classification (Good Sleep vs. Not Good Sleep)
- **Features**: 5 engineered features from sensor data
- **Performance**: 94% accuracy on test set

#### 3. Web Application (`app.py`)
- **Framework**: Streamlit
- **UI Components**: Interactive sliders, real-time metrics, visualizations
- **AI Integration**: Google Gemini API for personalized recommendations
- **Reporting**: Exportable sleep analysis reports

### Feature Engineering

The system creates the following features from sensor data:

1. **Duration in Bed (minutes)**: Total time from first to last sensor reading
2. **Sleep Onset Latency (minutes)**: Time until first period of inactivity
3. **In Bed Awake Duration (minutes)**: Time spent active during sleep period
4. **Night Time Phone Usage (minutes)**: Active phone usage during sleep hours
5. **Phone Unlock Count**: Number of times phone was unlocked during night

## 🔌 API Documentation

### AI Expert System

The `AIExpertSystem` class provides AI-powered sleep recommendations:

```python
class AIExpertSystem:
    def __init__(self, api_key: Optional[str] = None)
    def generate_personalized_suggestions(self, user_data: Dict, 
                                        prediction_confidence: float, 
                                        expert_data: Optional[Dict] = None) -> List[Dict]
```

#### Parameters
- `user_data`: Dictionary containing sleep metrics
- `prediction_confidence`: Model's confidence in poor sleep prediction
- `expert_data`: Optional clinical factors (stress, medication, etc.)

#### Returns
List of recommendation dictionaries with:
- `title`: Recommendation title
- `priority`: "high", "medium", or "low"
- `explanation`: Why the recommendation matters
- `action`: Specific action to take
- `science`: Scientific backing for the recommendation

### Model Prediction

```python
# Load the trained model
model = joblib.load('sleep_quality_binary_model.pkl')

# Prepare features
features = ['duration_in_bed_minutes', 'sleep_onset_latency_minutes', 
           'in_bed_awake_duration_minutes', 'night_time_phone_usage_day_minutes', 
           'phone_unlock_count_day']

# Make prediction
prediction = model.predict(input_data[features])
probability = model.predict_proba(input_data[features])
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
4. **Add tests** (if applicable)
5. **Commit your changes**
   ```bash
   git commit -m "Add: description of your changes"
   ```
6. **Push to the branch**
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Create a Pull Request**

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to new functions
- Include error handling for new features
- Test your changes thoroughly
- Update documentation as needed

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Kaggle Datasets**: For providing the sensor and sleep quality data
- **Google Gemini AI**: For powering the AI recommendation system
- **Streamlit**: For the web application framework
- **XGBoost**: For the machine learning algorithm
- **Open Source Community**: For the various Python libraries used

## 📞 Support

If you encounter any issues or have questions:

1. **Check the documentation** in this README
2. **Review the Jupyter notebook** for detailed implementation
3. **Open an issue** on the repository
4. **Contact the development team**

---

**Note**: This application is for educational and research purposes. For medical sleep issues, please consult with a qualified sleep specialist or healthcare provider.



