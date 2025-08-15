# **Birth Rate Forecasting for Scotland**

## **Overview**
This project is an end-to-end predictive analytics solution designed to forecast birth rates across **Scotland** at a granular level, broken down by **NHS Health Board** and **maternal age group**. It tackles the limitations of national-level forecasts by offering regional and demographic details for improved resource allocation, public health planning, and policy formulation.

**Solution highlights:**
- Automated data pipelines for sourcing, cleaning, and processing birth registration & population data.
- Time-series forecasting models using **Facebook's Prophet** library.
- Interactive **Streamlit** dashboard for visualizing historical trends, forecasts, and uncertainty intervals.

*Developed during my internship at ITSOLERA as Task 1.*

---

## **Features**
- **Granular Forecasting**: Predicts birth rates for every combination of Scotland’s 14 NHS Health Boards and maternal age groups *(15-19, 20-24, etc.)*.
- **Data Integration**: Combines data from *National Records of Scotland (NRS)* and *Office for National Statistics (ONS)*.
- **Model Training**: Trains Prophet models per segment, handling trends, seasonality, and holidays.
- **Interactive Dashboard** *(Streamlit)*:
  - Time-series plots, showing forecasts and uncertainty bands.
  - Heatmaps for historical patterns.
  - Key metrics — recent birth rates, year-on-year changes.
  - Filters for **NHS boards** and **age groups**.
- **Scalable Pipeline**: Automated scripts for data preprocessing, training, and forecasting, outputting to CSV files.

---

## **Technologies Used**
- **Programming Language:** Python 3.10+
- **Libraries:**
  - Data Processing: `pandas`, `numpy`
  - Forecasting: `prophet`
  - Visualization: `plotly`, `matplotlib`
  - Web App: `streamlit`
- **Data Sources:** NRS birth registrations, ONS mid-year population estimates

---

## **Installation**

1. **Clone the repository:**
    ```
    git clone https://github.com/your-username/birth-rate-forecasting-scotland.git
    cd birth-rate-forecasting-scotland
    ```

2. **Create a virtual environment** (optional but recommended):
    ```
    python -m venv env
    source env/bin/activate  # On Windows: env\Scripts\activate
    ```

3. **Install dependencies:**
    ```
    pip install -r requirements.txt
    ```

> *Ensure you have the necessary data files (Excel sheets for births & population) in the project directory or update paths in the scripts accordingly.*

---

## **Usage**


### Running the Data Pipeline & Forecasting
Run the script to process data and generate forecasts:

python app.py
- Outputs `forecasts.csv` with predictions for all segments.

### Launching the Dashboard
Start the Streamlit app:
streamlit run app.py:- Open your browser to the provided URL (typically `http://localhost:8501`).
- Use the sidebar filters to select **NHS Board** and **age group**.
- Explore time-series plots, heatmaps, and key metrics.

---

## **Project Structure**

| File | Description |
|------|-------------|
| `app.py` | Main Streamlit dashboard script |
| `requirements.txt` | Python dependencies |
| Data files | e.g., `bt5-2023-births-time-series.xlsx`, `monthly-births-june-2025.xlsx` |
| Forecast outputs | e.g., `forecasts.csv` |
| Other scripts | Data cleaning, model training pipelines |

---

## **Approach**

1. **Data Acquisition & Preprocessing:**
    - Load and clean raw Excel files (NRS, ONS)
    - Merge datasets, calculate birth rates, reshape to long format

2. **Exploratory Data Analysis:**
    - Identify trends, seasonality, correlations

3. **Modeling:**
    - Use Prophet for time-series forecasting
    - Train per NHS board and age group
    - Generate 5-year forecasts with uncertainty intervals

4. **Dashboard Development:**
    - Intuitive UI, filters, interactive charts
    - Custom styling for professional look

---

## **Business Impact**
- **Supports planning** for maternity services, childcare, and education
- **Enables targeted interventions** for public health
- **Democratizes data access** via interactive tool, reduces reliance on static reporting

---

## **Limitations & Future Improvements**
- Currently **annual data** — can extend to monthly forecasts with better inputs
- Incorporate economic factors or migration data
- Deploy dashboard on cloud (Streamlit Sharing, Heroku) for wider access
