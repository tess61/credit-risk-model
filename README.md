# redit Risk Model for Bati Bank

    This project develops a credit scoring model for Bati Bank in partnership with an eCommerce platform to enable a buy-now-pay-later service. The model uses customer transaction data to predict credit risk, assign credit scores, and recommend optimal loan amounts and durations.

## Project Structure

```bash
    credit-risk-model/
    ├── .github/workflows/ci.yml # CI/CD pipeline configuration
    ├── data/ # Data storage (ignored by Git)
    │ ├── raw/ # Raw input data
    │ └── processed/ # Processed data for modeling
    ├── notebooks/
    │ └── 1.0-eda.ipynb # Exploratory data analysis
    ├── src/
    │ ├── **init**.py
    │ ├── data_processing.py # Feature engineering logic
    │ ├── train.py # Model training script
    │ ├── predict.py # Inference script
    │ └── api/
    │ ├── main.py # FastAPI application
    │ └── pydantic_models.py # Pydantic models for API
    ├── tests/
    │ └── test_data_processing.py # Unit tests
    ├── Dockerfile # Docker configuration
    ├── docker-compose.yml # Docker Compose for local deployment
    ├── requirements.txt # Python dependencies
    ├── .gitignore # Git ignore file
    └── README.md # Project documentation
```

## Credit Scoring Business Understanding

    How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?
    The Basel II Capital Accord emphasizes robust risk measurement and management to ensure financial institutions maintain adequate capital reserves against potential losses. It requires banks to use standardized or internal ratings-based approaches for credit risk assessment, which mandates transparency, interpretability, and rigorous documentation. For Bati Bank’s credit scoring model, this means prioritizing interpretable models (e.g., Logistic Regression with Weight of Evidence) to clearly explain how features like RFM metrics contribute to risk predictions. Well-documented models facilitate regulatory compliance, enabling auditors to verify that the model aligns with Basel II’s requirements for risk differentiation and capital allocation. Interpretability also builds trust with stakeholders, ensuring loan decisions are fair and justifiable.
    Why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?
    Since the dataset lacks a direct "default" label, a proxy variable (e.g., is_high_risk derived from RFM clustering) is necessary to approximate credit risk. This proxy identifies disengaged customers (low frequency, low monetary value) as high-risk, assuming they are more likely to default. This approach enables model training but introduces risks, such as misclassification if the proxy poorly correlates with actual default behavior. Business risks include approving loans to high-risk customers (increasing default rates) or rejecting low-risk customers (reducing revenue). Inaccurate proxies could also lead to regulatory scrutiny or biased lending practices, necessitating validation against real-world default data when available.
    What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?
    In a regulated financial context like Bati Bank’s, simple models like Logistic Regression with Weight of Evidence (WoE) offer interpretability, ease of regulatory compliance, and transparency in how features influence predictions, aligning with Basel II’s requirements. However, they may sacrifice predictive power for complex patterns. Complex models like Gradient Boosting provide higher accuracy and capture non-linear relationships but are less interpretable, making it harder to justify decisions to regulators or stakeholders. They also risk overfitting and require more computational resources. The trade-off involves balancing predictive performance with regulatory compliance and operational transparency, with simpler models often preferred for their auditability in regulated environments.

## Setup Instructions

1. Clone the repository:
   git clone <repository-url>
   cd credit-risk-model

2. Install dependencies:
   ```bash
       pip install -r requirements.txt
   ```
3. Run the FastAPI application:
   ```bash
       docker-compose up
   ```
4. Access the API:The API will be available at http://localhost:8000.

## Next Steps

    Conduct exploratory data analysis in notebooks/1.0-eda.ipynb.
    Implement feature engineering in src/data_processing.py.
    Train models using src/train.py and track experiments with MLflow.
    Deploy the model via the FastAPI endpoint in src/api/main.py.
