"""
HDI PREDICTOR - Complete Single File Application
Human Development Index Predictor with Machine Learning
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pickle
import warnings
warnings.filterwarnings('ignore')

# Flask imports
from flask import Flask, render_template_string, request, jsonify

# ============================================
# PART 1: DATA CREATION
# ============================================

def create_dataset():
    """Create HDI dataset"""
    data = {
        'Country': ['Norway', 'Switzerland', 'Australia', 'Germany', 'USA', 'UK', 'Canada', 'Japan',
                   'France', 'Italy', 'Spain', 'Singapore', 'India', 'China', 'Brazil', 'Mexico',
                   'South Africa', 'Indonesia', 'Pakistan', 'Bangladesh', 'Nigeria', 'Ethiopia',
                   'Kenya', 'Egypt', 'Turkey', 'Russia', 'Argentina', 'Saudi Arabia', 'Thailand', 'Vietnam'],
        'Life_Expectancy': [82.3, 83.4, 83.4, 81.3, 79.1, 81.6, 82.2, 84.4, 82.7, 83.5, 83.4, 84.9,
                          69.4, 77.1, 75.9, 75.0, 64.1, 71.7, 67.3, 72.3, 54.3, 66.2, 66.9, 71.8,
                          77.6, 72.6, 76.6, 74.7, 77.2, 75.5],
        'Mean_Years_Schooling': [12.6, 13.4, 13.3, 14.2, 13.4, 13.3, 13.4, 13.0, 11.6, 10.6, 10.4, 11.4,
                                6.5, 7.8, 8.6, 8.9, 9.8, 8.2, 4.6, 5.9, 4.9, 2.7, 5.4, 7.2,
                                8.4, 12.0, 10.9, 10.3, 7.5, 8.5],
        'Expected_Years_Schooling': [18.1, 16.3, 22.9, 17.5, 16.8, 17.8, 16.4, 15.3, 16.2, 16.5, 18.2, 16.8,
                                    12.3, 14.2, 15.4, 14.5, 13.8, 13.2, 8.6, 10.4, 9.0, 8.9, 10.3, 13.4,
                                    15.8, 15.4, 17.1, 15.1, 14.3, 12.7],
        'GNI_per_Capita': [68521, 64463, 46143, 51567, 54923, 43096, 47037, 40809, 42956, 38119, 36623, 91187,
                          6810, 17312, 15017, 19482, 12637, 11268, 5120, 4953, 5270, 2364, 4205, 11637,
                          27762, 26190, 20198, 50465, 16707, 8042],
        'HDI_Score': [0.957, 0.955, 0.946, 0.947, 0.926, 0.932, 0.929, 0.925, 0.903, 0.895, 0.904, 0.939,
                     0.645, 0.768, 0.765, 0.779, 0.709, 0.718, 0.557, 0.633, 0.539, 0.485, 0.601, 0.731,
                     0.820, 0.824, 0.845, 0.854, 0.777, 0.704]
    }
    
    df = pd.DataFrame(data)
    df['HDI_Category'] = pd.cut(df['HDI_Score'], bins=[0, 0.55, 0.70, 0.80, 1.0], 
                                labels=['Low', 'Medium', 'High', 'Very High'])
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/hdi_dataset.csv', index=False)
    print("✅ Dataset created successfully!")
    return df

# ============================================
# PART 2: MODEL TRAINING
# ============================================

def train_model():
    """Train the HDI prediction model"""
    print("\n" + "="*60)
    print("🌍 HDI PREDICTOR - TRAINING MODEL")
    print("="*60)
    
    df = pd.read_csv('data/hdi_dataset.csv')
    print(f"✅ Dataset loaded: {len(df)} countries")
    
    X = df[['Life_Expectancy', 'Mean_Years_Schooling', 'Expected_Years_Schooling', 'GNI_per_Capita']]
    y = df['HDI_Score']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\n📊 Model Performance:")
    print(f"   R² Score: {r2:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE: {mae:.4f}")
    
    os.makedirs('models', exist_ok=True)
    with open('models/hdi_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    create_visualizations(df, model, scaler, X_test, y_test, y_pred)
    print("✅ Model and visualizations saved!")
    return model, scaler

def create_visualizations(df, model, scaler, X_test, y_test, y_pred):
    """Create all visualizations"""
    os.makedirs('static', exist_ok=True)
    
    # 1. HDI Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['HDI_Score'], bins=20, kde=True, color='green')
    plt.title('Distribution of HDI Scores', fontsize=14)
    plt.xlabel('HDI Score')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig('static/hdi_distribution.png', bbox_inches='tight')
    plt.close()
    
    # 2. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    corr = df[['Life_Expectancy', 'Mean_Years_Schooling', 'Expected_Years_Schooling', 
               'GNI_per_Capita', 'HDI_Score']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix', fontsize=14)
    plt.savefig('static/correlation_heatmap.png', bbox_inches='tight')
    plt.close()
    
    # 3. GNI vs HDI
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='GNI_per_Capita', y='HDI_Score', hue='HDI_Category', size='HDI_Score', sizes=(50, 200))
    plt.title('GNI per Capita vs HDI Score', fontsize=14)
    plt.xlabel('GNI per Capita (USD PPP)')
    plt.ylabel('HDI Score')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.savefig('static/gni_vs_hdi.png', bbox_inches='tight')
    plt.close()
    
    # 4. Actual vs Predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual HDI Score')
    plt.ylabel('Predicted HDI Score')
    plt.title('Actual vs Predicted HDI Scores')
    plt.grid(True, alpha=0.3)
    plt.savefig('static/actual_vs_predicted.png', bbox_inches='tight')
    plt.close()
    
    # 5. Feature Importance
    plt.figure(figsize=(10, 6))
    features = ['Life Expectancy', 'Mean Schooling', 'Expected Schooling', 'GNI per Capita']
    importance = pd.DataFrame({'Feature': features, 'Coefficient': model.coef_})
    importance = importance.sort_values('Coefficient', ascending=True)
    plt.barh(importance['Feature'], importance['Coefficient'], color='steelblue')
    plt.xlabel('Coefficient')
    plt.title('Feature Importance')
    plt.grid(True, alpha=0.3)
    plt.savefig('static/feature_importance.png', bbox_inches='tight')
    plt.close()

# ============================================
# PART 3: FLASK APPLICATION
# ============================================

app = Flask(__name__)
model = None
scaler = None

def get_hdi_category(score):
    if score >= 0.800:
        return "Very High", "success", "🌟 Excellent development!"
    elif score >= 0.700:
        return "High", "primary", "📈 Good development!"
    elif score >= 0.550:
        return "Medium", "warning", "📊 Moderate development"
    else:
        return "Low", "danger", "📉 Needs intervention"

# HTML Templates
HOME_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>HDI Predictor</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body{background:#f8f9fa;padding-bottom:60px}
        .navbar-brand{font-weight:bold;font-size:1.5rem}
        .card{border-radius:15px;transition:transform 0.3s}
        .card:hover{transform:translateY(-5px)}
        .footer{position:fixed;bottom:0;width:100%;background:#f8f9fa;padding:10px 0;text-align:center}
        .btn-success:hover{transform:scale(1.05)}
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="/">🌍 HDI Predictor</a>
            <div class="collapse navbar-collapse">
                <ul class="navbar-nav ms-auto">
                    <li class="nav-item"><a class="nav-link active" href="/">Home</a></li>
                    <li class="nav-item"><a class="nav-link" href="/predict">Predict</a></li>
                    <li class="nav-item"><a class="nav-link" href="/visualizations">Visualizations</a></li>
                    <li class="nav-item"><a class="nav-link" href="/about">About</a></li>
                </ul>
            </div>
        </div>
    </nav>
    <div class="container mt-5">
        <div class="row">
            <div class="col-lg-8 mx-auto text-center">
                <h1 class="display-4 mb-4">🌍 Human Development Index Predictor</h1>
                <p class="lead mb-4">Predict HDI scores using Machine Learning based on life expectancy, education, and income indicators.</p>
                <div class="row mt-5">
                    <div class="col-md-4 mb-3">
                        <div class="card h-100 shadow-sm"><div class="card-body text-center">
                            <h3 class="text-primary">📊</h3><h5>Data Analysis</h5><p class="small">Analyze HDI indicators</p>
                        </div></div>
                    </div>
                    <div class="col-md-4 mb-3">
                        <div class="card h-100 shadow-sm"><div class="card-body text-center">
                            <h3 class="text-success">🤖</h3><h5>ML Prediction</h5><p class="small">Linear Regression model</p>
                        </div></div>
                    </div>
                    <div class="col-md-4 mb-3">
                        <div class="card h-100 shadow-sm"><div class="card-body text-center">
                            <h3 class="text-info">📱</h3><h5>Easy to Use</h5><p class="small">Simple interface</p>
                        </div></div>
                    </div>
                </div>
                <a href="/predict" class="btn btn-success btn-lg px-5 mt-4">🚀 Start Prediction</a>
            </div>
        </div>
    </div>
    <div class="footer"><span class="text-muted">HDI Predictor - Machine Learning Project</span></div>
</body>
</html>
"""

PREDICT_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>HDI Predictor - Predict</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body{background:#f8f9fa;padding-bottom:60px}
        .navbar-brand{font-weight:bold;font-size:1.5rem}
        .card{border-radius:15px}
        .display-1{font-size:4rem;font-weight:bold}
        .text-success{color:#28a745!important}
        .text-primary{color:#007bff!important}
        .text-warning{color:#ffc107!important}
        .text-danger{color:#dc3545!important}
        .progress{height:30px;border-radius:10px}
        .footer{position:fixed;bottom:0;width:100%;background:#f8f9fa;padding:10px 0;text-align:center}
        @keyframes fadeInUp{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
        .fade-in-up{animation:fadeInUp 0.5s ease-out}
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="/">🌍 HDI Predictor</a>
            <div class="collapse navbar-collapse">
                <ul class="navbar-nav ms-auto">
                    <li class="nav-item"><a class="nav-link" href="/">Home</a></li>
                    <li class="nav-item"><a class="nav-link active" href="/predict">Predict</a></li>
                    <li class="nav-item"><a class="nav-link" href="/visualizations">Visualizations</a></li>
                    <li class="nav-item"><a class="nav-link" href="/about">About</a></li>
                </ul>
            </div>
        </div>
    </nav>
    <div class="container mt-5">
        <div class="row">
            <div class="col-lg-8 mx-auto">
                <h2 class="text-center mb-4">📊 HDI Prediction</h2>
                {% if error %}
                <div class="alert alert-danger">{{ error }}</div>
                {% endif %}
                <div class="card shadow">
                    <div class="card-body">
                        <form method="POST">
                            <div class="row g-4">
                                <div class="col-md-6">
                                    <label class="form-label">Life Expectancy (years)</label>
                                    <input type="number" step="0.1" class="form-control" name="life_expectancy" required>
                                    <small class="text-muted">Range: 50-85 years</small>
                                </div>
                                <div class="col-md-6">
                                    <label class="form-label">Mean Years of Schooling</label>
                                    <input type="number" step="0.1" class="form-control" name="mean_schooling" required>
                                    <small class="text-muted">Range: 0-18 years</small>
                                </div>
                                <div class="col-md-6">
                                    <label class="form-label">Expected Years of Schooling</label>
                                    <input type="number" step="0.1" class="form-control" name="expected_schooling" required>
                                    <small class="text-muted">Range: 0-22 years</small>
                                </div>
                                <div class="col-md-6">
                                    <label class="form-label">GNI per Capita (USD PPP)</label>
                                    <input type="number" step="100" class="form-control" name="gni_per_capita" required>
                                    <small class="text-muted">Range: 500-100,000 USD</small>
                                </div>
                            </div>
                            <div class="text-center mt-4">
                                <button type="submit" class="btn btn-success btn-lg px-5">🎯 Predict HDI</button>
                                <button type="reset" class="btn btn-secondary btn-lg px-4 ms-2">🔄 Reset</button>
                            </div>
                        </form>
                        {% if prediction is not none %}
                        <div class="mt-5 p-4 border rounded bg-light fade-in-up">
                            <h4 class="text-center">📈 Prediction Result</h4>
                            <div class="text-center">
                                <div class="display-1 text-{{ color }}">{{ prediction }}</div>
                                <h3 class="text-{{ color }}">{{ category }}</h3>
                                <p>{{ description }}</p>
                                <div class="progress">
                                    <div class="progress-bar bg-{{ color }}" style="width: {{ prediction * 100 }}%;">
                                        {{ "%.1f"|format(prediction * 100) }}%
                                    </div>
                                </div>
                            </div>
                        </div>
                        {% endif %}
                    </div>
                </div>
                <div class="mt-4">
                    <h6>💡 Quick Test Values:</h6>
                    <div class="row g-2">
                        <div class="col-md-3"><button class="btn btn-outline-success btn-sm w-100" onclick="fillValues(80,14,16,50000)">Very High</button></div>
                        <div class="col-md-3"><button class="btn btn-outline-primary btn-sm w-100" onclick="fillValues(72,10,12,25000)">High</button></div>
                        <div class="col-md-3"><button class="btn btn-outline-warning btn-sm w-100" onclick="fillValues(65,7,8,10000)">Medium</button></div>
                        <div class="col-md-3"><button class="btn btn-outline-danger btn-sm w-100" onclick="fillValues(55,3,4,3000)">Low</button></div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <div class="footer"><span class="text-muted">HDI Predictor - Machine Learning Project</span></div>
    <script>
        function fillValues(life, meanSchool, expectedSchool, gni) {
            document.querySelector('[name="life_expectancy"]').value = life;
            document.querySelector('[name="mean_schooling"]').value = meanSchool;
            document.querySelector('[name="expected_schooling"]').value = expectedSchool;
            document.querySelector('[name="gni_per_capita"]').value = gni;
        }
    </script>
</body>
</html>
"""

VISUALIZATIONS_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>HDI Predictor - Visualizations</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body{background:#f8f9fa;padding-bottom:60px}
        .navbar-brand{font-weight:bold;font-size:1.5rem}
        .card{border-radius:15px}
        .footer{position:fixed;bottom:0;width:100%;background:#f8f9fa;padding:10px 0;text-align:center}
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="/">🌍 HDI Predictor</a>
            <div class="collapse navbar-collapse">
                <ul class="navbar-nav ms-auto">
                    <li class="nav-item"><a class="nav-link" href="/">Home</a></li>
                    <li class="nav-item"><a class="nav-link" href="/predict">Predict</a></li>
                    <li class="nav-item"><a class="nav-link active" href="/visualizations">Visualizations</a></li>
                    <li class="nav-item"><a class="nav-link" href="/about">About</a></li>
                </ul>
            </div>
        </div>
    </nav>
    <div class="container mt-5">
        <h2 class="text-center mb-4">📊 Data Visualizations</h2>
        <div class="row">
            <div class="col-md-6 mb-4">
                <div class="card shadow"><div class="card-body">
                    <h5 class="text-center">HDI Score Distribution</h5>
                    <img src="/static/hdi_distribution.png" class="img-fluid" alt="HDI Distribution">
                </div></div>
            </div>
            <div class="col-md-6 mb-4">
                <div class="card shadow"><div class="card-body">
                    <h5 class="text-center">Correlation Heatmap</h5>
                    <img src="/static/correlation_heatmap.png" class="img-fluid" alt="Correlation">
                </div></div>
            </div>
            <div class="col-md-6 mb-4">
                <div class="card shadow"><div class="card-body">
                    <h5 class="text-center">GNI per Capita vs HDI</h5>
                    <img src="/static/gni_vs_hdi.png" class="img-fluid" alt="GNI vs HDI">
                </div></div>
            </div>
            <div class="col-md-6 mb-4">
                <div class="card shadow"><div class="card-body">
                    <h5 class="text-center">Actual vs Predicted</h5>
                    <img src="/static/actual_vs_predicted.png" class="img-fluid" alt="Actual vs Predicted">
                </div></div>
            </div>
        </div>
        <div class="text-center"><a href="/predict" class="btn btn-success">🚀 Start Predicting</a></div>
    </div>
    <div class="footer"><span class="text-muted">HDI Predictor - Machine Learning Project</span></div>
</body>
</html>
"""

ABOUT_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>HDI Predictor - About</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body{background:#f8f9fa;padding-bottom:60px}
        .navbar-brand{font-weight:bold;font-size:1.5rem}
        .card{border-radius:15px}
        .footer{position:fixed;bottom:0;width:100%;background:#f8f9fa;padding:10px 0;text-align:center}
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="/">🌍 HDI Predictor</a>
            <div class="collapse navbar-collapse">
                <ul class="navbar-nav ms-auto">
                    <li class="nav-item"><a class="nav-link" href="/">Home</a></li>
                    <li class="nav-item"><a class="nav-link" href="/predict">Predict</a></li>
                    <li class="nav-item"><a class="nav-link" href="/visualizations">Visualizations</a></li>
                    <li class="nav-item"><a class="nav-link active" href="/about">About</a></li>
                </ul>
            </div>
        </div>
    </nav>
    <div class="container mt-5">
        <div class="row">
            <div class="col-lg-8 mx-auto">
                <h2 class="text-center mb-4">📖 About HDI Predictor</h2>
                <div class="card shadow"><div class="card-body">
                    <h4>🌍 What is HDI?</h4>
                    <p>The Human Development Index (HDI) is a composite statistic of life expectancy, education, and per capita income indicators.</p>
                    <h4 class="mt-4">🎯 Features Used</h4>
                    <ul>
                        <li><strong>Life Expectancy</strong> - Average years a person is expected to live</li>
                        <li><strong>Mean Years of Schooling</strong> - Average years of education received</li>
                        <li><strong>Expected Years of Schooling</strong> - Expected years of future education</li>
                        <li><strong>GNI per Capita</strong> - Gross National Income per person</li>
                    </ul>
                    <h4 class="mt-4">🤖 Machine Learning Model</h4>
                    <ul>
                        <li><strong>Algorithm:</strong> Linear Regression</li>
                        <li><strong>R² Score:</strong> ~0.90</li>
                        <li><strong>Features:</strong> 4 indicators</li>
                    </ul>
                    <h4 class="mt-4">📊 HDI Categories</h4>
                    <div class="row mt-2">
                        <div class="col-3"><div class="border p-2 text-center bg-success text-white"><strong>Very High</strong><br>≥ 0.800</div></div>
                        <div class="col-3"><div class="border p-2 text-center bg-primary text-white"><strong>High</strong><br>0.700-0.799</div></div>
                        <div class="col-3"><div class="border p-2 text-center bg-warning"><strong>Medium</strong><br>0.550-0.699</div></div>
                        <div class="col-3"><div class="border p-2 text-center bg-danger text-white"><strong>Low</strong><br>&lt; 0.550</div></div>
                    </div>
                </div></div>
            </div>
        </div>
    </div>
    <div class="footer"><span class="text-muted">HDI Predictor - Machine Learning Project</span></div>
</body>
</html>
"""

# Flask Routes
@app.route('/')
def home():
    return render_template_string(HOME_PAGE)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    category = None
    color = None
    description = None
    error = None
    
    if request.method == 'POST':
        try:
            life = float(request.form.get('life_expectancy', 0))
            mean_school = float(request.form.get('mean_schooling', 0))
            expected_school = float(request.form.get('expected_schooling', 0))
            gni = float(request.form.get('gni_per_capita', 0))
            
            if life <= 0 or mean_school <= 0 or expected_school <= 0 or gni <= 0:
                error = "Please enter valid positive values for all fields."
            else:
                features = np.array([[life, mean_school, expected_school, gni]])
                features_scaled = scaler.transform(features)
                prediction = round(model.predict(features_scaled)[0], 3)
                category, color, description = get_hdi_category(prediction)
                
        except Exception as e:
            error = f"Error: {str(e)}"
    
    return render_template_string(PREDICT_PAGE, prediction=prediction, category=category,
                                 color=color, description=description, error=error)

@app.route('/visualizations')
def visualizations():
    return render_template_string(VISUALIZATIONS_PAGE)

@app.route('/about')
def about():
    return render_template_string(ABOUT_PAGE)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        features = np.array([[
            float(data.get('life_expectancy', 0)),
            float(data.get('mean_schooling', 0)),
            float(data.get('expected_schooling', 0)),
            float(data.get('gni_per_capita', 0))
        ]])
        features_scaled = scaler.transform(features)
        prediction = float(model.predict(features_scaled)[0])
        category, _, _ = get_hdi_category(prediction)
        return jsonify({'success': True, 'prediction': round(prediction, 3), 'category': category})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == '__main__':
    print("\n🌍 HDI PREDICTOR")
    print("="*60)
    
    if not os.path.exists('data/hdi_dataset.csv'):
        create_dataset()
    
    if not os.path.exists('models/hdi_model.pkl'):
        model, scaler = train_model()
    else:
        with open('models/hdi_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print("✅ Model loaded successfully!")
    
    os.makedirs('static', exist_ok=True)
    
    print("\n🚀 Starting Flask Web Application...")
    print("📍 URL: http://127.0.0.1:5000")
    print("="*60)
    
    app.run(debug=True, host='127.0.0.1', port=5000)