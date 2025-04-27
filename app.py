from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import numpy as np
import sqlite3
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import os

app = Flask(__name__)
app.secret_key = "Manage_Matrix"

# Ensure the 'data' folder exists
os.makedirs('data', exist_ok=True)

# Database setup for events management
def init_db():
    conn = sqlite3.connect('database/events.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS events
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  event_name TEXT, 
                  event_date TEXT, 
                  location TEXT, 
                  attendees TEXT)''')
    conn.commit()
    conn.close()

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Sales Forecasting Page
@app.route('/sales_forecasting')
def sales_forecasting_page():
    return render_template('sales_forecasting.html')

# Inventory Management Page
@app.route('/inventory_management')
def inventory_management_page():
    return render_template('inventory_management.html')

# Shipping Cost Prediction Page
@app.route('/shipping_cost')
def shipping_cost_page():
    return render_template('shipping_cost.html')

# Events Management Page
@app.route('/events_management')
def events_management_page():
    return render_template('events_management.html')

# Sales Forecasting with ARIMA
@app.route('/sales_forecast', methods=['POST'])
def sales_forecast():
    try:
        if 'sales_file' not in request.files:
            flash('No file uploaded', 'danger')
            return redirect(url_for('sales_forecasting_page'))
        
        file = request.files['sales_file']
        if file.filename == '':
            flash('No file selected', 'danger')
            return redirect(url_for('sales_forecasting_page'))
        
        # Save and process the file
        filepath = os.path.join('data', file.filename)
        file.save(filepath)
        data = pd.read_csv(filepath)
        
        # Validate dataset
        if 'sales' not in data.columns:
            flash('Dataset must contain a "sales" column', 'danger')
            return redirect(url_for('sales_forecasting_page'))
        
        # ARIMA Model with order (10, 1, 1)
        model = ARIMA(data['sales'], order=(10, 1, 1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=3)  # Forecast next 3 months

        forecast_rounded = [round(value,3) for value in forecast.tolist()]
        
        return render_template('sales_forecasting.html', sales_forecast=forecast_rounded)
    
    except Exception as e:
        flash(f'Error in sales forecasting: {str(e)}', 'danger')
        return redirect(url_for('sales_forecasting_page'))

# Inventory Management (EOQ)
@app.route('/inventory_management', methods=['POST'])
def inventory_management():
    try:
        demand = float(request.form['demand'])
        holding_cost = float(request.form['holding_cost'])
        order_cost = float(request.form['order_cost'])
        
        # Validate inputs
        if demand <= 0 or holding_cost <= 0 or order_cost <= 0:
            flash('All inputs must be positive numbers', 'danger')
            return redirect(url_for('inventory_management_page'))
        
        # EOQ Formula
        eoq = np.sqrt((2 * demand * order_cost) / holding_cost)
        eoq_rounded = round(eoq,3)
        return render_template('inventory_management.html', eoq=eoq_rounded)
    
    except Exception as e:
        flash(f'Error in inventory management: {str(e)}', 'danger')
        return redirect(url_for('inventory_management_page'))

# Shipping Cost Prediction
@app.route('/shipping_cost', methods=['POST'])
def shipping_cost():
    try:
        if 'shipping_file' not in request.files:
            flash('No file uploaded', 'danger')
            return redirect(url_for('shipping_cost_page'))
        
        file = request.files['shipping_file']
        if file.filename == '':
            flash('No file selected', 'danger')
            return redirect(url_for('shipping_cost_page'))
        
        # Save and process the file
        filepath = os.path.join('data', file.filename)
        file.save(filepath)
        data = pd.read_csv(filepath)
        
        # Validate dataset
        if not all(col in data.columns for col in ['weight', 'distance', 'carrier', 'cost']):
            flash('Dataset must contain "weight", "distance", "carrier", and "cost" columns', 'danger')
            return redirect(url_for('shipping_cost_page'))
        
        # One-Hot Encoding for 'carrier' column
        encoder = OneHotEncoder(sparse_output=False, drop='first')  # Drop first to avoid multicollinearity
        carrier_encoded = encoder.fit_transform(data[['carrier']])
        carrier_encoded_df = pd.DataFrame(carrier_encoded, columns=encoder.get_feature_names_out(['carrier']))
        
        # Combine encoded carriers with other features
        X = pd.concat([data[['weight', 'distance']], carrier_encoded_df], axis=1)
        y = data['cost']
        
        # Train Random Forest Regression model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Predict cost for user input
        weight = float(request.form['weight'])
        distance = float(request.form['distance'])
        carrier = request.form['carrier']
        
        # Validate user input
        if weight <= 0 or distance <= 0:
            flash('Weight and distance must be positive numbers', 'danger')
            return redirect(url_for('shipping_cost_page'))
        
        # Encode user input carrier
        user_carrier_encoded = encoder.transform([[carrier]])
        
        # Create user input array for prediction
        user_input = np.array([[weight, distance]]).reshape(1, -1)
        user_input = np.hstack([user_input, user_carrier_encoded])
        
        # Predict cost
        predicted_cost = round(model.predict(user_input)[0],3)
        
        return render_template('shipping_cost.html', predicted_cost=f"${predicted_cost:.2f}")
    
    except Exception as e:
        flash(f'Error in shipping cost prediction: {str(e)}', 'danger')
        return redirect(url_for('shipping_cost_page'))

# Events Management
@app.route('/add_event', methods=['POST'])
def add_event():
    try:
        event_name = request.form['event_name']
        event_date = request.form['event_date']
        location = request.form['location']
        attendees = request.form['attendees']
        
        # Validate inputs
        if not event_name or not event_date or not location or not attendees:
            flash('All fields are required', 'danger')
            return redirect(url_for('events_management_page'))
        
        conn = sqlite3.connect('database/events.db')
        c = conn.cursor()
        c.execute("INSERT INTO events (event_name, event_date, location, attendees) VALUES (?, ?, ?, ?)",
                  (event_name, event_date, location, attendees))
        conn.commit()
        conn.close()
        
        flash('Event added successfully', 'success')
        return redirect(url_for('events_management_page'))
    
    except Exception as e:
        flash(f'Error in adding event: {str(e)}', 'danger')
        return redirect(url_for('events_management_page'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)