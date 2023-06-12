# Import relevant libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
 # Load dataset
df = pd.read_csv('ArrahML_project\ForMahasen-Table1.csv')
# Check for missing values
# Drop duplicate rows
df=df.drop_duplicates(keep='first')
df = df.drop('Unnamed: 4', axis=1)
X = df[['Aspect ratio ', 'Clearance ratio ', 'ø (angle in theta)']]
y = df['Drag Coeff. (Out put)']
# Split the dat a into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
clf = DecisionTreeRegressor(max_depth=6, min_samples_leaf=2)
clf.fit(X_train, y_train)
train_accuracy = clf.score(X_train, y_train)
test_accuracy = clf.score(X_test, y_test)

print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

# Predict the labels of the test set
# y_pred = logreg_model.predict(X_test)


# Create the Dash app
app = dash.Dash(__name__)
server = app.server

# Define the layout of the dashboard
app.layout = html.Div(
    children=[
    
    html.H1('Coefficient calculator'),
    
    # Wine quality prediction based on input feature values
    html.H3("Drag Cooficceint prediction "),
    html.Div([
        html.Label("Aspect ratio : "),
        dcc.Input(id='aspectRatio', type='number', required=True),    
        html.Br(),
        html.Label("Clearance ratio : "),
        dcc.Input(id='clearanceRatio', type='number', required=True), 
        html.Br(),
        html.Label("ø (angle in theta)"),
        dcc.Input(id='ø', type='number', required=True),
        html.Br(),
        
      
    ]),

    html.Div([
        html.Button('Predict', id='predict-button', n_clicks=0),
    ]),

    html.Div([
        html.H4("Predicted Cooefficient "),
        html.Div(id='prediction-output')
    ])
])



# Define the callback function to predict wine quality
@app.callback(
    Output(component_id='prediction-output', component_property='children'),
    [Input('predict-button', 'n_clicks')],
    [State('aspectRatio', 'value'),
     State('clearanceRatio', 'value'),
     State('ø', 'value'),
     ]
)
def predict_quality(n_clicks, aspectRatio, clearanceRatioy,ø):
    # Create input features array for prediction
    input_features = np.array([aspectRatio, clearanceRatioy,ø]).reshape(1, -1)

    # Predict the wine quality (0 = bad, 1 = good)
    prediction = clf.predict(input_features)

    # Return the prediction
    
    return 'Thisco is predicted to be.'+  str(prediction)
  


if __name__ == '__main__':
    app.run_server(debug=False)
