# -*- coding: utf-8 -*-
"""
Main Dash Application for Multi-dimensional Classification with MBCs

Implements Borchani + Gil-Begue + Benjumeda algorithms:
- MB-MBC (HITON): HITON-PC + HITON-MB with G² tests
- CB-MBC (wrapper): 3-phase wrapper approach
- TW-MBC (Benjumeda): bounded treewidth learning with exact inference
- Discriminative learning (CLL optimization)
- TSEM: tractable structural EM for missing data

Features:
- Data loading (CSV/Parquet/BIF)
- Algorithm configuration and training
- Structure visualization with cytoscape
- Inference and explanation
- Tractability analysis
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context, dash_table, ALL
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
import base64
import io
import traceback
import logging
import sys
import os
from datetime import datetime
from dash.exceptions import PreventUpdate

# Import CB-MBC and minimal utilities
from mbc.cb_mbc import learn_cb_mbc
from mbc.inference import predict_classes, explain_prediction
from mbc.params import compute_bic_score, validate_cpts

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

print("🚀 MBC DASHBOARD STARTING...")
print(f"Python: {sys.version}")
print(f"Dash version: {dash.__version__}")

# Initialize Dash app - Simplified for better browser compatibility
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        'https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css'
    ],
    assets_folder='/var/www/html/CIGModels/backend/cigmodelsdjango/cigmodelsdjangoapp/mbc-dash/assets',
    requests_pathname_prefix='/Model/LearningFromData/MBCDash/',
    suppress_callback_exceptions=True
)
app.title = "MBC-Dash: Multi-dimensional Bayesian Classifiers"
server = app.server
print("✅ Dash app created")

# Global variables to store data and models
app_data = {
    'df': None,
    'class_vars': [],
    'feature_vars': [],
    'models': {},
    'training_results': {},
    'current_model': None
}

# Notification system
def create_error_notification(message, header="Error"):
    return {'message': message, 'header': header, 'icon': 'danger'}

def create_info_notification(message, header="Info"):
    return {'message': message, 'header': header, 'icon': 'info'}

# Cytoscape stylesheet for graph visualization
cytoscape_stylesheet = [
    {
        'selector': 'node',
        'style': {
            'content': 'data(label)',
            'text-valign': 'center',
            'text-halign': 'center',
            'background-color': '#BFD7B5',
            'border-color': '#A3C4BC',
            'border-width': 2,
            'width': 60,
            'height': 60,
            'font-size': 12
        }
    },
    {
        'selector': 'node[type="class"]',
        'style': {
            'background-color': '#FF6B6B',
            'border-color': '#FF5252'
        }
    },
    {
        'selector': 'node[type="feature"]',
        'style': {
            'background-color': '#4ECDC4',
            'border-color': '#26A69A'
        }
    },
    {
        'selector': 'edge',
        'style': {
            'curve-style': 'bezier',
            'target-arrow-shape': 'triangle',
            'target-arrow-color': '#666',
            'line-color': '#666',
            'width': 2
        }
    },
    {
        'selector': 'edge[type="class"]',
        'style': {
            'line-color': '#FF6B6B',
            'target-arrow-color': '#FF6B6B'
        }
    },
    {
        'selector': 'edge[type="bridge"]',
        'style': {
            'line-color': '#FFA726',
            'target-arrow-color': '#FFA726',
            'width': 3
        }
    },
    {
        'selector': 'edge[type="feature"]',
        'style': {
            'line-color': '#4ECDC4',
            'target-arrow-color': '#4ECDC4'
        }
    }
]

# App layout - Vertical design like MPE Dash
app.layout = html.Div([
    # Link bar
    html.Div(
        className="link-bar",
        style={"textAlign": "center", "marginBottom": "20px"},
        children=[
            html.A(
                children=[
                    html.Img(
                        src="https://cig.fi.upm.es/wp-content/uploads/github.png",
                        style={"height": "24px", "marginRight": "8px"}
                    ),
                    "MBC GitHub"
                ],
                href="https://github.com/",
                target="_blank",
                className="btn btn-outline-info me-2"
            ),
            html.A(
                children=[
                    html.Img(
                        src="https://cig.fi.upm.es/wp-content/uploads/2023/11/cropped-logo_CIG.png",
                        style={"height": "24px", "marginRight": "8px"}
                    ),
                    "Documentation"
                ],
                href="https://github.com/",
                target="_blank",
                className="btn btn-outline-primary me-2"
            ),
        ]
    ),

    html.H1("MBC-Dash: Multi-dimensional Bayesian Classifiers",
            style={'textAlign': 'center', 'marginBottom': '20px'}),

    html.Div([
        html.P(
            "CB-MBC (3-phase wrapper) for multi-dimensional Bayesian classification.",
            style={"textAlign": "center", "maxWidth": "800px", "margin": "0 auto", "marginBottom": "30px"}
        )
    ]),

    dcc.Store(id='notification-store'),
    dcc.Store(id='main-content-trigger', data='init'),

    # Notification container
    html.Div(id='notification-container', style={
        'position': 'fixed',
        'bottom': '20px',
        'right': '20px',
        'zIndex': '1000',
        'width': '300px',
        'transition': 'all 0.3s ease-in-out',
        'transform': 'translateY(100%)',
        'opacity': '0'
    }),

    # Main content container
    html.Div(id="main-content", style={'padding': 20})
])

# Data section - Vertical layout like MPE Dash
def create_data_section():
    return html.Div([
        # (1) Dataset Upload
        html.Div(className="card", children=[
            html.Div([
                html.H3("1. Load Dataset (.csv)", style={'display': 'inline-block', 'marginRight': '10px', 'textAlign': 'center'}),
                dbc.Button(
                    html.I(className="fa fa-question-circle"),
                    id="help-button-data-upload",
                    color="link",
                    style={"display": "inline-block", "verticalAlign": "middle", "padding": "0", "marginLeft": "5px"}
                ),
            ], style={"textAlign": "center", "position": "relative"}),

            html.Div([
                html.Div([
                    html.Img(
                        src="https://img.icons8.com/ios-glyphs/40/cloud--v1.png",
                        className="upload-icon"
                    ),
                    html.Div("Drag and drop or select a CSV file", className="upload-text")
                ]),
                dcc.Upload(
                    id='upload-data',
                    children=html.Div([], style={'display': 'none'}),
                    className="upload-dropzone",
                    multiple=False
                ),
            ], className="upload-card"),

            html.Div(id='data-upload-status', style={'textAlign': 'center', 'color': 'green'}),
        ]),

        # (2) Variable Configuration
        html.Div(className="card", children=[
            html.Div([
                html.H3("2. Select Variables", style={'display': 'inline-block', 'marginRight': '10px', 'textAlign': 'center'}),
            ], style={"textAlign": "center", "position": "relative"}),

            html.Div(id='variable-config'),
        ]),

        # (3) Data Preview
        html.Div(className="card", children=[
            html.Div([
                html.H3("3. Data Preview", style={'display': 'inline-block', 'marginRight': '10px', 'textAlign': 'center'}),
            ], style={"textAlign": "center", "position": "relative"}),

            html.Div(id='data-preview')
        ])
    ])

# Algorithm section - Vertical layout like MPE Dash
def create_algorithm_section():
    return html.Div([
        # (4) Algorithm
        html.Div(className="card", children=[
            html.Div([
                html.H3("4. Algorithm", style={'display': 'inline-block', 'marginRight': '10px', 'textAlign': 'center'}),
            ], style={"textAlign": "center", "position": "relative"}),
            html.Div("Using CB-MBC (3-phase wrapper)", style={'textAlign': 'center', 'marginBottom': 20})
        ]),

        # (5) Parameters (CB-MBC only)
        html.Div(className="card", children=[
            html.Div([
                html.H3("5. CB-MBC Parameters", style={'display': 'inline-block', 'marginRight': '10px', 'textAlign': 'center'}),
            ], style={"textAlign": "center", "position": "relative"}),

            html.Div([
                html.Label("Max Iterations Phase II (T):"),
                dcc.Slider(id='cb-max-iter-slider', min=5, max=50, step=5, value=10,
                          marks={5: '5', 10: '10', 20: '20', 50: '50'}),

                html.Label("Evaluation Metric:"),
                dcc.Dropdown(
                    id='cb-metric-dropdown',
                    options=[
                        {'label': 'Global Accuracy', 'value': 'global'},
                        {'label': 'Mean-Hamming Distance', 'value': 'hamming'}
                    ],
                    value='global'
                )
            ], style={'marginBottom': 20})
        ]),

        # (6) Advanced options (removed for CB-only)
        html.Div(),

        # (7) Train Models
        html.Div([
            dbc.Button(
                [
                    html.I(className="fas fa-play-circle me-2"),
                    "Train CB-MBC"
                ],
                id='train-button',
                n_clicks=0,
                color="info",
                className="btn-lg",
                style={
                    'fontSize': '1.1rem',
                    'padding': '0.75rem 2rem',
                    'borderRadius': '8px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                    'transition': 'all 0.3s ease',
                    'backgroundColor': '#00A2E1',
                    'border': 'none',
                    'margin': '1rem 0',
                    'color': 'white',
                    'fontWeight': '500'
                }
            )
        ], style={'textAlign': 'center'}),

        html.Div(id='training-status', style={'marginTop': 20}),
        html.Div(id='training-results', style={'marginTop': 20})
    ])

# Inference section - Vertical layout like MPE Dash
def create_inference_section():
    return html.Div([
        # (8) Model Selection
        html.Div(className="card", children=[
            html.Div([
                html.H3("8. Model Selection", style={'display': 'inline-block', 'marginRight': '10px', 'textAlign': 'center'}),
            ], style={"textAlign": "center", "position": "relative"}),

            dcc.Dropdown(
                id='model-selection-dropdown',
                placeholder="Select a trained model...",
                style={'marginBottom': 20}
            ),
        ]),

        # (9) Evidence Input
        html.Div(className="card", children=[
            html.Div([
                html.H3("9. Evidence Input", style={'display': 'inline-block', 'marginRight': '10px', 'textAlign': 'center'}),
            ], style={"textAlign": "center", "position": "relative"}),

            html.Div(id='evidence-input'),

            html.Div([
                dbc.Button(
                    [
                        html.I(className="fas fa-search me-2"),
                        "Predict"
                    ],
                    id='predict-button',
                    n_clicks=0,
                    color="success",
                    className="btn-lg",
                    style={
                        'fontSize': '1.1rem',
                        'padding': '0.75rem 2rem',
                        'borderRadius': '8px',
                        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                        'transition': 'all 0.3s ease',
                        'backgroundColor': '#28a745',
                        'border': 'none',
                        'margin': '1rem 0',
                        'color': 'white',
                        'fontWeight': '500'
                    }
                )
            ], style={'textAlign': 'center'}),
        ]),

        # (10) Results
        html.Div(className="card", children=[
            html.Div([
                html.H4("Prediction Results", style={'display': 'inline-block', 'marginRight': '10px', 'textAlign': 'center'}),
            ], style={"textAlign": "center", "position": "relative"}),
            html.Div(id='prediction-results')
        ]),

        # (11) Explanation
        html.Div(className="card", children=[
            html.Div([
                html.H4("Explanation", style={'display': 'inline-block', 'marginRight': '10px', 'textAlign': 'center'}),
            ], style={"textAlign": "center", "position": "relative"}),
            html.Div(id='prediction-explanation')
        ])
    ])

# Tractability section - Vertical layout like MPE Dash
def create_tractability_section():
    return html.Div([
        # (12) Structure Visualization
        html.Div(className="card", children=[
            html.Div([
                html.H3("12. Model Structure Visualization", style={'display': 'inline-block', 'marginRight': '10px', 'textAlign': 'center'}),
            ], style={"textAlign": "center", "position": "relative"}),

            cyto.Cytoscape(
                id='structure-graph',
                elements=[],
                stylesheet=cytoscape_stylesheet,
                style={'width': '100%', 'height': '400px'},
                layout={'name': 'cose'}
            )
        ]),

        # (13) Tractability Metrics
        html.Div(className="card", children=[
            html.Div([
                html.H4("Tractability Metrics", style={'display': 'inline-block', 'marginRight': '10px', 'textAlign': 'center'}),
            ], style={"textAlign": "center", "position": "relative"}),
            html.Div(id='tractability-metrics')
        ]),

        # (14) Complexity Analysis
        html.Div(className="card", children=[
            html.Div([
                html.H4("Complexity Analysis", style={'display': 'inline-block', 'marginRight': '10px', 'textAlign': 'center'}),
            ], style={"textAlign": "center", "position": "relative"}),
            html.Div(id='complexity-analysis')
        ])
    ])

# Main content callback - Vertical layout like MPE Dash
@app.callback(Output('main-content', 'children'),
              Input('main-content-trigger', 'data'),
              prevent_initial_call=False)
def render_main_content(trigger_data):
    """Render all content vertically like MPE Dash"""
    logger.info(f"Rendering main content, trigger: {trigger_data}")

    try:
        # Return full content with all sections
        return html.Div([
            # Data section
            create_data_section(),

            # Algorithm section
            create_algorithm_section(),

            # Inference section
            create_inference_section(),

            # Tractability section
            create_tractability_section()
        ], style={'padding': '20px', 'maxWidth': '1200px', 'margin': '0 auto'})
    except Exception as e:
        logger.error(f"Error rendering main content: {e}")
        return html.Div([
            html.H2("❌ Error Loading MBC Dashboard", style={'textAlign': 'center', 'color': 'red'}),
            html.P(f"Error: {str(e)}", style={'textAlign': 'center', 'color': 'red'})
        ], style={'padding': '40px'})

# Data upload callback
@app.callback([Output('data-upload-status', 'children'),
               Output('variable-config', 'children'),
               Output('data-preview', 'children')],
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'))
def update_data_upload(contents, filename):
    if contents is None:
        return "", "", ""
    
    try:
        # Parse uploaded file
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        if filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif filename.endswith('.parquet'):
            df = pd.read_parquet(io.BytesIO(decoded))
        else:
            return html.Div("Unsupported file format. Please use CSV or Parquet.", 
                           style={'color': 'red'}), "", ""
        
        # Store data globally
        app_data['df'] = df
        
        # Create variable configuration interface
        var_config = html.Div([
            html.Label("Select Class Variables (≥1):"),
            dcc.Checklist(
                id='class-var-checklist',
                options=[{'label': f'Column {i}: {col}', 'value': i} 
                        for i, col in enumerate(df.columns)],
                value=[],
                style={'marginBottom': 10}
            ),
            
            html.Label("Select Feature Variables:"),
            dcc.Checklist(
                id='feature-var-checklist',
                options=[{'label': f'Column {i}: {col}', 'value': i} 
                        for i, col in enumerate(df.columns)],
                value=[],
                style={'marginBottom': 10}
            ),
            
            html.Button('Confirm Selection', id='confirm-vars-button', n_clicks=0,
                       style={'backgroundColor': '#FF9800', 'color': 'white', 'padding': '8px 16px',
                             'border': 'none', 'borderRadius': '3px', 'cursor': 'pointer'})
        ])
        
        # Create data preview
        preview = html.Div([
            html.H5(f"Dataset: {filename}"),
            html.P(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns"),
            html.P(f"Missing values: {df.isnull().sum().sum()}"),
            dash_table.DataTable(
                data=df.head(10).to_dict('records'),
                columns=[{"name": i, "id": i} for i in df.columns],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '10px'},
                style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
            )
        ])
        
        status = html.Div(f"✅ Successfully loaded {filename}", style={'color': 'green'})
        
        return status, var_config, preview
        
    except Exception as e:
        return html.Div(f"❌ Error loading file: {str(e)}", style={'color': 'red'}), "", ""

# Variable selection callback
@app.callback(Output('data-upload-status', 'children', allow_duplicate=True),
              [Input('confirm-vars-button', 'n_clicks')],
              [State('class-var-checklist', 'value'),
               State('feature-var-checklist', 'value')],
              prevent_initial_call=True)
def confirm_variable_selection(n_clicks, class_vars, feature_vars):
    if n_clicks == 0:
        return ""
    
    if not class_vars:
        return html.Div("❌ Please select at least one class variable", style={'color': 'red'})
    
    if not feature_vars:
        return html.Div("❌ Please select at least one feature variable", style={'color': 'red'})
    
    # Check for overlap
    if set(class_vars) & set(feature_vars):
        return html.Div("❌ Class and feature variables cannot overlap", style={'color': 'red'})
    
    # Store variable selection
    app_data['class_vars'] = class_vars
    app_data['feature_vars'] = feature_vars
    
    return html.Div(f"✅ Variables configured: {len(class_vars)} classes, {len(feature_vars)} features", 
                   style={'color': 'green'})

# Training callback
@app.callback([Output('training-status', 'children'),
               Output('training-results', 'children')],
              Input('train-button', 'n_clicks'),
              [State('cb-max-iter-slider', 'value'),
               State('cb-metric-dropdown', 'value')])
def train_models(n_clicks, cb_max_iter, cb_metric):
    if n_clicks == 0 or app_data['df'] is None:
        return "", ""
    
    if not app_data['class_vars'] or not app_data['feature_vars']:
        return html.Div("❌ Please configure variables first", style={'color': 'red'}), ""
    
    try:
        df = app_data['df']
        class_vars = app_data['class_vars']
        feature_vars = app_data['feature_vars']
        
        status_updates = []
        results = []
        app_data['models'] = {}
        app_data['training_results'] = {}
        
        # Train CB-MBC only
        status_updates.append("🔄 Training CB-MBC...")
        start_time = datetime.now()
        model = learn_cb_mbc(df, class_vars, feature_vars, metric=cb_metric, max_iterations=cb_max_iter)
        app_data['models'] = {'CB-MBC': model}
        training_time = (datetime.now() - start_time).total_seconds()

        # Evaluate model
        if hasattr(model, 'cpts') and model.cpts:
            graph = model.get_graph() if hasattr(model, 'get_graph') else (model.G if hasattr(model, 'G') else None)
            bic_score = compute_bic_score(df, model.cpts, graph) if graph is not None else 0.0
            validation = validate_cpts(model.cpts)
        else:
            bic_score = 0.0
            validation = {'valid': False, 'errors': ['No parameters learned']}

        # Structure info
        if hasattr(model, 'get_structure_info'):
            struct_info = model.get_structure_info()
        elif hasattr(model, 'get_graph'):
            struct_info = {'total_edges': model.get_graph().number_of_edges()}
        elif hasattr(model, 'G'):
            struct_info = {'total_edges': model.G.number_of_edges()}
        else:
            struct_info = {'total_edges': 0}

        tw_info = {'treewidth_estimate': 'Unknown', 'tractable': False}

        app_data['training_results'] = {
            'cb_mbc': {
                'training_time': training_time,
                'bic_score': bic_score,
                'structure_info': struct_info,
                'treewidth_info': tw_info,
                'validation': validation
            }
        }

        status_updates.append(f"✅ CB-MBC completed in {training_time:.2f}s")
        
        # Handle TSEM if selected
        if 'tsem' in advanced_options and df.isnull().sum().sum() > 0:
            status_updates.append("🔄 Running TSEM for missing data...")
            
            tsem_result = tsem(df, class_vars, feature_vars, tw_max=tw_max, max_iter=10, verbose=False)
            app_data['models']['TSEM'] = tsem_result['model']
            app_data['training_results']['tsem'] = {
                'converged': tsem_result['converged'],
                'iterations': tsem_result['iterations'],
                'completed_data': tsem_result['completed_data']
            }
            
            status_updates.append(f"✅ TSEM completed ({tsem_result['iterations']} iterations)")
        
        # Create results summary
        results_table_data = []
        for alg, result in app_data['training_results'].items():
            results_table_data.append({
                'Algorithm': alg.upper(),
                'Training Time (s)': f"{result.get('training_time', 0):.2f}",
                'BIC Score': f"{result.get('bic_score', 0):.2f}",
                'Edges': result.get('structure_info', {}).get('total_edges', 0),
                'Treewidth': result.get('treewidth_info', {}).get('treewidth_estimate', 'N/A'),
                'Tractable': '✅' if result.get('treewidth_info', {}).get('tractable', False) else '❌'
            })
        
        results_display = html.Div([
            html.H4("Training Results Summary"),
            dash_table.DataTable(
                data=results_table_data,
                columns=[{"name": i, "id": i} for i in results_table_data[0].keys()] if results_table_data else [],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'center', 'padding': '10px'},
                style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                style_data_conditional=[
                    {
                        'if': {'filter_query': '{Tractable} = ✅'},
                        'backgroundColor': '#E8F5E8',
                    },
                    {
                        'if': {'filter_query': '{Tractable} = ❌'},
                        'backgroundColor': '#FFE8E8',
                    }
                ]
            )
        ])
        
        status = html.Div([html.P(update) for update in status_updates])
        
        return status, results_display
        
    except Exception as e:
        error_msg = f"❌ Training failed: {str(e)}"
        logger.error(f"Training error: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return "", create_error_notification(error_msg, "Training Error")

# Model selection dropdown update
@app.callback(Output('model-selection-dropdown', 'options'),
              Input('training-results', 'children'))
def update_model_dropdown(training_results):
    if not app_data['models']:
        return []
    
    return [{'label': model_name, 'value': model_name} for model_name in app_data['models'].keys()]

# Evidence input generation
@app.callback(Output('evidence-input', 'children'),
              Input('model-selection-dropdown', 'value'))
def create_evidence_input(selected_model):
    if not selected_model or not app_data['feature_vars']:
        return ""
    
    df = app_data['df']
    feature_vars = app_data['feature_vars']
    
    inputs = []
    for i, feature_var in enumerate(feature_vars):
        col_name = df.columns[feature_var]
        unique_values = sorted(df.iloc[:, feature_var].dropna().unique())
        
        inputs.append(html.Div([
            html.Label(f"Feature {feature_var} ({col_name}):"),
            dcc.Dropdown(
                id=f'evidence-{feature_var}',
                options=[{'label': str(val), 'value': val} for val in unique_values],
                placeholder=f"Select {col_name}...",
                style={'marginBottom': 10}
            )
        ]))
    
    return html.Div(inputs)

# Prediction callback
@app.callback([Output('prediction-results', 'children'),
               Output('prediction-explanation', 'children')],
              Input('predict-button', 'n_clicks'),
              [State('model-selection-dropdown', 'value')] + 
              [State(f'evidence-{var}', 'value') for var in app_data.get('feature_vars', [])])
def make_prediction(n_clicks, selected_model, *evidence_values):
    if n_clicks == 0 or not selected_model:
        return "", ""
    
    try:
        model = app_data['models'][selected_model]
        feature_vars = app_data['feature_vars']
        class_vars = app_data['class_vars']
        
        # Build evidence dictionary
        evidence = {}
        for i, value in enumerate(evidence_values):
            if value is not None and i < len(feature_vars):
                evidence[feature_vars[i]] = value
        
        if not evidence:
            return html.Div("Please provide at least one evidence value", style={'color': 'orange'}), ""
        
        # Make prediction
        prediction = predict_classes(model, evidence)
        
        # Format results
        mpe_result = prediction['mpe']
        marginals = prediction['marginals']
        confidence = prediction['confidence']
        
        # Create results display
        results_data = []
        for class_var in class_vars:
            col_name = app_data['df'].columns[class_var]
            predicted_value = mpe_result.get(class_var, 'Unknown')
            
            if class_var in marginals:
                prob_dist = marginals[class_var]
                max_prob = max(prob_dist.values()) if prob_dist else 0.0
            else:
                max_prob = confidence
            
            results_data.append({
                'Class Variable': f"{class_var} ({col_name})",
                'Predicted Value': str(predicted_value),
                'Confidence': f"{max_prob:.3f}"
            })
        
        results_table = dash_table.DataTable(
            data=results_data,
            columns=[{"name": i, "id": i} for i in results_data[0].keys()],
            style_cell={'textAlign': 'center', 'padding': '10px'},
            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
        )
        
        # Generate explanation
        explanation = explain_prediction(model, evidence, mpe_result)
        
        explanation_content = html.Div([
            html.H5("Prediction Factors:"),
            html.Ul([
                html.Li(f"Variable {factor['variable']}: predicted {factor['predicted_value']} "
                       f"with probability {factor['probability']:.3f}")
                for factor in explanation['factors']
            ])
        ])
        
        return results_table, explanation_content
        
    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        logger.error(f"Prediction error: {e}")
        return "", create_error_notification(error_msg, "Prediction Error")

# Structure visualization callback
@app.callback([Output('structure-graph', 'elements'),
               Output('tractability-metrics', 'children'),
               Output('complexity-analysis', 'children')],
              Input('model-selection-dropdown', 'value'))
def update_structure_visualization(selected_model):
    if not selected_model or selected_model not in app_data['models']:
        return [], "", ""
    
    try:
        model = app_data['models'][selected_model]
        class_vars = app_data['class_vars']
        feature_vars = app_data['feature_vars']
        df = app_data['df']
        
        # Create cytoscape elements
        elements = []
        
        # Add nodes
        for var in class_vars + feature_vars:
            col_name = df.columns[var]
            node_type = 'class' if var in class_vars else 'feature'
            
            elements.append({
                'data': {
                    'id': str(var),
                    'label': f"{var}\n({col_name[:8]}...)" if len(col_name) > 8 else f"{var}\n({col_name})",
                    'type': node_type
                }
            })
        
        # Add edges
        if hasattr(model, 'G'):
            for edge in model.G.edges():
                parent, child = edge
                
                # Determine edge type
                if parent in class_vars and child in class_vars:
                    edge_type = 'class'
                elif parent in class_vars and child in feature_vars:
                    edge_type = 'bridge'
                else:
                    edge_type = 'feature'
                
                elements.append({
                    'data': {
                        'id': f"{parent}-{child}",
                        'source': str(parent),
                        'target': str(child),
                        'type': edge_type
                    }
                })
        elif hasattr(model, 'edges'):
            for edge in model.edges:
                parent, child = edge
                edge_type = 'bridge' if parent in class_vars else 'feature'
                
                elements.append({
                    'data': {
                        'id': f"{parent}-{child}",
                        'source': str(parent),
                        'target': str(child),
                        'type': edge_type
                    }
                })
        
        # Tractability metrics
        metrics = []
        
        if hasattr(model, 'get_treewidth_info'):
            tw_info = model.get_treewidth_info()
            metrics.extend([
                html.P(f"Treewidth Estimate: {tw_info['treewidth_estimate']}"),
                html.P(f"Tractable: {'✅ Yes' if tw_info['tractable'] else '❌ No'}"),
                html.P(f"Elimination Order: {tw_info.get('elimination_order', 'N/A')}")
            ])
        
        if hasattr(model, 'get_structure_info'):
            struct_info = model.get_structure_info()
            metrics.extend([
                html.P(f"Total Edges: {struct_info['total_edges']}"),
                html.P(f"Class Configurations: {struct_info.get('class_configs', 'N/A')}")
            ])
        
        # Complexity analysis
        n_classes = len(class_vars)
        n_features = len(feature_vars)
        
        # Estimate inference complexity
        if hasattr(model, 'treewidth_estimate'):
            tw = model.treewidth_estimate
            if isinstance(tw, (int, float)) and tw <= 10:
                inference_complexity = f"O(n * d^{tw})"
                tractable_note = "Exact inference is tractable"
            else:
                inference_complexity = f"O(d^{n_classes})"
                tractable_note = "May require approximation"
        else:
            inference_complexity = f"O(d^{n_classes})"
            tractable_note = "Complexity depends on structure"
        
        complexity_content = html.Div([
            html.P(f"Variables: {n_classes} classes, {n_features} features"),
            html.P(f"Inference Complexity: {inference_complexity}"),
            html.P(f"Note: {tractable_note}"),
            html.P(f"Learning Time: {app_data['training_results'].get(selected_model.lower().replace('-', '_').replace(' (discriminative)', ''), {}).get('training_time', 'N/A')} seconds")
        ])
        
        return elements, html.Div(metrics), complexity_content
        
    except Exception as e:
        error_msg = f"Visualization error: {str(e)}"
        logger.error(f"Visualization error: {e}")
        return [], "", create_error_notification(error_msg, "Visualization Error")

# Notification callback
@app.callback(
    [Output('notification-container', 'children'),
     Output('notification-container', 'style')],
    Input('notification-store', 'data')
)
def show_notification(data):
    if data is None:
        return None, {
            'position': 'fixed', 'bottom': '20px', 'right': '20px', 'zIndex': '1000',
            'width': '300px', 'transition': 'all 0.3s ease-in-out',
            'transform': 'translateY(100%)', 'opacity': '0'
        }

    toast = dbc.Toast(
        data['message'], header=data['header'], icon=data['icon'],
        is_open=True, dismissable=True,
        style={'width': '100%', 'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
               'borderRadius': '8px', 'marginBottom': '10px'}
    )

    container_style = {
        'position': 'fixed', 'bottom': '20px', 'right': '20px', 'zIndex': '1000',
        'width': '300px', 'transition': 'all 0.3s ease-in-out',
        'transform': 'translateY(0)', 'opacity': '1'
    }

    return toast, container_style

# Expose server for gunicorn
server = app.server

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8058)
