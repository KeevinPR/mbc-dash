# -*- coding: utf-8 -*-
"""
Dash Application for Multi-dimensional Bayesian Classifier (MBC)

This app allows training a multi-dimensional Bayesian network classifier on a dataset.
"""

import sys, os, io, base64, logging
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc
from dash import Input, Output, State, ALL
from dash.exceptions import PreventUpdate
import pandas as pd
import dash_cytoscape as cyto

# R interface setup
from rpy2 import robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector
from rpy2.robjects import conversion

# Create a local converter to avoid global state issues
pandas_converter = conversion.Converter('pandas converter')
pandas_converter += pandas2ri.converter

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

print("🚀 MBC DASHBOARD STARTING...")
print(f"Python: {sys.version}")
print(f"Dash version: {dash.__version__}")

# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
R_SRC_PATH = os.path.join(current_dir, "mbc.R")  # Path to the R script with MBC functions

# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        'https://bayes-interpret.com/Model/LearningFromData/MBCDash/assets/liquid-glass.css' # Apple Liquid Glass CSS
    ],
    requests_pathname_prefix='/Model/LearningFromData/MBCDash/',
    suppress_callback_exceptions=True
)
server = app.server

# Cytoscape stylesheet for network visualization - CIG Corporate Style
cytoscape_stylesheet = [
    {
        'selector': 'node',
        'style': {
            'content': 'data(label)',
            'text-valign': 'center',
            'text-halign': 'center',
            'background-color': '#E3F2FD',  # Light blue
            'border-color': '#90CAF9',
            'border-width': 2,
            'width': 50,
            'height': 50,
            'font-size': 11,
            'color': '#333',
            'text-wrap': 'wrap',
            'text-max-width': '80px'
        }
    },
    {
        'selector': 'node[type="class"]',
        'style': {
            'background-color': '#00A2E1',  # CIG Corporate Blue
            'border-color': '#0077A8',
            'shape': 'ellipse',
            'color': '#FFFFFF',  # White text for better contrast
            'font-weight': 'bold',
            'border-width': 3
        }
    },
    {
        'selector': 'node[type="feature"]',
        'style': {
            'background-color': '#E3F2FD',  # Very light blue
            'border-color': '#90CAF9',
            'shape': 'rectangle',
            'color': '#0077A8',
            'border-width': 2
        }
    },
    {
        'selector': 'edge',
        'style': {
            'curve-style': 'bezier',
            'target-arrow-shape': 'triangle',
            'target-arrow-color': '#666',
            'line-color': '#666',
            'width': 2,
            'opacity': 0.7
        }
    },
    {
        'selector': 'edge[type="class"]',
        'style': {
            'line-color': '#0077A8',  # Dark CIG blue
            'target-arrow-color': '#0077A8',
            'width': 2.5,
            'opacity': 0.8
        }
    },
    {
        'selector': 'edge[type="bridge"]',
        'style': {
            'line-color': '#00A2E1',  # CIG Corporate Blue
            'target-arrow-color': '#00A2E1',
            'width': 3,
            'opacity': 0.9
        }
    },
    {
        'selector': 'edge[type="feature"]',
        'style': {
            'line-color': '#90CAF9',  # Light blue
            'target-arrow-color': '#90CAF9',
            'width': 2,
            'opacity': 0.6
        }
    }
]

# Helper: Ensure R environment is ready and source the MBC R script
def ensure_r_ready():
    try:
        importr('bnlearn')
        importr('arules')
        # gRain is optional; load if available (for completeness of R functions)
        try:
            importr('gRain')
        except Exception:
            pass
    except Exception as e:
        raise RuntimeError(
            "Missing required R packages for MBC. Please install 'bnlearn', 'arules', and 'gRain' in R. "
            f"R error: {e}"
        )
    # Load the R script with MBC functions
    try:
        ro.r['source'](R_SRC_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to source R file {R_SRC_PATH}: {e}")

# App layout
app.layout = html.Div([
    # Data stores
    dcc.Store(id='mbc-dataset-store'),
    dcc.Store(id='mbc-columns-store'),
    dcc.Store(id='mbc-classes-selected', data=[]),
    dcc.Store(id='mbc-features-selected', data=[]),
    dcc.Store(id='mbc-network-store'),  # Store network structure
    dcc.Store(id='mbc-model-trained', data=False),  # Flag for trained model
    dcc.Store(id='mbc-feature-values-store'),  # Store feature values/levels
    dcc.Store(id='notification-store'),
    dcc.Store(id='dataset-change-detector', data=None),  # Detect dataset changes
    dcc.Store(id='previous-classes-selection', data=[]),  # Track previous class selections
    dcc.Store(id='previous-features-selection', data=[]),  # Track previous feature selections
    
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

    dcc.Loading(
        id="global-spinner", 
        type="default", 
        fullscreen=False,
        color="#00A2E1",
        style={
            "position": "fixed",
            "top": "50%",
            "left": "50%",
            "transform": "translate(-50%, -50%)",
            "zIndex": "999999"
        },
        children=html.Div([
            html.H1("Multi-dimensional Bayesian Classifier ", style={'textAlign': 'center'}),

            ########################################################
            # Info text
            ########################################################
            html.Div(
                className="link-bar",
                style={
                    "textAlign": "center",
                    "marginBottom": "20px"
                },
                children=[
                    html.A(
                        children=[
                            html.Img(
                                src="https://cig.fi.upm.es/wp-content/uploads/2023/11/cropped-logo_CIG.png",
                                style={"height": "24px", "marginRight": "8px"}
                            ),
                            "Proceeding Paper"
                        ],
                        href="https://cig.fi.upm.es/wp-content/uploads/2024/01/bielza_IJAR2011-2.pdf",
                        target="_blank",
                        className="btn btn-outline-primary me-2"
                    ),
                    html.A(
                        children=[
                            html.Img(
                                src="https://cig.fi.upm.es/wp-content/uploads/github.png",
                                style={"height": "24px", "marginRight": "8px"}
                            ),
                            "Dash Adapted GitHub"
                        ],
                        href="https://github.com/KeevinPR/mbc-dash",
                        target="_blank",
                        className="btn btn-outline-info me-2"
                    ),
                ]
            ),
            ########################################################
            # Short explanatory text
            ########################################################
            html.Div(
                [
                    html.P(
                        "Multi-dimensional Bayesian Classifiers (MBC) extend traditional Bayesian networks "
                        "to predict multiple class variables simultaneously. Upload your dataset, select "
                        "class and feature variables, and train an MBC model to see the learned structure "
                        "and performance metrics.",
                        style={"textAlign": "center", "maxWidth": "800px", "margin": "0 auto"}
                    )
                ],
                style={"marginBottom": "20px"}
            ),

            ########################################################
            # (A) Data upload
            ########################################################
            html.Div(className="card", children=[
                html.H3("1. Upload Dataset (CSV)", style={'textAlign': 'center'}),

                # Container "card"
                html.Div([
                    # Top part with icon and text
                    html.Div([
                        html.Img(
                            src="https://img.icons8.com/ios-glyphs/40/cloud--v1.png",
                            className="upload-icon"
                        ),
                        html.Div("Drag and drop or select a CSV file", className="upload-text")
                    ]),
                    
                    # Upload component
                    dcc.Upload(
                        id='mbc-upload-csv',
                        children=html.Div([], style={'display': 'none'}),
                        className="upload-dropzone",
                        multiple=False
                    ),
                ], className="upload-card"),

                # Use default dataset + help icon
                html.Div([
                    dcc.Checklist(
                        id='use-default-dataset',
                        options=[{'label': 'Use the default dataset', 'value': 'default'}],
                        value=[],
                        style={'display': 'inline-block', 'textAlign': 'center', 'marginTop': '10px'}
                    ),
                    dbc.Button(
                        html.I(className="fa fa-question-circle"),
                        id="help-button-default-dataset",
                        color="link",
                        style={"display": "inline-block", "marginLeft": "8px"}
                    ),
                ], style={'textAlign': 'center'}),

                # Upload status
                html.Div(id='mbc-upload-status', style={'textAlign': 'center', 'color': 'green'}),
            ]),

            # 2. Select Classes
            html.Div(className="card", children=[
                html.Div([
                    html.H3("2. Select Classes", style={'display': 'inline-block', 'marginRight': '10px', 'textAlign': 'center'}),
                    dbc.Button(
                        html.I(className="fa fa-question-circle"), 
                        id="help-button-mbc-classes",
                        color="link", 
                        style={"display": "inline-block", "verticalAlign": "middle", "padding": "0", "marginLeft": "5px"}
                    ),
                ], style={"textAlign": "center", "position": "relative"}),
                html.Div([
                    dbc.Button("Select All", id="mbc-select-all-classes", color="outline-primary", size="sm", style={'marginRight': '10px'}),
                    dbc.Button("Clear All", id="mbc-clear-classes", color="outline-secondary", size="sm"),
                ], style={'textAlign': 'center', 'marginBottom': '15px'}),
                html.Div(id='mbc-class-checkbox-container', style={
                    'maxHeight': '200px', 'overflowY': 'auto', 'border': '1px solid #ddd',
                    'borderRadius': '5px', 'padding': '10px', 'margin': '0 auto', 'width': '80%',
                    'backgroundColor': '#f8f9fa'
                }),
            ]),

            # 3. Select Features
            html.Div(className="card", children=[
                html.Div([
                    html.H3("3. Select Features", style={'display': 'inline-block', 'marginRight': '10px', 'textAlign': 'center'}),
                    dbc.Button(
                        html.I(className="fa fa-question-circle"), 
                        id="help-button-mbc-features",
                        color="link", 
                        style={"display": "inline-block", "verticalAlign": "middle", "padding": "0", "marginLeft": "5px"}
                    ),
                ], style={"textAlign": "center", "position": "relative"}),
                html.Div([
                    dbc.Button("Select All", id="mbc-select-all-features", color="outline-primary", size="sm", style={'marginRight': '10px'}),
                    dbc.Button("Clear All", id="mbc-clear-features", color="outline-secondary", size="sm"),
                ], style={'textAlign': 'center', 'marginBottom': '15px'}),
                html.Div(id='mbc-feature-checkbox-container', style={
                    'maxHeight': '220px', 'overflowY': 'auto', 'border': '1px solid #ddd',
                    'borderRadius': '5px', 'padding': '10px', 'margin': '0 auto', 'width': '80%',
                    'backgroundColor': '#f8f9fa'
                }),
                html.Div([
                    html.I(className="fa fa-info-circle", style={'marginRight': '5px', 'color': '#6c757d'}),
                    html.Span("Classes and features must be disjoint sets. When you select a variable as a class, it will disappear from the feature list (and vice versa).", 
                             style={'fontSize': '11px', 'color': '#6c757d'}),
                ], style={'textAlign': 'center', 'marginTop': '8px'}),
            ]),

            # 4. Options (Approach and others)
            html.Div(className="card", children=[
                html.Div([
                    html.H3("4. Algorithm Options", style={'display': 'inline-block', 'marginRight': '10px', 'textAlign': 'center'}),
                    dbc.Button(
                        html.I(className="fa fa-question-circle"), 
                        id="help-button-mbc-options",
                        color="link", 
                        style={"display": "inline-block", "verticalAlign": "middle", "padding": "0", "marginLeft": "5px"}
                    ),
                ], style={"textAlign": "center", "position": "relative"}),
                dbc.Row([
                    dbc.Col([
                        html.Label("Approach", style={'fontWeight': '500'}),
                        dbc.RadioItems(
                            id='mbc-approach-radio',
                            options=[
                                {'label': 'Filter (BIC)', 'value': 'filter'},
                                {'label': 'Wrapper (Accuracy)', 'value': 'wrapper'},
                            ],
                            value='filter', inline=True,
                        ),
                    ], md=6),
                    dbc.Col([
                        html.Label("Wrapper Measure", style={'fontWeight': '500'}),
                        dbc.RadioItems(
                            id='mbc-measure-radio',
                            options=[
                                {'label': 'Global accuracy', 'value': 'global'},
                                {'label': 'Average accuracy', 'value': 'average'},
                            ],
                            value='global', inline=True,
                        ),
                    ], md=6),
                ], style={'marginTop': '10px'}),
                dbc.Row([
                    dbc.Col([
                        html.Label("Discretization", style={'fontWeight': '500'}),
                        dbc.Select(
                            id='mbc-disc-method',
                            options=[
                                {'label': 'frequency (3 bins)', 'value': 'frequency'},
                                {'label': 'cluster (3 bins)', 'value': 'cluster'},
                                {'label': 'none (assume factors)', 'value': 'none'},
                            ],
                            value='frequency',
                            style={
                                'border': '1px solid #d0d7de',
                                'borderRadius': '6px',
                                'padding': '8px 12px',
                                'backgroundColor': 'rgba(255, 255, 255, 0.8)',
                                'backdropFilter': 'blur(10px)',
                                'boxShadow': '0 1px 3px rgba(0, 0, 0, 0.1)',
                                'transition': 'all 0.2s ease',
                                'fontSize': '14px'
                            }
                        ),
                    ], md=6),
                    # Hidden component to maintain train_split value (80% default)
                    dcc.Store(id='mbc-train-split', data=80),
                ], style={'marginTop': '10px'}),
            ]),

            # Run button
            html.Div([
                dbc.Button(
                    [
                        html.I(className="fas fa-play-circle me-2"),
                        "Train MBC Model"
                    ],
                    id='mbc-run-button',
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

            # 5. Network Visualization
            html.Div(id='mbc-network-container', style={'marginTop': '20px'}),

            # 6. Performance Results
            html.Div(id='mbc-results', style={'textAlign': 'center', 'marginBottom': '2rem'}),

            # 7. Individual Prediction Section
            html.Div(id='mbc-prediction-section', style={'marginTop': '20px'}),
        ])
    ),

    dbc.Popover(
        [
            dbc.PopoverHeader(
                [
                    "Help",
                    html.I(className="fa fa-info-circle ms-2", style={"color": "#0d6efd"})
                ],
                style={
                    "backgroundColor": "#f8f9fa",  # Light gray background
                    "fontWeight": "bold"
                }
            ),
            dbc.PopoverBody(
                [
                    html.P(
                        [
                            "For details and content of the dataset, check out: ",
                            html.A(
                                "asia10KClassmayus.csv",
                                href="https://github.com/KeevinPR/mbc-dash/blob/main/sample_data/asia10KClassmayus.csv",
                                target="_blank",
                                style={"textDecoration": "underline", "color": "#0d6efd"}
                            ),
                        ]
                    ),
                    html.Hr(),  # Horizontal rule for a modern divider
                    html.P("Feel free to upload your own dataset at any time.")
                ],
                style={
                    "backgroundColor": "#ffffff",
                    "borderRadius": "0 0 0.25rem 0.25rem"
                }
            ),
        ],
        id="help-popover-default-dataset",
        target="help-button-default-dataset",
        placement="right",
        is_open=False,
        trigger="hover"
    ),
    
    dbc.Popover(
        [
            dbc.PopoverHeader(
                [
                    "Class Variables",
                    html.I(className="fa fa-info-circle ms-2", style={"color": "#0d6efd"})
                ],
                style={"backgroundColor": "#f8f9fa", "fontWeight": "bold"}
            ),
            dbc.PopoverBody(
                [
                    html.P("Select the label columns (class variables) to predict."),
                    html.P("Key points:"),
                    html.Ul([
                        html.Li("Can select one or multiple class variables"),
                        html.Li("Classes can be binary or multi-valued categorical"),
                        html.Li("MBC will learn dependencies between classes"),
                        html.Li([html.Strong("Mutually exclusive: "), "Once selected here, the variable disappears from features"]),
                    ]),
                ],
                style={"backgroundColor": "#ffffff", "borderRadius": "0 0 0.25rem 0.25rem", "maxWidth": "320px"}
            ),
        ],
        id="help-popover-mbc-classes",
        target="help-button-mbc-classes",
        placement="right",
        is_open=False,
        trigger="hover",
        style={"position": "absolute", "zIndex": 1000, "marginLeft": "5px"}
    ),
    
    dbc.Popover(
        [
            dbc.PopoverHeader(
                [
                    "Feature Variables",
                    html.I(className="fa fa-info-circle ms-2", style={"color": "#0d6efd"})
                ],
                style={"backgroundColor": "#f8f9fa", "fontWeight": "bold"}
            ),
            dbc.PopoverBody(
                [
                    html.P("Select the input feature columns used for prediction."),
                    html.P("Important:"),
                    html.Ul([
                        html.Li("Features are used to predict class variables"),
                        html.Li("Numeric features will be automatically discretized"),
                        html.Li([html.Strong("Mutually exclusive: "), "Once selected here, the variable disappears from classes"]),
                        html.Li("The network will learn dependencies among features"),
                    ]),
                ],
                style={"backgroundColor": "#ffffff", "borderRadius": "0 0 0.25rem 0.25rem", "maxWidth": "320px"}
            ),
        ],
        id="help-popover-mbc-features",
        target="help-button-mbc-features",
        placement="right",
        is_open=False,
        trigger="hover",
        style={"position": "absolute", "zIndex": 1000, "marginLeft": "5px"}
    ),
    
    dbc.Popover(
        [
            dbc.PopoverHeader(
                [
                    "Algorithm Options",
                    html.I(className="fa fa-cog ms-2", style={"color": "#6c757d"})
                ],
                style={"backgroundColor": "#f8f9fa", "fontWeight": "bold"}
            ),
            dbc.PopoverBody(
                [
                    html.P("Configure the MBC learning algorithm:"),
                    html.P([html.Strong("Approach:"), " Filter uses BIC score, Wrapper uses classification accuracy"]),
                    html.P([html.Strong("Wrapper Measure:"), " Global (overall accuracy) or Average (per-class accuracy)"]),
                    html.P([html.Strong("Discretization:"), " Method to convert numeric features to categories"]),
                    html.Hr(),
                    html.Small([html.I(className="fa fa-info-circle me-1"), "Note: MBC uses an 80/20 train/validation split internally"], className="text-muted"),
                ],
                style={"backgroundColor": "#ffffff", "borderRadius": "0 0 0.25rem 0.25rem", "maxWidth": "350px"}
            ),
        ],
        id="help-popover-mbc-options",
        target="help-button-mbc-options",
        placement="right",
        is_open=False,
        trigger="hover",
        style={"position": "absolute", "zIndex": 1000, "marginLeft": "5px"}
    ),
    
    # Notification container (outside dcc.Loading to avoid interference)
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
])

# ---------- Callbacks for UI interactivity ----------

# 1A. Clear default checkbox when file is uploaded
@app.callback(
    Output('use-default-dataset', 'value'),
    Input('mbc-upload-csv', 'contents'),
    State('use-default-dataset', 'value'),
    prevent_initial_call=True
)
def clear_default_on_upload(contents, current_value):
    """Clear the 'Use default dataset' checkbox when a file is uploaded."""
    if contents is not None:
        logger.info("File uploaded, clearing default dataset checkbox")
        return []  # Clear the checkbox
    raise PreventUpdate

# 1B. Load CSV into data store (from upload or default)
@app.callback(
    Output('mbc-dataset-store', 'data'),
    Output('mbc-columns-store', 'data'),
    Output('mbc-upload-status', 'children'),
    Output('notification-store', 'data'),
    Output('dataset-change-detector', 'data'),
    Input('mbc-upload-csv', 'contents'),
    State('mbc-upload-csv', 'filename'),
    Input('use-default-dataset', 'value'),
    prevent_initial_call=False
)
def mbc_load_csv(contents, filename, default_value):
    """Load dataset from uploaded file or default CSV."""
    import time
    
    # Generate unique change ID for dataset change detection
    change_id = str(time.time())
    
    # Priority 1: Handle file upload (takes precedence over default)
    if contents:
        logger.info(f"Attempting to load uploaded CSV: {filename}")
        
        # Validate file extension
        if filename and not filename.lower().endswith('.csv'):
            return dash.no_update, dash.no_update, dash.no_update, {
                "message": "Please upload a .csv file. Other formats are not supported.",
                "header": "Invalid File Format",
                "icon": "warning"
            }, dash.no_update
        
        content_type, content_string = contents.split(',')
        try:
            decoded = base64.b64decode(content_string)
            # Read CSV into DataFrame
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            
            if df.empty:
                return dash.no_update, dash.no_update, dash.no_update, {
                    "message": "The uploaded CSV file is empty.",
                    "header": "Empty File",
                    "icon": "warning"
                }, dash.no_update
            
            # Check if dataset is too large
            if len(df) > 50000:
                notification = {
                    "message": f"Large dataset detected ({len(df)} rows). Performance may be affected.",
                    "header": "Large Dataset",
                    "icon": "warning"
                }
            else:
                notification = None
            
            logger.info(f"Valid CSV uploaded: {filename} with {len(df)} rows, {len(df.columns)} columns")
            
            return (
                {'records': df.to_dict('records'), 'columns': list(df.columns)},
                list(df.columns),
                f"✓ Loaded: {filename} ({len(df)} rows, {len(df.columns)} columns)",
                notification,
                change_id
            )
            
        except UnicodeDecodeError:
            logger.error(f"Error decoding file: {filename}")
            return dash.no_update, dash.no_update, dash.no_update, {
                "message": "Unable to decode the file. Please ensure it's a valid UTF-8 encoded CSV file.",
                "header": "File Encoding Error",
                "icon": "danger"
            }, dash.no_update
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            return dash.no_update, dash.no_update, dash.no_update, {
                "message": f"Error loading CSV: {str(e)}",
                "header": "CSV Loading Error",
                "icon": "danger"
            }, dash.no_update
    
    # Priority 2: Handle default checkbox (only if no file was uploaded)
    elif 'default' in default_value:
        try:
            # Path to default CSV file
            default_path = '/var/www/html/CIGModels/backend/cigmodelsdjango/cigmodelsdjangoapp/mbc-dash/sample_data/asia10KClassmayus.csv'
            
            # Validate default file exists
            if not os.path.exists(default_path):
                logger.error("Default CSV file not found")
                return dash.no_update, dash.no_update, dash.no_update, {
                    "message": "Default dataset file not found. Please upload your own CSV file.",
                    "header": "File Not Found",
                    "icon": "danger"
                }, dash.no_update
            
            # Read default CSV
            df = pd.read_csv(default_path)
            
            if df.empty:
                return dash.no_update, dash.no_update, dash.no_update, {
                    "message": "Default dataset is empty.",
                    "header": "Empty Dataset",
                    "icon": "danger"
                }, dash.no_update
            
            logger.info(f"Using default dataset: asia10KClassmayus.csv ({len(df)} rows, {len(df.columns)} columns)")
            
            return (
                {'records': df.to_dict('records'), 'columns': list(df.columns)},
                list(df.columns),
                f"✓ Loaded: asia10KClassmayus.csv (default) ({len(df)} rows, {len(df.columns)} columns)",
                None,
                change_id
            )
            
        except Exception as e:
            logger.error(f"Error loading default dataset: {e}")
            return dash.no_update, dash.no_update, dash.no_update, {
                "message": f"Error loading default dataset: {str(e)}",
                "header": "Invalid Default Dataset",
                "icon": "danger"
            }, dash.no_update
    
    # If neither default is checked nor any file is uploaded => do nothing
    raise PreventUpdate

# 1C. Reset app state when dataset changes
@app.callback(
    Output('mbc-results', 'children', allow_duplicate=True),
    Output('mbc-network-container', 'children', allow_duplicate=True),
    Output('mbc-prediction-section', 'children', allow_duplicate=True),
    Output('mbc-model-trained', 'data', allow_duplicate=True),
    Output('mbc-network-store', 'data', allow_duplicate=True),
    Output('previous-classes-selection', 'data', allow_duplicate=True),
    Output('previous-features-selection', 'data', allow_duplicate=True),
    Output('notification-store', 'data', allow_duplicate=True),
    Input('dataset-change-detector', 'data'),
    prevent_initial_call=True
)
def reset_app_state_on_dataset_change(change_id):
    """Reset app to clean state when dataset changes."""
    if change_id is not None:
        logger.info(f"Dataset changed (ID: {change_id}), resetting app state")
        
        # Create notification to inform user of reset
        notification = {
            "message": "Application state has been reset due to dataset change. Previous selections and results have been cleared.",
            "header": "State Reset",
            "icon": "info"
        }
        
        return (
            html.Div(),  # Clear results
            html.Div(),  # Clear network visualization
            html.Div(),  # Clear prediction section
            False,       # Reset model trained flag
            None,        # Clear network store
            [],          # Clear previous class selection
            [],          # Clear previous feature selection
            notification # Notify user of reset
        )
    raise PreventUpdate

# 1D. Clear checkboxes when dataset changes
@app.callback(
    Output({'type': 'mbc-class-checkbox', 'index': ALL}, 'value', allow_duplicate=True),
    Output({'type': 'mbc-feature-checkbox', 'index': ALL}, 'value', allow_duplicate=True),
    Input('dataset-change-detector', 'data'),
    State({'type': 'mbc-class-checkbox', 'index': ALL}, 'id'),
    State({'type': 'mbc-feature-checkbox', 'index': ALL}, 'id'),
    prevent_initial_call=True
)
def clear_checkboxes_on_dataset_change(change_id, class_ids, feature_ids):
    """Clear all checkbox selections when dataset changes to start fresh."""
    if change_id is not None:
        logger.info("Clearing all checkbox selections due to dataset change")
        # Return empty lists for all checkboxes to clear them
        class_clear = [[] for _ in class_ids]
        feature_clear = [[] for _ in feature_ids] 
        return class_clear, feature_clear
    raise PreventUpdate

# 2. Render class and feature checkboxes based on columns
# IMPORTANT: Classes and features must be DISJOINT sets
# If a column is selected as a class, it will NOT appear in features and vice versa
@app.callback(
    Output('mbc-class-checkbox-container', 'children'),
    Output('mbc-feature-checkbox-container', 'children'),
    Input('mbc-columns-store', 'data'),
    Input('mbc-classes-selected', 'data'),    # Changed to Input to trigger on changes
    Input('mbc-features-selected', 'data'),   # Changed to Input to trigger on changes
)
def render_checkboxes(columns, current_classes, current_features):
    if not columns:
        no_data_msg = html.Div("No dataset loaded.", style={'textAlign': 'center', 'color': '#666'})
        return no_data_msg, no_data_msg
    
    # Create sets for quick lookup
    class_set = set(current_classes or [])
    feature_set = set(current_features or [])
    
    # Create checklist items for classes
    # Only show columns that are NOT already selected as features
    class_checks = []
    for col in columns:
        # Skip if already selected as feature (makes them mutually exclusive)
        if col in feature_set:
            continue
            
        checked = col in class_set
        class_checks.append(
            dcc.Checklist(
                id={'type': 'mbc-class-checkbox', 'index': col},
                options=[{'label': f" {col}", 'value': col}],
                value=[col] if checked else [],
                style={'marginBottom': '5px'}
            )
        )
    
    # Create checklist items for features
    # Only show columns that are NOT already selected as classes
    feature_checks = []
    for col in columns:
        # Skip if already selected as class (makes them mutually exclusive)
        if col in class_set:
            continue
            
        checked = col in feature_set
        feature_checks.append(
            dcc.Checklist(
                id={'type': 'mbc-feature-checkbox', 'index': col},
                options=[{'label': f" {col}", 'value': col}],
                value=[col] if checked else [],
                style={'marginBottom': '5px'}
            )
        )
    
    # Arrange in two columns
    class_container = html.Div(class_checks, style={'columnCount': 2, 'margin': '0 auto', 'width': '80%'})
    feature_container = html.Div(feature_checks, style={'columnCount': 2, 'margin': '0 auto', 'width': '80%'})
    return class_container, feature_container

# Track selected classes
@app.callback(
    Output('mbc-classes-selected', 'data'),
    Input({'type': 'mbc-class-checkbox', 'index': ALL}, 'value')
)
def update_selected_classes(selected_lists):
    # Flatten the list of selected values
    classes = [val for sublist in selected_lists for val in sublist] if selected_lists else []
    return classes

# Track selected features
@app.callback(
    Output('mbc-features-selected', 'data'),
    Input({'type': 'mbc-feature-checkbox', 'index': ALL}, 'value')
)
def update_selected_features(selected_lists):
    features = [val for sublist in selected_lists for val in sublist] if selected_lists else []
    return features

# Select All / Clear All for classes
@app.callback(
    Output({'type': 'mbc-class-checkbox', 'index': ALL}, 'value'),
    Input('mbc-select-all-classes', 'n_clicks'),
    Input('mbc-clear-classes', 'n_clicks'),
    State({'type': 'mbc-class-checkbox', 'index': ALL}, 'id'),
    prevent_initial_call=True
)
def toggle_classes(select_all_click, clear_all_click, class_ids):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger == 'mbc-select-all-classes':
        return [[cid['index']] for cid in class_ids]  # select every class checkbox
    elif trigger == 'mbc-clear-classes':
        return [[] for _ in class_ids]               # clear all selections
    else:
        raise PreventUpdate

# Select All / Clear All for features
@app.callback(
    Output({'type': 'mbc-feature-checkbox', 'index': ALL}, 'value'),
    Input('mbc-select-all-features', 'n_clicks'),
    Input('mbc-clear-features', 'n_clicks'),
    State({'type': 'mbc-feature-checkbox', 'index': ALL}, 'id'),
    prevent_initial_call=True
)
def toggle_features(select_all_click, clear_all_click, feature_ids):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger == 'mbc-select-all-features':
        return [[fid['index']] for fid in feature_ids]
    elif trigger == 'mbc-clear-features':
        return [[] for _ in feature_ids]
    else:
        raise PreventUpdate

# Popover toggles for help
@app.callback(
    Output("help-popover-default-dataset", "is_open"),
    Input("help-button-default-dataset", "n_clicks"),
    State("help-popover-default-dataset", "is_open")
)
def toggle_help_default_dataset(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("help-popover-mbc-classes", "is_open"),
    Input("help-button-mbc-classes", "n_clicks"),
    State("help-popover-mbc-classes", "is_open")
)
def toggle_help_classes(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("help-popover-mbc-features", "is_open"),
    Input("help-button-mbc-features", "n_clicks"),
    State("help-popover-mbc-features", "is_open")
)
def toggle_help_features(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("help-popover-mbc-options", "is_open"),
    Input("help-button-mbc-options", "n_clicks"),
    State("help-popover-mbc-options", "is_open")
)
def toggle_help_options(n, is_open):
    if n:
        return not is_open
    return is_open

# 5. Run MBC model training and evaluation
@app.callback(
    Output('mbc-results', 'children'),
    Output('mbc-network-store', 'data'),
    Output('mbc-model-trained', 'data'),
    Output('mbc-feature-values-store', 'data'),
    Output('notification-store', 'data', allow_duplicate=True),
    Input('mbc-run-button', 'n_clicks'),
    State('mbc-dataset-store', 'data'),
    State('mbc-classes-selected', 'data'),
    State('mbc-features-selected', 'data'),
    State('mbc-approach-radio', 'value'),
    State('mbc-measure-radio', 'value'),
    State('mbc-train-split', 'data'),
    State('mbc-disc-method', 'value'),
    prevent_initial_call=True
)
def run_mbc(n_clicks, dataset_store, classes, features, approach, measure, train_split, disc_method):
    if not n_clicks:
        raise PreventUpdate
    # Validate selections
    if not dataset_store:
        return html.Div("No dataset loaded.", style={'color': 'red'}), None, False, None, {
            "message": "Please upload a dataset or use the default dataset.",
            "header": "No Dataset",
            "icon": "danger"
        }
    if not classes:
        return html.Div("Please select at least one class variable.", style={'color': 'red'}), None, False, None, {
            "message": "No class variables selected. Please select at least one class variable.",
            "header": "Configuration Error",
            "icon": "warning"
        }
    if not features:
        return html.Div("Please select at least one feature variable.", style={'color': 'red'}), None, False, None, {
            "message": "No feature variables selected. Please select at least one feature variable.",
            "header": "Configuration Error",
            "icon": "warning"
        }
    if set(classes) & set(features):
        return html.Div("A column cannot be both class and feature.", style={'color': 'red'}), None, False, None, {
            "message": "Classes and features must be distinct. A variable cannot be both a class and a feature.",
            "header": "Variable Overlap Error",
            "icon": "danger"
        }

    # Convert stored dataset back to DataFrame
    try:
        df_all = pd.DataFrame.from_records(dataset_store['records'], columns=dataset_store['columns'])
    except Exception as e:
        return html.Div(f"Error reading dataset: {e}", style={'color': 'red'}), None, False, None, {
            "message": f"Error reading dataset: {str(e)}",
            "header": "Dataset Error",
            "icon": "danger"
        }

    # Prepare R environment and source MBC functions
    try:
        ensure_r_ready()
    except Exception as e:
        return html.Div(str(e), style={'color': 'red'}), None, False, None, {
            "message": str(e),
            "header": "R Environment Error",
            "icon": "danger"
        }

    # Convert all string columns to categorical in pandas first (before R conversion)
    for col in df_all.columns:
        if df_all[col].dtype == 'object':  # String columns
            df_all[col] = df_all[col].astype('category')
    
    # Transfer data to R using the converter context
    with pandas_converter.context():
        r_df = pandas_converter.py2rpy(df_all)
    ro.globalenv['df_all'] = r_df
    ro.globalenv['classes'] = StrVector(classes)
    ro.globalenv['features'] = StrVector(features)
    # Basic preprocessing in R: drop NA, ensure all columns are factors
    ro.r('''
        df_all <- as.data.frame(df_all)
        df_all <- na.omit(df_all)
        # Convert all character columns to factors
        for (col_name in names(df_all)) {
            if (is.character(df_all[[col_name]])) {
                df_all[[col_name]] <- as.factor(df_all[[col_name]])
            }
        }
        # Ensure class and feature columns are factors
        for (cls in classes) {
            df_all[[cls]] <- as.factor(df_all[[cls]])
        }
        for (feat in features) {
            if (!is.numeric(df_all[[feat]])) {
                df_all[[feat]] <- as.factor(df_all[[feat]])
            }
        }
    ''')
    # Discretize numeric features if requested
    if disc_method in ("frequency", "cluster"):
        # Use arules::discretize for numeric features
        ro.globalenv['disc_method'] = disc_method
        ro.r('''
            for (feat in features) {
                if (!is.factor(df_all[[feat]])) {
                    # Try discretization; if it fails, convert to factor as is
                    tryCatch({
                        df_all[[feat]] <- discretize(df_all[[feat]], method = disc_method, breaks = 3)
                    }, error = function(e) {
                        df_all[[feat]] <- as.factor(df_all[[feat]])
                    })
                }
            }
        ''')
    else:
        # No discretization; just ensure numeric are treated as factors (if already factors, this does nothing)
        ro.r('''
            for (feat in features) {
                if (!is.factor(df_all[[feat]])) {
                    df_all[[feat]] <- as.factor(df_all[[feat]])
                }
            }
        ''')

    # Train/validation split
    ro.globalenv['train_ratio'] = train_split / 100.0
    ro.r('''
        set.seed(123)  # for reproducibility
        N <- nrow(df_all)
        train_indices <- sample(N, floor(N * train_ratio))
        train_df <- df_all[train_indices, c(classes, features), drop=FALSE]
        val_df   <- df_all[-train_indices, c(classes, features), drop=FALSE]
    ''')

    # Train MBC model using selected approach
    try:
        if approach == 'filter':
            ro.r('MBC_model <- learn_MBC(train_df, classes, features)')
        else:  # wrapper approach
            # Use wrapper with chosen measure (global or average accuracy)
            ro.globalenv['measure'] = measure
            ro.r('MBC_model <- learn_MBC_wrapper2(train_df, val_df, classes, features, measure = measure, verbose = FALSE)')
    except Exception as e:
        return html.Div(f"Error during MBC training: {e}", style={'color': 'red'}), None, False, None, {
            "message": f"Error during MBC training: {str(e)}",
            "header": "Training Error",
            "icon": "danger"
        }

    # Get the learned network structure (arc list)
    try:
        # Get arcs from the model
        ro.r('arcs_matrix <- arcs(MBC_model)')
        
        # Convert to pandas DataFrame
        with pandas_converter.context():
            arcs_r = ro.r('as.data.frame(arcs_matrix, stringsAsFactors=FALSE)')
            # The conversion happens automatically, just assign it
            arcs_df = arcs_r
            
        # If it's still an R object, convert it manually
        if not isinstance(arcs_df, pd.DataFrame):
            with pandas_converter.context():
                arcs_df = pandas_converter.rpy2py(arcs_df)
        
        if arcs_df.empty or len(arcs_df) == 0:
            structure_str = "(No arcs in the learned network.)"
            network_data = {'arcs': [], 'classes': classes, 'features': features}
        else:
            arc_list = [f"{row['from']} -> {row['to']}" for _, row in arcs_df.iterrows()]
            structure_str = "Learned MBC structure (arcs):\n" + "\n".join(arc_list)
            # Store network data for visualization
            network_data = {
                'arcs': arcs_df.to_dict('records'),
                'classes': classes,
                'features': features
            }
    except Exception as e:
        logger.error(f"Error retrieving network structure: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        structure_str = f"(Could not retrieve network structure: {str(e)})"
        network_data = None

    # Make predictions on validation set
    try:
        ro.r('pred_df <- predict_MBC_dataset_veryfast(MBC_model, val_df, classes, features)')
        perf = ro.r('test_multidimensional(val_df, pred_df, classes)')
        global_acc = float(perf.rx2('global')[0])
        avg_acc = float(perf.rx2('average')[0])
        per_class_acc = list(perf.rx2('per_class'))
    except Exception as e:
        return html.Div(f"Error during prediction/evaluation: {e}", style={'color': 'red'}), None, False, None, {
            "message": f"Error during prediction/evaluation: {str(e)}",
            "header": "Prediction Error",
            "icon": "danger"
        }
    
    # Extract feature levels/values for prediction UI
    try:
        feature_values = {}
        for feat in features:
            # Get levels of each feature variable from the processed data
            levels_r = ro.r(f'levels(df_all[["{feat}"]])')
            if levels_r != ro.NULL:
                feature_values[feat] = list(levels_r)
            else:
                # If not a factor, get unique values
                unique_vals = ro.r(f'unique(as.character(df_all[["{feat}"]]))')
                feature_values[feat] = list(unique_vals)
    except Exception as e:
        logger.warning(f"Could not extract feature values: {e}")
        feature_values = {feat: [] for feat in features}

    # Build results card with identical style to probExplainer
    per_class_rows = [
        html.Tr([
            html.Td(cls_name, style={'textAlign': 'center'}), 
            html.Td(f"{acc:.4f}", style={'textAlign': 'center'})
        ])
        for cls_name, acc in zip(classes, per_class_acc)
    ]
    results_card = dbc.Card(
        dbc.CardBody([
            html.H4("MBC Results", className="card-title", style={'textAlign': 'center', 'marginBottom': '20px'}),
        
        # Network Structure Section
        html.Div([
            html.H5("Learned Network Structure", style={'textAlign': 'center', 'color': '#00A2E1', 'marginBottom': '10px'}),
            html.Pre(structure_str, style={
                'textAlign': 'left', 
                'whiteSpace': 'pre-wrap',
                'backgroundColor': '#f8f9fa',
                'padding': '15px',
                'borderRadius': '5px',
                'border': '1px solid #dee2e6',
                'maxHeight': '200px',
                'overflowY': 'auto'
            }),
        ], style={'marginBottom': '20px'}),
        
        html.Hr(style={'margin': '20px 0'}),
        
        # Performance Metrics Section
        html.Div([
            html.H5("Validation Performance", style={'textAlign': 'center', 'color': '#00A2E1', 'marginBottom': '15px'}),
            
            # Overall metrics
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.I(className="fa fa-check-circle", style={'color': '#28a745', 'marginRight': '8px', 'fontSize': '18px'}),
                            html.Strong("Global Accuracy: ", style={'fontSize': '16px'}),
                            html.Span(f"{global_acc:.4f}", style={'fontSize': '16px', 'color': '#00A2E1', 'fontWeight': 'bold'})
                        ], style={'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#e7f3ff', 'borderRadius': '5px'})
                    ], md=6),
                    dbc.Col([
                        html.Div([
                            html.I(className="fa fa-bar-chart", style={'color': '#17a2b8', 'marginRight': '8px', 'fontSize': '18px'}),
                            html.Strong("Average Accuracy: ", style={'fontSize': '16px'}),
                            html.Span(f"{avg_acc:.4f}", style={'fontSize': '16px', 'color': '#00A2E1', 'fontWeight': 'bold'})
                        ], style={'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#e7f3ff', 'borderRadius': '5px'})
                    ], md=6),
                ], style={'marginBottom': '20px'})
            ]),
            
            # Per-class accuracies table
            html.Div([
                html.H6("Per-Class Accuracies", style={'textAlign': 'center', 'marginBottom': '10px'}),
                dbc.Table(
                    [
                        html.Thead(html.Tr([
                            html.Th("Class Variable", style={'textAlign': 'center'}), 
                            html.Th("Accuracy", style={'textAlign': 'center'})
                        ])),
                        html.Tbody(per_class_rows)
                    ],
                    bordered=True,
                    striped=True,
                    hover=True,
                    responsive=True,
                    className="mt-2",
                    style={'width': '70%', 'margin': '0 auto'}
                ),
            ])
        ]),
        ]),
        className="mt-3"
    )
    return results_card, network_data, True, feature_values, {
        'header': 'Success',
        'message': 'MBC model trained successfully!',
        'icon': 'success'
    }

# 6. Visualize learned network structure
@app.callback(
    Output('mbc-network-container', 'children'),
    Input('mbc-network-store', 'data'),
    prevent_initial_call=True
)
def visualize_network(network_data):
    """Create interactive network visualization using Cytoscape"""
    if not network_data:
        return html.Div()
    
    classes = network_data.get('classes', [])
    features = network_data.get('features', [])
    arcs = network_data.get('arcs', [])
    
    # Build Cytoscape elements
    elements = []
    
    # Add nodes - Classes
    for cls in classes:
        elements.append({
            'data': {
                'id': cls,
                'label': cls,
                'type': 'class'
            }
        })
    
    # Add nodes - Features
    for feat in features:
        elements.append({
            'data': {
                'id': feat,
                'label': feat,
                'type': 'feature'
            }
        })
    
    # Add edges with type classification
    for arc in arcs:
        from_node = arc['from']
        to_node = arc['to']
        
        # Determine edge type
        if from_node in classes and to_node in classes:
            edge_type = 'class'  # Class-to-class edge
        elif from_node in classes and to_node in features:
            edge_type = 'bridge'  # Class-to-feature edge
        else:
            edge_type = 'feature'  # Feature-to-feature edge
        
        elements.append({
            'data': {
                'id': f"{from_node}-{to_node}",
                'source': from_node,
                'target': to_node,
                'type': edge_type
            }
        })
    
    # Create the visualization card with identical style to probExplainer
    network_card = dbc.Card(
        dbc.CardBody([
            html.H4("🔗 Learned Network Structure", className="card-title", style={'textAlign': 'center', 'color': '#00A2E1', 'marginBottom': '20px'}),
        html.Div([
            cyto.Cytoscape(
                id='mbc-network-graph',
                elements=elements,
                stylesheet=cytoscape_stylesheet,
                style={
                    'width': '100%',
                    'height': '500px',
                    'border': '1px solid #ddd',
                    'borderRadius': '5px',
                    'backgroundColor': '#ffffff'
                },
                layout={
                    'name': 'cose',  # Force-directed layout
                    'animate': True,
                    'animationDuration': 500,
                    'nodeRepulsion': 8000,
                    'idealEdgeLength': 100,
                    'edgeElasticity': 100,
                    'nestingFactor': 5,
                    'gravity': 80,
                    'numIter': 1000,
                    'randomize': False
                }
            ),
        ], style={'marginBottom': '15px'}),
        
        # Legend and instructions
        html.Div([
            html.Hr(style={'margin': '15px 0'}),
            html.Div([
                html.Div([
                    html.I(className="fa fa-info-circle", style={'marginRight': '8px', 'color': '#00A2E1'}),
                    html.B("Legend: ", style={'color': '#333'}),
                ], style={'display': 'inline-block', 'marginRight': '15px'}),
                html.Span("● Class nodes", style={'marginRight': '15px', 'color': '#00A2E1', 'fontWeight': 'bold'}),
                html.Span("▢ Feature nodes", style={'marginRight': '15px', 'color': '#90CAF9', 'fontWeight': '500'}),
                html.Span("→ Class→Feature (bridge)", style={'color': '#00A2E1', 'fontWeight': '500'})
            ], style={'textAlign': 'center', 'fontSize': '14px', 'marginBottom': '10px'}),
            html.Div([
                html.I(className="fa fa-hand-pointer-o", style={'marginRight': '5px', 'color': '#6c757d'}),
                html.Small([
                    "Zoom: scroll | Pan: drag background | Move nodes: drag individual nodes"
                ], className="text-muted")
            ], style={'textAlign': 'center'})
        ], style={
            'backgroundColor': '#f8f9fa',
            'padding': '15px',
            'borderRadius': '5px',
            'border': '1px solid #dee2e6'
        })
        ]),
        className="mt-3"
    )
    
    return network_card

# 7. Render individual prediction section
@app.callback(
    Output('mbc-prediction-section', 'children'),
    Input('mbc-model-trained', 'data'),
    State('mbc-feature-values-store', 'data'),
    State('mbc-classes-selected', 'data'),
    State('mbc-features-selected', 'data'),
    prevent_initial_call=True
)
def render_prediction_section(model_trained, feature_values, classes, features):
    """Render the individual prediction UI after model is trained"""
    if not model_trained or not feature_values:
        return html.Div()
    
    # Create dropdowns for each feature
    feature_inputs = []
    for feat in features:
        values = feature_values.get(feat, [])
        if not values:
            continue
        
        feature_inputs.append(
            dbc.Row([
                dbc.Col([
                    html.Label(feat, style={'fontWeight': '500', 'marginBottom': '5px'})
                ], md=3),
                dbc.Col([
                    dbc.Select(
                        id={'type': 'mbc-predict-input', 'index': feat},
                        options=[{'label': str(v), 'value': str(v)} for v in values],
                        placeholder=f"Select {feat}...",
                        style={
                            'border': '1px solid #d0d7de',
                            'borderRadius': '6px',
                            'padding': '8px 12px',
                            'backgroundColor': 'rgba(255, 255, 255, 0.8)',
                        }
                    )
                ], md=9),
            ], style={'marginBottom': '15px'})
        )
    
    prediction_card = dbc.Card(
        dbc.CardBody([
            html.H4([
                html.I(className="fa fa-magic", style={'marginRight': '10px', 'color': '#00A2E1'}),
                "Individual Case Prediction"
            ], className="card-title", style={'textAlign': 'center', 'marginBottom': '20px'}),
            
            html.P(
                "Enter values for the features below to predict the class variables for a new instance.",
                style={'textAlign': 'center', 'color': '#666', 'marginBottom': '20px'}
            ),
            
            html.Div(feature_inputs, style={'maxWidth': '700px', 'margin': '0 auto'}),
            
            html.Div([
                dbc.Button(
                    [
                        html.I(className="fas fa-brain me-2"),
                        "Predict Classes"
                    ],
                    id='mbc-predict-button',
                    color="success",
                    className="btn-lg",
                    style={
                        'fontSize': '1rem',
                        'padding': '0.6rem 1.5rem',
                        'borderRadius': '8px',
                        'marginTop': '10px'
                    }
                )
            ], style={'textAlign': 'center', 'marginTop': '20px'}),
            
            html.Div(id='mbc-prediction-output', style={'marginTop': '25px'})
        ]),
        className="mt-3",
        style={'backgroundColor': '#f8f9fa', 'border': '2px solid #00A2E1'}
    )
    
    return prediction_card

# 8. Make individual prediction
@app.callback(
    Output('mbc-prediction-output', 'children'),
    Input('mbc-predict-button', 'n_clicks'),
    State({'type': 'mbc-predict-input', 'index': ALL}, 'value'),
    State({'type': 'mbc-predict-input', 'index': ALL}, 'id'),
    State('mbc-classes-selected', 'data'),
    State('mbc-features-selected', 'data'),
    prevent_initial_call=True
)
def make_individual_prediction(n_clicks, feature_values, feature_ids, classes, features):
    """Use predict_MBC_case to predict classes for a single instance"""
    if not n_clicks:
        raise PreventUpdate
    
    # Check if all features have values
    if not all(feature_values):
        return dbc.Alert(
            "Please select values for all features before predicting.",
            color="warning",
            style={'marginTop': '15px'}
        )
    
    # Build feature dictionary
    evidence = {}
    for val, id_dict in zip(feature_values, feature_ids):
        feat_name = id_dict['index']
        evidence[feat_name] = val
    
    # Call R predict_MBC_case function
    try:
        # Create a single-row data.frame in R with the feature values
        ro.globalenv['classes'] = StrVector(classes)
        ro.globalenv['features'] = StrVector(features)
        
        # Create the case data frame with proper factor levels from the trained model
        # This is crucial: factors must have all levels that the model was trained with
        ro.r('''
        # Initialize empty data frame with proper structure
        case_df <- data.frame(matrix(ncol = length(features), nrow = 1))
        colnames(case_df) <- features
        
        # Set each feature value with the correct levels from the model
        for (feat in features) {
            # Get the levels from the trained model
            if (feat %in% names(MBC_model)) {
                feat_levels <- dimnames(MBC_model[[feat]]$prob)[[1]]
            } else {
                # If feature is not in model CPTs, get from training data (df_all)
                feat_levels <- levels(df_all[[feat]])
            }
            case_df[[feat]] <- factor(NA, levels = feat_levels)
        }
        ''')
        
        # Now set the actual values
        for feat in features:
            val = evidence[feat]
            ro.r(f'case_df[["{feat}"]][1] <- "{val}"')
        
        # Predict using the trained model (MBC_model is still in R environment)
        ro.r('prediction <- predict_MBC_case(MBC_model, case_df, classes, features)')
        
        # Debug: print the prediction in R console
        ro.r('print("=== DEBUG: Prediction output ===")')
        ro.r('print(prediction)')
        ro.r('print(class(prediction))')
        ro.r('print(str(prediction))')
        
        # Extract the predicted values directly using R vectors
        # This is more reliable than data.frame conversion
        pred_names = list(ro.r('names(prediction)'))
        pred_values = list(ro.r('as.character(unname(prediction))'))
        
        # Build dictionary
        predicted_classes = {}
        for name, value in zip(pred_names, pred_values):
            predicted_classes[name] = value
        
        # Log for debugging
        logger.info(f"Predicted classes: {predicted_classes}")
        
        # Format the results
        result_rows = []
        for cls, pred_val in predicted_classes.items():
            result_rows.append(
                html.Tr([
                    html.Td(html.Strong(cls), style={'textAlign': 'center', 'padding': '10px'}),
                    html.Td(
                        html.Span(pred_val, style={
                            'backgroundColor': '#d4edda', 
                            'color': '#155724',
                            'padding': '5px 15px',
                            'borderRadius': '5px',
                            'fontWeight': 'bold'
                        }),
                        style={'textAlign': 'center', 'padding': '10px'}
                    )
                ])
            )
        
        results_table = dbc.Table(
            [
                html.Thead(html.Tr([
                    html.Th("Class Variable", style={'textAlign': 'center', 'backgroundColor': '#e7f3ff'}),
                    html.Th("Predicted Value", style={'textAlign': 'center', 'backgroundColor': '#e7f3ff'})
                ])),
                html.Tbody(result_rows)
            ],
            bordered=True,
            striped=True,
            hover=True,
            responsive=True,
            style={'margin': '20px auto', 'maxWidth': '600px'}
        )
        
        return html.Div([
            html.H5(
                [
                    html.I(className="fa fa-check-circle", style={'color': '#28a745', 'marginRight': '8px'}),
                    "Prediction Results"
                ],
                style={'textAlign': 'center', 'color': '#00A2E1', 'marginBottom': '15px'}
            ),
            results_table,
            html.Div([
                html.I(className="fa fa-info-circle", style={'marginRight': '5px', 'color': '#17a2b8'}),
                html.Small(
                    "These predictions are based on the Most Probable Explanation (MPE) using the trained MBC model.",
                    style={'color': '#666'}
                )
            ], style={'textAlign': 'center', 'marginTop': '10px'})
        ])
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return dbc.Alert(
            f"Error making prediction: {str(e)}",
            color="danger",
            style={'marginTop': '15px'}
        )

# Notification system callback
@app.callback(
    [Output('notification-container', 'children'),
     Output('notification-container', 'style')],
    Input('notification-store', 'data')
)
def show_notification(data):
    """Display notifications with Bootstrap toasts and animations"""
    if data is None:
        return None, {
            'position': 'fixed',
            'bottom': '20px',
            'right': '20px',
            'zIndex': '1000',
            'width': '300px',
            'transition': 'all 0.3s ease-in-out',
            'transform': 'translateY(100%)',
            'opacity': '0'
        }
    
    # Create toast with animation
    toast = dbc.Toast(
        data['message'],
        header=data['header'],
        icon=data['icon'],
        is_open=True,
        dismissable=True,
        style={
            'width': '100%',
            'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
            'borderRadius': '8px',
            'marginBottom': '10px'
        }
    )
    
    # Style to show notification with animation
    container_style = {
        'position': 'fixed',
        'bottom': '20px',
        'right': '20px',
        'zIndex': '1000',
        'width': '300px',
        'transition': 'all 0.3s ease-in-out',
        'transform': 'translateY(0)',
        'opacity': '1'
    }
    
    return toast, container_style

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8058)
