# -*- coding: utf-8 -*-
"""
Main Dash Application for Multi-dimensional Classification with MBCs

- TW-MBC (Benjumeda): bounded treewidth learning with exact inference
"""

import sys
import os
import io
import base64
import logging

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc
from dash import Input, Output, State, ALL
from dash.exceptions import PreventUpdate

import pandas as pd

# ==== (MBCTree) R interop imports ====
from rpy2 import robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector
pandas2ri.activate()

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

print("🚀 MBC DASHBOARD STARTING...")
print(f"Python: {sys.version}")
print(f"Dash version: {dash.__version__}")

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
R_SRC_PATH = os.path.join(current_dir, "mbctree.R")  # <-- keep R code here unchanged

# -----------------------------------------------------------------------------
# Dash app
# -----------------------------------------------------------------------------
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css",
    ],
    assets_folder='/var/www/html/CIGModels/backend/cigmodelsdjango/cigmodelsdjangoapp/mbc-dash/assets',
    requests_pathname_prefix='/Model/LearningFromData/MBCDash/',
    suppress_callback_exceptions=True,
    title="MBC-Dash: Multi-dimensional Bayesian Classifiers",
)
server = app.server
print("✅ MBC DASHBOARD STARTED")

# -----------------------------------------------------------------------------
# Simple notification payload helper (optional)
# -----------------------------------------------------------------------------
def create_error_notification(message, header="Error"):
    return {"message": message, "header": header, "icon": "danger"}

# -----------------------------------------------------------------------------
# R environment check + source original R
# -----------------------------------------------------------------------------
def ensure_r_ready():
    """
    Load required R packages and source the original mbctree.R without modifying it.
    If something is missing in the R environment, we raise a clear error.
    """
    try:
        # Core pkgs used by your R file
        importr('bnlearn')
        importr('utiml')
        importr('FSelector')
        importr('foreign')
        importr('arules')
        importr('mldr.datasets')
        importr('foreach')
        importr('caret')
        importr('doParallel')
        # Optional often used in compare_models:
        # importr('e1071'); importr('randomForest')
    except Exception as e:
        raise RuntimeError(
            "Missing required R packages. Please install the R packages used by mbctree.R "
            "(bnlearn, utiml, FSelector, mldr.datasets, foreach, caret, doParallel, arules, foreign, etc.). "
            f"R error: {e}"
        )

    try:
        ro.r['source'](R_SRC_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to source R file at {R_SRC_PATH}. Error: {e}")

# -----------------------------------------------------------------------------
# Layout
# -----------------------------------------------------------------------------
app.layout = html.Div(
    [
        # stores
        dcc.Store(id='mbc-dataset-store'),
        dcc.Store(id='mbc-columns-store'),
        dcc.Store(id='mbc-classes-selected', data=[]),
        dcc.Store(id='mbc-features-selected', data=[]),
        dcc.Store(id='notification-store'),

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
                "zIndex": "999999",
            },
            children=html.Div(
                [
                    # 1. Upload
                    html.Div(
                        className="card",
                        children=[
                            html.H3("MBCTree — 1. Upload Dataset (CSV)", style={'textAlign': 'center'}),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Img(
                                                src="https://img.icons8.com/ios-glyphs/40/upload.png",
                                                className="upload-icon",
                                            ),
                                            html.Div("Drag and drop or select a CSV file", className="upload-text"),
                                        ]
                                    ),
                                    dcc.Upload(
                                        id='mbc-upload-csv',
                                        children=html.Div([], style={'display': 'none'}),
                                        className="upload-dropzone",
                                        multiple=False,
                                    ),
                                    html.Div(
                                        id='mbc-upload-status',
                                        style={'textAlign': 'center', 'marginTop': '10px', 'color': '#6c757d'},
                                    ),
                                ],
                                className="upload-card",
                            ),
                        ],
                    ),

                    # 2. Classes
                    html.Div(
                        className="card",
                        children=[
                            html.Div(
                                [
                                    html.H3(
                                        "MBCTree — 2. Select Classes",
                                        style={'display': 'inline-block', 'marginRight': '10px', 'textAlign': 'center'},
                                    ),
                                    dbc.Button(
                                        html.I(className="fa fa-question-circle"),
                                        id="help-button-mbc-classes",
                                        color="link",
                                        style={"display": "inline-block", "verticalAlign": "middle", "padding": "0", "marginLeft": "5px"},
                                    ),
                                ],
                                style={"textAlign": "center", "position": "relative"},
                            ),
                            html.Div(
                                [
                                    dbc.Button("Select All", id="mbc-select-all-classes", color="outline-primary", size="sm", style={'marginRight': '10px'}),
                                    dbc.Button("Clear All", id="mbc-clear-classes", color="outline-secondary", size="sm"),
                                ],
                                style={'textAlign': 'center', 'marginBottom': '15px'},
                            ),
                            html.Div(
                                id='mbc-class-checkbox-container',
                                style={
                                    'maxHeight': '200px',
                                    'overflowY': 'auto',
                                    'border': '1px solid #ddd',
                                    'borderRadius': '5px',
                                    'padding': '10px',
                                    'margin': '0 auto',
                                    'width': '80%',
                                    'backgroundColor': '#f8f9fa',
                                },
                            ),
                        ],
                    ),

                    # 3. Features
                    html.Div(
                        className="card",
                        children=[
                            html.Div(
                                [
                                    html.H3(
                                        "MBCTree — 3. Select Features",
                                        style={'display': 'inline-block', 'marginRight': '10px', 'textAlign': 'center'},
                                    ),
                                    dbc.Button(
                                        html.I(className="fa fa-question-circle"),
                                        id="help-button-mbc-features",
                                        color="link",
                                        style={"display": "inline-block", "verticalAlign": "middle", "padding": "0", "marginLeft": "5px"},
                                    ),
                                ],
                                style={"textAlign": "center", "position": "relative"},
                            ),
                            html.Div(
                                [
                                    dbc.Button("Select All", id="mbc-select-all-features", color="outline-primary", size="sm", style={'marginRight': '10px'}),
                                    dbc.Button("Clear All", id="mbc-clear-features", color="outline-secondary", size="sm"),
                                ],
                                style={'textAlign': 'center', 'marginBottom': '15px'},
                            ),
                            html.Div(
                                id='mbc-feature-checkbox-container',
                                style={
                                    'maxHeight': '220px',
                                    'overflowY': 'auto',
                                    'border': '1px solid #ddd',
                                    'borderRadius': '5px',
                                    'padding': '10px',
                                    'margin': '0 auto',
                                    'width': '80%',
                                    'backgroundColor': '#f8f9fa',
                                },
                            ),
                            html.Div(
                                [
                                    html.I(className="fa fa-info-circle", style={'marginRight': '5px', 'color': '#6c757d'}),
                                    html.Span("Classes cannot be used as features.", style={'fontSize': '11px', 'color': '#6c757d'}),
                                ],
                                style={'textAlign': 'center', 'marginTop': '8px'},
                            ),
                        ],
                    ),

                    # 4. Options
                    html.Div(
                        className="card",
                        children=[
                            html.H3("MBCTree — 4. Options", style={'textAlign': 'center'}),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Label("Approach", style={'fontWeight': '500'}),
                                            dbc.RadioItems(
                                                id='mbc-approach-radio',
                                                options=[
                                                    {'label': 'Filter (BIC)', 'value': 'filter'},
                                                    {'label': 'Wrapper', 'value': 'wrapper'},
                                                ],
                                                value='filter',
                                                inline=True,
                                            ),
                                        ],
                                        md=6,
                                    ),
                                    dbc.Col(
                                        [
                                            html.Label("Wrapper Measure", style={'fontWeight': '500'}),
                                            dbc.RadioItems(
                                                id='mbc-measure-radio',
                                                options=[
                                                    {'label': 'Global accuracy', 'value': 'global'},
                                                    {'label': 'Average accuracy', 'value': 'average'},
                                                ],
                                                value='global',
                                                inline=True,
                                            ),
                                        ],
                                        md=6,
                                    ),
                                ],
                                style={'marginTop': '10px'},
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Label("Train / Validation split (%)", style={'fontWeight': '500'}),
                                            dcc.Slider(
                                                id='mbc-train-split',
                                                min=50,
                                                max=90,
                                                value=80,
                                                step=1,
                                                marks={i: f"{i}%" for i in [50, 60, 70, 80, 90]},
                                            ),
                                        ],
                                        md=8,
                                    ),
                                    dbc.Col(
                                        [
                                            html.Label("Discretization", style={'fontWeight': '500'}),
                                            dbc.Select(
                                                id='mbc-disc-method',
                                                options=[
                                                    {'label': 'frequency (k=3)', 'value': 'frequency'},
                                                    {'label': 'cluster (k=3)', 'value': 'cluster'},
                                                    {'label': 'none (assume already factors)', 'value': 'none'},
                                                ],
                                                value='frequency',
                                            ),
                                        ],
                                        md=4,
                                    ),
                                ],
                                style={'marginTop': '10px'},
                            ),
                        ],
                    ),

                    # Run
                    html.Div(
                        [
                            dbc.Button(
                                [html.I(className="fas fa-play-circle me-2"), "Run MBCTree"],
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
                                    'fontWeight': '500',
                                },
                            )
                        ],
                        style={'textAlign': 'center'},
                    ),

                    html.Br(),
                    html.Div(id='mbc-results', style={'textAlign': 'center'}),
                ]
            ),
        ),

        # Popovers
        dbc.Popover(
            [
                dbc.PopoverHeader(
                    ["Classes", html.I(className="fa fa-info-circle ms-2", style={"color": "#0d6efd"})],
                    style={"backgroundColor": "#f8f9fa", "fontWeight": "bold"},
                ),
                dbc.PopoverBody(
                    ["Select the binary/multi-valued label columns (e.g., C1…Cd). These will be predicted by MBCTree."]
                ),
            ],
            id="help-popover-mbc-classes",
            target="help-button-mbc-classes",
            placement="right",
            is_open=False,
            trigger="hover",
        ),
        dbc.Popover(
            [
                dbc.PopoverHeader(
                    ["Features", html.I(className="fa fa-info-circle ms-2", style={"color": "#0d6efd"})],
                    style={"backgroundColor": "#f8f9fa", "fontWeight": "bold"},
                ),
                dbc.PopoverBody(
                    ["Select input attributes (X1…Xm). If numeric, choose a discretization method unless your data are already factors."]
                ),
            ],
            id="help-popover-mbc-features",
            target="help-button-mbc-features",
            placement="right",
            is_open=False,
            trigger="hover",
        ),
    ]
)

# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------

# Upload CSV -> store
@app.callback(
    Output('mbc-dataset-store', 'data'),
    Output('mbc-columns-store', 'data'),
    Output('mbc-upload-status', 'children'),
    Input('mbc-upload-csv', 'contents'),
    State('mbc-upload-csv', 'filename'),
    prevent_initial_call=True
)
def mbc_load_csv(contents, filename):
    if not contents:
        raise PreventUpdate
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        if df.empty:
            return dash.no_update, dash.no_update, "Uploaded file is empty."
        return (
            {'records': df.to_dict('records'), 'columns': list(df.columns)},
            list(df.columns),
            f"Loaded: {filename} ({len(df)} rows)"
        )
    except Exception as e:
        return dash.no_update, dash.no_update, f"Failed to read CSV: {e}"

# Render checkboxes
@app.callback(
    Output('mbc-class-checkbox-container', 'children'),
    Output('mbc-feature-checkbox-container', 'children'),
    Input('mbc-columns-store', 'data'),
    State('mbc-classes-selected', 'data'),
    State('mbc-features-selected', 'data'),
)
def mbc_render_checkboxes(cols, prev_classes, prev_features):
    if not cols:
        return (
            html.Div("No dataset loaded", style={'textAlign': 'center', 'color': '#666'}),
            html.Div("No dataset loaded", style={'textAlign': 'center', 'color': '#666'})
        )

    def _mk_checklist(prefix, values, selected):
        boxes = []
        for c in values:
            boxes.append(
                html.Div(
                    [
                        dcc.Checklist(
                            id={'type': f'{prefix}-checkbox', 'index': c},
                            options=[{'label': f' {c}', 'value': c}],
                            value=[c] if c in (selected or []) else [],
                            style={'margin': '0'},
                        )
                    ],
                    style={'display': 'inline-block', 'width': '50%', 'marginBottom': '5px'}
                )
            )
        return html.Div(boxes, style={'columnCount': '2', 'columnGap': '20px'})

    class_area = _mk_checklist('mbc-class', cols, prev_classes)
    feat_candidates = [c for c in cols if c not in (prev_classes or [])]
    feat_area = _mk_checklist('mbc-feature', feat_candidates, prev_features)
    return class_area, feat_area

# Track selections
@app.callback(
    Output('mbc-classes-selected', 'data'),
    Input({'type': 'mbc-class-checkbox', 'index': ALL}, 'value')
)
def track_mbc_classes(vals):
    selected = []
    for v in vals or []:
        if v:
            selected.extend(v)
    return selected

@app.callback(
    Output('mbc-features-selected', 'data'),
    Input({'type': 'mbc-feature-checkbox', 'index': ALL}, 'value')
)
def track_mbc_features(vals):
    selected = []
    for v in vals or []:
        if v:
            selected.extend(v)
    return selected

# Select/Clear buttons
@app.callback(
    Output({'type': 'mbc-class-checkbox', 'index': ALL}, 'value'),
    Input('mbc-select-all-classes', 'n_clicks'),
    Input('mbc-clear-classes', 'n_clicks'),
    State({'type': 'mbc-class-checkbox', 'index': ALL}, 'id'),
    prevent_initial_call=True
)
def mbc_toggle_classes(select_all, clear_all, ids):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    if ctx.triggered[0]['prop_id'].startswith('mbc-select-all-classes'):
        return [[cid['index']] for cid in ids]
    return [[] for _ in ids]

@app.callback(
    Output({'type': 'mbc-feature-checkbox', 'index': ALL}, 'value'),
    Input('mbc-select-all-features', 'n_clicks'),
    Input('mbc-clear-features', 'n_clicks'),
    State({'type': 'mbc-feature-checkbox', 'index': ALL}, 'id'),
    prevent_initial_call=True
)
def mbc_toggle_features(select_all, clear_all, ids):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    if ctx.triggered[0]['prop_id'].startswith('mbc-select-all-features'):
        return [[fid['index']] for fid in ids]
    return [[] for _ in ids]

# Help popovers
@app.callback(
    Output("help-popover-mbc-classes", "is_open"),
    Input("help-button-mbc-classes", "n_clicks"),
    State("help-popover-mbc-classes", "is_open")
)
def toggle_help_mbc_classes(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("help-popover-mbc-features", "is_open"),
    Input("help-button-mbc-features", "n_clicks"),
    State("help-popover-mbc-features", "is_open")
)
def toggle_help_mbc_features(n, is_open):
    if n:
        return not is_open
    return is_open

# Train + evaluate via your R functions (unchanged)
@app.callback(
    Output('mbc-results', 'children'),
    Output('notification-store', 'data', allow_duplicate=True),
    Input('mbc-run-button', 'n_clicks'),
    State('mbc-dataset-store', 'data'),
    State('mbc-classes-selected', 'data'),
    State('mbc-features-selected', 'data'),
    State('mbc-approach-radio', 'value'),
    State('mbc-measure-radio', 'value'),
    State('mbc-train-split', 'value'),
    State('mbc-disc-method', 'value'),
    prevent_initial_call=True
)
def mbc_run(n_clicks, dataset_store, classes, features, approach, measure, train_split, disc_method):
    if not n_clicks:
        raise PreventUpdate

    # Validations
    if not dataset_store or not dataset_store.get('records'):
        return html.Div("No dataset loaded.", style={'color': 'red'}), create_error_notification(
            "Load a CSV dataset first.", "Dataset Required"
        )
    if not classes:
        return html.Div("Select at least one class column.", style={'color': 'red'}), create_error_notification(
            "Select at least one class column.", "Configuration Error"
        )
    if not features:
        return html.Div("Select at least one feature column.", style={'color': 'red'}), create_error_notification(
            "Select at least one feature column.", "Configuration Error"
        )
    overlap = set(classes) & set(features)
    if overlap:
        return (
            html.Div(f"Columns cannot be both classes and features: {', '.join(overlap)}", style={'color': 'red'}),
            create_error_notification("Classes and features must be disjoint.", "Configuration Error"),
        )

    # Build pandas DataFrame
    try:
        df = pd.DataFrame.from_records(dataset_store['records'], columns=dataset_store['columns'])
    except Exception as e:
        return html.Div(f"Invalid dataset: {e}", style={'color': 'red'}), create_error_notification(
            f"Invalid dataset: {e}", "Dataset Error"
        )

    # Ensure R ready & source your unchanged file
    try:
        ensure_r_ready()
    except Exception as e:
        return html.Div(str(e), style={'color': 'red'}), create_error_notification(str(e), "R Environment Error")

    # Push DF + params to R and call your original functions EXACTLY.
    try:
        # Send data to R
        r_df = pandas2ri.py2rpy(df)
        ro.globalenv['df_all'] = r_df
        ro.globalenv['classes'] = StrVector(classes)
        ro.globalenv['features'] = StrVector(features)

        # Basic R pre-processing (na.omit, discretization to factors)
        ro.r('''
            suppressMessages(library(bnlearn))
            set.seed(123)
            df_all <- as.data.frame(df_all)
            df_all <- na.omit(df_all)
            for (nm in classes) { df_all[[nm]] <- as.factor(df_all[[nm]]) }
        ''')

        if disc_method == 'frequency':
            ro.r('''
                for (nm in features) {
                  if (!is.factor(df_all[[nm]])) {
                    tryCatch({
                      df_all[[nm]] <- discretize(df_all[[nm]], method="frequency", breaks=3)
                    }, error=function(e) { df_all[[nm]] <- as.factor(df_all[[nm]]) })
                  }
                }
            ''')
        elif disc_method == 'cluster':
            ro.r('''
                for (nm in features) {
                  if (!is.factor(df_all[[nm]])) {
                    tryCatch({
                      df_all[[nm]] <- discretize(df_all[[nm]], method="cluster", breaks=3)
                    }, error=function(e) { df_all[[nm]] <- as.factor(df_all[[nm]]) })
                  }
                }
            ''')
        else:
            ro.r('''
                for (nm in features) {
                  if (!is.factor(df_all[[nm]])) { df_all[[nm]] <- as.factor(df_all[[nm]]) }
                }
            ''')

        # Split train/validation
        ro.globalenv['train_pct'] = train_split / 100.0
        ro.r('''
            N <- nrow(df_all)
            idx <- sample.int(N, floor(N * train_pct))
            train_df <- df_all[idx, c(classes, features), drop=FALSE]
            val_df   <- df_all[-idx, c(classes, features), drop=FALSE]
        ''')

        # === CALL YOUR FUNCTIONS UNCHANGED ===
        # learn_MBCTree
        if approach == 'filter':
            ro.r('MBCTree_model <- learn_MBCTree(train_df, val_df, classes, features, filter=TRUE, measure="global", verbose=FALSE)')
        else:
            ro.globalenv['measure'] = measure
            ro.r('MBCTree_model <- learn_MBCTree(train_df, val_df, classes, features, filter=FALSE, measure=measure, verbose=FALSE)')

        # info_MBCTree
        tree_str = ro.r('info_MBCTree(MBCTree_model)')
        tree_str = str(tree_str[0]) if len(tree_str) else "(structure unavailable)"

        # predict_MBCTree_dataset_veryfast
        ro.r('pred <- predict_MBCTree_dataset_veryfast(MBCTree_model, val_df, classes, features)')

        # test_multidimensional
        perf = ro.r('test_multidimensional(pred$true, pred$out, classes)')
        global_acc = float(perf.rx2('global')[0])
        avg_acc    = float(perf.rx2('average')[0])
        per_class  = list(perf.rx2('per_class'))

        # Pretty output
        per_class_rows = [html.Tr([html.Td(cn), html.Td(f"{float(acc):.4f}")]) for cn, acc in zip(classes, per_class)]

        results_card = dbc.Card(
            dbc.CardBody(
                [
                    html.H4("MBCTree Results", className="card-title"),
                    html.Pre(tree_str, style={'textAlign': 'left', 'whiteSpace': 'pre-wrap'}),
                    html.Hr(),
                    html.H5("Validation Performance"),
                    html.P([html.B("Global accuracy: "), f"{global_acc:.4f}"]),
                    html.P([html.B("Average accuracy: "), f"{avg_acc:.4f}"]),
                    dbc.Table(
                        [html.Thead(html.Tr([html.Th("Class"), html.Th("Accuracy")]))] +
                        [html.Tbody(per_class_rows)],
                        bordered=True, striped=True, hover=True, responsive=True
                    ),
                ]
            ),
            className="mt-3",
        )
        return results_card, None

    except Exception as e:
        return html.Div(f"Error running MBCTree: {e}", style={'color': 'red'}), create_error_notification(
            f"Error running MBCTree: {str(e)}", "Computation Error"
        )

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8058)
