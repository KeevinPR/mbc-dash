# -*- coding: utf-8 -*-
"""
Main Dash Application for Multi-dimensional Classification with MBCs

- TW-MBC (Benjumeda): bounded treewidth learning with exact inference



"""

import dash



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
print("✅ MBC DASHBOARD STARTED")

# Expose server for gunicorn
server = app.server

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8058)
