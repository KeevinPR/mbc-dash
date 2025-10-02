#!/bin/bash
# Start script for MBC-Dash application
# Activates conda environment and runs the Dash app

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate conda environment
echo "Activating mbc_dash conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mbc_dash

# Check if activation was successful
if [ $? -ne 0 ]; then
    echo "❌ Failed to activate mbc_dash environment"
    echo "Please run: conda env create -f environment.yml"
    exit 1
fi

echo "✅ Environment activated"
echo "Starting MBC-Dash application..."
echo "Access the application at: http://localhost:8058"
echo "Press Ctrl+C to stop the server"
echo ""

# Run the Dash application
python dash_mbc.py

