#!/bin/bash

echo "Starting CoSMIC"
echo "========================"

# Start the backend server
python -Xfrozen_modules=off demo.py --host 0.0.0.0 --port 3000