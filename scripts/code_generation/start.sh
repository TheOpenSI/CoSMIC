#!/bin/sh

# check if requirements.txt exists
if [ -f /usr/src/app/requirements.txt ]; then
    echo "Installing requirements..."
    pip3 install -r /usr/src/app/requirements.txt
fi

# check if main.py exists
if [ -f /usr/src/app/main.py ]; then
    echo "Executing main.py..."
    python3 /usr/src/app/main.py
else
    echo "Nothing to execute, shutting down."
fi