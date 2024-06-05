# tested with echo only
#!/bin/bash

for file_number in {0..9}; do
    # will work with default config with .env only
    command="python generate_data.py --file $file_number"
    echo "Running: $command"
    # $command
done
