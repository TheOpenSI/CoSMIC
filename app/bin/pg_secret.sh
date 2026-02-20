#!/bin/bash


# -------------------------------------------------------------------------------------------------------------
# File: pg_secret.sh
# Project: Open Source Institute-Cognitive System of Machine Intelligent Computing (OpenSI-CoSMIC)
# Contributors:
#     Bing Tran <u3295557@canberra.edu.au>
# 
# Copyright (c) 2026 Open Source Institute
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without
# limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so, subject to the following
# conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
# LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# -------------------------------------------------------------------------------------------------------------


# --- Safety Flags ---
set -euo pipefail


# --- Utils ---
source "./bin/utils/pg_log.sh"
source "./bin/utils/pg_perms.sh"


# --- Core Vars ---
SECRET_DIR="./docker/secret"
CONFIG_DIR="./docker/config"
SECRET_PREFIXES=("postgres" "pgadmin")


# Ensure the secret path exists
for DIR in "${SECRET_DIR}" "${CONFIG_DIR}"; do
    if [[ ! -d "${DIR}" ]]; then
        pg_log -l=3 -m="[⚠️ ${DIR}] path not found. Creating..."
        mkdir -p "${DIR}"
    fi
done


# Dynamic Processing
for SECRET_PREFIX in "${SECRET_PREFIXES[@]}"; do
    # Prepare casing
    LOWER_SECRET_PREFIX="${SECRET_PREFIX,,}"
    UPPER_SECRET_PREFIX="${SECRET_PREFIX^^}"

    # Load `.env` files with given prefix
    mapfile -t ENV_FILES < <(find "${SECRET_DIR}" -type f -name "${LOWER_SECRET_PREFIX}_*.env" 2>/dev/null)
    for ENV_PATH in "${ENV_FILES[@]}"; do
        pg_log -l=2 -m="📦 Loading config file: [${ENV_PATH}] (cosmic-${LOWER_SECRET_PREFIX})"

        # Auto export all defined variables from here on, then turn it off immidiately
        set -a
        # shellcheck source=/dev/null
        # Bypass `shellcheck` non-constant source warning as we're doing it dynamically
        source "${ENV_PATH}"
        set +a
        sleep 1.5s
    done

    # Set permissions for secret (.txt) and config (.json) files with given prefix
    mapfile -t TARGET_FILES < <(find "$SECRET_DIR" "$CONFIG_DIR" -type f \( -name "${LOWER_SECRET_PREFIX}_*.txt" -o -name "${LOWER_SECRET_PREFIX}_*.json" \) 2>/dev/null)
    for TARGET_FILE in "${TARGET_FILES[@]}"; do
        if pg_perms_get -p="${UPPER_SECRET_PREFIX}"; then
            pg_log -l=2 -m="🔒 Updating targeted file permission: [${TARGET_FILE}] (cosmic-${LOWER_SECRET_PREFIX})"
            pg_perms_set -p="${UPPER_SECRET_PREFIX}" -f="${TARGET_FILE}"
            sleep 1.5s
        else
            pg_log -l=4 -m="❌ Skipping targeted file [$TARGET_FILE]: <<\$${UPPER_SECRET_PREFIX}_UID/GID/CHMOD>> vars not found."
            break
        fi
    done
done


# Startup
sleep 1.5s
pg_log -l=1 -m="🚀 Firing up CoSMIC services from Docker..."

sleep 1.5s
sudo docker compose up -d --build
