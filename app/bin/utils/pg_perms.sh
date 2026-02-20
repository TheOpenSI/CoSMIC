#!/bin/bash


# -------------------------------------------------------------------------------------------------------------
# File: pg_perms.sh
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


# @FUNCTION: pg_perms_get
# @USAGE: pg_perms_get -p=<PREFIX>
# @DESCRIPTION:
#   Checks for the existence of UID, GID, and CHMOD variables in the environment
#   using a dynamic prefix search (indirect expansion).
# @ARG -p|--prefix: The uppercase service prefix (e.g., POSTGRES).
# @ARG -h|--help:   Display function usage.
# @RETURN: 0
#   If all variables exist, 1 if any are missing.
pg_perms_get() {
    local prefix=""

    for arg in "$@"; do
        case "$arg" in
            -p=*|--prefix=*)
                prefix="${arg#*=}"
                ;;
            -h|--help)
                echo "Usage: pg_perms_get -p=<PREFIX>"
                return 0
                ;;
        esac
    done

    [[ -z "$prefix" ]] && { pg_log -l=4 -m="Error: -p/--prefix required"; return 1; }

    local get_uid="${prefix}_UID"
    local get_gid="${prefix}_GID" 
    local get_perm="${prefix}_CHMOD"

    [[ -n "${!get_uid:-}" && -n "${!get_gid:-}" && -n "${!get_perm:-}" ]]
}


# @FUNCTION: pg_perms_set
# @USAGE: pg_perms_set -p=<PREFIX> -f=<PATH>
# @DESCRIPTION:
#   Executes `sudo chown` and `sudo chmod` on a file path based on the variables
#   associated with the provided prefix.
# @ARG -p|--prefix: The uppercase service prefix.
# @ARG -f|--file:   The full path to the target file.
# @ARG -h|--help:   Display function usage.
# @RETURN: 0
#   On success, non-zero if sudo commands fail.
pg_perms_set() {
    local prefix=""
    local file=""

    for arg in "$@"; do
        case "$arg" in
            -p=*|--prefix=*)
                prefix="${arg#*=}"
                ;;

            -f=*|--file=*)
                file="${arg#*=}"
                ;;

            -h|--help) 
                echo "Usage: pg_perms_set -p=<PREFIX> -f=<PATH>"
                return 0
                ;;
        esac
    done

    local set_uid="${prefix}_UID"
    local set_gid="${prefix}_GID"
    local set_perm="${prefix}_CHMOD"

    sudo chown "${!set_uid}:${!set_gid}" "$file"
    sudo chmod "${!set_perm}" "$file"
}
