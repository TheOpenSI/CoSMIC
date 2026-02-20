#!/bin/bash


# -------------------------------------------------------------------------------------------------------------
# File: pg_log.sh
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


# --- Log Level Colour ---
CLR_RED='\033[0;31m'
CLR_GRN='\033[0;32m'
CLR_YLW='\033[1;33m'
CLR_RST='\033[0m'


# @FUNCTION: pg_log
# @USAGE: pg_log -l=<1-4> -m="<TEXT>"
# @DESCRIPTION:
#   Outputs a color-coded message to STDOUT.
#   Levels: 
#       1 = LOG     (Standard)
#       2 = SUCCESS (Green)
#       3 = WARNING (Yellow)
#       4 = ERROR   (Red)
# @ARG -l|--level: The numeric severity level.
# @ARG -m|--msg:   The string content to be displayed.
# @ARG -h|--help:  Display function usage.
pg_log() {
    local lvl=1
    local msg=""
    local clr="${CLR_RST}"

    for arg in "$@"; do
        case "$arg" in
            -l=*|--level=*)
                lvl="${arg#*=}"
                ;;
            -m=*|--msg=*)
                msg="${arg#*=}"
                ;;
            -h|--help)
                echo "Usage: pg_log -l=<1-4> -m='<TEXT>'"
                return 0
                ;;
        esac
    done

    case "$lvl" in
        2)
            clr="${CLR_GRN}"
            ;;
        3)
            clr="${CLR_YLW}"
            ;;
        4)
            clr="${CLR_RED}"
            ;;
        *)
            clr="${CLR_RST}"
            ;;
    esac

    echo -e "${clr}${msg}${CLR_RST}"
}
