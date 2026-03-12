#!/usr/bin/env zsh


################################################################################
#
# cosmic_demo.zsh - Demo OpenSI-CoSMIC Platform Setup Script
#
# @description Automates OpenSI-CoSMIC platform setup (demo version) by copying
#              example files from the examples directory to properly organized
#              docker/secrets, docker/configs, and backend/cores directories
#              with filename cleaning.
#
# @usage
#   ./cosmic_demo.sh
#   LOG_LEVEL=1 ./cosmic_demo.sh
#
# @examples
#   # Run with default INFO logging
#   ./cosmic_demo.sh
#
#   # Run with WARNING level (less verbose)
#   LOG_LEVEL=1 ./cosmic_demo.sh
#
#   # Run with ERROR level (most verbose)
#   LOG_LEVEL=2 ./cosmic_demo.sh
#
# @env LOG_LEVEL (optional) Sets the logging verbosity (0=INFO, 1=WARNING, 2=ERROR)
#
# @exitcode 0 If OpenSI-CoSMIC platform setup (demo version) completed
#             successfully.
# @exitcode 1 If any critical operation failed.
#
################################################################################


setopt ERR_EXIT           # Exit on error
setopt NO_UNSET           # Error on undefined variables
setopt PIPE_FAIL          # Pipe failure causes script failure
setopt GLOB_SUBST         # Allow parameter expansion within glob patterns


# Script metadata
SCRIPT_DIR="${0:a:h}"
readonly SCRIPT_DIR

# Find the project root by searching upward for the examples directory
# This allows scripts to be nested in subdirectories like [scripts/demos]
find_project_root() {
	local current_dir="$SCRIPT_DIR"
	local max_depth=10
	local depth=0
	
	while (( depth < max_depth )); do
		if [[ -d "${current_dir}/examples" ]]; then
			print "$current_dir"
			return 0
		fi
		current_dir="${current_dir:h}"  # Zsh-native: parent directory
		((depth++))
	done
	
	# Fallback to parent of script directory if examples not found
	print "${SCRIPT_DIR:h}"
	return 1
}

SCRIPT_VERSION="0.1"
readonly SCRIPT_VERSION
ROOT_DIR="$(find_project_root)"
readonly ROOT_DIR
BACKEND_DIR="${ROOT_DIR}/backend"
readonly BACKEND_DIR
EXAMPLES_DIR="${ROOT_DIR}/examples"
readonly EXAMPLES_DIR
DOCKER_DIR="${ROOT_DIR}/docker"
readonly DOCKER_DIR
DOCKER_SECRETS="${DOCKER_DIR}/secrets"
readonly DOCKER_SECRETS
DOCKER_CONFIGS="${DOCKER_DIR}/configs"
readonly DOCKER_CONFIGS


# Color codes for logging (ANSI escape sequences)
COLOR_RESET=$'\e[0m'
readonly COLOR_RESET
COLOR_BLUE=$'\e[0;34m'
readonly COLOR_BLUE
COLOR_YELLOW=$'\e[1;33m'
readonly COLOR_YELLOW
COLOR_RED=$'\e[0;31m'
readonly COLOR_RED


# Logging level constants
LOG_INFO=0
readonly LOG_INFO
LOG_WARNING=1
readonly LOG_WARNING
LOG_ERROR=2
readonly LOG_ERROR


# Current logging level (can be overridden by environment variable)
CURRENT_LOG_LEVEL="${LOG_LEVEL:-$LOG_INFO}"


################################################################################
#                               Logging Functions                              #
################################################################################
#
# @description Print a timestamped log message with color coding and level filtering
#
# Outputs formatted log messages to stderr with timestamp, emoji indicator,
# and color coding. Respects the CURRENT_LOG_LEVEL setting to filter verbose output.
#
# @arg $1 int - Logging level constant (LOG_INFO=0, LOG_WARNING=1, LOG_ERROR=2)
# @arg $2 str - Message text to log
#
# @exitcode 0 Always succeeds
#
# @see log_info
# @see log_warning
# @see log_error
#
# @example
#   log_message "$LOG_INFO" "Process started"
#   log_message "$LOG_ERROR" "Critical failure occurred"
log_message() {
	local level="$1"
	local message="$2"
	local timestamp
	timestamp=$(date '+%Y-%m-%d %H:%M:%S')
	
	case "$level" in
		$LOG_INFO)
			if (( CURRENT_LOG_LEVEL <= LOG_INFO )); then
				print -u 2 "${COLOR_BLUE}[${timestamp}] ℹ INFO${COLOR_RESET}: ${message}"
			fi
			;;
		$LOG_WARNING)
			if (( CURRENT_LOG_LEVEL <= LOG_WARNING )); then
				print -u 2 "${COLOR_YELLOW}[${timestamp}] ⚠ WARNING${COLOR_RESET}: ${message}"
			fi
			;;
		$LOG_ERROR)
			print -u 2 "${COLOR_RED}[${timestamp}] ✗ ERROR${COLOR_RESET}: ${message}"
			;;
	esac
}

# @description Log an informational message at INFO level
# @arg $1 str - Message text to log
# @exitcode 0 Always succeeds
# @example log_info "Initialization started"
log_info() {
	log_message "$LOG_INFO" "$1"
}

# @description Log a warning message at WARNING level
# @arg $1 str - Message text to log
# @exitcode 0 Always succeeds
# @example log_warning "Configuration file not found, using defaults"
log_warning() {
	log_message "$LOG_WARNING" "$1"
}

# @description Log an error message at ERROR level
# @arg $1 str - Message text to log
# @exitcode 0 Always succeeds
# @example log_error "Failed to create required directory"
log_error() {
	log_message "$LOG_ERROR" "$1"
}


################################################################################
#								Utility Functions							   #
################################################################################
#
# @description Validate that required source directories exist
#
# Checks for the existence of the examples directory and its required subdirectories
# (backend, database, gui). Issues warnings for missing subdirectories but does not
# fail the entire validation process to allow partial setups.
#
# @exitcode 0 If examples directory exists
# @exitcode 1 If examples directory does not exist
#
# @example
#   if validate_source_directories; then
#       log_info "Validation passed"
#   fi
validate_source_directories() {
	log_info "Validating source directories..."
	
	if [[ ! -d "$EXAMPLES_DIR" ]]; then
		log_error "Examples directory not found: [$EXAMPLES_DIR]"
		return 1
	fi
	
	local required_subdirs=("backend" "database" "gui")
	for subdir in "${required_subdirs[@]}"; do
		if [[ ! -d "$EXAMPLES_DIR/$subdir" ]]; then
			log_warning "Subdirectory not found: [$EXAMPLES_DIR/$subdir]"
		fi
	done
	
	log_info "Source directories validated successfully"
	return 0
}

# @description Create docker directory structure if it doesn't exist
# Initializes the nested docker directory structure with proper subdirectories
# for secrets and configs. Required before copying files.
# @exitcode 0 If directories are created successfully
# @exitcode 1 If directory creation fails
# @example create_docker_directories
create_docker_directories() {
	log_info "Creating docker directory structure..."
	
	mkdir -p "$DOCKER_SECRETS/database" || return 1
	mkdir -p "$DOCKER_SECRETS/gui"		|| return 1
	mkdir -p "$DOCKER_CONFIGS/gui"		|| return 1
	
	log_info "Docker directory structure created"
	return 0
}

# @description Remove .example suffix and prefix from filename
#
# Transforms filenames according to Docker setup conventions:
# - Removes .example suffix (pgadmin_svr.example.json -> pgadmin_svr.json)
# - Removes service prefix (pgadmin_svr.json -> svr.json)
#
# @arg $1 str - Original filename to process
# @stdout Cleaned filename with transformations applied
# @example
#   cleaned=$(get_cleaned_filename "pgadmin_svr.example.json")
#   echo "$cleaned"  # Output: svr.json
get_cleaned_filename() {
	local filename="$1"

	# Get extension after last dot
	local extension="${filename##*.}"

	# Remove extension
	local basename="${filename%.*}"
	
	# Remove .example suffix
	basename="${basename%.example}"
	# Remove prefix (everything up to and including first underscore)
	basename="${basename#*_}"

	print "${basename}.${extension}"
}

# @description Copy file with cleaned name and log the operation
#
# Copies a single configuration file to the destination directory while applying
# filename transformations. Creates destination directory if needed and logs
# success or failure messages.
#
# @arg $1 str - Full path to source file
# @arg $2 str - Destination directory path
# @arg $3 str - Description for logging
# @exitcode 0 If file is copied successfully
# @exitcode 1 If source file not found or copy fails
# @example
#   copy_file "/path/to/cosmic_app.example.env" "/docker/secrets/backend" "backend env file"
copy_file() {
	local source="$1"
	local target="$2"
	local description="$3"
	
	if [[ ! -f "$source" ]]; then
		log_warning "Source file not found: [$source]"
		return 1
	fi
	
	local filename="$(basename "$source")"
	local cleaned_filename="$(get_cleaned_filename "$filename")"
	local destination="${target}/${cleaned_filename}"
	
	if cp -p "$source" "$destination" 2>/dev/null; then
		log_info "Copied <<$description: $filename>> ==> <<$cleaned_filename>>"
		return 0
	else
		log_error "Failed to copy <<$description>>: <<$source>>"
		return 1
	fi
}


################################################################################
#							Main Copy Functions								   #
################################################################################
#
# @description Copy [backend] docker service files to specified directories
# Processes cosmic_*.example.env files from the [backend] examples directory
# and copies them to the [backend/cores] directory with cleaned names.
# @exitcode 0 Always succeeds (non-fatal if directory doesn't exist)
# @example copy_backend_files
copy_backend_files() {
	log_info "Processing [backend] docker service files..."
	
	if [[ ! -d "$EXAMPLES_DIR/backend" ]]; then
		log_warning "[backend] directory not found, skipping..."
		return 0
	fi
	
	local source_files=()
	source_files=( "$EXAMPLES_DIR/backend"/cosmic_*.example.env(N) )
	
	if (( ${#source_files[@]} == 0 )); then
		log_warning "No [backend] example files found"
		return 0
	fi
	
	for source_file in "${source_files[@]}"; do
		copy_file "$source_file" "$BACKEND_DIR/cores" "required files for [backend] docker service" "" || true
	done
	
	return 0
}

# @description Copy [database] docker service files to [docker/secrets/database]
# Processes postgres_*.example.txt files from the [database] examples directory
# and copies them to the [docker/secrets/database] directory with cleaned names.
# @exitcode 0 Always succeeds (non-fatal if directory doesn't exist)
# @example copy_database_files
copy_database_files() {
	log_info "Processing [database] docker service files..."
	
	if [[ ! -d "$EXAMPLES_DIR/database" ]]; then
		log_warning "[database] directory not found, skipping..."
		return 0
	fi
	
	local source_files=()
	source_files=( "$EXAMPLES_DIR/database"/postgres_*.example.txt(N) )
	
	if (( ${#source_files[@]} == 0 )); then
		log_warning "No [database] example files found"
		return 0
	fi
	
	mkdir -p "$DOCKER_SECRETS/database" || return 1
	
	for source_file in "${source_files[@]}"; do
		copy_file "$source_file" "$DOCKER_SECRETS/database" "required files for [database] docker service" || true
	done
	
	return 0
}

# @description Copy [gui] docker service files to appropriate docker directories
# Processes pgadmin_*.example.txt files to [docker/secrets/gui] and
# pgadmin_*.example.json files to [docker/configs/gui], with cleaned names.
# @exitcode 0 Always succeeds (non-fatal if directory doesn't exist)
# @example copy_gui_files
copy_gui_files() {
	log_info "Processing [gui] docker service files..."
	
	if [[ ! -d "$EXAMPLES_DIR/gui" ]]; then
		log_warning "[gui] directory not found, skipping..."
		return 0
	fi
	
	local txt_files=()
	txt_files=( "$EXAMPLES_DIR/gui"/pgadmin_*.example.txt(N) )
	
	if (( ${#txt_files[@]} > 0 )); then
		mkdir -p "$DOCKER_SECRETS/gui" || return 1
		for txt_file in "${txt_files[@]}"; do
			copy_file "$txt_file" "$DOCKER_SECRETS/gui" "required files for [gui] docker service" || true
		done
	else
		log_warning "[.txt] required files for [gui] docker service not found"
	fi
	
	local json_files=()
	json_files=( "$EXAMPLES_DIR/gui"/pgadmin_*.example.json(N) )
	
	if (( ${#json_files[@]} > 0 )); then
		mkdir -p "$DOCKER_CONFIGS/gui" || return 1
		for json_file in "${json_files[@]}"; do
			copy_file "$json_file" "$DOCKER_CONFIGS/gui" "required files for [gui] docker service" || true
		done
	else
		log_warning "[.json] required files for [gui] docker service not found"
	fi
	
	return 0
}


################################################################################
#								Main Execution								   #
################################################################################
#
# @description Main orchestration function
#
# Coordinates all setup steps in the proper sequence:
# 1. Validates source directories exist
# 2. Loads environment variables from .env
# 3. Creates docker directory structure
# 4. Copies configuration files with transformations
# 5. Starts Docker Compose with proper build flags
#
# @exitcode 0 If all operations complete successfully
# @exitcode 1 If any critical operation fails
# @example main
main() {
	log_info "Starting Docker setup script v$SCRIPT_VERSION"
	log_info "Root directory: $ROOT_DIR"
	
	if ! validate_source_directories; then
		log_error "Validation failed"
		return 1
	fi
	
	if ! create_docker_directories; then
		log_error "Failed to create docker directories"
		return 1
	fi
	
	copy_backend_files
	copy_database_files
	copy_gui_files
	
	log_info "File copying completed"
	log_info "Running Docker Compose..."
	
	if builtin cd "$ROOT_DIR" && docker compose up -d --build; then
		log_info "Docker Compose started successfully"
		return 0
	else
		log_error "Docker Compose failed to start"
		return 1
	fi
}

trap 'log_error "Script failed at line $LINENO"; exit 1' ERR
main "$@"
exit $?
