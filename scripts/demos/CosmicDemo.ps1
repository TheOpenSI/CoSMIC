#Requires -Version 7.0
#Requires -PSEdition Core


<#
.SYNOPSIS
Demo OpenSI-CoSMIC Platform Setup Script

.DESCRIPTION
Automates OpenSI-CoSMIC platform setup (demo version) by copying example files
from the examples directory to properly organized docker/secrets, docker/configs,
and backend/cores directories with filename cleaning.

.PARAMETER LogLevel
Sets the logging verbosity level. Controls which message types are displayed.
Valid values: Info (0), Warning (1), Error (2). Default is Info for complete visibility.

.EXAMPLE
.\CosmicDemo.ps1
Runs the script with default INFO level logging, showing all messages.

.EXAMPLE
.\CosmicDemo.ps1 -LogLevel Warning
Runs the script showing only WARNING and ERROR, reducing output verbosity.

.EXAMPLE
.\CosmicDemo.ps1 -LogLevel Error
Runs the script showing only critical ERROR, useful for CI/CD pipelines.

.NOTES
Author: Bing Tran
Version: 0.1
Requires: PowerShell 7.0+ (Core Edition)
Platform: Windows 10/11 with Docker Desktop installed

.LINK
https://docs.docker.com/compose/

.INPUTS
None. This script does not accept pipeline input.

.OUTPUTS
Console output with timestamped log messages. All output goes to stderr for
proper stream handling in pipelines.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $false, Position = 0)]
    [ValidateSet('Info', 'Warning', 'Error')]
    [string]$LogLevel = 'Info'
)

################################################################################
#                            Script Configuration                              #
################################################################################

$ErrorActionPreference = 'Stop'
$VerbosePreference = 'Continue'


# Script metadata
$script:ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Find the project root by searching upward for the examples directory
# This allows scripts to be nested in subdirectories like [scripts/demos]
function Find-ProjectRoot
{
    [CmdletBinding()]
    [OutputType([string])]
    param()

    $currentDir = $script:ScriptDir
    $maxDepth = 10
    $depth = 0

    while ($depth -lt $maxDepth)
    {
        $examplesPath = Join-Path -Path $currentDir -ChildPath 'examples'

        if (Test-Path -Path $examplesPath -PathType Container)
        {
            return $currentDir
        }

        $parentDir = Split-Path -Parent $currentDir

        # Break if we've reached the filesystem root
        if ($parentDir -eq $currentDir)
        {
            break
        }

        $currentDir = $parentDir
        $depth++
    }

    # Fallback to parent of script directory if examples not found
    return (Split-Path -Parent $script:ScriptDir)
}

$script:Version         = '0.1'
$script:RootDir         = Find-ProjectRoot
$script:BackendDir      = Join-Path $script:RootDir 'backend'
$script:ExamplesDir     = Join-Path $script:RootDir 'examples'
$script:DockerDir       = Join-Path $script:RootDir 'docker'
$script:DockerSecrets   = Join-Path $script:DockerDir 'secrets'
$script:DockerConfigs   = Join-Path $script:DockerDir 'configs'


# Color codes for console output
$script:Colors = @{
    Reset   = 0
    Blue    = 34
    Yellow  = 33
    Red     = 31
}


# Logging levels
$script:LogLevels = @{
    'Info'    = 0
    'Warning' = 1
    'Error'   = 2
}


# Current logging level
$script:CurrentLogLevel = $script:LogLevels[$LogLevel]


################################################################################
#                               Logging Functions                              #
################################################################################

<#
.SYNOPSIS
Writes a formatted log message to the console with timestamp and level indicator.

.DESCRIPTION
Logs messages with configurable levels (Info, Warning, Error) with timestamps,
emoji indicators, and color-coded output to stderr. Respects the current logging
level to filter verbose output appropriately.

.PARAMETER Level
The logging level: 'Info', 'Warning', or 'Error'. Controls both the output
formatting and whether the message is displayed based on current log level.

.PARAMETER Message
The message content to log. Can contain any text and will be included after
the timestamp and level indicator.

.EXAMPLE
Write-LogMessage -Level Info -Message "Starting process"
Writes an info message with blue color to stderr.

.EXAMPLE
Write-LogMessage -Level Error -Message "Critical failure occurred"
Writes an error message with red color to stderr (always displayed).

.OUTPUTS
None. This function outputs to stderr only.
#>
function Write-LogMessage
{
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [ValidateSet('Info', 'Warning', 'Error')]
        [string]$Level,

        [Parameter(Mandatory = $true)]
        [string]$Message
    )

    $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    $levelNumeric = $script:LogLevels[$Level]

    # Only output if current level allows it
    if ($levelNumeric -lt $script:CurrentLogLevel)
    {
        return
    }

    $colorCode = $script:Colors['Reset']
    $emoji = ''

    switch ($Level)
    {
        'Info'
        {
            $colorCode  = $script:Colors['Blue']
            $emoji      = 'ℹ'
            $levelText  = 'INFO'
        }
        'Warning'
        {
            $colorCode  = $script:Colors['Yellow']
            $emoji      = '⚠'
            $levelText  = 'WARNING'
        }
        'Error'
        {
            $colorCode  = $script:Colors['Red']
            $emoji      = '✗'
            $levelText  = 'ERROR'
        }
    }

    $formattedMessage = "`e[0;${colorCode}m[$timestamp] $emoji $levelText`e[0m: $Message"
    [Console]::Error.WriteLine($formattedMessage)
}

<#
.SYNOPSIS
Convenience function for Info level logging.

.PARAMETER Message
The message to log at Info level.

.EXAMPLE
Write-Info "Process completed successfully"

.OUTPUTS
None. Output goes to stderr.
#>
function Write-Info
{
    [CmdletBinding()]
    param([Parameter(Mandatory = $true)][string]$Message)
    Write-LogMessage -Level Info -Message $Message
}

<#
.SYNOPSIS
Convenience function for Warning level logging.

.PARAMETER Message
The message to log at Warning level.

.EXAMPLE
Write-WarningLog "File not found, skipping..."

.OUTPUTS
None. Output goes to stderr.
#>
function Write-WarningLog
{
    [CmdletBinding()]
    param([Parameter(Mandatory = $true)][string]$Message)
    Write-LogMessage -Level Warning -Message $Message
}

<#
.SYNOPSIS
Convenience function for Error level logging.

.PARAMETER Message
The message to log at Error level.

.EXAMPLE
Write-ErrorLog "Failed to create directory"

.OUTPUTS
None. Output goes to stderr.
#>
function Write-ErrorLog
{
    [CmdletBinding()]
    param([Parameter(Mandatory = $true)][string]$Message)
    Write-LogMessage -Level Error -Message $Message
}


################################################################################
#                               Utility Functions                              #
################################################################################

<#
.SYNOPSIS
Validates that required source directories exist.

.DESCRIPTION
Checks for the existence of the examples directory and its required subdirectories
(backend, database, gui). Issues warnings for missing subdirectories but does not
fail the entire validation process to allow partial setups.

.OUTPUTS
System.Boolean
Returns $true if validation passes, $false if critical directories are missing.

.EXAMPLE
$isValid = Test-SourceDirectory
if ($isValid) { Write-Info "Validation passed" }
#>
function Test-SourceDirectory
{
    [CmdletBinding()]
    [OutputType([bool])]
    param()

    Write-Info 'Validating source directories...'

    if (-not (Test-Path -Path $script:ExamplesDir -PathType Container))
    {
        Write-ErrorLog "Examples directory not found: [$script:ExamplesDir]"
        return $false
    }

    $requiredSubdirs = @('backend', 'database', 'gui')
    foreach ($subdir in $requiredSubdirs)
    {
        $subdirPath = Join-Path $script:ExamplesDir $subdir
        if (-not (Test-Path -Path $subdirPath -PathType Container))
        {
            Write-WarningLog "Subdirectory not found: [$subdirPath]"
        }
    }

    Write-Info 'Source directories validated successfully'
    return $true
}

<#
.SYNOPSIS
Creates the required Docker directory structure.

.DESCRIPTION
Creates nested docker directory structure for secrets and configs. Handles
creation of all subdirectories needed for proper file organization. Non-fatal
if directories already exist.

.OUTPUTS
System.Boolean
Returns $true on success, $false if directory creation fails.

.EXAMPLE
$success = Test-DockerDirectory
if (-not $success) { exit 1 }
#>
function Test-DockerDirectory
{
    [CmdletBinding()]
    [OutputType([bool])]
    param()

    Write-Info 'Creating docker directory structure...'

    try
    {
        $directoriesToCreate = @(
            (Join-Path $script:DockerSecrets 'database'),
            (Join-Path $script:DockerSecrets 'gui'),
            (Join-Path $script:DockerConfigs 'gui')
        )

        foreach ($dir in $directoriesToCreate)
        {
            if (-not (Test-Path -Path $dir -PathType Container))
            {
                New-Item -Path $dir -ItemType Directory -Force | Out-Null
            }
        }

        Write-Info 'Docker directory structure created'
        return $true
    } catch
    {
        Write-ErrorLog "Failed to create docker directories: [$_]"
        return $false
    }
}

<#
.SYNOPSIS
Cleans filename by removing .example suffix and service prefix.

.DESCRIPTION
Transforms filenames according to Docker setup conventions:
- Removes .example suffix (pgadmin_svr.example.json -> pgadmin_svr.json)
- Removes service prefix (pgadmin_svr.json -> svr.json)

This ensures clean, minimal filenames for configuration files.

.PARAMETER Filename
The original filename to process.

.OUTPUTS
System.String
The cleaned filename with all transformations applied.

.EXAMPLE
Get-CleanedFilename 'pgadmin_svr.example.json'
# Returns: svr.json

.EXAMPLE
Get-CleanedFilename 'cosmic_app.example.env'
# Returns: app.env
#>
function Get-CleanedFilename
{
    [CmdletBinding()]
    [OutputType([string])]
    param(
        [Parameter(Mandatory = $true)]
        [string]$Filename
    )

    # Split filename and extension
    $name = [System.IO.Path]::GetFileNameWithoutExtension($Filename)
    $extension = [System.IO.Path]::GetExtension($Filename)

    # Remove .example suffix if present
    $name = $name -replace '\.example$', ''

    # Remove prefix (everything up to and including underscore)
    if ($name -match '^[^_]+_(.+)$')
    {
        $name = $matches[1]
    }

    return "$name$extension"
}

<#
.SYNOPSIS
Copies a file to destination with cleaned filename and logs operation.

.DESCRIPTION
Copies a single file while renaming it according to Docker setup conventions.
Logs success or failure with descriptive messages. Handles the complete
transformation and logging workflow for file copying.

.PARAMETER SourcePath
Full path to source file. Must exist.

.PARAMETER TargetPath
Directory where file will be copied. Will be created if needed.

.PARAMETER Description
Human-readable description of the file for logging purposes.

.OUTPUTS
System.Boolean
Returns $true on success, $false if source file not found or copy fails.

.EXAMPLE
Copy-ConfigFile -SourcePath 'examples/gui/pgadmin_svr.example.json' `
                 -TargetPath 'docker/configs/gui' `
                 -Description 'GUI JSON config'
#>
function Copy-ConfigFile
{
    [CmdletBinding()]
    [OutputType([bool])]
    param(
        [Parameter(Mandatory = $true)]
        [string]$SourcePath,

        [Parameter(Mandatory = $true)]
        [string]$TargetPath,

        [Parameter(Mandatory = $true)]
        [string]$Description
    )

    if (-not (Test-Path -Path $SourcePath -PathType Leaf))
    {
        Write-WarningLog "Source file not found: [$SourcePath]"
        return $false
    }

    try
    {
        $filename = Split-Path -Leaf $SourcePath
        $cleanedFilename = Get-CleanedFilename $filename
        $destinationPath = Join-Path $TargetPath $cleanedFilename

        Copy-Item -Path $SourcePath -Destination $destinationPath -Force
        Write-Info "Copied <<$Description`: $filename>> ==> <<$cleanedFilename>>"
        return $true
    } catch
    {
        Write-ErrorLog "Failed to copy <<$Description`>>: <<$_>>"
        return $false
    }
}


################################################################################
#                               Main Copy Functions                            #
################################################################################

<#
.SYNOPSIS
Copies [backend] docker service files to specified directories.

.DESCRIPTION
Processes all cosmic_*.example.env files from the [backend] examples directory
and copies them to the [backend/cores] directory at the project root. The
[backend/cores] directory is expected to exist by default and is not created
by this function.

The [backend/cores] directory must exist in the project structure. If it doesn't
exist, the copy operation will fail with an error message, but the script will
continue processing other file types since this is a non-fatal operation.

Files are transformed during the copy operation using the standard filename
transformation rules: the .example suffix is removed and the service prefix
(everything before the first underscore) is stripped from the filename.

Non-fatal if the [backend] examples directory is missing—the script will issue
a warning and continue with other file types.

.OUTPUTS
[bool] Always returns $true (non-fatal operation)

.EXAMPLE
Copy-BackendFile

.NOTES
Files copied: examples/backend/cosmic_*.example.env
Destination: <root>/backend/cores/
Transformation applied: cosmic_app.example.env -> app.env

The backend/cores directory is pre-existing and is not created by this function.
#>
function Copy-BackendFile
{
    [CmdletBinding()]
    [OutputType([bool])]
    param()

    Write-Info 'Processing [backend] docker service files...'

    $backendDir = Join-Path $script:ExamplesDir 'backend'
    if (-not (Test-Path -Path $backendDir -PathType Container))
    {
        Write-WarningLog '[backend] directory not found, skipping...'
        return $true
    }

    $sourceFiles = @(Get-ChildItem -Path $backendDir -Filter 'cosmic_*.example.env' -ErrorAction SilentlyContinue)

    if ($sourceFiles.Count -eq 0)
    {
        Write-WarningLog 'No [backend] example files found'
        return $true
    }

    $coresDir = Join-Path $script:BackendDir 'cores'
    foreach ($sourceFile in $sourceFiles)
    {
        Copy-ConfigFile -SourcePath $sourceFile.FullName -TargetPath $coresDir -Description 'required files for [backend] docker service'
    }

    return $true
}

<#
.SYNOPSIS
Copies [database] docker service files to [docker/secrets/database].

.DESCRIPTION
Processes postgres_*.example.txt files from the [database] examples directory.
Files are copied to [docker/secrets/database] with cleaned names.

.OUTPUTS
System.Boolean
Returns $true on completion (non-fatal if directory missing).

.EXAMPLE
Copy-DatabaseFile
#>
function Copy-DatabaseFile
{
    [CmdletBinding()]
    [OutputType([bool])]
    param()

    Write-Info 'Processing [database] docker service files...'

    $databaseDir = Join-Path $script:ExamplesDir 'database'
    if (-not (Test-Path -Path $databaseDir -PathType Container))
    {
        Write-WarningLog '[database] directory not found, skipping...'
        return $true
    }

    $sourceFiles = @(Get-ChildItem -Path $databaseDir -Filter 'postgres_*.example.txt' -ErrorAction SilentlyContinue)

    if ($sourceFiles.Count -eq 0)
    {
        Write-WarningLog 'No [database] example files found'
        return $true
    }

    $destDir = Join-Path $script:DockerSecrets 'database'
    New-Item -Path $destDir -ItemType Directory -Force | Out-Null

    foreach ($sourceFile in $sourceFiles)
    {
        Copy-ConfigFile -SourcePath $sourceFile.FullName -TargetPath $destDir -Description 'required files for [database] docker service'
    }

    return $true
}

<#
.SYNOPSIS
Copies [gui] docker service files to appropriate docker directories.

.DESCRIPTION
Processes pgadmin_*.example.txt files to [docker/secrets/gui] and
pgadmin_*.example.json files to [docker/configs/gui]. Handles both file types
with appropriate destination directories and logging.

.OUTPUTS
System.Boolean
Returns $true on completion (non-fatal if directory missing).

.EXAMPLE
Copy-GuiFile
#>
function Copy-GuiFile
{
    [CmdletBinding()]
    [OutputType([bool])]
    param()

    Write-Info 'Processing [gui] docker service files...'

    $guiDir = Join-Path $script:ExamplesDir 'gui'
    if (-not (Test-Path -Path $guiDir -PathType Container))
    {
        Write-WarningLog '[gui] directory not found, skipping...'
        return $true
    }

    # Copy .txt files to secrets
    $txtFiles = @(Get-ChildItem -Path $guiDir -Filter 'pgadmin_*.example.txt' -ErrorAction SilentlyContinue)

    if ($txtFiles.Count -gt 0)
    {
        $destDir = Join-Path $script:DockerSecrets 'gui'
        New-Item -Path $destDir -ItemType Directory -Force | Out-Null
        foreach ($txtFile in $txtFiles)
        {
            Copy-ConfigFile -SourcePath $txtFile.FullName -TargetPath $destDir -Description 'required files for [gui] docker service'
        }
    } else
    {
        Write-WarningLog '[.txt] required files for [gui] docker service not found'
    }

    # Copy .json files to configs
    $jsonFiles = @(Get-ChildItem -Path $guiDir -Filter 'pgadmin_*.example.json' -ErrorAction SilentlyContinue)

    if ($jsonFiles.Count -gt 0)
    {
        $destDir = Join-Path $script:DockerConfigs 'gui'
        New-Item -Path $destDir -ItemType Directory -Force | Out-Null
        foreach ($jsonFile in $jsonFiles)
        {
            Copy-ConfigFile -SourcePath $jsonFile.FullName -TargetPath $destDir -Description 'required files for [gui] docker service'
        }
    } else
    {
        Write-WarningLog '[.json] required files for [gui] docker service not found'
    }

    return $true
}


################################################################################
#                               Main Execution                                 #
################################################################################

<#
.SYNOPSIS
Main orchestration function that coordinates all setup steps.

.DESCRIPTION
Runs through the complete setup process in order:
1. Validates source directories
2. Creates docker directory structure
3. Copies all configuration files
4. Starts Docker Compose

Handles error states and provides detailed logging throughout.

.OUTPUTS
System.Int32
Returns 0 on success, 1 on failure.

.EXAMPLE
$exitCode = Invoke-CosmicDemo
#>
function Invoke-CosmicDemo
{
    [CmdletBinding()]
    [OutputType([int])]
    param()

    Write-Info "Starting Docker setup script v$script:Version"
    Write-Info "Root directory: $script:RootDir"

    # Validate prerequisites
    if (-not (Test-SourceDirectory))
    {
        Write-ErrorLog 'Validation failed'
        return 1
    }

    # Create directory structure
    if (-not (Test-DockerDirectory))
    {
        Write-ErrorLog 'Failed to create docker directories'
        return 1
    }

    # Copy configuration files
    Copy-BackendFile
    Copy-DatabaseFile
    Copy-GuiFile

    Write-Info 'File copying completed'
    Write-Info 'Running Docker Compose...'

    # Start Docker Compose
    try
    {
        Push-Location -Path $script:RootDir
        docker compose up -d --build

        if ($LASTEXITCODE -eq 0)
        {
            Write-Info 'Docker Compose started successfully'
            return 0
        } else
        {
            Write-ErrorLog 'Docker Compose failed to start'
            return 1
        }
    } catch
    {
        Write-ErrorLog "Docker Compose execution failed: $_"
        return 1
    } finally
    {
        Pop-Location
    }
}


# Error handler
trap
{
    Write-ErrorLog "Script failed: $_"
    exit 1
}


# Execute main function
$exitCode = Invoke-CosmicDemo
exit $exitCode
