#!/bin/bash
set -e

# Default values
# Default values
TOOL=""
TOOL_PROVIDED_BY_ARG=false
PROJECT_PATH="" # If set, scope is project
PROJECT_PATH="" # If set, scope is project
USE_GLOBAL=false
OVERWRITE=false
INSTALL_DIR=""
SKIP_PROMPTS=false
LAUNCH_DIR=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Helper function to print error and exit
die() {
    echo "Error: $1" >&2
    exit 1
}

# Helper function to check if a directory is writable
is_writable() {
    if [ -w "$1" ]; then
        return 0
    else
        return 1
    fi
}

# Determine default install location
determine_install_dir() {
    if [ -d "/data" ] && is_writable "/data"; then
        echo "/data/opd-mcp"
    elif [ -d "/localhome" ] && is_writable "/localhome"; then
        echo "/localhome/opd-mcp"
    else
        echo "$HOME/opd-mcp"
    fi
}

# Usage help
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --tool <cline|roo|continue>   Specify the tool to configure"
    echo "  --project [path]              Configure for specific project (defaults to current dir if path omitted)"
    echo "  --overwrite                   Overwrite existing installation and configs without prompting"
    echo "  --location <path>             Specify install location (optional)"
    echo "  --help                        Show this help message"
    exit 0
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --tool) TOOL="$2"; TOOL_PROVIDED_BY_ARG=true; shift ;;
        --project) 
            # Optional argument handling is tricky in bash.
            # If next arg exists and doesn't start with -, take it. Else default to .
            if [[ -n "$2" && "$2" != -* ]]; then
                PROJECT_PATH="$2"
                shift
            else
                PROJECT_PATH="."
            fi
            ;;
        --overwrite) OVERWRITE=true ;;
        --location) INSTALL_DIR="$2"; shift ;;
        --editor) EDITOR_NAME="$2"; shift ;;
        --launch-dir) LAUNCH_DIR="$2"; shift ;;
        --help) usage ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

echo "=== opd-mcp Installer ==="

# Prerequisites check
find_python() {
    for ver in "3.12" "3.11" "3.10"; do
        if command -v "python$ver" &> /dev/null; then
            echo "python$ver"
            return
        fi
    done
    if command -v python3 &> /dev/null; then
        echo "python3"
        return
    fi
}

PYTHON_CMD=$(find_python)

if [ -z "$PYTHON_CMD" ]; then
    die "python3 is not installed."
fi
echo "Using Python: $PYTHON_CMD"

# Interactive Prompts
if [ -z "$INSTALL_DIR" ]; then
    DEFAULT_DIR=$(determine_install_dir)
    if [ "$TOOL" == "" ] && [ "$PROJECT_PATH" == "" ] && [ "$OVERWRITE" == "false" ]; then 
        # Only prompt if fully interactive or location not set
        read -p "Install location [$DEFAULT_DIR]: " INPUT_DIR
        INSTALL_DIR=${INPUT_DIR:-$DEFAULT_DIR}
    else
        INSTALL_DIR=$DEFAULT_DIR
    fi
fi

if [ -z "$TOOL" ]; then
    echo "Which tool config would you like to use?"
    select t in "cline" "roo" "continue" "none"; do
        TOOL=$t
        break
    done
fi

# Helper to map generic editor aliases to directory names and display names
# Returns "DirName" corresponding to "Code", "Cursor", "Windsurf", "Antigravity"
# We try to be case-insensitive for input but precise for output.
resolve_editor_dir() {
    local input=$(echo "$1" | tr '[:upper:]' '[:lower:]')
    case "$input" in
        code|vscode) echo "Code" ;;
        cursor) echo "Cursor" ;;
        windsurf) echo "Windsurf" ;;
        antigravity) echo "Antigravity" ;;
        vscodium|codium) echo "VSCodium" ;;
        *) echo "$1" ;; # Return as-is if unknown, assuming user knows the dir name
    esac
}

# Editor Selection Logic
if [ "$TOOL" != "none" ]; then
    # If using arguments, default to global if project path not set
    if [ "$TOOL_PROVIDED_BY_ARG" = true ]; then
         if [ -z "$PROJECT_PATH" ]; then
             USE_GLOBAL=true
         fi
         # If EDITOR arg provided, use it. If not, we might prompt or default?
         # User said: "default vscode".
         if [ -z "$EDITOR_NAME" ]; then
             EDITOR_NAME="Code"
         fi
    elif [ -z "$PROJECT_PATH" ]; then
        # Interactive mode: ask Scope
        echo "Where should the tool configuration be placed?"
        echo "1) Global (User Settings)"
        echo "2) Project (Current Directory: $LAUNCH_DIR)"
        echo "3) Custom Project Path"
        read -p "Select option [1]: " SCOPE_OPT
        SCOPE_OPT=${SCOPE_OPT:-1}
        
        if [ "$SCOPE_OPT" == "1" ]; then
            USE_GLOBAL=true
        elif [ "$SCOPE_OPT" == "2" ]; then
            PROJECT_PATH="."
        else
            read -p "Enter project path: " PROJECT_PATH
        fi
    fi
    
    if [ "$USE_GLOBAL" = true ]; then
        # For non-Continue tools (Cline, Roo, etc.), we need to know WHICH Editor.
        # Continue usually uses ~/.continue which is shared? Or does it?
        # User said: "for each extension that has this issue". 
        # Continue seems to use ~/.continue so it might be editor-agnostic?
        # But let's verify. Yes, Continue uses ~/.continue usually.
        # So this only applies if tool IS NOT "continue".
        
        if [ "$TOOL" != "continue" ]; then
             if [ -z "$EDITOR_NAME" ] && [ "$TOOL_PROVIDED_BY_ARG" = false ]; then
                # Interactive Editor Selection
                echo "Which editor are you using?"
                options=("VS Code" "Cursor" "Windsurf" "Antigravity" "VSCodium" "Other")
                select opt in "${options[@]}"; do
                    case $opt in
                        "VS Code") EDITOR_NAME="Code"; break ;;
                        "Cursor") EDITOR_NAME="Cursor"; break ;;
                        "Windsurf") EDITOR_NAME="Windsurf"; break ;;
                        "Antigravity") EDITOR_NAME="Antigravity"; break ;;
                        "VSCodium") EDITOR_NAME="VSCodium"; break ;;
                        "Other") 
                            read -p "Enter the configuration directory name (e.g. Code - OSS): " EDITOR_NAME
                            break 
                            ;;
                        *) echo "Invalid option";;
                    esac
                done
             elif [ -z "$EDITOR_NAME" ]; then
                 # Default
                 EDITOR_NAME="Code"
             fi
             
             # Resolve to Title Case just in case
             EDITOR_DIR=$(resolve_editor_dir "$EDITOR_NAME")
        fi
    fi
fi

echo "Installing to: $INSTALL_DIR"

# Upgrade/Install Logic
if [ -d "$INSTALL_DIR" ]; then
    echo "Detected existing installation at $INSTALL_DIR."
    if [ "$OVERWRITE" = false ]; then
        read -p "Overwrite existing installation? This will replace files. (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborting installation."
            exit 0
        fi
    else
        echo "Overwriting existing installation..."
    fi
    # Proceed to copy over
else
    mkdir -p "$INSTALL_DIR"
fi

# Generate Config EARLY (so user can respond to prompts before waiting for install)
if [ "$TOOL" != "none" ]; then
    echo "Generating configuration for $TOOL..."
    
    GEN_ARGS=""
    if [ "$OVERWRITE" = true ]; then
        GEN_ARGS="--force"
    fi
    GEN_CMD="$PYTHON_CMD scripts/generate_configs.py $GEN_ARGS"

    # Determine output location based on TOOL and PROJECT_PATH
    
    if [ -n "$PROJECT_PATH" ]; then
        # Project Scope
        
        # Resolve path using LAUNCH_DIR
        TARGET_CONFIG_DIR="$PROJECT_PATH"
        if [[ "$PROJECT_PATH" != /* ]]; then
            TARGET_CONFIG_DIR="$LAUNCH_DIR/$PROJECT_PATH"
        fi
        
        echo "Generating project config in $TARGET_CONFIG_DIR..."
        if [ "$TOOL" == "continue" ]; then
             TARGET_CONFIG_DIR="$TARGET_CONFIG_DIR/.continue"
        fi
        
        $GEN_CMD --input universal_config.json --output "$TARGET_CONFIG_DIR" --tool "$TOOL" --cwd "$INSTALL_DIR"

    else
        USE_GLOBAL=true
    fi
    
    if [ "$USE_GLOBAL" = true ]; then
         # Global
        echo "Global configuration path detection."
        
        if [ "$TOOL" == "continue" ]; then
             TARGET_CONFIG_DIR="$HOME/.continue"
        else
             # Generic Editor Path Logic
             # Mac: ~/Library/Application Support/$EDITOR_DIR/User/globalStorage/$EXT_ID/settings
             # Linux: ~/.config/$EDITOR_DIR/User/globalStorage/$EXT_ID/settings
             
             if [ "$TOOL" == "cline" ]; then
                 EXT_ID="saoudrizwan.claude-dev"
             elif [ "$TOOL" == "roo" ]; then
                 EXT_ID="rooveterinaryinc.roo-cline"
             fi
             
             if [[ "$OSTYPE" == "darwin"* ]]; then
                 BASE_DIR="$HOME/Library/Application Support/$EDITOR_DIR/User/globalStorage/$EXT_ID/settings"
             else
                 # Assume Linux/WSL
                 BASE_DIR="$HOME/.config/$EDITOR_DIR/User/globalStorage/$EXT_ID/settings"
             fi
             TARGET_CONFIG_DIR="$BASE_DIR"
        fi
        
        # We rely on generate_configs.py to handle prompts now, unless OVERWRITE (force) is set.
        echo "Generating global config in $TARGET_CONFIG_DIR..."
        # Ensure parent dirs exist? generate_configs does makedirs for output.
        $GEN_CMD --input universal_config.json --output "$TARGET_CONFIG_DIR" --tool "$TOOL" --cwd "$INSTALL_DIR"
    fi
    
    echo "Config generation complete."
fi

# Copy files
echo "Copying files..."
# Assume we are running INSIDE the extracted archive or a directory containing the sources.
# Since we are creating a single-file installer, this script will be extracted along with the data.
# Files to copy are in PWD (the extracted dir).

FILES=("mcp_server.py" "ingest.py" "requirements.txt" "universal_config.json" "scripts" "dependencies" "data" "storage" "models" "release_common")

for item in "${FILES[@]}"; do
    if [ -e "./$item" ]; then
        # Check if we should warn about overwriting user data dirs?
        # If overwrite is true, we overwrite files. But data/storage?
        # Usually we preserve data/storage if they exist and are not boilerplate.
        if [[ "$item" == "data" || "$item" == "storage" || "$item" == "models" ]]; then
             mkdir -p "$INSTALL_DIR/$item"
             cp -rn "./$item/"* "$INSTALL_DIR/$item/" 2>/dev/null || true
        elif [ "$item" == "dependencies" ]; then
             rm -rf "$INSTALL_DIR/$item"
             cp -r "./$item" "$INSTALL_DIR/"
        else
             cp -r "./$item" "$INSTALL_DIR/"
        fi
    fi
done

# Setup venv
cd "$INSTALL_DIR"
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
fi

echo "Installing dependencies..."
DEP_DIR="$INSTALL_DIR/dependencies"

if [ ! -d "$DEP_DIR" ] || [ -z "$(ls -A "$DEP_DIR" 2>/dev/null)" ]; then
    die "Packaged dependencies not found. Please use the packaged installer build that includes wheels."
fi

# Install wheel first to handle legacy setup.py builds
./venv/bin/pip install --no-index --find-links "$DEP_DIR" --no-cache-dir wheel

./venv/bin/pip install --no-index --find-links "$DEP_DIR" --no-cache-dir -r requirements.txt

echo "Installation complete!"
echo "MCP Server installed at: $INSTALL_DIR"
if [ "$TOOL" != "none" ]; then
    echo "Config generated for $TOOL"
    echo "Please restart your client/editor to load the changes."
fi
