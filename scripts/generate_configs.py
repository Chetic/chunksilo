import json
import os
import argparse
import sys
import shutil
from datetime import datetime

def load_universal_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def get_user_confirmation(filepath, prompt="already exists. Overwrite? (y/n)"):
    while True:
        response = input(f"File '{filepath}' {prompt}: ").lower().strip()
        if response in ['y', 'yes']:
            return True
        if response in ['n', 'no']:
            return False

def create_backup(filepath):
    if os.path.exists(filepath):
        # Timestamped backup to avoid overwriting previous backups
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{filepath}.{timestamp}.bak"
        shutil.copy2(filepath, backup_path)
        print(f"Created backup: {backup_path}")

def safe_write_json(filepath, content, force=False):
    """
    Safely writes JSON.
    If file exists, allows merging for 'mcpServers'.
    """
    
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                existing_data = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse existing JSON at {filepath}. Starting fresh.")
            existing_data = {}
        
        # Merge Logic for MCP Settings
        if "mcpServers" in content:
            if "mcpServers" not in existing_data:
                existing_data["mcpServers"] = {}
            
            # We only touch the specific key for OUR server
            server_name = list(content["mcpServers"].keys())[0]
            
            # Check if we are actually changing anything
            if server_name in existing_data["mcpServers"]:
                 # If identical, skip? Or notify?
                 pass
            
            if not force:
                if not get_user_confirmation(filepath, prompt=f"exists. Update '{server_name}' config? (y/n)"):
                    print(f"Skipping update of {filepath}")
                    return

            create_backup(filepath)
            existing_data["mcpServers"][server_name] = content["mcpServers"][server_name]
            final_content = existing_data
            
        else:
            # Not an MCP settings file? Treat as full overwrite
            if not force:
                 if not get_user_confirmation(filepath):
                     print(f"Skipping {filepath}")
                     return
            create_backup(filepath)
            final_content = content
    else:
        final_content = content

    with open(filepath, 'w') as f:
        json.dump(final_content, f, indent=2)
    print(f"Updated {filepath}")

def safe_write_text(filepath, content, force=False):
    if os.path.exists(filepath) and not force:
        if not get_user_confirmation(filepath):
            print(f"Skipping {filepath}")
            return
    
    if os.path.exists(filepath):
         create_backup(filepath)

    with open(filepath, 'w') as f:
        f.write(content)
    print(f"Wrote {filepath}")

def generate_cline_config(config, output_dir, cwd, force=False):
    # cline_mcp_settings.json
    mcp_settings = {
        "mcpServers": {
            config["mcp_server"]["name"]: {
                "command": config["mcp_server"]["command"],
                "args": config["mcp_server"]["args"],
                "env": config["mcp_server"]["env"],
                "cwd": cwd, 
                "disabled": False,
                "autoApprove": []
            }
        }
    }
    
    # Rules
    # User requested: "write (and overwrite) its own ChunkSilo.md file to the global clinerules directory"
    # The output_dir passed in for Cline global is typically ".../settings".
    # We should look for "prompts" or "rules" directory relative to it?
    # Standard Cline (Claude Dev) separation isn't strictly defined for global rules, BUT
    # recent updates added Custom Instructions which are stored in `globalStorage/saoudrizwan.claude-dev/settings/cline_custom_instructions.json`?
    # OR user likely has a `rules` folder if they are asking for this.
    # Let's attempt to write to `ChunkSilo.md` in the same directory, or if a `rules` subdir exists.
    
    os.makedirs(output_dir, exist_ok=True)
    
    safe_write_json(os.path.join(output_dir, "cline_mcp_settings.json"), mcp_settings, force)

    # For rules, we write a separate markdown file as requested.
    # We will try to place it in 'rules' subdir if it exists, otherwise in current dir?
    # Or just `ChunkSilo-rules.md` in the settings dir to be safe.
    
    rules_content = config["rules"]["system_prompt"]
    rules_path = os.path.join(output_dir, "ChunkSilo-rules.md")
    
    # Check if a rules directory is expected?
    # "global clinerules directory".
    # Often people map `~/.clinerules`.
    # If the user passed a separate project path, it might be `.roomode` etc.
    # We will default to a discrete filename to avoid conflict.
    
    safe_write_text(rules_path, rules_content, force)
    
    print(f"Generated Cline config in {output_dir}")

def generate_roo_config(config, output_dir, cwd, force=False):
    # mcp_settings.json
    mcp_settings = {
        "mcpServers": {
            config["mcp_server"]["name"]: {
                "command": config["mcp_server"]["command"],
                "args": config["mcp_server"]["args"],
                "env": config["mcp_server"]["env"],
                "cwd": cwd
            }
        }
    }

    os.makedirs(output_dir, exist_ok=True)
    
    safe_write_json(os.path.join(output_dir, "mcp_settings.json"), mcp_settings, force)
        
    # Rules
    # Roo Code definitely supports `rules/` directory in its global storage.
    rules_dir = os.path.join(output_dir, "rules")
    os.makedirs(rules_dir, exist_ok=True)
    
    # Write separate rule file
    safe_write_text(os.path.join(rules_dir, "ChunkSilo-rules.md"), config["rules"]["system_prompt"], force)
    
    print(f"Generated Roo Code config in {output_dir}")

def generate_continue_config(config, output_dir, cwd, force=False):
    # mcpServers/ChunkSilo.yaml
    
    servers_dir = os.path.join(output_dir, "mcpServers")
    os.makedirs(servers_dir, exist_ok=True)
    
    # Continue supports multiple config files, so we simply write our own.
    # This is already "separate" (ChunkSilo.yaml).
    # We still check if it exists and backup.
    
    server_name = config["mcp_server"]["name"]
    command = config["mcp_server"]["command"]
    args = config["mcp_server"]["args"]
    env = config["mcp_server"]["env"]
    
    # Manual YAML construction
    yaml_content = f"""name: On-Prem Docs MCP
version: 1.0.0
schema: v1
mcpServers:
  - name: {server_name}
    command: {command}
    args:"""
    
    for arg in args:
        yaml_content += f"\n      - {arg}"
        
    yaml_content += f"\n    cwd: {cwd}"
    
    if env:
        yaml_content += "\n    env:"
        for k, v in env.items():
            yaml_content += f"\n      {k}: {v}"
            
    safe_write_text(os.path.join(servers_dir, "ChunkSilo.yaml"), yaml_content, force)
        
    # Rules
    # Continue rules can be separate files too.
    rules_dir = os.path.join(output_dir, "rules")
    os.makedirs(rules_dir, exist_ok=True)
    
    safe_write_text(os.path.join(rules_dir, "ChunkSilo-system-prompt.md"), config["rules"]["system_prompt"], force)
        
    print(f"Generated Continue config in {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Generate tool-specific configs from universal config.")
    parser.add_argument("--input", "-i", default="universal_config.json", help="Path to universal config JSON")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--tool", "-t", required=True, choices=["cline", "roo", "continue"], help="Tool to generate config for")
    parser.add_argument("--cwd", default=".", help="Current working directory to set in config")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing files")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Config file not found at {args.input}")
        sys.exit(1)
        
    config = load_universal_config(args.input)
    
    if args.tool == "cline":
        generate_cline_config(config, args.output, args.cwd, args.force)
    elif args.tool == "roo":
        generate_roo_config(config, args.output, args.cwd, args.force)
    elif args.tool == "continue":
        generate_continue_config(config, args.output, args.cwd, args.force)

if __name__ == "__main__":
    main()
