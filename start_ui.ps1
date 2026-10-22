# Start MiniMax Studio in WSL2. Open the printed login link in your browser; Ctrl+C stops the app.
#   .\start_ui.ps1            # port 8080
#   .\start_ui.ps1 -Port 8090
param(
    [int]$Port = 8080,
    [string]$Distro = "Ubuntu"
)

$ErrorActionPreference = "Stop"
$wslRoot = (wsl -d $Distro -- wslpath -a ($PSScriptRoot -replace '\\', '/')).Trim()
wsl -d $Distro -- bash -lc "MINIMAX_UI_PORT=$Port bash '$wslRoot/ui/run_ui.sh'"
