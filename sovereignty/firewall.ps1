# Egress control for the Sovereign Workbench (AGENTS.md §10.1).
#
# §10 specifies nftables. The demo machine runs Windows 11 with no Linux
# userland, so the equivalent control is Windows Firewall: default-deny
# outbound on all profiles, with loopback and the local subnet allowed.
#
# Loopback is never filtered by Windows Firewall, so Ollama on 127.0.0.1:11434
# and Uvicorn on 127.0.0.1:8080 keep working while this is active.
#
#   .\firewall.ps1 -Enable     # blocks ALL internet access on this machine
#   .\firewall.ps1 -Disable    # restores normal outbound
#   .\firewall.ps1 -Status
#
# Run from an Administrator PowerShell. -Enable will cut this machine off the
# internet until you run -Disable. That is the point; do not run it mid-download.

[CmdletBinding()]
param(
    [switch]$Enable,
    [switch]$Disable,
    [switch]$Status
)

$RuleName = "sovereign-allow-localsubnet"
$LogFile  = "$env:windir\system32\LogFiles\Firewall\pfirewall.log"

function Assert-Admin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    $pr = New-Object Security.Principal.WindowsPrincipal($id)
    if (-not $pr.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        Write-Error "Run this from an Administrator PowerShell."
        exit 1
    }
}

if ($Enable) {
    Assert-Admin
    Write-Host "Blocking all outbound traffic except the local subnet..." -ForegroundColor Yellow

    if (-not (Get-NetFirewallRule -DisplayName $RuleName -ErrorAction SilentlyContinue)) {
        New-NetFirewallRule -DisplayName $RuleName -Direction Outbound -Action Allow `
            -RemoteAddress LocalSubnet -Profile Any | Out-Null
    }

    Set-NetFirewallProfile -All -DefaultOutboundAction Block `
        -LogBlocked True -LogFileName $LogFile -LogMaxSizeKilobytes 4096

    Write-Host "Outbound: BLOCK. Drop log: $LogFile" -ForegroundColor Green
    exit 0
}

if ($Disable) {
    Assert-Admin
    Set-NetFirewallProfile -All -DefaultOutboundAction Allow
    try { Remove-NetFirewallRule -DisplayName $RuleName -ErrorAction Stop } catch {}
    Write-Host "Outbound: ALLOW (restored). Drop logging left enabled." -ForegroundColor Green
    exit 0
}

# Default: -Status
Get-NetFirewallProfile -All |
    Select-Object Name, Enabled, DefaultOutboundAction, LogBlocked, LogFileName |
    Format-Table -AutoSize
if (Test-Path $LogFile) {
    $kb = [math]::Round((Get-Item $LogFile).Length / 1KB, 1)
    Write-Host "drop log present: $LogFile ($kb KB)"
} else {
    Write-Host "drop log MISSING: $LogFile" -ForegroundColor Yellow
}
