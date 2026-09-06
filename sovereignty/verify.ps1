# Pre-demo assertion script (AGENTS.md §10.5, §13). Exits 0 only if contained.
# Run this on stage: .\sovereignty\verify.ps1

$ErrorActionPreference = "Continue"
$fails = 0
$root  = Split-Path -Parent $PSScriptRoot

function Check($name, $ok, $detail) {
    if ($ok) {
        Write-Host ("  PASS  {0}" -f $name) -ForegroundColor Green
    } else {
        Write-Host ("  FAIL  {0} -- {1}" -f $name, $detail) -ForegroundColor Red
        $script:fails++
    }
}
function Warn($name, $detail) {
    Write-Host ("  WARN  {0} -- {1}" -f $name, $detail) -ForegroundColor Yellow
}

Write-Host "`nSOVEREIGNTY VERIFICATION`n" -ForegroundColor Cyan

# 1. Outbound default-deny on every profile
$profiles = Get-NetFirewallProfile -All
$blocked  = @($profiles | Where-Object { $_.DefaultOutboundAction -ne "Block" })
Check "firewall outbound is default-DENY" ($blocked.Count -eq 0) `
      ("allowing: " + (($blocked | ForEach-Object Name) -join ", "))

# 2. Drop logging on, log readable by the monitor
$nolog = @($profiles | Where-Object { $_.LogBlocked -ne "True" })
Check "drop logging enabled" ($nolog.Count -eq 0) `
      ("not logging: " + (($nolog | ForEach-Object Name) -join ", "))

$logFile = "$env:windir\system32\LogFiles\Firewall\pfirewall.log"
Check "drop log present" (Test-Path $logFile) "$logFile missing"

# 3. Zero established sockets to anything off-box
$ext = @(Get-NetTCPConnection -State Established -ErrorAction SilentlyContinue |
    Where-Object {
        $ip = $_.RemoteAddress
        -not ($ip -match '^(127\.|::1$|10\.|192\.168\.|169\.254\.|0\.0\.0\.0$)' -or
              $ip -match '^172\.(1[6-9]|2[0-9]|3[01])\.')
    })
Check "zero established external sockets" ($ext.Count -eq 0) `
      ("open: " + (($ext | ForEach-Object { "$($_.RemoteAddress):$($_.RemotePort)" }) -join ", "))

# 4. Audit chain intact.
# The venv interpreter explicitly, never a bare `python`: this script is meant to
# be run from an ELEVATED shell (it reads the firewall drop log), and a fresh
# elevated PowerShell does not have .venv on PATH. Bare `python` there resolves
# to system/Anaconda Python, which lacks this project's deps -- so the check
# fails on an import error while the chain is perfectly intact, and the demo
# gets halted for nothing.
Push-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
$auditOut = & $py -m core.audit verify "workspace\.audit\audit.jsonl"
$auditOk  = $LASTEXITCODE -eq 0
Pop-Location
Check "audit chain intact" $auditOk $auditOut

# 5. Ollama, if running, must be loopback-only. Not fatal until leg 2 exists.
$ollama = @(Get-NetTCPConnection -State Listen -LocalPort 11434 -ErrorAction SilentlyContinue)
if ($ollama.Count -eq 0) {
    Warn "ollama loopback bind" "nothing listening on 11434 (expected until demo leg 2)"
} else {
    $bad = @($ollama | Where-Object { $_.LocalAddress -notin @("127.0.0.1", "::1") })
    Check "ollama bound to loopback only" ($bad.Count -eq 0) `
          ("bound to: " + (($bad | ForEach-Object LocalAddress) -join ", "))
}

# 6. VRAM headroom (AGENTS.md §4.1, §17).
# A warning, not a failure: this script's job is containment, and a squeezed
# GPU is a performance fault, not a sovereignty one. It belongs here anyway
# because Windows gives no other signal -- overcommitting VRAM does not raise
# OOM on WDDM, it spills to shared system memory and inference simply starts
# taking minutes. Measured: a run at 5896/6141 MiB ran 23x slower with no
# error. If this warns on stage, close Chrome/Discord/Electron (§4.2.7) before
# blaming the model.
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $smi) {
    Warn "VRAM headroom" "nvidia-smi not on PATH; cannot check"
} else {
    $line = (& nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits) |
            Select-Object -First 1
    $used, $total = $line -split ',' | ForEach-Object { [int]$_.Trim() }
    $free = $total - $used
    # §4.1 reserves 0.8 GB for display+CUDA context; the rest is the model budget.
    if ($free -lt 820) {
        Warn "VRAM headroom" ("only ${free} MiB free of ${total} MiB -- a resident model will spill to shared memory and crawl")
    } else {
        Check "VRAM headroom for one model" $true ""
        Write-Host ("        ${free} MiB free of ${total} MiB") -ForegroundColor DarkGray
    }
}

Write-Host ""
if ($fails -eq 0) {
    Write-Host "CONTAINED -- $($fails) failures`n" -ForegroundColor Green
    exit 0
}
Write-Host "NOT CONTAINED -- $fails failure(s)`n" -ForegroundColor Red
exit 1
