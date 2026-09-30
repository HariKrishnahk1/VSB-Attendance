<#
.SYNOPSIS
    Automatically fetches and pulls updates from https://github.com/HariKrishnahk1/VSB-Attendance.

.DESCRIPTION
    Checks if there are incoming commits on origin/main. If working tree is clean, pulls fast-forward changes.
    Can be run once or continuously with -Watch.

.PARAMETER Watch
    If specified, keeps running and checks for updates every IntervalSeconds.

.PARAMETER IntervalSeconds
    Interval between update checks when running in Watch mode (default: 300 seconds / 5 minutes).
#>

param (
    [switch]$Watch,
    [int]$IntervalSeconds = 300
)

function Sync-Repository {
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Checking for updates from origin/main..." -ForegroundColor Cyan

    # Fetch origin
    git fetch origin main 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Failed to fetch from origin. Check your internet connection or git permissions."
        return
    }

    $localCommit = (git rev-parse HEAD).Trim()
    $remoteCommit = (git rev-parse origin/main).Trim()

    if ($localCommit -eq $remoteCommit) {
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Already up to date." -ForegroundColor Green
        return
    }

    # Check if we have uncommitted local changes
    $status = git status --porcelain
    if ($status) {
        Write-Warning "[$(Get-Date -Format 'HH:mm:ss')] Local changes detected in working tree. Skipping pull to avoid merge conflicts."
        return
    }

    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] New commits detected. Pulling latest changes..." -ForegroundColor Yellow
    $pullOutput = git pull --ff-only origin main
    Write-Host $pullOutput -ForegroundColor Green
}

if ($Watch) {
    Write-Host "Starting continuous repository auto-sync (checking every $IntervalSeconds seconds)..." -ForegroundColor Cyan
    Write-Host "Press Ctrl+C to stop." -ForegroundColor Gray
    while ($true) {
        Sync-Repository
        Start-Sleep -Seconds $IntervalSeconds
    }
} else {
    Sync-Repository
}
