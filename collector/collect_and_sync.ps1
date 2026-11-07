param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern("^[A-Za-z0-9_-]+$")]
    [string]$Participant,

    [Parameter(Mandatory = $true)]
    [ValidatePattern("^[A-Za-z0-9_-]+$")]
    [string]$Session,

    [int]$Trials = 10,

    [double]$Duration = 2.0,

    [ValidateSet(50, 100)]
    [int]$SampleRate = 100,

    [string]$PiTarget = "zero@lhnt.local",

    [switch]$Resume,

    [switch]$SyncOnly
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$localParticipantDirectory = Join-Path $repoRoot "data\raw\$Participant"
$localSessionDirectory = Join-Path $localParticipantDirectory $Session
$remoteSessionDirectory = "/home/zero/emg-imu-data/$Participant/$Session"

if (-not $SyncOnly) {
    $remoteCommand = @(
        "python3 /home/zero/collect_full_session.py"
        "--participant $Participant"
        "--session $Session"
        "--trials $Trials"
        "--duration $Duration"
        "--sample-rate $SampleRate"
    ) -join " "

    if ($Resume) {
        $remoteCommand += " --resume"
    }

    Write-Host "`nStarting collection on $PiTarget..."
    & ssh -t $PiTarget $remoteCommand

    if ($LASTEXITCODE -ne 0) {
        throw "Collection did not finish normally. Data already recorded remains on the Pi. Use -SyncOnly to retrieve it."
    }
}

New-Item -ItemType Directory -Force $localParticipantDirectory | Out-Null

Write-Host "`nCopying the session from the Pi..."
& scp -r "${PiTarget}:${remoteSessionDirectory}" $localParticipantDirectory

if ($LASTEXITCODE -ne 0) {
    throw "The file transfer failed. The original data is still stored on the Pi."
}

$localCount = @(
    Get-ChildItem `
        -Path $localSessionDirectory `
        -Filter "*.csv" `
        -File `
        -Recurse
).Count

$remoteCountText = & ssh $PiTarget `
    "find '$remoteSessionDirectory' -type f -name '*.csv' | wc -l"

if ($LASTEXITCODE -ne 0) {
    throw "Could not verify the remote file count."
}

$remoteCountLine = $remoteCountText | Select-Object -Last 1
$remoteCount = [int]($remoteCountLine.Trim())

Write-Host "`nLocal CSV count:  $localCount"
Write-Host "Remote CSV count: $remoteCount"

if ($localCount -ne $remoteCount) {
    throw "File counts do not match. Keep the Pi data and investigate before committing."
}

$expectedCount = 7 * $Trials

if (-not $SyncOnly -and $localCount -ne $expectedCount) {
    Write-Warning "Expected $expectedCount CSV files for seven classes, but found $localCount."
}

Write-Host "`nSync complete."
Write-Host "Laptop: $localSessionDirectory"
Write-Host "The original files remain on the Pi."