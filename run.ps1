# Dev launcher: starts FastAPI (:8000) and Vite (:5173) together.
# Ctrl+C stops both. Windows equivalent of run.sh.

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$py = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) {
    Write-Error "No venv at $root\.venv - create one and run: pip install -r requirements.txt"
    exit 1
}

$web = Join-Path $root 'web'
if (-not (Test-Path (Join-Path $web 'node_modules'))) {
    Write-Host "Installing frontend deps..."
    Push-Location $web
    try { & npm install } finally { Pop-Location }
}

# Start-Process with -NoNewWindow pipes stdout/stderr into this console so
# both servers' logs interleave here, matching run.sh's behavior.
# Only app sources are worth a restart. Left alone, WatchFiles also
# walks web/ and .venv/ at startup (tens of thousands of files) and
# reloads the API whenever Vite rewrites its dep cache under
# web/node_modules. The patterns are attached with `=` because Click
# glob-expands every bare argument on Windows, which would turn
# `web/*` into the whole directory listing before uvicorn sees it.
Write-Host "Starting FastAPI on http://127.0.0.1:8000"
$api = Start-Process -FilePath $py `
    -ArgumentList '-m', 'uvicorn', 'server:app', '--host', '127.0.0.1', '--port', '8000', '--reload', `
        '--reload-exclude=web/*', '--reload-exclude=.venv/*', '--reload-exclude=.pytest_cache/*' `
    -WorkingDirectory $root -PassThru -NoNewWindow

# Do not let the browser reach the SPA before the API answers. The
# first /api/auth/status decides logged-in vs. the login screen, and a
# request that lands while uvicorn is still importing comes back as a
# connection error — which the UI used to read as "signed out" and
# answer with the login screen. 120 x 250 ms = 30 s ceiling; a backend
# that dies in that window is a real failure the console shows anyway.
Write-Host "Waiting for FastAPI..."
$apiReady = $false
for ($i = 0; $i -lt 120 -and -not $api.HasExited; $i++) {
    try {
        $probe = Invoke-WebRequest -Uri 'http://127.0.0.1:8000/api/settings' `
            -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
        if ($probe.StatusCode -eq 200) { $apiReady = $true; break }
    } catch {
        Start-Sleep -Milliseconds 250
    }
}
if (-not $apiReady) {
    Write-Host "FastAPI did not answer within 30s; starting Vite anyway"
}

Write-Host "Starting Vite on http://127.0.0.1:5173"
# npm on Windows is npm.cmd; Start-Process can't invoke .cmd scripts
# directly, so route through cmd /c.
$webProc = Start-Process -FilePath 'cmd.exe' `
    -ArgumentList '/c', 'npm', 'run', 'dev', '--', '--host', '127.0.0.1', '--port', '5173' `
    -WorkingDirectory $web -PassThru -NoNewWindow

try {
    # Poll instead of Wait-Process: if a server dies on startup (a port
    # clash or an import error) its PID is already gone and
    # Wait-Process -Id throws, skipping the cleanup that stops the other
    # half of the pair.
    while (-not $api.HasExited -and -not $webProc.HasExited) {
        Start-Sleep -Milliseconds 500
    }
}
finally {
    foreach ($p in @($api, $webProc)) {
        if ($p -and -not $p.HasExited) {
            Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
        }
    }
}
