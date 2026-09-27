# Starts the web API on :8000 and the Vite dev server on :5173 (hot reload, proxies /api to :8000).
# Usage: powershell scripts/dev.ps1        Ctrl+C stops both.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$web = Join-Path $root 'web'

if (-not (Test-Path (Join-Path $web 'node_modules'))) {
    Write-Host 'Installing web dependencies...'
    Push-Location $web
    try { npm install } finally { Pop-Location }
}

Write-Host 'API  -> http://127.0.0.1:8000  (s2c.web.server:app, --reload)'
$api = Start-Process -FilePath 'uv' -ArgumentList 'run', 'uvicorn', 's2c.web.server:app', '--port', '8000', '--reload' `
    -WorkingDirectory $root -NoNewWindow -PassThru

try {
    Write-Host 'Web  -> http://localhost:5173'
    Push-Location $web
    npm run dev
}
finally {
    Pop-Location
    if ($api -and -not $api.HasExited) {
        # uvicorn --reload spawns a child process; stop the whole tree.
        & taskkill /PID $api.Id /T /F | Out-Null
    }
    Write-Host 'Stopped the API and the web dev server.'
}
