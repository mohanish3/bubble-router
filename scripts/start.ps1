param(
    [string]$Host = "127.0.0.1",
    [int]$Port = 8090,
    [switch]$NoWarm
)

$root = Split-Path $PSScriptRoot -Parent

if (-not (Test-Path "$root\model-router.json")) {
    Write-Error "model-router.json not found."
    Write-Host "Copy config\model-router.example.json to model-router.json and fill in your model paths."
    exit 1
}

Push-Location $root

if ($NoWarm) { $env:BUBBLE_ROUTER_NO_WARM = "1" }

try {
    python -m uvicorn model_router.main:app --host $Host --port $Port
} finally {
    Pop-Location
    if ($NoWarm) { Remove-Item Env:BUBBLE_ROUTER_NO_WARM -ErrorAction SilentlyContinue }
}
