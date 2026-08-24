param(
    [ValidateSet("86", "89")]
    [string]$Architecture = "86"
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$buildRoot = Join-Path $projectRoot ("build-sm" + $Architecture)

cmake -S $projectRoot -B $buildRoot "-DCMAKE_CUDA_ARCHITECTURES=$Architecture"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

cmake --build $buildRoot --config Release
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "KudaCuda built for sm_$Architecture in $buildRoot"
