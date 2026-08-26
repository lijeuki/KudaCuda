param(
    [ValidateSet("86", "89")]
    [string]$Architecture = "86",
    [switch]$SkipVsEnvironment
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$buildRoot = Join-Path $projectRoot ("build-sm" + $Architecture + "-ninja")

# CUDA 13.3 is installed on Windows, but this setup does not register a CUDA
# toolset that CMake's Visual Studio generator can discover. Enter the VS
# developer environment once, then use the Ninja generator explicitly.
if (-not $SkipVsEnvironment -and $env:VSCMD_VER -eq $null) {
    $vsDevCmd = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
    if (Test-Path -LiteralPath $vsDevCmd) {
        $command = "`"$vsDevCmd`" -arch=x64 -host_arch=x64 && powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Architecture $Architecture -SkipVsEnvironment"
        & cmd.exe /d /s /c $command
        exit $LASTEXITCODE
    }
}

$cudaRoot = $env:CUDA_PATH
$nvcc = if ($cudaRoot) { Join-Path $cudaRoot "bin\nvcc.exe" } else { (Get-Command nvcc.exe -ErrorAction SilentlyContinue).Source }
$ninjaCandidates = @(
    (Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"),
    (Get-Command ninja.exe -ErrorAction SilentlyContinue).Source
)
$ninja = $ninjaCandidates | Where-Object { $_ -and (Test-Path -LiteralPath $_) } | Select-Object -First 1

if (-not $nvcc -or -not (Test-Path -LiteralPath $nvcc)) {
    throw "nvcc.exe was not found. Install the CUDA Toolkit or set CUDA_PATH."
}
if (-not $ninja) {
    throw "ninja.exe was not found. Install the Visual Studio CMake tools."
}

cmake -S $projectRoot -B $buildRoot -G Ninja `
    "-DCMAKE_BUILD_TYPE=Release" `
    "-DCMAKE_MAKE_PROGRAM=$ninja" `
    "-DCMAKE_CUDA_COMPILER=$nvcc" `
    "-DCMAKE_CUDA_ARCHITECTURES=$Architecture"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

cmake --build $buildRoot --config Release
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "KudaCuda built for sm_$Architecture in $buildRoot"
