# Replaces the results in this repo with a new Kaggle output zip and refreshes the README.
# Usage (from the repo folder):
#   powershell -ExecutionPolicy Bypass -File tools\update_results.ps1
#   powershell -ExecutionPolicy Bypass -File tools\update_results.ps1 -Zip "C:\path\to\file.zip"
param([string]$Zip = "")
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not $Zip) {
    $found = Get-ChildItem "$HOME\Downloads" -Filter *.zip | Where-Object { $_.Name -like "*artifact*" } |
             Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $found) { throw "No *artifacts*.zip found in Downloads. Run again with -Zip `"<path to the Kaggle zip>`"" }
    $Zip = $found.FullName
}
Write-Host "Using $Zip"

$tmp = Join-Path $env:TEMP "loomguard_unzip"
Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
Expand-Archive -Path $Zip -DestinationPath $tmp
$res = Get-ChildItem $tmp -Recurse -Filter results.json | Select-Object -First 1
if (-not $res) { throw "This zip has no results.json, so it is not the LoomGuard Kaggle output." }
$src = $res.DirectoryName

Get-ChildItem models -Directory -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force
Remove-Item figures, samples, results.json, RESULTS.md -Recurse -Force -ErrorAction SilentlyContinue
Copy-Item "$src\*" . -Recurse -Force
Remove-Item $tmp -Recurse -Force

python tools/fill_readme.py
