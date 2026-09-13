$py = "C:\opt\nse_gapbot\venv\Scripts\python.exe"
$dir = "C:\home\ubuntu\pattern_screens"
& $py "$dir\build_screens.py"
& $py "$dir\build_page.py"

# Push the freshly built artifact + chart data to the public GitHub relay repo.
# A scheduled cloud routine pulls this repo and republishes the live Artifact —
# see project_the_ledge_screener memory for the full pipeline.
Set-Location $dir
git add output.html charts_*.json screens_data.json *.py template.html
git commit -m "Auto-build $(Get-Date -Format 'yyyy-MM-dd HH:mm')" --quiet
if ($LASTEXITCODE -eq 0) {
    git push origin main --quiet
} else {
    Write-Output "Nothing to commit (no data changes) - skipping push"
}
