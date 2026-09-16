$py = "C:\opt\nse_gapbot\venv\Scripts\python.exe"
$dir = "C:\home\ubuntu\pattern_screens"

# Re-attempt the bhav/mcap downloads here too (both are idempotent -- they skip
# dates already on disk) in case the shared 18:30/18:35 fleet-wide download
# tasks ran before NSE had posted that day's bhavcopy. This gives a second,
# later chance to catch up before building, without touching the shared
# fleet-wide schedule those tasks run on (see project_the_ledge_screener memory,
# 2026-09-16 update, for why this was added).
& $py "C:\opt\nse_gapbot\download_bhav.py"
& $py "C:\opt\nse_gapbot\download_mcap.py"

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
