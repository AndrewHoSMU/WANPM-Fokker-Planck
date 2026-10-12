# Sec. 4.3 re-run: mixed bank, Muon split, M = 1000, the manuscript's 100,000 epochs; 3 seeds, one core each.
$py = 'D:\PythonProjects\WANPM_FP\venv\Scripts\python.exe'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
New-Item -ItemType Directory -Force (Join-Path $here 'logs') | Out-Null
$procs = foreach ($s in 0, 1, 2) {
    $tag = "mixed_100k_s$s"
    $log = Join-Path $here "logs\$tag.log"
    $argList = @("`"$(Join-Path $here 'dp2d_lab.py')`"", '--tag', $tag, '--seed', $s, '--epochs', '100000')
    Start-Process -FilePath $py -ArgumentList $argList -WorkingDirectory $here `
        -RedirectStandardOutput $log -RedirectStandardError "$log.err" -NoNewWindow -PassThru
}
$procs | Wait-Process
Get-ChildItem (Join-Path $here 'logs') -Filter 'mixed_100k*.log' | ForEach-Object { "--- $($_.Name)"; Get-Content $_.FullName -Tail 2 }
