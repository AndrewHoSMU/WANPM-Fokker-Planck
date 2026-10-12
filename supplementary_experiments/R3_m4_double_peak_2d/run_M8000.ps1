# Sec. 4.3 re-run: mixed bank, batch size M = 8000, 3 seeds on the GPU.
$py = 'D:\PythonProjects\WANPM_FP\venv\Scripts\python.exe'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
New-Item -ItemType Directory -Force (Join-Path $here 'logs') | Out-Null
$procs = foreach ($s in 0, 1, 2) {
    $tag = "mixed_M8000_s$s"
    $log = Join-Path $here "logs\$tag.log"
    $argList = @("`"$(Join-Path $here 'dp2d_lab.py')`"", '--tag', $tag, '--seed', $s, '--M', '8000', '--device', 'cuda')
    Start-Process -FilePath $py -ArgumentList $argList -WorkingDirectory $here `
        -RedirectStandardOutput $log -RedirectStandardError "$log.err" -NoNewWindow -PassThru
}
$procs | Wait-Process
Get-ChildItem (Join-Path $here 'logs') -Filter 'mixed_M8000*.log' | ForEach-Object { "--- $($_.Name)"; Get-Content $_.FullName -Tail 2 }
