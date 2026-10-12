# Sec. 4.3 re-run: mixed bank x 3 seeds + plane-wave-only control, in parallel (one core each).
$py = 'D:\PythonProjects\WANPM_FP\venv\Scripts\python.exe'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
New-Item -ItemType Directory -Force (Join-Path $here 'logs') | Out-Null
$runs = @(
    @{ tag = 'mixed_s0'; a = @('--seed', '0') },
    @{ tag = 'mixed_s1'; a = @('--seed', '1') },
    @{ tag = 'mixed_s2'; a = @('--seed', '2') },
    @{ tag = 'sin_s0';   a = @('--seed', '0', '--bank', 'sin:200:1') }
)
$procs = foreach ($r in $runs) {
    $log = Join-Path $here "logs\$($r.tag).log"
    $argList = @("`"$(Join-Path $here 'dp2d_lab.py')`"", '--tag', $r.tag) + $r.a
    Start-Process -FilePath $py -ArgumentList $argList -WorkingDirectory $here `
        -RedirectStandardOutput $log -RedirectStandardError "$log.err" -NoNewWindow -PassThru
}
$procs | Wait-Process
Get-ChildItem (Join-Path $here 'logs') -Filter *.log | ForEach-Object { "--- $($_.Name)"; Get-Content $_.FullName -Tail 3 }
