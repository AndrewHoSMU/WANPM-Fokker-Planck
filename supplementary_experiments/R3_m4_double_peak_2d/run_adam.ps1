# Sec. 4.3 re-run with the manuscript's generator optimizer (Adam, lr 1e-2):
# mixed bank + sigma-annealing, 3 seeds at M = 1000 (CPU) and 3 at M = 8000 (GPU).
$py = 'D:\PythonProjects\WANPM_FP\venv\Scripts\python.exe'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
New-Item -ItemType Directory -Force (Join-Path $here 'logs') | Out-Null
$procs = foreach ($m in @(@{ M = '1000'; dev = 'cpu' }, @{ M = '8000'; dev = 'cuda' })) {
    foreach ($s in 0, 1, 2) {
        $tag = "adam_mixed_M$($m.M)_s$s"
        $log = Join-Path $here "logs\$tag.log"
        $argList = @("`"$(Join-Path $here 'dp2d_lab.py')`"", '--tag', $tag, '--seed', $s,
                     '--opt', 'adam', '--M', $m.M, '--device', $m.dev)
        Start-Process -FilePath $py -ArgumentList $argList -WorkingDirectory $here `
            -RedirectStandardOutput $log -RedirectStandardError "$log.err" -NoNewWindow -PassThru
    }
}
$procs | Wait-Process
Get-ChildItem (Join-Path $here 'logs') -Filter 'adam_mixed*.log' | ForEach-Object { Get-Content $_.FullName -Tail 1 }
