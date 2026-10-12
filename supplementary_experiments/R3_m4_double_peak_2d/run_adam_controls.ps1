# Controls for the Sec. 4.3 run (Adam 1e-2, M = 1000, 20k epochs), 3 seeds each:
#   adam_sin_*      plane-wave bank only, with sigma-annealing
#   adam_noanneal_* mixed bank, trained at sigma^2 = 0.16 throughout
$py = 'D:\PythonProjects\WANPM_FP\venv\Scripts\python.exe'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$variants = @(@{ name = 'adam_sin'; extra = @('--bank', 'sin:200:1') },
              @{ name = 'adam_noanneal'; extra = @('--no-anneal') })
$procs = foreach ($v in $variants) {
    foreach ($s in 0, 1, 2) {
        $tag = "$($v.name)_M1000_s$s"
        $log = Join-Path $here "logs\$tag.log"
        $argList = @("`"$(Join-Path $here 'dp2d_lab.py')`"", '--tag', $tag, '--seed', $s, '--opt', 'adam') + $v.extra
        Start-Process -FilePath $py -ArgumentList $argList -WorkingDirectory $here `
            -RedirectStandardOutput $log -RedirectStandardError "$log.err" -NoNewWindow -PassThru
    }
}
$procs | Wait-Process
Get-ChildItem (Join-Path $here 'logs') -Filter 'adam_*_M1000_s*.log' | ForEach-Object { Get-Content $_.FullName -Tail 1 }
