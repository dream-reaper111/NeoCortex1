$nssm = "nssm"
$svcName = "PineMimicTrader"
$python  = "V:\venvs\tradingai\Scripts\python.exe"      # change to your venv python
$workdir = "C:\Users\death\source\repos\tensorflowtest\tensorflowtest"
$script  = "server.py"

& $nssm install $svcName $python $script
& $nssm set $svcName AppDirectory $workdir
& $nssm set $svcName AppEnvironmentExtra API_HOST=0.0.0.0
& $nssm set $svcName AppEnvironmentExtra API_PORT=8000
& $nssm set $svcName AppEnvironmentExtra TV_WEBHOOK_SECRET=super-secret-here
& $nssm set $svcName AppEnvironmentExtra PAPER=true
& $nssm set $svcName Start SERVICE_AUTO_START
& $nssm start $svcName
