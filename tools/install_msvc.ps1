# VS Build Tools silent install (correct --add syntax, one component per --add)
# Run elevated (UAC will prompt once).
$exe = "D:\Tools\downloads\vs_BuildTools.exe"
$args = @(
    "--quiet", "--norestart", "--nocache", "--wait",
    "--add", "Microsoft.VisualStudio.Workload.VCTools",
    "--add", "Microsoft.VisualStudio.Component.VC.Tools.ARM64",
    "--add", "Microsoft.VisualStudio.Component.Windows11SDK.26100"
)
$p = Start-Process $exe -ArgumentList $args -Verb RunAs -PassThru -Wait
Write-Output ("exit=" + $p.ExitCode)
