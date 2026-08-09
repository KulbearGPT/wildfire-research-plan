[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$RunDirectory,
    [string]$EnvironmentPrefix = 'D:\WildFire Project\.conda-envs\wsts-res18-t1'
)

$ErrorActionPreference = 'Stop'
$script:TranscriptStarted = $false

function Invoke-CheckedNative {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    Write-Host ("COMMAND: {0} {1}" -f $Executable, ($Arguments -join ' '))
    $PreviousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $Executable @Arguments
        $NativeExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
    if ($NativeExitCode -ne 0) {
        throw "Command failed with exit code ${NativeExitCode}: $Executable $($Arguments -join ' ')"
    }
}

function Export-CheckedNative {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$OutputPath
    )
    Write-Host ("COMMAND: {0} {1} > {2}" -f $Executable, ($Arguments -join ' '), $OutputPath)
    $PreviousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $Executable @Arguments 2>&1 | Tee-Object -FilePath $OutputPath
        $NativeExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
    if ($NativeExitCode -ne 0) {
        throw "Command failed with exit code ${NativeExitCode}: $Executable $($Arguments -join ' ')"
    }
}

try {
    $ReproductionRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
    $RepositoryRoot = (Resolve-Path (Join-Path $ReproductionRoot '..\..')).Path
    $LockPath = Join-Path $ReproductionRoot 'upstream.lock.json'
    $Lock = Get-Content -Raw -LiteralPath $LockPath | ConvertFrom-Json
    $PinnedUrl = [string]$Lock.code.url
    $PinnedCommit = [string]$Lock.code.commit
    $UpstreamRoot = Join-Path $ReproductionRoot '.local\WildfireSpreadTS'
    $RunDirectory = [System.IO.Path]::GetFullPath($RunDirectory)
    $EnvironmentPrefix = [System.IO.Path]::GetFullPath($EnvironmentPrefix)

    New-Item -ItemType Directory -Force -Path $RunDirectory | Out-Null
    Start-Transcript -Path (Join-Path $RunDirectory 'bootstrap.log') -Append | Out-Null
    $script:TranscriptStarted = $true

    Write-Host "Pinned upstream: $PinnedUrl@$PinnedCommit"
    Write-Host "Pinned environment prefix: $EnvironmentPrefix"
    Write-Host "Current process Python is never invoked; all Python commands use conda run --prefix."

    if (Test-Path -LiteralPath $UpstreamRoot) {
        if (-not (Test-Path -LiteralPath (Join-Path $UpstreamRoot '.git'))) {
            throw "Upstream path exists but is not a Git checkout: $UpstreamRoot"
        }
        $Dirty = & git -C $UpstreamRoot status --porcelain
        if ($LASTEXITCODE -ne 0) {
            throw "Unable to inspect existing upstream checkout: $UpstreamRoot"
        }
        if ($Dirty) {
            throw "Existing upstream checkout is dirty; refusing to overwrite it: $UpstreamRoot"
        }
    }
    else {
        New-Item -ItemType Directory -Force -Path (Split-Path $UpstreamRoot -Parent) | Out-Null
        Invoke-CheckedNative git @('clone', '--no-checkout', $PinnedUrl, $UpstreamRoot)
    }

    Invoke-CheckedNative git @('-C', $UpstreamRoot, 'fetch', '--force', 'origin', $PinnedCommit)
    Invoke-CheckedNative git @('-C', $UpstreamRoot, 'checkout', '--detach', $PinnedCommit)
    $ActualCommit = (& git -C $UpstreamRoot rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $ActualCommit -ne $PinnedCommit) {
        throw "Pinned checkout mismatch: expected $PinnedCommit, got $ActualCommit"
    }
    $DirtyAfterCheckout = & git -C $UpstreamRoot status --porcelain
    if ($LASTEXITCODE -ne 0 -or $DirtyAfterCheckout) {
        throw "Pinned upstream checkout is not clean after checkout: $UpstreamRoot"
    }

    $EnvironmentPython = Join-Path $EnvironmentPrefix 'python.exe'
    if ((Test-Path -LiteralPath $EnvironmentPrefix) -and -not (Test-Path -LiteralPath $EnvironmentPython)) {
        throw "Environment prefix exists but has no python.exe; refusing to replace it: $EnvironmentPrefix"
    }
    if (-not (Test-Path -LiteralPath $EnvironmentPython)) {
        New-Item -ItemType Directory -Force -Path (Split-Path $EnvironmentPrefix -Parent) | Out-Null
        Invoke-CheckedNative conda @('create', '--prefix', $EnvironmentPrefix, 'python=3.10.4', 'pip', '-y')
    }

    # Lightning Fabric 2.0.1 imports pkg_resources at runtime. Setuptools >=82
    # removes that module, so pin the last pre-removal series as an environment
    # compatibility constraint without changing any scientific dependency.
    Invoke-CheckedNative conda @(
        'run', '--no-capture-output', '--prefix', $EnvironmentPrefix,
        'python', '-m', 'pip', 'install', 'setuptools==80.9.0'
    )
    Export-CheckedNative conda @(
        'run', '--prefix', $EnvironmentPrefix, 'python', '-c',
        'import setuptools,pkg_resources; print(setuptools.__version__); print(pkg_resources.__file__)'
    ) (Join-Path $RunDirectory 'setuptools-pkg-resources-preflight.txt')

    $RequirementsPath = Join-Path $UpstreamRoot 'requirements.txt'
    $RequirementsSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $RequirementsPath).Hash.ToLowerInvariant()
    $InstallMarkerPath = Join-Path $EnvironmentPrefix '.wsts-res18-t1-bootstrap.json'
    $InstallRequired = -not (Test-Path -LiteralPath $InstallMarkerPath)
    if (-not $InstallRequired) {
        $Marker = Get-Content -Raw -LiteralPath $InstallMarkerPath | ConvertFrom-Json
        if ($Marker.upstream_commit -ne $PinnedCommit -or $Marker.requirements_sha256 -ne $RequirementsSha256) {
            throw "Existing environment marker does not match the frozen upstream requirements: $InstallMarkerPath"
        }
    }

    if ($InstallRequired) {
        Invoke-CheckedNative conda @(
            'run', '--no-capture-output', '--prefix', $EnvironmentPrefix,
            'python', '-m', 'pip', 'install', 'torch==2.0.0', 'torchvision==0.15.1',
            '--index-url', 'https://download.pytorch.org/whl/cu118'
        )
        Invoke-CheckedNative conda @(
            'run', '--no-capture-output', '--prefix', $EnvironmentPrefix,
            'python', '-m', 'pip', 'install', '--requirement', $RequirementsPath,
            '--extra-index-url', 'https://download.pytorch.org/whl/cu118'
        )
        Invoke-CheckedNative conda @(
            'run', '--no-capture-output', '--prefix', $EnvironmentPrefix,
            'python', '-m', 'pip', 'check'
        )
    }

    # The Windows torch 2.0.0+cu118 wheel requires NumPy C API 0x10, while the
    # authors' 1.22.3 pin exposes 0x0f. Use the smallest bounded compatibility
    # override that restores the bidirectional bridge required by their loader.
    Invoke-CheckedNative conda @(
        'run', '--no-capture-output', '--prefix', $EnvironmentPrefix,
        'python', '-m', 'pip', 'install', 'numpy==1.23.5'
    )
    Export-CheckedNative conda @(
        'run', '--prefix', $EnvironmentPrefix, 'python', '-c',
        'import numpy as np,torch; assert tuple(map(int,np.__version__.split(chr(46))))==(1,23,5); a=np.zeros((1,),dtype=np.float32); t=torch.from_numpy(a); b=t.numpy(); assert b.shape==(1,); print(np.__version__); print(t); print(b)'
    ) (Join-Path $RunDirectory 'numpy-torch-bridge-preflight.txt')
    Invoke-CheckedNative conda @(
        'run', '--no-capture-output', '--prefix', $EnvironmentPrefix,
        'python', '-m', 'pip', 'check'
    )

    @{
        upstream_commit = $PinnedCommit
        requirements_sha256 = $RequirementsSha256
        python = '3.10.4'
        setuptools = '80.9.0'
        numpy_environment_compatibility = '1.23.5'
        torch = '2.0.0+cu118'
        torchvision = '0.15.1+cu118'
    } | ConvertTo-Json | Set-Content -Encoding UTF8 -LiteralPath $InstallMarkerPath

    Export-CheckedNative conda @('list', '--prefix', $EnvironmentPrefix) (Join-Path $RunDirectory 'conda-list.txt')
    Export-CheckedNative conda @(
        'run', '--prefix', $EnvironmentPrefix, 'python', '-m', 'pip', 'freeze'
    ) (Join-Path $RunDirectory 'pip-freeze.txt')
    Export-CheckedNative conda @(
        'run', '--prefix', $EnvironmentPrefix, 'python', '-c',
        'import platform,sys; print(sys.version); print(sys.executable); print(platform.platform())'
    ) (Join-Path $RunDirectory 'python-platform.txt')
    Export-CheckedNative conda @(
        'run', '--prefix', $EnvironmentPrefix, 'python', '-c',
        'import torch,torchvision; print(torch.__version__); print(torchvision.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)'
    ) (Join-Path $RunDirectory 'torch-runtime.txt')
    Export-CheckedNative conda @(
        'run', '--prefix', $EnvironmentPrefix, 'python', '-c',
        'from torchvision.models import ResNet18_Weights,resnet18; w=ResNet18_Weights.IMAGENET1K_V1; resnet18(weights=w); print(w.url)'
    ) (Join-Path $RunDirectory 'encoder-cache.txt')

    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        Export-CheckedNative nvidia-smi @('-q') (Join-Path $RunDirectory 'gpu-driver.txt')
    }
    else {
        'nvidia-smi unavailable' | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $RunDirectory 'gpu-driver.txt')
    }
    Get-CimInstance Win32_Processor | Format-List * | Out-File -Encoding utf8 (Join-Path $RunDirectory 'cpu.txt')
    Get-CimInstance Win32_ComputerSystem | Select-Object Manufacturer, Model, TotalPhysicalMemory | Format-List | Out-File -Encoding utf8 (Join-Path $RunDirectory 'ram.txt')
    $RunDriveName = ([System.IO.Path]::GetPathRoot($RunDirectory)).TrimEnd('\').TrimEnd(':')
    Get-PSDrive -Name $RunDriveName | Select-Object Name, Root, Used, Free | Format-List | Out-File -Encoding utf8 (Join-Path $RunDirectory 'disk.txt')

    @{
        status = 'pass'
        upstream_url = $PinnedUrl
        upstream_commit = $ActualCommit
        upstream_root = $UpstreamRoot
        environment_prefix = $EnvironmentPrefix
        requirements_sha256 = $RequirementsSha256
        setuptools = '80.9.0'
        numpy_environment_compatibility = '1.23.5'
        run_directory = $RunDirectory
    } | ConvertTo-Json | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $RunDirectory 'bootstrap.json')
    Write-Host "Bootstrap complete: $RunDirectory"
}
finally {
    if ($script:TranscriptStarted) {
        Stop-Transcript | Out-Null
    }
}
