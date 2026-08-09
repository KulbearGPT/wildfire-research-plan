[CmdletBinding()]
param(
    [string]$RunDirectory,
    [switch]$ValidateEnvironmentOnly,
    [switch]$ValidateCheckoutOnly,
    [switch]$ValidateRuntimeOnly,
    [string]$ValidationUpstreamRoot
)

$ErrorActionPreference = 'Stop'
$script:TranscriptStarted = $false
$EnvironmentPrefix = [System.IO.Path]::GetFullPath(
    'D:\WildFire Project\.conda-envs\wsts-res18-t1'
).TrimEnd('\')

function Get-NormalizedPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    return [System.IO.Path]::GetFullPath($Path).TrimEnd('\')
}

function Assert-IsolatedEnvironment {
    param([switch]$AllowMissing)

    $ActivePrefix = $null
    if ($env:CONDA_PREFIX) {
        $ActivePrefix = Get-NormalizedPath $env:CONDA_PREFIX
        if ($ActivePrefix -ieq $EnvironmentPrefix) {
            throw "Target prefix must not be the active Conda environment: $EnvironmentPrefix"
        }
    }

    $CondaBaseOutput = & conda info --base 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to determine the Conda base prefix"
    }
    $CondaBasePrefix = Get-NormalizedPath ([string]($CondaBaseOutput | Select-Object -Last 1))
    if ($CondaBasePrefix -ieq $EnvironmentPrefix) {
        throw "Target prefix must not be the Conda base prefix: $EnvironmentPrefix"
    }

    $EnvironmentPython = Join-Path $EnvironmentPrefix 'python.exe'
    if ((Test-Path -LiteralPath $EnvironmentPrefix) -and -not (Test-Path -LiteralPath $EnvironmentPython)) {
        throw "Environment prefix exists but has no python.exe; refusing to replace it: $EnvironmentPrefix"
    }
    if (-not (Test-Path -LiteralPath $EnvironmentPython)) {
        if ($AllowMissing) {
            return $null
        }
        throw "Environment prefix does not contain python.exe: $EnvironmentPrefix"
    }

    $PythonProbe = & $EnvironmentPython -c 'import sys;print(sys.prefix);print(*sys.version_info[:3],sep=chr(46))' 2>&1
    if ($LASTEXITCODE -ne 0 -or $PythonProbe.Count -lt 2) {
        throw "Unable to verify the target prefix Python runtime: $EnvironmentPython"
    }
    $PythonPrefix = Get-NormalizedPath ([string]$PythonProbe[0])
    $PythonVersion = ([string]$PythonProbe[1]).Trim()
    if ($PythonPrefix -ine $EnvironmentPrefix) {
        throw "Target python sys.prefix mismatch: expected $EnvironmentPrefix, got $PythonPrefix"
    }
    if ($PythonVersion -ne '3.10.4') {
        throw "Target python version mismatch: expected 3.10.4, got $PythonVersion"
    }

    return [pscustomobject][ordered]@{
        environment_prefix = $EnvironmentPrefix
        python_prefix = $PythonPrefix
        python_version = $PythonVersion
        conda_base_prefix = $CondaBasePrefix
        active_prefix = $ActivePrefix
    }
}

function Get-VerifiedRuntimeMetadata {
    param([Parameter(Mandatory = $true)][string]$EnvironmentPython)

    $PythonCommand = 'import json,sys;from importlib.metadata import version;s=lambda *x:bytes(x).decode();print(json.dumps(dict(python=chr(46).join(map(str,sys.version_info[:3])),setuptools=version(s(115,101,116,117,112,116,111,111,108,115)),numpy=version(s(110,117,109,112,121)),torch=version(s(116,111,114,99,104)),torchvision=version(s(116,111,114,99,104,118,105,115,105,111,110)))))'
    $RuntimeMetadataOutput = & $EnvironmentPython -c $PythonCommand 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to verify installed runtime versions: $RuntimeMetadataOutput"
    }
    $RuntimeMetadata = ([string]($RuntimeMetadataOutput | Select-Object -Last 1)) | ConvertFrom-Json
    $ExpectedRuntime = [ordered]@{
        python = '3.10.4'
        setuptools = '80.9.0'
        numpy = '1.23.5'
        torch = '2.0.0+cu118'
        torchvision = '0.15.1+cu118'
    }
    foreach ($PackageName in $ExpectedRuntime.Keys) {
        if ($RuntimeMetadata.$PackageName -ne $ExpectedRuntime[$PackageName]) {
            throw "Runtime version mismatch for ${PackageName}: expected $($ExpectedRuntime[$PackageName]), got $($RuntimeMetadata.$PackageName)"
        }
    }
    return $RuntimeMetadata
}

function Assert-OfficialCheckout {
    param(
        [Parameter(Mandatory = $true)][string]$CheckoutRoot,
        [Parameter(Mandatory = $true)][string]$PinnedUrl,
        [Parameter(Mandatory = $true)][string]$PinnedCommit,
        [switch]$RequirePinnedCommit
    )

    $CheckoutRoot = Get-NormalizedPath $CheckoutRoot
    if (-not (Test-Path -LiteralPath (Join-Path $CheckoutRoot '.git'))) {
        throw "Upstream path is not a Git checkout: $CheckoutRoot"
    }
    $Dirty = & git -C $CheckoutRoot status --porcelain
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to inspect existing upstream checkout: $CheckoutRoot"
    }
    if ($Dirty) {
        throw "Existing upstream checkout is dirty; refusing to overwrite it: $CheckoutRoot"
    }
    $ActualOrigin = (& git -C $CheckoutRoot remote get-url origin).Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to determine upstream origin URL: $CheckoutRoot"
    }
    if ($ActualOrigin -cne $PinnedUrl) {
        throw "Upstream origin URL mismatch: expected $PinnedUrl, got $ActualOrigin"
    }

    $ActualCommit = (& git -C $CheckoutRoot rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to determine upstream commit: $CheckoutRoot"
    }
    if ($RequirePinnedCommit -and $ActualCommit -ne $PinnedCommit) {
        throw "Pinned checkout mismatch: expected $PinnedCommit, got $ActualCommit"
    }

    return [pscustomobject][ordered]@{
        upstream_root = $CheckoutRoot
        upstream_origin = $ActualOrigin
        upstream_commit = $ActualCommit
    }
}

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
    $LockPath = Join-Path $ReproductionRoot 'upstream.lock.json'
    $Lock = Get-Content -Raw -LiteralPath $LockPath | ConvertFrom-Json
    $PinnedUrl = [string]$Lock.code.url
    $PinnedCommit = [string]$Lock.code.commit
    $UpstreamRoot = Get-NormalizedPath (Join-Path $ReproductionRoot '.local\WildfireSpreadTS')

    $ValidationModeCount = @(
        $ValidateEnvironmentOnly,
        $ValidateCheckoutOnly,
        $ValidateRuntimeOnly
    ).Where({ $_ }).Count
    if ($ValidationModeCount -gt 1) {
        throw "Choose only one dry validation mode"
    }
    if ($ValidationUpstreamRoot -and -not $ValidateCheckoutOnly) {
        throw "ValidationUpstreamRoot is allowed only with ValidateCheckoutOnly"
    }
    if ($ValidateEnvironmentOnly) {
        $EnvironmentValidation = Assert-IsolatedEnvironment
        $EnvironmentValidation | ConvertTo-Json -Compress
        return
    }
    if ($ValidateCheckoutOnly) {
        $CheckoutRoot = if ($ValidationUpstreamRoot) {
            Get-NormalizedPath $ValidationUpstreamRoot
        }
        else {
            $UpstreamRoot
        }
        $CheckoutValidation = Assert-OfficialCheckout `
            -CheckoutRoot $CheckoutRoot `
            -PinnedUrl $PinnedUrl `
            -PinnedCommit $PinnedCommit `
            -RequirePinnedCommit
        $CheckoutValidation | ConvertTo-Json -Compress
        return
    }
    if ($ValidateRuntimeOnly) {
        $EnvironmentValidation = Assert-IsolatedEnvironment
        $RuntimeMetadata = Get-VerifiedRuntimeMetadata `
            -EnvironmentPython (Join-Path $EnvironmentPrefix 'python.exe')
        $RuntimeMetadata | ConvertTo-Json -Compress
        return
    }
    if ([string]::IsNullOrWhiteSpace($RunDirectory)) {
        throw "RunDirectory is required for bootstrap execution"
    }

    # This guard runs before conda create or the first pip invocation. A missing
    # target is allowed only until the dedicated prefix is created below.
    $EnvironmentValidation = Assert-IsolatedEnvironment -AllowMissing
    $RunDirectory = Get-NormalizedPath $RunDirectory

    New-Item -ItemType Directory -Force -Path $RunDirectory | Out-Null
    Start-Transcript -Path (Join-Path $RunDirectory 'bootstrap.log') -Append | Out-Null
    $script:TranscriptStarted = $true

    Write-Host "Pinned upstream: $PinnedUrl@$PinnedCommit"
    Write-Host "Pinned environment prefix: $EnvironmentPrefix"
    Write-Host "Current process Python is never invoked; all Python commands use conda run --prefix."

    if (-not (Test-Path -LiteralPath $UpstreamRoot)) {
        New-Item -ItemType Directory -Force -Path (Split-Path $UpstreamRoot -Parent) | Out-Null
        Invoke-CheckedNative git @('clone', '--no-checkout', $PinnedUrl, $UpstreamRoot)
    }
    $CheckoutBeforeFetch = Assert-OfficialCheckout `
        -CheckoutRoot $UpstreamRoot `
        -PinnedUrl $PinnedUrl `
        -PinnedCommit $PinnedCommit

    Invoke-CheckedNative git @('-C', $UpstreamRoot, 'fetch', '--force', 'origin', $PinnedCommit)
    Invoke-CheckedNative git @('-C', $UpstreamRoot, 'checkout', '--detach', $PinnedCommit)
    $CheckoutValidation = Assert-OfficialCheckout `
        -CheckoutRoot $UpstreamRoot `
        -PinnedUrl $PinnedUrl `
        -PinnedCommit $PinnedCommit `
        -RequirePinnedCommit
    $ActualCommit = $CheckoutValidation.upstream_commit
    $ActualOrigin = $CheckoutValidation.upstream_origin

    $EnvironmentPython = Join-Path $EnvironmentPrefix 'python.exe'
    if (-not (Test-Path -LiteralPath $EnvironmentPython)) {
        New-Item -ItemType Directory -Force -Path (Split-Path $EnvironmentPrefix -Parent) | Out-Null
        Invoke-CheckedNative conda @('create', '--prefix', $EnvironmentPrefix, 'python=3.10.4', 'pip', '-y')
    }
    $EnvironmentValidation = Assert-IsolatedEnvironment

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

    $RuntimeMetadata = Get-VerifiedRuntimeMetadata -EnvironmentPython $EnvironmentPython

    @{
        upstream_origin = $ActualOrigin
        upstream_commit = $ActualCommit
        requirements_sha256 = $RequirementsSha256
        python = $RuntimeMetadata.python
        setuptools = $RuntimeMetadata.setuptools
        numpy_environment_compatibility = $RuntimeMetadata.numpy
        torch = $RuntimeMetadata.torch
        torchvision = $RuntimeMetadata.torchvision
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
        upstream_url = $ActualOrigin
        upstream_commit = $ActualCommit
        upstream_root = $UpstreamRoot
        environment_prefix = $EnvironmentPrefix
        python_prefix = $EnvironmentValidation.python_prefix
        python_version = $EnvironmentValidation.python_version
        conda_base_prefix = $EnvironmentValidation.conda_base_prefix
        requirements_sha256 = $RequirementsSha256
        setuptools = $RuntimeMetadata.setuptools
        numpy_environment_compatibility = $RuntimeMetadata.numpy
        torch = $RuntimeMetadata.torch
        torchvision = $RuntimeMetadata.torchvision
        run_directory = $RunDirectory
    } | ConvertTo-Json | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $RunDirectory 'bootstrap.json')
    Write-Host "Bootstrap complete: $RunDirectory"
}
finally {
    if ($script:TranscriptStarted) {
        Stop-Transcript | Out-Null
    }
}
