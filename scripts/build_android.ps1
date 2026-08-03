param(
    [ValidateSet('Debug', 'Release')]
    [string]$Configuration = 'Debug',
    [string]$BuildRoot = (Join-Path $env:TEMP 'smartwaste-vision-android-build')
)

$ErrorActionPreference = 'Stop'

if ($BuildRoot -match '[^\x00-\x7F]') {
    throw "BuildRoot must contain only ASCII characters: $BuildRoot"
}

$projectRoot = Split-Path -Parent $PSScriptRoot
$sourceMobileRoot = Join-Path $projectRoot 'apps\mobile'
$mobileRoot = $sourceMobileRoot
$artifactRoot = Join-Path $projectRoot 'artifacts\mobile'
$task = if ($Configuration -eq 'Release') { 'app:assembleRelease' } else { 'app:assembleDebug' }
$apkName = if ($Configuration -eq 'Release') { 'app-release.apk' } else { 'app-debug.apk' }
$destinationName = if ($Configuration -eq 'Release') {
    'smartwaste-vision-release.apk'
} else {
    'smartwaste-vision-debug.apk'
}

New-Item -ItemType Directory -Path $BuildRoot -Force | Out-Null
New-Item -ItemType Directory -Path $artifactRoot -Force | Out-Null

# Flutter's Windows AOT snapshotter can corrupt non-ASCII project paths even
# when Gradle's output directory is redirected. Release builds therefore run
# from a clean ASCII-only staging copy; debug builds can stay in place.
if ($Configuration -eq 'Release' -and $sourceMobileRoot -match '[^\x00-\x7F]') {
    $resolvedBuildRoot = (Resolve-Path -LiteralPath $BuildRoot).Path.TrimEnd('\')
    $stagingRoot = Join-Path $resolvedBuildRoot 'release-project'

    if (Test-Path -LiteralPath $stagingRoot) {
        $resolvedStagingRoot = (Resolve-Path -LiteralPath $stagingRoot).Path
        $expectedPrefix = "$resolvedBuildRoot\"
        if (-not $resolvedStagingRoot.StartsWith(
            $expectedPrefix,
            [System.StringComparison]::OrdinalIgnoreCase
        )) {
            throw "Refusing to clean staging path outside BuildRoot: $resolvedStagingRoot"
        }
        Remove-Item -LiteralPath $resolvedStagingRoot -Recurse -Force
    }

    New-Item -ItemType Directory -Path $stagingRoot -Force | Out-Null
    & robocopy.exe $sourceMobileRoot $stagingRoot /E `
        /XD `
            (Join-Path $sourceMobileRoot '.dart_tool') `
            (Join-Path $sourceMobileRoot '.idea') `
            (Join-Path $sourceMobileRoot 'build') `
            (Join-Path $sourceMobileRoot 'android\.gradle') `
            (Join-Path $sourceMobileRoot 'android\.kotlin') `
        /XF '.flutter-plugins-dependencies' `
        /NFL /NDL /NJH /NJS /NP | Out-Null
    $robocopyExitCode = $LASTEXITCODE
    if ($robocopyExitCode -gt 7) {
        throw "Could not stage the Flutter project (robocopy exit code $robocopyExitCode)."
    }

    $mobileRoot = $stagingRoot
    Write-Host "Release staging directory: $mobileRoot"
}

$androidRoot = Join-Path $mobileRoot 'android'

$previousBuildRoot = $env:SMARTWASTE_ANDROID_BUILD_DIR
$originalJavaHome = $env:JAVA_HOME
$env:SMARTWASTE_ANDROID_BUILD_DIR = (Resolve-Path -LiteralPath $BuildRoot).Path
if ([string]::IsNullOrWhiteSpace($env:JAVA_HOME)) {
    $androidStudioJbr = 'C:\Program Files\Android\Android Studio\jbr'
    if (-not (Test-Path -LiteralPath $androidStudioJbr)) {
        throw 'JAVA_HOME is empty and the Android Studio JBR could not be found.'
    }
    $env:JAVA_HOME = $androidStudioJbr
}

try {
    Push-Location $mobileRoot
    try {
        & flutter pub get
        if ($LASTEXITCODE -ne 0) {
            throw "flutter pub get failed with exit code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }

    Push-Location $androidRoot
    try {
        & .\gradlew.bat $task
        if ($LASTEXITCODE -ne 0) {
            throw "Gradle task $task failed with exit code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }

    $sourceApk = Join-Path $BuildRoot "app\outputs\flutter-apk\$apkName"
    if (-not (Test-Path -LiteralPath $sourceApk)) {
        throw "Gradle completed but APK was not found: $sourceApk"
    }
    $destinationApk = Join-Path $artifactRoot $destinationName
    Copy-Item -LiteralPath $sourceApk -Destination $destinationApk -Force
    $hash = Get-FileHash -LiteralPath $destinationApk -Algorithm SHA256
    Write-Host "APK: $destinationApk"
    Write-Host "Size: $((Get-Item -LiteralPath $destinationApk).Length) bytes"
    Write-Host "SHA256: $($hash.Hash.ToLowerInvariant())"
} finally {
    if ($null -eq $previousBuildRoot) {
        Remove-Item Env:SMARTWASTE_ANDROID_BUILD_DIR -ErrorAction SilentlyContinue
    } else {
        $env:SMARTWASTE_ANDROID_BUILD_DIR = $previousBuildRoot
    }
    if ($null -eq $originalJavaHome) {
        Remove-Item Env:JAVA_HOME -ErrorAction SilentlyContinue
    } else {
        $env:JAVA_HOME = $originalJavaHome
    }
}
