# Tideway's native Atmos playback helper.
#
# Windows Media Player decodes Dolby Atmos (E-AC-3 JOC) through Media
# Foundation with Dolby's decoder MFT and hands the resulting object
# audio to the endpoint's spatial renderer ("Dolby Atmos for
# Headphones"). That object stage lives inside the OS pipeline — PyAV
# only ever sees the 5.1/7.1 core bed — so the only way to hear real
# height objects is to let Windows play the file.
#
# This script hosts Windows.Media.Playback.MediaPlayer (the same engine
# the Media Player app uses) and speaks a line protocol over stdin so
# Tideway can control it. Every reply line is prefixed with @@ so the
# decoder MFT's own console chatter can be ignored by the reader.

param([string]$Ignored)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.Encoding]::UTF8

[Windows.Media.Playback.MediaPlayer, Windows.Media, ContentType = WindowsRuntime] | Out-Null
[Windows.Media.Core.MediaSource, Windows.Media.Core, ContentType = WindowsRuntime] | Out-Null
[Windows.Media.Devices.MediaDevice, Windows.Media.Devices, ContentType = WindowsRuntime] | Out-Null
[Windows.Media.Audio.SpatialAudioDeviceConfiguration, Windows.Media.Audio, ContentType = WindowsRuntime] | Out-Null
[Windows.Media.Audio.SpatialAudioFormatSubtype, Windows.Media.Audio, ContentType = WindowsRuntime] | Out-Null
[Windows.Devices.Enumeration.DeviceInformation, Windows.Devices.Enumeration, ContentType = WindowsRuntime] | Out-Null

$player = New-Object Windows.Media.Playback.MediaPlayer
$player.AudioCategory = [Windows.Media.Playback.MediaPlayerAudioCategory]::Media
$player.Volume = 1.0
$player.IsMuted = $false
$player.CommandManager.IsEnabled = $false

# Adaptive (DASH) sources. MediaPlayer cannot start on a file source
# that is still growing, but it streams segments happily when handed an
# AdaptiveMediaSource. Creating one is the one WinRT async call this
# script cannot make: Windows PowerShell cannot await the
# IAsyncOperation it returns, so a small compiled helper does it and
# hands back a ready MediaSource. Missing DLL = file sources only.
$script:adaptiveAvailable = $false
try {
    $dashDll = Join-Path $PSScriptRoot 'dash_source.dll'
    if (Test-Path -LiteralPath $dashDll) {
        Add-Type -Path $dashDll
        [Tideway.DashSource] | Out-Null
        $script:adaptiveAvailable = $true
    }
} catch {
    $script:adaptiveAvailable = $false
}

$script:path = ''
$script:sourceKind = 'file'
# MediaPlayer exposes exactly one gain knob — a linear 0..1 volume —
# so the user's slider and loudness normalization have to share it.
# Keeping them as separate inputs and composing on every change is
# what makes them independent: the slider means the same thing
# whether leveling is on or off, and a normalization change doesn't
# move the slider.
$script:userVolume = 1.0
$script:gainDb = 0.0

function Apply-OutputLevel {
    $linear = $script:userVolume * [Math]::Pow(10.0, $script:gainDb / 20.0)
    if ($linear -lt 0.0) { $linear = 0.0 }
    # Unity is the ceiling: there is no amplifier past it, so a
    # normalization gain that would need headroom is capped here.
    if ($linear -gt 1.0) { $linear = 1.0 }
    $player.Volume = $linear
}

function Get-GainApplied {
    if ($script:userVolume -le 0.0 -or $player.Volume -le 0.0) { return 0.0 }
    return 20.0 * [Math]::Log10($player.Volume / $script:userVolume)
}

function Send([hashtable]$payload) {
    $payload.ok = $true
    $json = $payload | ConvertTo-Json -Compress -Depth 4
    [Console]::Out.WriteLine('@@' + $json)
    [Console]::Out.Flush()
}

function SendError([string]$message) {
    $json = @{ ok = $false; error = $message } | ConvertTo-Json -Compress
    [Console]::Out.WriteLine('@@' + $json)
    [Console]::Out.Flush()
}

function Get-CodecModules {
    # Which decoder DLLs are actually loaded in this process. The Dolby
    # DD+ MFT appearing here is the proof that the file went through
    # Dolby's decoder rather than a fallback that downmixes.
    try {
        return @(
            (Get-Process -Id $PID).Modules |
                Where-Object { $_.ModuleName -match 'Dolby|ec3|ac3|atmos' } |
                ForEach-Object { $_.ModuleName } | Sort-Object -Unique
        )
    } catch {
        return @()
    }
}

function Get-RenderDeviceName {
    # The endpoint the OS pipeline is rendering to. MediaPlayer leaves
    # AudioDevice null until explicitly assigned, so resolve the current
    # default render endpoint and read its friendly name ("Speakers
    # (Realtek(R) Audio)") out of the MMDevices registry properties.
    try {
        $id = [Windows.Media.Devices.MediaDevice]::GetDefaultAudioRenderId(
            [Windows.Media.Devices.AudioDeviceRole]::Default)
        if ($id -match '\{0\.0\.0\.00000000\}\.(\{[0-9a-fA-F-]+\})') {
            $key = "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\MMDevices\Audio\Render\$($Matches[1])\Properties"
            $props = Get-ItemProperty -Path $key -ErrorAction Stop
            $desc = $props.'{a45c254e-df1c-4efd-8020-67d146a850e0},2'
            $name = $props.'{b3f8fa53-0004-438e-9003-51a46e139bfc},6'
            if ($desc -and $name) { return "$desc ($name)" }
            if ($name) { return $name }
            if ($desc) { return $desc }
        }
        return [string]$id
    } catch { }
    return ''
}

function Get-SpatialInfo {
    # Device-level Spatial Sound state. MediaPlayer always hands decoded
    # object audio to whatever spatialiser the endpoint has active, so
    # "this file is Atmos" and "you are hearing Atmos" are different
    # questions. Only a Dolby Atmos format places the height objects;
    # Windows Sonic and DTS virtualise the 7.1 bed instead, and with
    # spatial off the bed goes straight to the endpoint. The slug is
    # what Python classifies on — '' means spatial is off.
    try {
        $id = [Windows.Media.Devices.MediaDevice]::GetDefaultAudioRenderId(
            [Windows.Media.Devices.AudioDeviceRole]::Default)
        $cfg = [Windows.Media.Audio.SpatialAudioDeviceConfiguration]::GetForDeviceId($id)
        $s = [Windows.Media.Audio.SpatialAudioFormatSubtype]
        $active = ([string]$cfg.ActiveSpatialAudioFormat).ToUpperInvariant()
        $format = ''
        if ($active) {
            if ($active -eq ([string]$s::DolbyAtmosForHeadphones).ToUpperInvariant()) {
                $format = 'atmos-headphones'
            } elseif ($active -eq ([string]$s::DolbyAtmosForHomeTheater).ToUpperInvariant()) {
                $format = 'atmos-home-theater'
            } elseif ($active -eq ([string]$s::DolbyAtmosForSpeakers).ToUpperInvariant()) {
                $format = 'atmos-speakers'
            } elseif ($active -eq ([string]$s::WindowsSonic).ToUpperInvariant()) {
                $format = 'windows-sonic'
            } elseif ($active -eq ([string]$s::DTSHeadphoneX).ToUpperInvariant()) {
                $format = 'dts-headphone-x'
            } elseif ($active -eq ([string]$s::DTSXForHomeTheater).ToUpperInvariant()) {
                $format = 'dts-x'
            } elseif ($active -eq ([string]$s::DTSXUltra).ToUpperInvariant()) {
                $format = 'dts-x-ultra'
            } else {
                # A spatialiser this build doesn't know. Keep the raw
                # GUID so diagnostics can identify it without a code
                # change; Python treats it as non-Atmos.
                $format = $active.ToLowerInvariant()
            }
        }
        return @{
            supported = [bool]$cfg.IsSpatialAudioSupported
            format    = $format
        }
    } catch {
        # Spatial Sound is optional: older Windows builds, virtual or
        # disconnected endpoints, and drivers without a spatialiser all
        # land here.
        return @{ supported = $false; format = '' }
    }
}

function Get-State {
    $session = $player.PlaybackSession
    $spatial = Get-SpatialInfo
    return @{
        state    = [string]$session.PlaybackState
        pos_ms   = [int64]$session.Position.TotalMilliseconds
        dur_ms   = [int64]$session.NaturalDuration.TotalMilliseconds
        device   = Get-RenderDeviceName
        path     = $script:path
        # 'adaptive' when the source is a DASH presentation served over
        # loopback, 'file' when MediaPlayer opened a finished file.
        source_kind = $script:sourceKind
        adaptive_available = $script:adaptiveAvailable
        # Active spatial format of the render endpoint — the piece that
        # decides whether the JOC objects this pipeline decodes are
        # actually being placed as height audio.
        spatial_supported = $spatial.supported
        spatial_format    = $spatial.format
        # Applied level and the normalization offset that produced it.
        # `gain_applied_db` is what the listener actually got, which
        # differs from `gain_db` whenever the request would have
        # needed headroom past unity volume.
        volume          = [double]$player.Volume
        gain_db         = [double]$script:gainDb
        gain_applied_db = [double](Get-GainApplied)
        # @() keeps a single module as a JSON array instead of a bare
        # string — Python callers iterate this unconditionally.
        decoders = @(Get-CodecModules)
    }
}

function Wait-ForSource {
    # CreateFromUri is async: NaturalDuration stays 0 until the source
    # opens, and Position jumps once the pipeline is live. Give it a
    # bounded window so Play() is called on a ready player.
    for ($i = 0; $i -lt 40; $i++) {
        if ($player.PlaybackSession.NaturalDuration.TotalMilliseconds -gt 0) { return }
        Start-Sleep -Milliseconds 100
    }
}

while ($true) {
    $line = [Console]::In.ReadLine()
    if ($null -eq $line) { break }
    if ([string]::IsNullOrWhiteSpace($line)) { continue }
    try {
        $msg = $line | ConvertFrom-Json
        switch ($msg.cmd) {
            'load' {
                $script:path = [string]$msg.path
                if ($msg.adaptive) {
                    if (-not $script:adaptiveAvailable) {
                        throw 'adaptive source unavailable: dash_source.dll not loaded'
                    }
                    # DASH: MediaPlayer fetches segments itself, so it
                    # starts on the first buffered one instead of
                    # waiting for the whole track.
                    $player.Source = [Tideway.DashSource]::Create($script:path, 20000)
                    $script:sourceKind = 'adaptive'
                } else {
                    $player.Source = [Windows.Media.Core.MediaSource]::CreateFromUri([Uri]$script:path)
                    $script:sourceKind = 'file'
                }
                Wait-ForSource
                if ($msg.autoplay) { $player.Play() }
                Send (Get-State)
            }
            'play' { $player.Play(); Send (Get-State) }
            'pause' { $player.Pause(); Send (Get-State) }
            'seek' { $player.PlaybackSession.Position = [TimeSpan]::FromMilliseconds([double]$msg.ms); Send (Get-State) }
            'volume' {
                $script:userVolume = [Math]::Max(0.0, [Math]::Min(1.0, [double]$msg.value / 100.0))
                Apply-OutputLevel
                Send (Get-State)
            }
            'gain' {
                $script:gainDb = [double]$msg.value
                Apply-OutputLevel
                Send (Get-State)
            }
            'mute' { $player.IsMuted = [bool]$msg.value; Send (Get-State) }
            'state' { Send (Get-State) }
            'stop' {
                $player.Pause()
                $player.Source = $null
                $script:path = ''
                Send (Get-State)
            }
            'quit' {
                $player.Pause()
                $player.Source = $null
                $player.Dispose()
                Send @{ state = 'closed' }
                break
            }
            default { SendError "unknown command: $($msg.cmd)" }
        }
        if ($msg.cmd -eq 'quit') { break }
    } catch {
        SendError ($_.Exception.Message)
    }
}
