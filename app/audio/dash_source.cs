// Adaptive (DASH) source interop for Tideway's native Atmos helper.
//
// Windows.Media.Playback.MediaPlayer can only play a still-growing
// streamed track if it is given an AdaptiveMediaSource (the same DASH
// client Edge uses) rather than a file source: a file/HTTP source is
// not opened until its full body has arrived. Creating one is an
// IAsyncOperation, which Windows PowerShell cannot await — its binder
// cannot convert the returned __ComObject to the closed generic
// interface — so the one async call lives here and the helper gets a
// finished MediaSource back.
//
// Built against the Windows SDK's union metadata; see build_dash_source
// in the repo notes. The DLL is loaded by winrt_player.ps1 with
// Add-Type -Path and is only used on the native Atmos path.

using System;
using Windows.Media.Core;
using Windows.Media.Streaming.Adaptive;

namespace Tideway
{
    public static class DashSource
    {
        // Status of the last Create call, for diagnostics: "Success",
        // "ManifestDownloadFailure", "UnsupportedManifestContentType",
        // ... or "timeout".
        public static string LastStatus = "";

        public static MediaSource Create(string url, int timeoutMs)
        {
            LastStatus = "";
            var operation = AdaptiveMediaSource.CreateFromUriAsync(new Uri(url));
            var task = operation.AsTask();
            if (!task.Wait(timeoutMs))
            {
                LastStatus = "timeout";
                throw new TimeoutException(
                    "adaptive source creation timed out after " + timeoutMs + " ms");
            }
            AdaptiveMediaSourceCreationResult result = task.Result;
            LastStatus = result.Status.ToString();
            if (result.Status != AdaptiveMediaSourceCreationStatus.Success
                || result.MediaSource == null)
            {
                string detail = result.ExtendedError == null
                    ? ""
                    : ": " + result.ExtendedError.Message;
                throw new InvalidOperationException(
                    "adaptive manifest rejected (" + result.Status + ")" + detail);
            }
            return MediaSource.CreateFromAdaptiveMediaSource(result.MediaSource);
        }
    }
}
