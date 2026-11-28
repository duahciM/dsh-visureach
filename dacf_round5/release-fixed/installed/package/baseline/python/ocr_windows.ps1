param([string]$ImagePath, [string]$Language)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$asTask = [System.WindowsRuntimeSystemExtensions].GetMethods() |
    Where-Object { $_.Name -eq 'AsTask' -and $_.IsGenericMethod -and $_.GetParameters().Count -eq 1 } |
    Select-Object -First 1
function Wait-Result($operation, $resultType) {
    $task = $asTask.MakeGenericMethod($resultType).Invoke($null, @($operation))
    $task.GetAwaiter().GetResult()
}
$storageFile = [Windows.Storage.StorageFile,Windows.Storage,ContentType=WindowsRuntime]
$streamType = [Windows.Storage.Streams.IRandomAccessStream,Windows.Storage.Streams,ContentType=WindowsRuntime]
$decoderType = [Windows.Graphics.Imaging.BitmapDecoder,Windows.Graphics.Imaging,ContentType=WindowsRuntime]
$bitmapType = [Windows.Graphics.Imaging.SoftwareBitmap,Windows.Graphics.Imaging,ContentType=WindowsRuntime]
$resultType = [Windows.Media.Ocr.OcrResult,Windows.Foundation,ContentType=WindowsRuntime]
$engineType = [Windows.Media.Ocr.OcrEngine,Windows.Foundation,ContentType=WindowsRuntime]
$file = Wait-Result ($storageFile::GetFileFromPathAsync($ImagePath)) $storageFile
$stream = Wait-Result ($file.OpenAsync([Windows.Storage.FileAccessMode,Windows.Storage,ContentType=WindowsRuntime]::Read)) $streamType
try {
    $decoder = Wait-Result ($decoderType::CreateAsync($stream)) $decoderType
    $bitmap = Wait-Result ($decoder.GetSoftwareBitmapAsync()) $bitmapType
    $engine = $engineType::TryCreateFromUserProfileLanguages()
    if ($Language) {
        $lang = [Windows.Globalization.Language,Windows.Globalization,ContentType=WindowsRuntime]::new($Language)
        $engine = $engineType::TryCreateFromLanguage($lang)
    }
    if ($null -eq $engine) { throw 'No matching Windows OCR language is installed.' }
    $result = Wait-Result ($engine.RecognizeAsync($bitmap)) $resultType
    $output = foreach ($line in $result.Lines) {
        foreach ($word in $line.Words) {
            $r = $word.BoundingRect
            @{text=$word.Text; native_xywh=@($r.X,$r.Y,$r.Width,$r.Height)}
        }
    }
    ConvertTo-Json -InputObject @($output | Where-Object { $null -ne $_ }) -Compress -Depth 4
} finally {
    if ($null -ne $bitmap) { $bitmap.Dispose() }
    $stream.Dispose()
}
