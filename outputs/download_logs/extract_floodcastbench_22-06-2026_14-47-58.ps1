$ErrorActionPreference = 'Stop'
$zip = 'C:\Users\Student\Desktop\wissam\a-iat-research\data\FloodCastBench.zip'
$target = 'C:\Users\Student\Desktop\wissam\a-iat-research\data\FloodCastBench'
$temp = 'C:\Users\Student\Desktop\wissam\a-iat-research\data\FloodCastBench_extracting'
$done = 'C:\Users\Student\Desktop\wissam\a-iat-research\outputs\download_logs\extract_floodcastbench_22-06-2026_14-47-58.done.txt'
$expectedMd5 = 'c43f3009c82e212ef21a65739f4ada3d'
"Starting extraction: $(Get-Date -Format o)" | Out-File -FilePath $done -Encoding utf8
"Zip: $zip" | Out-File -FilePath $done -Append -Encoding utf8
if (-not (Test-Path -LiteralPath $zip)) { throw "Missing zip: $zip" }
if (Test-Path -LiteralPath $target) { throw "Target already exists, aborting: $target" }
if (Test-Path -LiteralPath $temp) { throw "Temporary extraction folder already exists, aborting: $temp" }
$hash = (Get-FileHash -Algorithm MD5 -Path $zip).Hash.ToLowerInvariant()
"md5: $hash" | Out-File -FilePath $done -Append -Encoding utf8
"md5_expected: $expectedMd5" | Out-File -FilePath $done -Append -Encoding utf8
if ($hash -ne $expectedMd5) { throw "MD5 mismatch" }
New-Item -ItemType Directory -Force -Path $temp | Out-Null
& tar.exe -xf $zip -C $temp
$exit = $LASTEXITCODE
"tar_exit_code: $exit" | Out-File -FilePath $done -Append -Encoding utf8
if ($exit -ne 0) { throw "tar extraction failed with exit code $exit" }
$items = @(Get-ChildItem -LiteralPath $temp -Force)
if ($items.Count -eq 1 -and $items[0].PSIsContainer -and $items[0].Name -eq 'FloodCastBench') {
    Move-Item -LiteralPath $items[0].FullName -Destination $target
    Remove-Item -LiteralPath $temp -Force
} else {
    Move-Item -LiteralPath $temp -Destination $target
}
$tiffCount = (Get-ChildItem -LiteralPath $target -Recurse -File -Include *.tif,*.tiff -ErrorAction SilentlyContinue | Measure-Object).Count
"target: $target" | Out-File -FilePath $done -Append -Encoding utf8
"tiff_count: $tiffCount" | Out-File -FilePath $done -Append -Encoding utf8
"Finished: $(Get-Date -Format o)" | Out-File -FilePath $done -Append -Encoding utf8
