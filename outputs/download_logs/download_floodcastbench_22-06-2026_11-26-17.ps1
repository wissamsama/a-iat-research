$ErrorActionPreference = 'Stop'
$url = 'https://zenodo.org/records/14017092/files/FloodCastBench.zip?download=1'
$zip = 'C:\Users\Student\Desktop\wissam\a-iat-research\data\FloodCastBench.zip'
$done = 'C:\Users\Student\Desktop\wissam\a-iat-research\outputs\download_logs\download_floodcastbench_22-06-2026_11-26-17.done.txt'
$expectedMd5 = 'c43f3009c82e212ef21a65739f4ada3d'
"Starting download: $(Get-Date -Format o)" | Out-File -FilePath $done -Encoding utf8
"Target: $zip" | Out-File -FilePath $done -Append -Encoding utf8
& curl.exe -L -C - --retry 30 --retry-delay 20 --connect-timeout 60 --speed-time 120 --speed-limit 1024 -o $zip $url
$exit = $LASTEXITCODE
"curl_exit_code: $exit" | Out-File -FilePath $done -Append -Encoding utf8
if ($exit -eq 0) {
    $hash = (Get-FileHash -Algorithm MD5 -Path $zip).Hash.ToLowerInvariant()
    "md5: $hash" | Out-File -FilePath $done -Append -Encoding utf8
    "md5_expected: $expectedMd5" | Out-File -FilePath $done -Append -Encoding utf8
    "md5_ok: $($hash -eq $expectedMd5)" | Out-File -FilePath $done -Append -Encoding utf8
}
"Finished: $(Get-Date -Format o)" | Out-File -FilePath $done -Append -Encoding utf8
