$datasets = @("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter")
$maxRetries = 10

foreach ($d in $datasets) {
    $attempt = 0
    do {
        $attempt++
        Write-Host "`n[$d] Attempt $attempt..." -ForegroundColor Cyan
        echo "" | python download.py C:/deepfake-detector/data/ffpp -d $d -c c40 -t videos --server EU2
        $exit = $LASTEXITCODE
        if ($exit -ne 0) {
            Write-Host "[$d] Failed (exit $exit), retrying in 10s..." -ForegroundColor Yellow
            Start-Sleep -Seconds 10
        }
    } while ($exit -ne 0 -and $attempt -lt $maxRetries)

    if ($exit -eq 0) {
        Write-Host "[$d] Done." -ForegroundColor Green
    } else {
        Write-Host "[$d] Gave up after $maxRetries attempts." -ForegroundColor Red
    }
}
