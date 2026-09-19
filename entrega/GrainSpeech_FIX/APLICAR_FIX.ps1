param(
    [string]$Destino = ""
)

if (-not $Destino) {
    Write-Host "Uso: .\APLICAR_FIX.ps1 -Destino 'C:\caminho\para\GrainSpeechStudio_pacote'" -ForegroundColor Yellow
    exit 1
}

$DestinoPath = Resolve-Path $Destino -ErrorAction SilentlyContinue
if (-not $DestinoPath or -not (Test-Path $DestinoPath)) {
    Write-Host "Erro: Pasta de destino '$Destino' não encontrada!" -ForegroundColor Red
    exit 1
}

$AppDir = Join-Path $DestinoPath "app"
if (-not (Test-Path $AppDir)) {
    Write-Host "Erro: Pasta 'app' não encontrada em $DestinoPath" -ForegroundColor Red
    exit 1
}

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$BackupDir = Join-Path $DestinoPath "app_backup_$Timestamp"
Write-Host "Criando backup dos arquivos originais em: $BackupDir" -ForegroundColor Cyan
New-Item -ItemType Directory -Path $BackupDir -Force | Out-Null
Copy-Item -Path "$AppDir\*" -Destination $BackupDir -Recurse -Force

Write-Host "Copiando arquivos corrigidos para $AppDir..." -ForegroundColor Green
$ScriptDir = $PSScriptRoot
Copy-Item -Path "$ScriptDir\app\*" -Destination $AppDir -Recurse -Force

Write-Host "✅ FIX APLICADO COM SUCESSO!" -ForegroundColor Green
Write-Host "Você já pode iniciar o GrainSpeech Studio e usar o Upload MP3 + Treino Rápido." -ForegroundColor White
