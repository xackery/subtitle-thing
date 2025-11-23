@echo off
setlocal

REM Get current directory
pushd "%~dp1"
set "CURDIR=%cd%"
for %%F in ("%~1") do set "CURFILE=%%~nxF"
echo Current directory: %CURDIR%
echo Current file: %CURFILE%

REM Run Docker command
docker run --rm --gpus all ^
    -v "%CURDIR%:/data" ^
    systran-faster-whisper:latest ^
    "/data/%CURFILE%" ^
    --model-size large-v3 ^
    --device cuda ^
    --compute-type float16 ^
    --language en ^
    --output-dir /data
pause