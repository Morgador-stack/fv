@echo off
rem Faithful 12-image bench x2 loops (crash-night params: 300 layers / 100k samples / solid)
cd /d D:\WorkSpace\ZCWorkspace\forza-vinyl
set EXE=fv\target-l\release
set IMGS=refs\forza-painter-fh6\src\data\fh6_vinyl_resources\Vinyls
set OUT=bench\results\r3
if not exist %OUT% mkdir %OUT%
for %%L in (1 2 3 4 5 6 7 8 9 10 11 12) do (
  for %%R in (
    "Community_Vinyls_1\16.png c1"
    "Community_Vinyls_3\22.png c3"
    "Flames\14.png fl"
    "Gradient_Shapes\11.png gs"
    "Lower_Letters_1\10.png ll"
    "Nature\15.png na"
    "Paint_Splats\31.png ps"
    "Primitives\15.png pr"
    "Racing_Icons\3.png ri"
    "Stripes\24.png st"
    "Tribal\16.png tr"
    "Upper_Letters_1\14.png ul"
  ) do (
    for /f "tokens=1,2" %%A in (%%R) do (
      echo === loop%%L flatten %%B ===
      %EXE%\examples\flatten.exe %IMGS%\%%A %OUT%\%%B_flat.png
      if errorlevel 1 ( echo FAIL flatten %%B ) else (
        echo === loop%%L fit %%B ===
        %EXE%\fvfit-gpu.exe %OUT%\%%B_flat.png --layers 300 --samples 100000 --opt-res 328 --bg auto --out-prefix %OUT%\%%B
        if errorlevel 1 ( echo FAIL fit %%B ) else ( echo DONE %%B )
      )
    )
  )
)
echo BENCH_ALL_DONE
