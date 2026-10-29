@echo off
rem Faithful fc_full x3 back-to-back (crash-night command, restored workspace paths)
cd /d D:\WorkSpace\ZCWorkspace\forza-vinyl\fv
set EXE=target-l\release\fvfit-gpu.exe
set IMG=..\refs\forza-painter-fh6\assets\imgs\ayylmao.png
set R=..\bench\results\r3
if not exist %R% mkdir %R%
setlocal enabledelayedexpansion
for %%N in (1 2 3 4) do (
  echo === fc_full run %%N start %date% %time% ===
  %EXE% %IMG% --resume ..\bench\results\r3\fc_f2000.json --layers 3000 --auto-soft --samples 1000000 --opt-res 328 --bg auto --out-prefix %R%\fc_full
  echo === fc_full run %%N exit=!errorlevel! %time% ===
)
echo FC_ALL_DONE
