@echo off
echo Terminating user applications using the GPU...
echo WARNING: Terminating system processes can cause instability. Only terminate user applications.
echo Close applications normally if possible to avoid data loss.

:: List of process names to terminate (adjust as needed)
set "processes=Notion Calendar.exe Slack.exe brave.exe Code.exe WindowsTerminal.exe OneDrive.exe Muse.exe msedgewebview2.exe ArmouryCrate.exe"

:: Terminate each process
for %%p in (%processes%) do (
    taskkill /F /IM "%%p"
)

echo All specified processes have been terminated.
pause