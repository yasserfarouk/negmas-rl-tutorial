
REM Repository: projects/negmas-rl-tutorial
echo Restoring: projects/negmas-rl-tutorial
if exist "projects\negmas-rl-tutorial\.git" (
    echo   Directory already exists, skipping...
) else (
    REM Create parent directory if needed
    if not exist "projects" mkdir "projects"
    
    REM Clone the repository
    git clone "git@github.com:yasserfarouk/negmas-rl-tutorial.git" "projects\negmas-rl-tutorial"
    if errorlevel 1 (
        echo   Failed to clone
    ) else (
        echo   Successfully cloned
        
        REM Checkout the original branch if not already on it
        cd "projects\negmas-rl-tutorial"
        git checkout "main" 2>nul
        if errorlevel 1 (
            echo   Could not checkout branch: main
        ) else (
            echo   Checked out branch: main
        )
        cd ..\..
    )
)
echo.


REM Repository: projects/negmas-rl-tutorial
echo Restoring: projects/negmas-rl-tutorial
if exist "projects\negmas-rl-tutorial\.git" (
    echo   Directory already exists, skipping...
) else (
    REM Create parent directory if needed
    if not exist "projects" mkdir "projects"
    
    REM Clone the repository
    git clone "git@github.com:yasserfarouk/negmas-rl-tutorial.git" "projects\negmas-rl-tutorial"
    if errorlevel 1 (
        echo   Failed to clone
    ) else (
        echo   Successfully cloned
        
        REM Checkout the original branch if not already on it
        cd "projects\negmas-rl-tutorial"
        git checkout "main" 2>nul
        if errorlevel 1 (
            echo   Could not checkout branch: main
        ) else (
            echo   Checked out branch: main
        )
        cd ..\..
    )
)
echo.

