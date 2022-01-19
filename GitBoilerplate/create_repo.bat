@echo off

CHOICE /C:YN /M "Press Y for Yes, N for No"

IF ERRORLEVEL ==2 (
ECHO pressed N
GOTO :cancel
)

IF ERRORLEVEL ==1 (
ECHO pressed Y
GOTO :routine
)
ELSE (
GOTO :cancel
)

:routine

ECHO processing...

cd /d D:/
mkdir first_git_repo
cd D:/first_git_repo
git init --bare ../first_git_repo_local_server.git
git init
echo new_file  > text.txt
git add .
git commit -m "added text.txt"
git remote add origin D:/first_git_repo_local_server.git
git push --set-upstream origin master
git log


:cancel

ECHO finished...

pause
exit