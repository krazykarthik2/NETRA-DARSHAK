REM delete folder apk 
rmdir /s /q apk/netradarshak/src
REM create folder apk
mkdir apk/netradarshak/src
REM copy src folder content to apk folder
xcopy src apk/netradarshak/src /s /e
REM add a bin folder to apk folder
cd apk/netradarshak
briefcase build android