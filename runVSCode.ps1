& 'C:/Softwares/Microsoft/Visual Studio/2022/BuildTools/Common7/Tools/Launch-VsDevShell.ps1' -Arch amd64 -HostArch amd64 -SkipAutomaticLocation

chcp 65001 > $null

Start-Process code -ArgumentList "." -WindowStyle Hidden

exit