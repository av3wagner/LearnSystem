#cs ----------------------------------------------------------------------------
 AutoIt Version: 3.3.10.2
 22.03.2025
 https://trycatchdebug.net/news/1280763/autoit-running-batch-files-with-arguments
 https://www.autoitscript.com/forum/topic/201867-bat-file-to-autoit-exe/
 https://trycatchdebug.net/news/1280763/autoit-running-batch-files-with-arguments
 https://www.shellhacks.com/run-batch-file-bat-on-startup-in-windows/
 ;Run(@ScriptDir & "\execute_python_file.bat","",@SW_HIDE)
 ;RunWait(@ComSpec & " /C " & @ScriptDir & "\execute_python_file.bat", "", @SW_SHOW)
 netstat -aon | find "8083"
#ce ----------------------------------------------------------------------------

ShellExecute(@ScriptDir & "\execute_streamlit.bat")

