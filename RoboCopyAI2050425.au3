;#!!!# https://www.tecchannel.de/a/robocopy-daten-schnell-und-einfach-unter-windows-sichern,2033515,2
  ;;;****************************************************************;;;
  ;;;***  MSD                                                     ***;;;
  ;;;***  ALEXANDER WAGNER                                        ***;;;
  ;;;***  STUDIEN-NAME   : BACKUP_ROBOCOPY                        ***;;;
  ;;;***  STUDIEN-NUMMER :                                        ***;;;
  ;;;***  SPONSOR        :                                        ***;;;
  ;;;***  ARBEITSBEGIN   : 01.01.2008                             ***;;;
  ;;;****************************************************************;;;
  ;;;*--------------------------------------------------------------*;;;
  ;;;*---  PROGRAMM      : RoboCopy_IPYNB2023                    ---*;;;
  ;;;*---  Parent        :  					                 ---*;;;
  ;;;*---  BESCHREIBUNG  : Sicherung Alle Dateien                ---*;;;
  ;;;*---                :                                       ---*;;;
  ;;;*---                :                                       ---*;;;
  ;;;*---  VERSION   VOM : 04.05.2023                            ---*;;;
  ;;;*--   KORREKTUR VOM : 05.07.2023                            ---*;;;
  ;;;*--                 :                                       ---*;;;
  ;;;*---  INPUT         : C:\IPYNB, H:\IPYNB                    ---*;;;
  ;;;*---                :                                       ---*;;;
  ;;;*---  OUTPUT        :                                       ---*;;;
  ;;;*--------------------------------------------------------------*;;;
  ;;;****************************************************************;;;
  ;;;*                                                              *;;;
  ;;;*                                                              *;;;
  ;;;*--------------------------------------------------------------*;;;
  ;;;*                                                              *;;;
  ;;;*--------------------------------------------------------------*;;;
  ;;;*---   DATUM        :       JOB                             ---*;;;
  ;;;*--------------------------------------------------------------*;;;
  ;;;*---  26.04.2023     : C:\IPNB-> D:                         ---*;;;
  ;;;*---  30.04.2023     : C:\IPYNBgesamt-> G:\BACKUM\IPYNB...  ---*;;;
  ;;;*---  04.05.2023     : C:\IPYNBgesamt\+IPYNB2023            ---*;;;
  ;;;*---  05.07.2023     : C:\IPYNBgesamt                       ---*;;;
  ;;;*---  16.07.2023     : C:\IPYNBgesamt                       ---*;;;
  ;;;*---  16.08.2023     : C:\IPYNBgesamt                       ---*;;;
  ;;;*---  25.04.2025     : C:\AsfendiyarovAI\LearnSystem        ---*;;;
  ;;;*--------------------------------------------------------------*;;;
  ;;;****************************************************************;;;


#include <WindowsConstants.au3>
#include <GuiConstantsEx.au3>
#include <Date.au3>
#include <array.au3>
#include <File.au3>
#include <String.au3>
#include <Excel.au3>

Global $usbdb_dir = 'C:\Robocopy\robocopy.exe'
Global $DateTime, $FileSize, $DATE, $PREFIX, $INPREFIX

$INPREFIX="C:\IPYNBgesamt2025\AProjekte\AktuellProjekte\LearnSystem"
;$PREFIX  ="C:\AsfendiyarovAI\LearnSystem"
$PREFIX  ="C:\AsfendiyarovAI\LearnSystem"

$sSystemDate = @YEAR & "/" & @MON & "/" & @MDAY
$day=string(@MDAY)
$mon=string(@MON)
$year=string(@YEAR)
Global $date= $year & $mon & $day
Global $oShell = ObjCreate("shell.application"), $FileCount = 1

MAIN($INPREFIX, $PREFIX, "C:\AsfendiyarovAI\Log" & $date & ".txt")

Func MAIN($Directory, $OutDir, $LogFile)
	DateTime()
	FileWriteLine($LogFile, "StartTime " & $DateTime & @CRLF)
	If FileExists($LogFile) Then
	   FileDelete($LogFile)
	EndIf

	;DirRemove($OutDir, 1)
	DirCreate($OutDir)

	$rob = RunWait(@ComSpec & ' /c C:\Robocopy\Robocopy "' & $Directory  & '" "' &  $OutDir & '" /S /NP /V /Tee /TS /XO /FP *.* /R:1 /W:1 /LOG+:"' & $Logfile & '"')
	DateTime()
	FileWriteLine($LogFile, "EndTime " & $DateTime & @CRLF)
EndFunc

Func DateTime()
	$day=string(@MDAY)
	$mon=string(@MON)
	$year=string(@YEAR)
	Local $Delim="-"
	Local $xTime=$Delim & @HOUR & $Delim & @Min & $Delim & @Sec
	Global $DateTime=$year & $mon & $day & $xTime
EndFunc

Func Datum()
	Local $Delim=":"
	Local $xTime=@HOUR & $Delim & @Min & $Delim & @Sec
	xDate()
	Global $Datum = $Date & " " & $xTime
EndFunc

Func xDate()
	$day=string(@MDAY)
	$mon=string(@MON)
	$year=string(@YEAR)
	Global $Date= $day & "." & $mon & "." & $year
EndFunc

