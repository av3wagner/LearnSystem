#include <GUIConstantsEx.au3>
#include <WindowsConstants.au3>

Global $ctrl_start, $ctrl_end

#Region ### START Koda GUI section ### Form=
$Settings = GUICreate("Settings", 615, 438, 192, 124)
$mFile = GUICtrlCreateMenu("&File")
$mNew = GUICtrlCreateMenuItem("&New", $mFile)
GUICtrlSetState(-1, $GUI_DEFBUTTON)
$mOpen = GUICtrlCreateMenuItem("&Open", $mFile)
$mSave = GUICtrlCreateMenuItem("&Save", $mFile)
$mSaveas = GUICtrlCreateMenu("&Save as", $mFile, 1)
GUICtrlCreateMenuItem("", $mFile, 4)
$mExit = GUICtrlCreateMenuItem("&Exit", $mFile)
$mABS = GUICtrlCreateMenu("&ABS")
$mEndpoint_ABS = GUICtrlCreateMenu("&Endpoint", $mABS, 1)
$mWavelengths_ABS = GUICtrlCreateMenuItem("Wavelenghts", $mEndpoint_ABS)
$PlateType_ABS = GUICtrlCreateMenuItem("Plate Type", $mEndpoint_ABS)
$ReadArea_ABS = GUICtrlCreateMenuItem("Read Area", $mEndpoint_ABS)
$Shake_ABS = GUICtrlCreateMenuItem("Shake", $mEndpoint_ABS)
$SpeedRead_ABS = GUICtrlCreateMenuItem("Speed Read", $mEndpoint_ABS)
$MoreSettings_ABS = GUICtrlCreateMenuItem("More Settings", $mEndpoint_ABS)
$mKinetic_ABS = GUICtrlCreateMenu("&Kinetic", $mABS, 1)
$mWellScan_ABS = GUICtrlCreateMenu("&Well Scan", $mABS, 1)
$mSpectrum_ABS = GUICtrlCreateMenu("&Spectrum", $mABS, 1)
$mFRET = GUICtrlCreateMenu("&FRET")
$mEndpoint_FRET = GUICtrlCreateMenu("&Endpoint", $mFRET, 1)
$mWavelengths_FRET = GUICtrlCreateMenuItem("Wavelenghts", $mEndpoint_FRET)
$PlateType_FRET = GUICtrlCreateMenuItem("Plate Type", $mEndpoint_FRET)
$ReadArea_FRET = GUICtrlCreateMenuItem("Read Area", $mEndpoint_FRET)
$PTMuOptics_FRET = GUICtrlCreateMenuItem("PTM and Optics", $mEndpoint_FRET)
$Shake_FRET = GUICtrlCreateMenuItem("Shake", $mEndpoint_FRET)
$MoreSettings_FRET = GUICtrlCreateMenuItem("More Settings", $mEndpoint_FRET)
$mKinetic_FRET = GUICtrlCreateMenu("&Kinetic", $mFRET, 1)
$mWellScan_FRET = GUICtrlCreateMenu("&Well Scan", $mFRET, 1)
$mLUM = GUICtrlCreateMenu("&LUM")
$mEndpoint_LUM = GUICtrlCreateMenu("&Endpoint", $mLUM, 1)
$mWavelengths_LUM = GUICtrlCreateMenuItem("Wavelenghts", $mEndpoint_LUM)
$PlateType_LUM = GUICtrlCreateMenuItem("Plate Type", $mEndpoint_LUM)
$ReadArea_LUM = GUICtrlCreateMenuItem("Read Area", $mEndpoint_LUM)
$PTMuOptics_LUM = GUICtrlCreateMenuItem("PTM and Optics", $mEndpoint_LUM)
$Shake_LUM = GUICtrlCreateMenuItem("Shake", $mEndpoint_LUM)
$MoreSettings_LUM = GUICtrlCreateMenuItem("More Settings", $mEndpoint_LUM)
$mKinetic_LUM = GUICtrlCreateMenu("&Kinetic", $mLUM, 1)
$mWellScan_LUM = GUICtrlCreateMenu("&Well Scan", $mLUM, 1)
$mSpectrum_LUM = GUICtrlCreateMenu("&Spectrum", $mLUM, 1)
$mFragezeichen = GUICtrlCreateMenu("&?")
$mHelp_Fragezeichen = GUICtrlCreateMenuItem("Help" & @TAB & "F1", $mFragezeichen)
$Button_Cancel = GUICtrlCreateButton("Cancel", 450, 370, 70, 20)
$Button_OK = GUICtrlCreateButton("OK", 530, 370, 70, 20)
GUICtrlSetState(-1, $GUI_FOCUS)
GUISetState(@SW_SHOW)
#EndRegion ### END Koda GUI section ###

Func Wavelengths()
    DeletePrevCtrls()
    $ctrl_start = GUICtrlCreateDummy()
    $Label_Wavelenghts = GUICtrlCreateLabel("Wavelength Settings", 10, 10, 170, 20)
    $Label_NumberW = GUICtrlCreateLabel("Number of Wavelenghts:", 10, 40, 170, 20)
    $Input_Wavelenghts = GUICtrlCreateInput("", 150, 37, 70, 20)
    GUICtrlCreateUpdown($Input_Wavelenghts)
    $Label_Lm1 = GUICtrlCreateLabel("Lm1", 120, 70, 20, 20)
    $Input_Lm1 = GUICtrlCreateInput("", 150, 67, 70, 20)
    $Label_nm = GUICtrlCreateLabel("nm", 230, 70, 70, 20)
    $ctrl_end = GUICtrlCreateDummy()
EndFunc   ;==>Wavelengths
Func PlateTyp()
    DeletePrevCtrls()
    $ctrl_start = GUICtrlCreateDummy()
    $Label_PlateType = GUICtrlCreateLabel("Plate Type Settings", 10, 10, 170, 20)
    $Label_NumberP = GUICtrlCreateLabel("Number of Plates:", 10, 40, 170, 20)
    $Input_NumberP = GUICtrlCreateInput("", 120, 37, 70, 20)
    $ctrl_end = GUICtrlCreateDummy()
EndFunc   ;==>PlateTyp
Func ReadArea()
    DeletePrevCtrls()
    $ctrl_start = GUICtrlCreateDummy()
    $Label_ReadArea = GUICtrlCreateLabel("Read Area Settings", 10, 10, 170, 20)
    $Checkbox_allWells = GUICtrlCreateCheckbox("Select All", 500, 10, 97, 17)
    Local $Pic_Wellplate = GUICtrlCreatePic("C:\Users\Admin\Pictures\Wellplatepic.jpg", 70, 40, 467, 320)
    $ctrl_end = GUICtrlCreateDummy()
    GUISetState(@SW_SHOW)
EndFunc   ;==>ReadArea
Func PTMuOptics()
    DeletePrevCtrls()
    $ctrl_start = GUICtrlCreateDummy()
    $Label_PTMuOptics = GUICtrlCreateLabel("PTM and Optics Settings", 10, 10, 170, 20)
    $ctrl_end = GUICtrlCreateDummy()
EndFunc   ;==>PTMuOptics
Func Shake()
    DeletePrevCtrls()
    $ctrl_start = GUICtrlCreateDummy()
    $Label_Shake = GUICtrlCreateLabel("Shake Settings", 10, 10, 170, 20)
    $Label_Shake_Question = GUICtrlCreateLabel("Should the plate be shaken?", 10, 40, 170, 20)
    $Checkbox_Shake_Yes = GUICtrlCreateCheckbox("Yes", 160, 37, 170, 20)
    $Checkbox_Shake_No = GUICtrlCreateCheckbox("No", 250, 37, 170, 20)
    $Label_Shakespeed = GUICtrlCreateLabel("Shaking speed:", 10, 70, 170, 20)
    $Input_Shakespeed = GUICtrlCreateInput("", 120, 70, 70, 20)
    $ctrl_end = GUICtrlCreateDummy()
EndFunc   ;==>Shake

Func DeletePrevCtrls()
    For $i = $ctrl_start To $ctrl_end
        GUICtrlDelete($i)
    Next
EndFunc

While 1 ; FEHLER: Ausgeführte FUnktionen schließen sich nicht und überlagern die Oberfläche
    $nMsg = GUIGetMsg()
    Switch $nMsg
        Case $GUI_EVENT_CLOSE, $Button_Cancel, $mExit
            Exit

        Case $mOpen ; AUSSTEHEND: "OK" Button verlinken
            FileOpenDialog(" Open some File...", @WindowsDir, "All(*.*)")
        Case $mWavelengths_ABS
            Wavelengths()
        Case $mWavelengths_FRET
            Wavelengths()
        Case $mWavelengths_LUM
            Wavelengths()

        Case $PlateType_ABS
            PlateTyp()
        Case $PlateType_FRET
            PlateTyp()
        Case $PlateType_LUM
            PlateTyp()

        Case $ReadArea_ABS
            ReadArea()
        Case $ReadArea_FRET
            ReadArea()
        Case $ReadArea_LUM
            ReadArea()

        Case $PTMuOptics_FRET
            PTMuOptics()
        Case $PTMuOptics_LUM
            PTMuOptics()

        Case $Shake_ABS
            Shake()
        Case $Shake_FRET
            Shake()
        Case $Shake_LUM
            Shake()
            Exit
    EndSwitch
WEnd