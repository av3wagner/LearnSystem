#include <GuiMenu.au3>
#include <WinAPIError.au3>
#include <WindowsConstants.au3>
#include <GUIConstantsEx.au3>


GUIRegisterMsg($WM_CONTEXTMENU, "_WM_CONTEXTMENU")
GUIRegisterMsg($WM_COMMAND, "_WM_COMMAND")

Global $idMenuItem = 10000, $counter = 0
Global $hGUI = GUICreate("Menu test", 400, 300)
Global $idListview = GUICtrlCreateListView("", 2, 2, 396, 200)
Global $hListview = GUICtrlGetHandle($idListview)
GUISetState(@SW_SHOW)

Global $hMenuContext = _GUICtrlMenu_CreatePopup($MNS_MODELESS)
_GUICtrlMenu_AddMenuItem($hMenuContext, "Menu item", $idMenuItem)


Global $hTimer = TimerInit()
Do
    If TimerDiff($hTimer)>200 Then
        $counter += 1
         ConsoleWrite($counter & " " )
         $hTimer = TimerInit()
    EndIf
Until GUIGetMsg() = $GUI_EVENT_CLOSE

Exit

Func _WM_CONTEXTMENU($hWnd, $iMsg, $wParam, $lParam)
    #forceref $hWnd, $iMsg, $lParam
    Switch $hWnd
        Case $hGUI
            Switch $wParam
                Case $hListview
                    _GUICtrlMenu_TrackPopupMenu($hMenuContext, $hWnd)
            EndSwitch
    EndSwitch
EndFunc

Func _WM_COMMAND($hWnd, $iMsg, $wParam, $lParam)
    #forceref $hWnd, $iMsg, $lParam
    Switch $wParam
        Case $idMenuItem
            ConsoleWrite( @CRLF & "_GUICtrlMenu_TrackPopupMenu menu item clicked = " & $idMenuItem & @CRLF)
    EndSwitch
    Return $GUI_RUNDEFMSG
EndFunc