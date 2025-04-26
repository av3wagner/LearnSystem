#include <GUIConstantsEx.au3>

; Create an array to hold the ControlIDs of the MenuItems
Global $aMenuItem[3]

$hGUI = GUICreate("Test", 500, 500)

; Create the menu
$mMenu = GUICtrlCreateMenu("File")
For $i = 0 To 2
    $aMenuItem[$i] = GUICtrlCreateMenuItem("Item " & $i, $mMenu)
Next

; Clicking this adds an item to the menu
$hButton = GUICtrlCreateButton("Add Menu Item", 10, 10, 80, 30)

GUISetState()

While 1

    $iMsg = GUIGetMsg()
    Switch $iMsg
        Case $GUI_EVENT_CLOSE
            Exit
        Case $hButton
            ; Resize the array
            $iCount = UBound($aMenuItem)
            ReDim $aMenuItem[$iCount + 1]
            ; And add the new item ControlID
            $aMenuItem[$iCount] = GUICtrlCreateMenuItem("Item " & $iCount, $mMenu)
        Case Else
            ; Run through the array to see if we have a match
            For $i = 0 To UBound($aMenuItem) - 1
                If $iMsg = $aMenuItem[$i] Then
                    ; We do, so read the text of the menu item
                    $sText = GUICtrlRead($iMsg, 1)
                    ; And display it
                    ConsoleWrite($sText & @CRLF)
                    ; No point in looking any further
                    ExitLoop
                EndIf
            Next
    EndSwitch

WEnd