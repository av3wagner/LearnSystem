#include <GUIConstantsEx.au3>
#include <GuiMenu.au3>

$hGui = GUICreate('Read Menu Item', 400, 300)
$mnuFile = GUICtrlCreateMenu('&File')
$mnuFileExit = GUICtrlCreateMenuItem('Exit', $mnuFile)
GUISetState()

; read the text of the menu item
$hMenu = _GUICtrlMenu_GetMenu($hGui)
$sText = _GUICtrlMenu_GetItemText($hMenu, $mnuFileExit, False)
MsgBox(0, 'Menu item text', $sText)

While 1
    $msg = GUIGetMsg()
	$sText = _GUICtrlMenu_GetItemText($hMenu, $mnuFileExit, False)
    MsgBox(0, 'Menu item text', $sText)
    ;If $msg = $GUI_EVENT_CLOSE Or $msg = $mnuFileExit Then ExitLoop
WEnd