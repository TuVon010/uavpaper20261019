param(
  [Parameter(Mandatory=$true)][string]$InputPptx,
  [Parameter(Mandatory=$true)][string]$OutputStem
)
$ErrorActionPreference='Stop'
$pptApplication=$null
$pptDeck=$null
try {
  $pptApplication=New-Object -ComObject PowerPoint.Application
  $pptDeck=$pptApplication.Presentations.Open($InputPptx,-1,0,0)
  Write-Output ('Native PowerPoint opened: slides={0}, objects={1}' -f $pptDeck.Slides.Count,$pptDeck.Slides.Item(1).Shapes.Count)
  $pptDeck.Slides.Item(1).Export(($OutputStem+'.png'),'PNG',2880,2200)
  $pptDeck.SaveAs(($OutputStem+'.pdf'),32)
  Write-Output 'PNG and PDF exported'
} finally {
  if($null -ne $pptDeck){
    $pptDeck.Close()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($pptDeck) | Out-Null
  }
  if($null -ne $pptApplication){
    if($pptApplication.Presentations.Count -eq 0){$pptApplication.Quit()}
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($pptApplication) | Out-Null
  }
}
