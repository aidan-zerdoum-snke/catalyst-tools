
param(
  # Datasets you want to run (IDs from your nnUNet v2 setup)
  [int[]]$Datasets = @(501),

  # Configurations to run
  [string[]]$Configs  = @("3d_fullres"),   # e.g., @("3d_fullres","3d_lowres","3d_cascade_fullres")

  # Plans identifier to keep topology consistent
  [string]$Plans      = "nnUNetResEncUNetMPlans",

  # Epoch counts
  [int[]]$Epochs      = @(10,20,30,40,50),

  # Epoch -> trainer class mapping
  [hashtable]$TrainerMap = @{
    5  = "nnUNetTrainer_5epochs"           # if available; else use your custom 5-epoch trainer
    10 = "nnUNetTrainer_10epochs"
    20 = "nnUNetTrainer_20epochs"
    30 = "nnUNetTrainerV2_30epochs"        # custom
    40 = "nnUNetTrainerV2_40epochs"        # custom
    50 = "nnUNetTrainer_50epochs"
  },

  # final vs best checkpoint for prediction
  [ValidateSet("final","best")] [string]$Checkpoint = "final",

  # resume training if interrupted
  [switch]$Resume,

  # MedPy evaluator settings
  [string]$PythonExe   = "python",
  [string]$MedPyScript = "evaluate_medpy.py",
  [int[]]$Labels       = $null,             # e.g., @(1,2,3)
  [switch]$KeepLargestCC
)

function Resolve-DatasetDir([int]$DatasetId) {
  $rawRoot = $env:nnUNet_raw
  if ([string]::IsNullOrEmpty($rawRoot)) {
    throw "nnUNet_raw environment variable not set."
  }
  $dsGlob = ("Dataset{0:d3}*" -f $DatasetId)
  $dsDir = Get-ChildItem -Path $rawRoot -Directory -Filter $dsGlob | Select-Object -First 1
  if (-not $dsDir) { throw "Dataset folder not found for id $DatasetId at $rawRoot" }
  return $dsDir.FullName
}

foreach ($D in $Datasets) {
  $dsDir   = Resolve-DatasetDir -DatasetId $D
  $imagesTs = Join-Path $dsDir "imagesTs"
  $labelsTs = Join-Path $dsDir "labelsTs"
  if (-not (Test-Path $imagesTs)) { throw "imagesTs not found: $imagesTs" }
  if (-not (Test-Path $labelsTs)) { throw "labelsTs not found: $labelsTs" }

  foreach ($Config in $Configs) {
    foreach ($E in $Epochs) {
      if (-not $TrainerMap.ContainsKey($E)) {
        throw "No trainer mapped for ${E} epochs. Add an entry to -TrainerMap."
      }
      $TR = $TrainerMap[$E]
      Write-Host "=== Training Dataset $D | Config: $Config | $E epochs (trainer: $TR) ==="

      # Cascade guard: skip cascade training unless lowres trained first
      if ($Config -eq "3d_cascade_fullres") {
        $lowresPath = Join-Path $env:nnUNet_results ("Dataset{0:d3}*/${TR}__${Plans}__3d_lowres" -f $D)
        if (-not (Get-ChildItem -Path $lowresPath -ErrorAction SilentlyContinue)) {
          Write-Warning "Skipping 3d_cascade_fullres: 3d_lowres not found for Dataset $D and trainer $TR."
          continue
        }
      }

      $trainArgs = @($D, $Config, "all", "-p", $Plans, "-tr", $TR)
      if ($Resume) { $trainArgs += "--c" }
      nnUNetv2_train @trainArgs

      # Predict to per-epoch+config folder
      $out = Join-Path $dsDir ("predictionsTs/{0}/ep{1}" -f $Config, $E)
      New-Item -ItemType Directory -Force -Path $out | Out-Null

      Write-Host "=== Predicting Dataset $D | Config: $Config | to $out ==="
      $chkFile = "checkpoint_{0}.pth" -f $Checkpoint
      nnUNetv2_predict -i $imagesTs -o $out -d $D -c $Config -f all -p $Plans -chk $chkFile

      # MedPy evaluation
      Write-Host "=== MedPy evaluation (Config=$Config, ep=$E) ==="
      $pyArgs = @($MedPyScript, "--gt", $labelsTs, "--pred", $out, "--out", $out)
      if ($Labels) {
        $pyArgs += @("--labels")
        $pyArgs += $Labels
      }
      if ($KeepLargestCC) { $pyArgs += "--keep-largest-cc" }
      & $PythonExe @pyArgs

      # Index line
      $csvOut   = Join-Path $out "all_metrics_per_case.csv"
      $jsonOut  = Join-Path $out "all_metrics_summary.json"
      $index    = Join-Path $dsDir "predictionsTs/INDEX.tsv"
      $indexRow = "{0}`t{1}`t{2}`t{3}`t{4}" -f $D, $Config, $E, $csvOut, $jsonOut
      Add-Content -Path $index -Value $indexRow

      Write-Host "Done: D=$D | Config=$Config | epochs=$E"
    }
  }
}
``
