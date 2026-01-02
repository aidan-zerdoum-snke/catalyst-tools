
param(
  # Datasets you want to run (IDs from your nnUNet v2 setup)
  [int[]]$Datasets = @(501),            # e.g., @(501, 307)

  # nnU-Net configuration and plans
  [string]$Config   = "3d_fullres",
  [string]$Plans    = "nnUNetResEncUNetMPlans",

  # Epoch counts to test
  [int[]]$Epochs    = @(10, 20, 30, 40, 50),

  # Map epochs -> trainer class names (built-in or custom)
  [hashtable]$TrainerMap = @{
    5  = "nnUNetTrainer_5epochs"
    10 = "nnUNetTrainer_10epochs"
    20 = "nnUNetTrainer_20epochs"
    50 = "nnUNetTrainer_50epochs"
  },

  # Choose which checkpoint to use for prediction: final or best
  [ValidateSet("final","best")] [string]$Checkpoint = "final",

  # Resume interrupted training
  [switch]$Resume,

  # MedPy evaluator settings
  [string]$PythonExe   = "python",          # path to Python
  [string]$MedPyScript = "evaluate_medpy.py",
  [int[]]$Labels       = $null,             # e.g., @(1,2,3)
  [switch]$KeepLargestCC                     # apply largest-connected-component filter
)

# --- Helper: resolve dataset directory under nnUNet_raw ---
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

# --- Main loop ---
foreach ($D in $Datasets) {
  $dsDir = Resolve-DatasetDir -DatasetId $D
  $imagesTs = Join-Path $dsDir "imagesTs"
  $labelsTs = Join-Path $dsDir "labelsTs"
  if (-not (Test-Path $imagesTs)) { throw "imagesTs not found: $imagesTs" }
  if (-not (Test-Path $labelsTs)) { throw "labelsTs not found: $labelsTs" }

  foreach ($E in $Epochs) {
    if (-not $TrainerMap.ContainsKey($E)) {
      throw "No trainer mapped for ${E} epochs. Add an entry to -TrainerMap."
    }
    $TR = $TrainerMap[$E]
    Write-Host "=== Training Dataset $D | $Config | $E epochs (trainer: $TR) ==="

    $trainArgs = @($D, $Config, "all", "-p", $Plans, "-tr", $TR)
    if ($Resume) { $trainArgs += "--c" }   # resume from last checkpoint if available (v2)
    # Optional: save softmax at final validation for later ensembling/best-config workflows
    # $trainArgs += "--npz"   # not required for MedPy evaluation

    # Train all folds sequentially
    nnUNetv2_train @trainArgs

    # Predict test set to per-epoch folder
    $out = Join-Path $dsDir ("predictionsTs/ep{0}" -f $E)
    New-Item -ItemType Directory -Force -Path $out | Out-Null

    Write-Host "=== Predicting Dataset $D to $out ==="
    $chkFile = "checkpoint_{0}.pth" -f $Checkpoint  # final or best
    nnUNetv2_predict -i $imagesTs -o $out -d $D -c $Config -f all -p $Plans -chk $chkFile

    # Evaluate with your MedPy script
    Write-Host "=== MedPy evaluation (ep$E) ==="
    $pyArgs = @($MedPyScript, "--gt", $labelsTs, "--pred", $out, "--out", $out)
    if ($Labels) {
      $pyArgs += @("--labels")
      $pyArgs += $Labels
    }
    if ($KeepLargestCC) { $pyArgs += "--keep-largest-cc" }

    & $PythonExe @pyArgs

    # Index: append a compact row so you can spot outputs fast
    $csvOut   = Join-Path $out "all_metrics_per_case.csv"
    $jsonOut  = Join-Path $out "all_metrics_summary.json"
    $indexRow = "{0}`t{1}`t{2}`t{3}" -f $D, $E, $csvOut, $jsonOut
    Add-Content -Path (Join-Path $dsDir "predictionsTs/INDEX.tsv") -Value $indexRow

    Write-Host "Done: D=$D | epochs=$E"
    }
}
