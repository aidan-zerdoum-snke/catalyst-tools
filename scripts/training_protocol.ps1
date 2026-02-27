param(
  # Datasets you want to run (IDs from your nnUNet v2 setup)
  [int[]]$Datasets = @(312, 313, 314, 315, 316, 317, 318, 320),

  # Configurations to run
  [string[]]$Configs  = @("3d_fullres"),   # e.g., @("3d_fullres","3d_lowres","3d_cascade_fullres")

  # Plans identifier to keep topology consistent
  [string]$Plans      = "nnUNetResEncUNetMPlans",

  # Only 10 epochs (you can change if needed, but script assumes 10-epoch trainer)
  [int]$Epochs        = 10,

  # Trainer class for 10 epochs
  [string]$Trainer10  = "nnUNetTrainer_10epochs",

  # final vs best checkpoint for prediction
  [ValidateSet("final","best")] [string]$Checkpoint = "final",

  # resume training if interrupted
  [switch]$Resume,

  # Path pattern for testing folder relative to Dataset dir; can be absolute if preferred.
  # Expected layout: <testingDir>/images  and  <testingDir>/labels   (filenames match the image IDs)
  [string]$TestingSubdir = "testing",

  # Optional subfolder names under testing (override if your structure differs)
  [string]$TestingImagesSubdir = "images",
  [string]$TestingLabelsSubdir = "labels"
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
  $dsDir = Resolve-DatasetDir -DatasetId $D

  # --- testing directory (used for validation after training) ---
  $testingDir  = Join-Path $dsDir $TestingSubdir
  $imagesTest  = Join-Path $testingDir $TestingImagesSubdir
  $labelsTest  = Join-Path $testingDir $TestingLabelsSubdir

  if (-not (Test-Path $imagesTest)) { throw "Testing images not found: $imagesTest" }
  if (-not (Test-Path $labelsTest)) { Write-Warning "Testing labels not found: $labelsTest (predictions will still run, metrics are not computed here)." }

  foreach ($Config in $Configs) {
    $TR = $Trainer10
    Write-Host "=== Training Dataset $D | Config: $Config | 10 epochs (trainer: $TR) | fold: 0 ==="

    # Cascade guard: skip cascade training unless lowres trained first (same behavior as before)
    if ($Config -eq "3d_cascade_fullres") {
      $lowresGlob = ("Dataset{0:d3}*/${TR}__${Plans}__3d_lowres" -f $D)
      $lowresPath = Join-Path $env:nnUNet_results $lowresGlob
      if (-not (Get-ChildItem -Path $lowresPath -ErrorAction SilentlyContinue)) {
        Write-Warning "Skipping 3d_cascade_fullres: 3d_lowres not found for Dataset $D and trainer $TR."
        continue
      }
    }

    # --- Train only fold 0 ---
    $trainArgs = @($D, $Config, 0, "-p", $Plans, "-tr", $TR)
    if ($Resume) { $trainArgs += "--c" }
    nnUNetv2_train @trainArgs

    # --- Predict on the dedicated testing folder ---
    $predOut = Join-Path $dsDir ("predictions_testing/{0}/fold0_ep10" -f $Config)
    New-Item -ItemType Directory -Force -Path $predOut | Out-Null

    Write-Host "=== Predicting on testing set for Dataset $D | Config: $Config | to $predOut ==="
    $chkFile = "checkpoint_{0}.pth" -f $Checkpoint
    nnUNetv2_predict -i $imagesTest -o $predOut -d $D -c $Config -f 0 -p $Plans -chk $chkFile

    # --- Index line (for quick bookkeeping; paths for predictions only) ---
    $indexPath = Join-Path $dsDir "predictions_testing/INDEX.tsv"
    $indexRow  = "{0}`t{1}`t{2}`t{3}" -f $D, $Config, 0, $predOut
    Add-Content -Path $indexPath -Value $indexRow

    Write-Host "Done: D=$D | Config=$Config | epochs=10 | fold=0"
  }
}