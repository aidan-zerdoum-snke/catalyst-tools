import slicer
from slicer.ScriptedLoadableModule import *
from slicer.i18n import tr as _, translate

import os, sys, traceback, tempfile
from pathlib import Path
import numpy as np
import qt, ctk
import vtk
import pandas as pd

from SlicerNNUNetLib import Widget

# --- BEGIN: Inline IJV CSA helper (no external imports needed beyond numpy/scipy) ---
import numpy as np
from scipy import ndimage
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

class IJVCSALogic:

    @staticmethod
    def compute_axial_cross_sections(mask: np.ndarray, spacing: np.ndarray):
        """
        Compute per-slice CSA and centroid for axial planes.
        mask is assumed from slicer.util.arrayFromVolume(...) -> shape (K, J, I).
        Returns:
          z_idx (P,) int   : axial slice indices K that contain foreground
          areas (P,) float : CSA per slice [mm^2]
          pts_ijk (P,3)    : centroids in IJK (i, j, k)
        """
        spacing = np.asarray(spacing, dtype=float)
        K = int(mask.shape[0])  # axial slices

        z_idx, areas, pts_ijk = [], [], []
        for k in range(K):
            sl = (mask[k, :, :] > 0).astype(np.uint8)  # (J, I)
            if sl.sum() == 0:
                continue

            # axial pixel area = sx * sy (I×J)
            area = float(sl.sum()) * float(spacing[0] * spacing[1])

            # center_of_mass on (rows=J, cols=I) -> (cy, cx) == (j, i)
            cy, cx = ndimage.center_of_mass(sl)
            if not np.all(np.isfinite([cx, cy])):
                continue

            z_idx.append(k)
            areas.append(area)
            # IJK = (i, j, k) = (cx, cy, k)
            pts_ijk.append([float(cx), float(cy), float(k)])

        if not z_idx:
            return np.array([], dtype=int), np.array([], dtype=float), np.empty((0, 3), dtype=float)

        return (
            np.asarray(z_idx, dtype=int),
            np.asarray(areas, dtype=float),
            np.asarray(pts_ijk, dtype=float),
        )
    
    @staticmethod
    def largest_component(mask: np.ndarray) -> np.ndarray:
        # Use scipy.ndimage.label to avoid skimage dependency.
        if mask.sum() == 0:
            return mask.astype(np.uint8, copy=False)
        lbl, num = ndimage.label(mask.astype(np.uint8), structure=np.ones((3,3,3), dtype=np.uint8))
        if num <= 1:
            return (lbl > 0).astype(np.uint8)
        # Keep the label with largest voxel count (exclude 0)
        counts = np.bincount(lbl.ravel())
        counts[0] = 0
        largest = counts.argmax()
        return (lbl == largest).astype(np.uint8)

    @staticmethod
    def compute_volume_ml(mask: np.ndarray, spacing: np.ndarray) -> float:
        voxel_mm3 = float(spacing[0] * spacing[1] * spacing[2])
        return float(int(mask.sum()) * voxel_mm3 / 1000.0)

    @staticmethod
    def skeleton_centerline(mask: np.ndarray) -> np.ndarray:
        """
        Simple axial fallback: per-slice COM across z. Returns Nx3 (x,y,z) voxel coords.
        """
        m = (mask > 0).astype(np.uint8)
        Z = m.shape[2]
        pts = []
        for z in range(Z):
            sl = m[:, :, z]
            if sl.sum() == 0:
                continue
            com = ndimage.center_of_mass(sl)
            if not np.all(np.isfinite(com)):
                continue
            x, y = com
            pts.append([x, y, float(z)])
        if not pts:
            return np.empty((0, 3), dtype=float)

        pts = np.array(pts, dtype=float)
        # prune big jumps
        if len(pts) > 1:
            deltas = np.diff(pts, axis=0)
            d = np.linalg.norm(deltas, axis=1)
            keep = np.insert(d < 10.0, 0, True)  # allow ~10 voxel jumps
            pts = pts[keep]
        return pts

    @staticmethod
    def resample_equal_arclength(points_vox: np.ndarray, spacing: np.ndarray,
                                 step_mm: float = 2.0,
                                 smooth_window: int = 9, smooth_poly: int = 3):
        """
        Return pts_vox_resampled (Mx3), t_mm (Mx3 tangents), s_mm (M,)
        """
        if len(points_vox) < 2:
            return points_vox, None, np.array([0.0] * max(len(points_vox), 1))

        pts_mm = points_vox.astype(float) * spacing
        deltas_mm = np.diff(pts_mm, axis=0)
        seg_len_mm = np.linalg.norm(deltas_mm, axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg_len_mm)])
        total_len = s[-1]
        if total_len < 1e-6:
            return points_vox[[0]], None, np.array([0.0])

        s_query = np.arange(0.0, total_len + 1e-6, step_mm)
        f_x = interp1d(s, pts_mm[:, 0], kind="linear")
        f_y = interp1d(s, pts_mm[:, 1], kind="linear")
        f_z = interp1d(s, pts_mm[:, 2], kind="linear")
        pts_mm_res = np.stack([f_x(s_query), f_y(s_query), f_z(s_query)], axis=1)

        # Smooth + compute tangents
        k = min(smooth_window, (len(pts_mm_res) // 2) * 2 - 1)  # odd <= len
        if k >= 5:
            px = savgol_filter(pts_mm_res[:, 0], window_length=k, polyorder=smooth_poly)
            py = savgol_filter(pts_mm_res[:, 1], window_length=k, polyorder=smooth_poly)
            pz = savgol_filter(pts_mm_res[:, 2], window_length=k, polyorder=smooth_poly)
            pts_mm_res = np.stack([px, py, pz], axis=1)

        t_mm = np.gradient(pts_mm_res, s_query, axis=0)
        norms = np.linalg.norm(t_mm, axis=1) + 1e-12
        t_mm = t_mm / norms[:, None]

        pts_vox_res = pts_mm_res / spacing
        return pts_vox_res, t_mm, s_query

    @staticmethod
    def _orthonormal_basis_perp(t_mm: np.ndarray):
        a = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(a, t_mm)) > 0.9:
            a = np.array([0.0, 1.0, 0.0])
        u = np.cross(t_mm, a)
        u /= (np.linalg.norm(u) + 1e-12)
        v = np.cross(t_mm, u)
        v /= (np.linalg.norm(v) + 1e-12)
        return u, v

    @staticmethod
    def sample_plane_area(mask: np.ndarray, spacing: np.ndarray, center_vox: np.ndarray, t_mm: np.ndarray,
                          plane_size_mm: float = 25.0, res_mm: float = 0.5, slab_thickness_mm: float = 0.0) -> float:
        """
        Sample mask on plane ⟂ t_mm through center_vox.
        """
        if t_mm is None or np.linalg.norm(t_mm) < 1e-6:
            return 0.0

        # Ensure numpy arrays for broadcasting
        spacing = np.asarray(spacing, dtype=float)
        center_vox = np.asarray(center_vox, dtype=float)
        t_mm = np.asarray(t_mm, dtype=float)

        u_mm, v_mm = IJVCSALogic._orthonormal_basis_perp(t_mm)
        half = plane_size_mm / 2.0
        xs = np.arange(-half, half + 1e-6, res_mm)
        ys = np.arange(-half, half + 1e-6, res_mm)
        X, Y = np.meshgrid(xs, ys, indexing="xy")
        H, W = X.shape

        n_slabs = max(1, int(round(slab_thickness_mm / max(res_mm, 1e-6))))
        offsets_t = np.linspace(-0.5 * slab_thickness_mm, 0.5 * slab_thickness_mm, n_slabs)

        sampled = np.zeros((H, W), dtype=np.uint8)
        for dt in offsets_t:
            mm_offsets = (X[..., None] * u_mm[None, None, :] +
                          Y[..., None] * v_mm[None, None, :] +
                          dt * t_mm[None, None, :])
            vox_offsets = mm_offsets / spacing[None, None, :]   # requires numpy spacing
            coords_vox = center_vox[None, None, :] + vox_offsets
            coords = [coords_vox[..., i].ravel() for i in range(3)]
            vals = ndimage.map_coordinates(mask.astype(float), coords, order=0, mode="nearest")
            vals = (vals.reshape(H, W) > 0).astype(np.uint8)
            sampled = np.maximum(sampled, vals)

        area_mm2 = sampled.sum() * (res_mm ** 2)
        return float(area_mm2)
  
    @staticmethod
    def compute_cross_sections(mask: np.ndarray, spacing: np.ndarray,
                               step_mm: float = 2.0,
                               plane_size_mm: float = 25.0,
                               res_mm: float = 0.5,
                               slab_thickness_mm: float = 0.0):
        spacing = np.asarray(spacing, dtype=float)  # <- critical

        cl_vox = IJVCSALogic.skeleton_centerline(mask)
        if len(cl_vox) < 2:
            com = np.array(ndimage.center_of_mass(mask)) if mask.sum() > 0 else np.array([0, 0, 0], dtype=float)
            s_mm = np.array([0.0])
            areas = np.array([
                IJVCSALogic.sample_plane_area(mask, spacing, com, np.array([0, 0, 1.0]),
                                              plane_size_mm, res_mm, slab_thickness_mm)
            ])
            return s_mm, areas, np.array([com])

        pts_vox_res, t_mm, s_mm = IJVCSALogic.resample_equal_arclength(cl_vox, spacing, step_mm=step_mm)

        areas = []
        for i in range(len(pts_vox_res)):
            t = t_mm[i] if t_mm is not None else np.array([0, 0, 1.0])
            a = IJVCSALogic.sample_plane_area(mask, spacing, pts_vox_res[i], t,
                                              plane_size_mm, res_mm, slab_thickness_mm)
            areas.append(a)

        return s_mm, np.array(areas), pts_vox_res

    @staticmethod
    def run_summary(mask: np.ndarray, spacing, step_mm=2.0, plane_size_mm=25.0, res_mm=0.5, slab_mm=0.0):
        """
        Returns a pandas DataFrame summary row.
        """
        import pandas as pd

        mask = np.asarray(mask).astype(np.uint8, copy=False)
        spacing = np.asarray(spacing, dtype=float)

        mask = IJVCSALogic.largest_component(mask)

        if mask.sum() == 0:
            return pd.DataFrame([{
                "volume_ml": 0.0,
                "mean_area_mm2": float('nan'),
                "std_area_mm2": float('nan'),
                "min_area_mm2": float('nan'),
                "max_area_mm2": float('nan'),
                "num_samples": 0,
                "spacing_x_mm": float(spacing[0]),
                "spacing_y_mm": float(spacing[1]),
                "spacing_z_mm": float(spacing[2]),
            }])

        volume_ml = IJVCSALogic.compute_volume_ml(mask, spacing)
        s_mm, areas_mm2, _ = IJVCSALogic.compute_cross_sections(
            mask, spacing,
            step_mm=step_mm, plane_size_mm=plane_size_mm, res_mm=res_mm, slab_thickness_mm=slab_mm
        )

        valid = areas_mm2[(areas_mm2 > 0) & np.isfinite(areas_mm2)]
        if valid.size:
            mean_area = float(valid.mean()); std_area = float(valid.std())
            min_area  = float(valid.min());  max_area = float(valid.max())
        else:
            mean_area = std_area = min_area = max_area = float('nan')

        return pd.DataFrame([{
            "volume_ml": float(volume_ml),
            "mean_area_mm2": mean_area,
            "std_area_mm2": std_area,
            "min_area_mm2": min_area,
            "max_area_mm2": max_area,
            "num_samples": int(len(areas_mm2)),
            "spacing_x_mm": float(spacing[0]),
            "spacing_y_mm": float(spacing[1]),
            "spacing_z_mm": float(spacing[2]),
        }])
# --- END: Inline IJV CSA helper ---


class SlicerNNUNet(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("nnUNet")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Segmentation")]
        self.parent.dependencies = []
        self.parent.contributors = ["Thibault Pelletier (Kitware SAS)"]
        self.parent.helpText = _(
            "This extension is meant to streamline the integration of nnUnet based models into 3D Slicer.<br>"
            "It allows for quick and reliable nnUNet dependency installation in 3D Slicer environment and provides"
            " simple logic to launch nnUNet prediction on given directories.<br><br>"
            "The installation steps are based on the work done in the "
            '<a href="https://github.com/lassoan/SlicerTotalSegmentator/">Slicer Total Segmentator extension</a>'
        )
        self.parent.acknowledgementText = _(
            "This module was originally co-financed by the "
            '<a href="https://orthodontie-ffo.org/">Fédération Française d\'Orthodontie</a> '
            "(FFO) as part of the "
            '<a href="https://github.com/gaudot/SlicerDentalSegmentator/">Dental Segmentator</a>'
            " developments and the "
            '<a href="https://rhu-cosy.com/en/accueil-english/">Cure Overgrowth Syndromes</a>'
            " (COSY) RHU Project."
        )


class SlicerNNUNetWidget(ScriptedLoadableModuleWidget):
    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        self.logic = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)
        widget = Widget()
        self.logic = widget.logic
        self.layout.addWidget(widget)

        # --- IJV Space Health add-on UI (below existing nnUNet widget) ---
        ijvBox = ctk.ctkCollapsibleButton()
        ijvBox.text = "IJV Space Health (experimental)"
        ijvLayout = qt.QFormLayout(ijvBox)

        # Segmentation selector
        self.ijvSegSelector = slicer.qMRMLNodeComboBox()
        self.ijvSegSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
        self.ijvSegSelector.selectNodeUponCreation = False
        self.ijvSegSelector.noneEnabled = True
        self.ijvSegSelector.addEnabled = False
        self.ijvSegSelector.removeEnabled = False
        self.ijvSegSelector.setMRMLScene(slicer.mrmlScene)
        self.ijvSegSelector.toolTip = "Select IJV segmentation to analyze"
        ijvLayout.addRow("Segmentation:", self.ijvSegSelector)

        # Reference ultrasound (or other input) volume selector
        self.ijvRefVolumeSelector = slicer.qMRMLNodeComboBox()
        self.ijvRefVolumeSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.ijvRefVolumeSelector.selectNodeUponCreation = False
        self.ijvRefVolumeSelector.noneEnabled = True
        self.ijvRefVolumeSelector.addEnabled = False
        self.ijvRefVolumeSelector.removeEnabled = False
        self.ijvRefVolumeSelector.setMRMLScene(slicer.mrmlScene)
        self.ijvRefVolumeSelector.toolTip = "Select the ultrasound input volume (reference geometry)"
        ijvLayout.addRow("Reference volume:", self.ijvRefVolumeSelector)

        # Output folder
        self.ijvOutputDir = ctk.ctkPathLineEdit()
        self.ijvOutputDir.filters = ctk.ctkPathLineEdit.Dirs
        self.ijvOutputDir.currentPath = slicer.app.defaultScenePath
        ijvLayout.addRow("Output folder:", self.ijvOutputDir)

        # Compute button
        self.ijvComputeBtn = qt.QPushButton("Compute IJV Cross‑Sections")
        self.ijvComputeBtn.toolTip = "Runs ijv_cross_sections.py on the selected segmentation and creates a Table (+ CSV)."
        self.ijvComputeBtn.clicked.connect(self.onComputeIJVCrossSections)
        ijvLayout.addRow(self.ijvComputeBtn)
        
        self.layout.addWidget(ijvBox)
        

    def onComputeIJVCrossSections(self):
        segNode = self.ijvSegSelector.currentNode()
        if not segNode:
            slicer.util.errorDisplay("Select a segmentation node first.")
            return

        # Reference volume preference
        refVolume = self.ijvRefVolumeSelector.currentNode()
        fallbackToSegGeom = refVolume is None

        # --- Pick the IJV segment robustly ---
        segmentation = segNode.GetSegmentation()
        ids = vtk.vtkStringArray()
        segmentation.GetSegmentIDs(ids)
        if ids.GetNumberOfValues() == 0:
            slicer.util.errorDisplay("No segments in this segmentation.")
            return

        # 1) Prefer a segment literally named "3" (your schema: IJV == 3)
        chosenSegmentID = None
        for i in range(ids.GetNumberOfValues()):
            sid = ids.GetValue(i)
            nm = segmentation.GetSegment(sid).GetName().strip().lower()
            if nm == "3":
                chosenSegmentID = sid
                break

        # 2) Otherwise, any name containing "ijv"
        if chosenSegmentID is None:
            for i in range(ids.GetNumberOfValues()):
                sid = ids.GetValue(i)
                nm = segmentation.GetSegment(sid).GetName().strip().lower()
                if "ijv" in nm or "internal jugular" in nm:
                    chosenSegmentID = sid
                    break

        # 3) If still not found, stop (do not silently use index 3 or "first segment")
        if chosenSegmentID is None:
            all_names = [segmentation.GetSegment(ids.GetValue(i)).GetName()
                        for i in range(ids.GetNumberOfValues())]
            slicer.util.errorDisplay(
                "Could not find IJV segment. Expected a segment named '3' or containing 'IJV'.\n\n"
                "Available segments:\n- " + "\n- ".join(all_names)
            )
            return

        segName = segmentation.GetSegment(chosenSegmentID).GetName()
        slicer.util.showStatusMessage(
            f"Computing IJV CSA on segment: '{segName}' (ID={chosenSegmentID})", 3000
        )

        # --- Get a pure binary mask for ONLY that segment in the correct grid ---
        # Use the reference volume geometry if provided, else segmentation geometry
        if fallbackToSegGeom:
            ijv_mask = slicer.util.arrayFromSegmentBinaryLabelmap(segNode, chosenSegmentID, None)
            refGridNode = segNode   # for transform attachment
        else:
            ijv_mask = slicer.util.arrayFromSegmentBinaryLabelmap(segNode, chosenSegmentID, refVolume)
            refGridNode = refVolume

        # Ensure binary and keep largest component
        ijv_mask = (ijv_mask > 0).astype(np.uint8)
        ijv_mask = IJVCSALogic.largest_component(ijv_mask)

        # Sanity: mask must be strictly {0,1}
        u = np.unique(ijv_mask)
        if not (len(u) <= 2 and set(u).issubset({0, 1})):
            slicer.util.errorDisplay(
                f"Selected segment should be binary but found label values: {u}.\n"
                "This indicates multiple structures were included; aborting."
            )
            return

        # Geometry from the reference grid
        tmpLM = None
        try:
            if fallbackToSegGeom:
                # Export to a temporary labelmap to read spacing/orientation in one shot
                tmpLM = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode', 'tmp_for_spacing')
                idsOne = vtk.vtkStringArray()
                idsOne.InsertNextValue(chosenSegmentID)
                slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
                    segNode, idsOne, tmpLM, None
                )
                spacing = np.asarray(tmpLM.GetSpacing(), dtype=float)
                ijkToRas = vtk.vtkMatrix4x4()
                tmpLM.GetIJKToRASMatrix(ijkToRas)
            else:
                spacing = np.asarray(refVolume.GetSpacing(), dtype=float)
                ijkToRas = vtk.vtkMatrix4x4()
                refVolume.GetIJKToRASMatrix(ijkToRas)

            # --- Axial per-slice CSA + centroids (viewer-aligned) ---
            z_idx, areas_mm2, pts_ijk = IJVCSALogic.compute_axial_cross_sections(ijv_mask, spacing)
            if len(z_idx) == 0:
                slicer.util.infoDisplay("No voxels found in the IJV segment after largest-component filtering.")
                return

            # IJK->RAS (local) converter
            def ijk_to_ras_local(p_ijk):
                v = [float(p_ijk[0]), float(p_ijk[1]), float(p_ijk[2]), 1.0]
                ras = [0.0, 0.0, 0.0, 0.0]
                ijkToRas.MultiplyPoint(v, ras)
                return [ras[0], ras[1], ras[2]]

            pts_ras_local = [ijk_to_ras_local(p) for p in pts_ijk]

            # ---- Build a single per-slice table (z-index, CSA, centroid in RAS local) ----
            full_df = pd.DataFrame({
                "slice_k": z_idx.astype(int),
                "csa_mm2": areas_mm2.astype(float),
                "ras_x": [float(p[0]) for p in pts_ras_local],
                "ras_y": [float(p[1]) for p in pts_ras_local],
                "ras_z": [float(p[2]) for p in pts_ras_local],
                "spacing_x_mm": spacing[0],
                "spacing_y_mm": spacing[1],
                "spacing_z_mm": spacing[2],
            })

            tableNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLTableNode",
                f"{segNode.GetName()} IJV CSA (Axial per-slice)"
            )
            self._dfToTable(tableNode, full_df)

            # ---- Markups: one point per slice at centroid, label with CSA ----
            csaFids = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsFiducialNode",
                f"{segNode.GetName()} IJV CSA (Axial centroids)"
            )
            # Attach the same transform as the reference grid node so everything stays aligned
            csaFids.SetAndObserveTransformNodeID(refGridNode.GetTransformNodeID())

            csaFidsDisp = csaFids.GetDisplayNode()
            if csaFidsDisp:
                csaFidsDisp.SetVisibility3D(True)
                csaFidsDisp.SetVisibility2D(True)
                csaFidsDisp.SetGlyphScale(1.2)  # slightly smaller
                csaFidsDisp.SetTextScale(1.1)
                csaFidsDisp.SetPointLabelsVisibility(True)

            for k, a_mm2, p_ras in zip(z_idx, areas_mm2, pts_ras_local):
                csaFids.AddControlPoint(p_ras)
                idx = csaFids.GetNumberOfControlPoints() - 1
                csaFids.SetNthControlPointLabel(idx, f"k={int(k)}: {a_mm2:.1f} mm²")

            # ---- Save CSV to the chosen output directory ----
            outDir = Path(self.ijvOutputDir.currentPath)
            try:
                outDir.mkdir(parents=True, exist_ok=True)
                csvPath = outDir / f"{segNode.GetName()}_IJV_CSA_axial_per_slice.csv"
                full_df.to_csv(csvPath, index=False)
            except Exception as save_e:
                slicer.util.warningDisplay(f"Failed to save CSV: {save_e}")

            # Make viewers show the same grid used for geometry (nice for QA)
            if not fallbackToSegGeom and refVolume is not None:
                lm = slicer.app.layoutManager()
                for viewName in ("Red", "Yellow", "Green"):
                    sv = lm.sliceWidget(viewName).sliceLogic()
                    sv.GetSliceCompositeNode().SetBackgroundVolumeID(refVolume.GetID())

            slicer.util.infoDisplay("IJV axial per-slice CSA complete.")

        except Exception as e:
            slicer.util.errorDisplay(
                f"Failed to compute IJV cross-sections.\n\n{e}\n\n{traceback.format_exc()}"
            )
        finally:
            # Clean up the temporary labelmap (if created)
            if 'tmpLM' in locals() and tmpLM is not None:
                slicer.mrmlScene.RemoveNode(tmpLM)

    def _dfToTable(self, tableNode, df):
        import pandas as pd
        import numpy as np
        from vtk import vtkFloatArray, vtkIntArray, vtkStringArray

        vtk_table = vtk.vtkTable()

        # Create VTK columns from pandas df
        for col in df.columns:
            series = df[col]
            if pd.api.types.is_integer_dtype(series):
                arr = vtkIntArray()
            elif pd.api.types.is_float_dtype(series):
                arr = vtkFloatArray()
            else:
                arr = vtkStringArray()

            arr.SetName(str(col))

            # Insert values
            for v in series.astype(object).tolist():
                if isinstance(arr, vtkIntArray):
                    arr.InsertNextValue(0 if v is None or v == "" else int(v))
                elif isinstance(arr, vtkFloatArray):
                    arr.InsertNextValue(np.nan if v is None or v == "" else float(v))
                else:
                    arr.InsertNextValue("" if v is None else str(v))

            vtk_table.AddColumn(arr)

        # Instead of SetTable(), modify underlying object
        tableNode.SetAndObserveTable(vtk_table)
        tableNode.Modified()

def onReload(self):
    """
    Customization of reload to allow reloading of the SlicerNNUNetLib files.
    """
    import imp

    packageName = "SlicerNNUNetLib"
    submoduleNames = ["Signal", "Parameter", "InstallLogic", "SegmentationLogic", "Widget"]
    f, filename, description = imp.find_module(packageName)
    package = imp.load_module(packageName, f, filename, description)
    for submoduleName in submoduleNames:
        print(f"Reloading {packageName}.{submoduleName}")
        f, filename, description = imp.find_module(submoduleName, package.__path__)
        try:
            imp.load_module(packageName + '.' + submoduleName, f, filename, description)
        finally:
            f.close()

    ScriptedLoadableModuleWidget.onReload(self)


class SlicerNNUNetTest(ScriptedLoadableModuleTest):
    def runTest(self):
        from pathlib import Path
        from SlicerNNUNetLib import InstallLogic

        try:
            from SlicerPythonTestRunnerLib import RunnerLogic, RunSettings, isRunningInTestMode
        except ImportError:
            slicer.util.warningDisplay("Please install SlicerPythonTestRunner extension to run the self tests.")
            return

        if InstallLogic().getInstalledNNUnetVersion() is None:
            slicer.util.warningDisplay("Please install nnUNet to run the self tests of this extension.")
            return

        currentDirTest = Path(__file__).parent.joinpath("Testing")
        results = RunnerLogic().runAndWaitFinished(
            currentDirTest,
            RunSettings(extraPytestArgs=RunSettings.pytestFileFilterArgs("*TestCase.py") + ["-m not slow"]),
            doRunInSubProcess=not isRunningInTestMode()
        )

        if results.failuresNumber:
            raise AssertionError(f"Test failed: \n{results.getFailingCasesString()}")

        slicer.util.delayDisplay(f"Tests OK. {results.getSummaryString()}")
