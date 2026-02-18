import slicer
from slicer.ScriptedLoadableModule import *
from slicer.i18n import tr as _, translate
from SlicerNNUNetLib import Widget

import numpy as np
import pandas as pd
from scipy import ndimage

import qt, ctk
import vtk
import traceback
from pathlib import Path


# -----------------------------
# Logic helpers
# -----------------------------
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
        if mask.sum() == 0:
            return mask.astype(np.uint8, copy=False)
        lbl, num = ndimage.label(mask.astype(np.uint8), structure=np.ones((3, 3, 3), dtype=np.uint8))
        if num <= 1:
            return (lbl > 0).astype(np.uint8)
        counts = np.bincount(lbl.ravel())
        counts[0] = 0
        largest = counts.argmax()
        return (lbl == largest).astype(np.uint8)


# -----------------------------
# Module shell
# -----------------------------
class SlicerNNUNet(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("nnUNet")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Segmentation")]
        self.parent.dependencies = []
        self.parent.contributors = ["Thibault Pelletier (Kitware SAS)"]


# -----------------------------
# Widget
# -----------------------------
class SlicerNNUNetWidget(ScriptedLoadableModuleWidget):
    def __init__(self, parent=None) -> None:
        ScriptedLoadableModuleWidget.__init__(self, parent)
        self.logic = None

        # Per-run state for slice annotations
        self._metricsByK = {}              # k -> dict with ijv/cca/angle
        self._summaryText = ""             # upper-left
        self._annotationObservers = []     # [(sliceNode, tag), ...]
        self._annotationGridNode = None    # volume/labelmap to convert RAS->IJK(k)

        # Center-of-segment CSA labels (Markups)
        self._ijvFids = None
        self._ccaFids = None

    # -----------------------------
    # UI setup
    # -----------------------------
    def setup(self) -> None:
        ScriptedLoadableModuleWidget.setup(self)
        widget = Widget()
        self.logic = widget.logic
        self.layout.addWidget(widget)

        ijvBox = ctk.ctkCollapsibleButton()
        ijvBox.text = "IJV Space Health (experimental)"
        ijvLayout = qt.QFormLayout(ijvBox)

        # Segmentation selector
        self.segSelector = slicer.qMRMLNodeComboBox()
        self.segSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
        self.segSelector.selectNodeUponCreation = False
        self.segSelector.noneEnabled = True
        self.segSelector.addEnabled = False
        self.segSelector.removeEnabled = False
        self.segSelector.setMRMLScene(slicer.mrmlScene)
        self.segSelector.toolTip = "Select segmentation containing IJV and CCA segments"
        ijvLayout.addRow("Segmentation:", self.segSelector)

        # Reference volume selector (optional)
        self.refVolumeSelector = slicer.qMRMLNodeComboBox()
        self.refVolumeSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.refVolumeSelector.selectNodeUponCreation = False
        self.refVolumeSelector.noneEnabled = True
        self.refVolumeSelector.addEnabled = False
        self.refVolumeSelector.removeEnabled = False
        self.refVolumeSelector.setMRMLScene(slicer.mrmlScene)
        self.refVolumeSelector.toolTip = "Optional reference volume to define geometry and k-indexing (recommended)"
        ijvLayout.addRow("Reference volume:", self.refVolumeSelector)

        # Segment name inputs (user-specified)
        self.ijvSegmentEdit = qt.QLineEdit()
        self.ijvSegmentEdit.setPlaceholderText("e.g. IJV or 3")
        self.ijvSegmentEdit.toolTip = "IJV segment name (case-insensitive exact match) OR SegmentID"
        ijvLayout.addRow("IJV segment:", self.ijvSegmentEdit)

        self.ccaSegmentEdit = qt.QLineEdit()
        self.ccaSegmentEdit.setPlaceholderText("e.g. CCA or 2")
        self.ccaSegmentEdit.toolTip = "CCA segment name (case-insensitive exact match) OR SegmentID"
        ijvLayout.addRow("CCA segment:", self.ccaSegmentEdit)

        # Side selector
        self.sideCombo = qt.QComboBox()
        self.sideCombo.addItems(["Left", "Right"])
        self.sideCombo.toolTip = (
            "Angle reference:\n"
            "- Left: angle from Vector.right (+X in RAS)\n"
            "- Right: angle from Vector.left (-X in RAS)\n\n"
            "Convention:\n"
            "- CCW is positive\n"
            "- CW is negative"
        )
        ijvLayout.addRow("Side:", self.sideCombo)

        # Output folder
        self.outputDir = ctk.ctkPathLineEdit()
        self.outputDir.filters = ctk.ctkPathLineEdit.Dirs
        self.outputDir.currentPath = slicer.app.defaultScenePath
        ijvLayout.addRow("Output folder:", self.outputDir)

        # Compute button
        self.computeBtn = qt.QPushButton("Compute IJV/CCA CSA + Angle")
        self.computeBtn.toolTip = "Computes per-slice CSA+centroids for IJV & CCA, plus angle between them."
        self.computeBtn.clicked.connect(self.onCompute)
        ijvLayout.addRow(self.computeBtn)

        self.layout.addWidget(ijvBox)

    # -----------------------------
    # Segment selection helpers
    # -----------------------------
    def _listSegments(self, segmentation, ids: vtk.vtkStringArray):
        items = []
        for i in range(ids.GetNumberOfValues()):
            sid = ids.GetValue(i)
            nm = segmentation.GetSegment(sid).GetName()
            items.append(f"{nm} (ID={sid})")
        return items

    def _findSegmentIdByNameOrId(self, segmentation, ids: vtk.vtkStringArray, userText: str):
        """
        Accepts either:
          - a SegmentID (exact match), or
          - a Segment Name (case-insensitive exact match)
        Returns segmentID or None.
        """
        if not userText:
            return None
        key = userText.strip()
        if not key:
            return None

        # 1) exact match on SegmentID
        for i in range(ids.GetNumberOfValues()):
            sid = ids.GetValue(i)
            if sid == key:
                return sid

        # 2) case-insensitive exact match on name
        key_l = key.lower()
        for i in range(ids.GetNumberOfValues()):
            sid = ids.GetValue(i)
            nm = segmentation.GetSegment(sid).GetName()
            if nm and nm.strip().lower() == key_l:
                return sid

        return None

    # -----------------------------
    # RAS/IJK helpers
    # -----------------------------
    def _makeIJKToRASConverter(self, ijkToRas: vtk.vtkMatrix4x4):
        def ijk_to_ras(p_ijk):
            v4 = [float(p_ijk[0]), float(p_ijk[1]), float(p_ijk[2]), 1.0]
            ras4 = [0.0, 0.0, 0.0, 0.0]
            ijkToRas.MultiplyPoint(v4, ras4)
            return np.array([ras4[0], ras4[1], ras4[2]], dtype=float)
        return ijk_to_ras

    def _angleFromRefDeg(self, v_xy: np.ndarray, ref_xy: np.ndarray) -> float:
        """
        Signed angle from ref -> v in XY plane, degrees in (-180, +180].
        Convention:
          - CCW is positive
          - CW is negative
        """
        v = np.asarray(v_xy, dtype=float)
        r = np.asarray(ref_xy, dtype=float)

        nv = np.linalg.norm(v)
        nr = np.linalg.norm(r)
        if nv < 1e-9 or nr < 1e-9:
            return float("nan")

        v = v / nv
        r = r / nr

        dot = float(r[0] * v[0] + r[1] * v[1])
        cross = float(r[0] * v[1] - r[1] * v[0])

        ang = float(np.degrees(np.arctan2(cross, dot)))  # in [-180, 180]
        if ang <= -180.0:
            ang += 360.0
        return ang

    # -----------------------------
    # Center-of-segment CSA labels (Markups)
    # -----------------------------
    def _removeCenterLabelNodes(self):
        for n in (self._ijvFids, self._ccaFids):
            if n is not None and slicer.mrmlScene.IsNodePresent(n):
                slicer.mrmlScene.RemoveNode(n)
        self._ijvFids = None
        self._ccaFids = None

    def _createCenterLabelNodes(self, baseName: str, transformNodeID: str):
        self._ijvFids = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsFiducialNode", f"{baseName} IJV CSA @ COM"
        )
        self._ccaFids = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsFiducialNode", f"{baseName} CCA CSA @ COM"
        )

        for node in (self._ijvFids, self._ccaFids):
            node.CreateDefaultDisplayNodes()
            node.SetAndObserveTransformNodeID(transformNodeID)

            disp = node.GetDisplayNode()
            if disp:
                disp.SetVisibility(True)
                disp.SetVisibility2D(True)
                disp.SetVisibility3D(False)
                disp.SetGlyphScale(1.5)
                disp.SetTextScale(1.2)
                disp.SetPointLabelsVisibility(True)

    def _populateCenterLabelNodes(self, rows):
        if self._ijvFids is None or self._ccaFids is None:
            return

        for r in rows:
            pIJV = [float(r["ijv_ras_x"]), float(r["ijv_ras_y"]), float(r["ijv_ras_z"])]
            pCCA = [float(r["cca_ras_x"]), float(r["cca_ras_y"]), float(r["cca_ras_z"])]

            self._ijvFids.AddControlPoint(pIJV)
            ijvIdx = self._ijvFids.GetNumberOfControlPoints() - 1
            self._ijvFids.SetNthControlPointLabel(ijvIdx, f"{r['ijv_csa_mm2']:.1f} mm²")

            self._ccaFids.AddControlPoint(pCCA)
            ccaIdx = self._ccaFids.GetNumberOfControlPoints() - 1
            self._ccaFids.SetNthControlPointLabel(ccaIdx, f"{r['cca_csa_mm2']:.1f} mm²")

    # -----------------------------
    # Slice annotation management
    # -----------------------------
    def _clearSliceAnnotationsAndObservers(self):
        # Remove old observers
        for sliceNode, tag in self._annotationObservers:
            try:
                sliceNode.RemoveObserver(tag)
            except Exception:
                pass
        self._annotationObservers = []
        self._annotationGridNode = None

        # Remove old center labels
        self._removeCenterLabelNodes()

        # Clear all annotations
        lm = slicer.app.layoutManager()
        if not lm:
            return
        for viewName in ("Red", "Yellow", "Green"):
            sw = lm.sliceWidget(viewName)
            if not sw:
                continue
            ca = sw.sliceView().cornerAnnotation()
            ca.SetText(vtk.vtkCornerAnnotation.UpperRight, "")
            ca.SetText(vtk.vtkCornerAnnotation.UpperLeft, "")
            sw.sliceView().scheduleRender()

    def _installSliceObservers(self, gridNode):
        """
        gridNode must be a vtkMRMLVolumeNode with GetRASToIJKMatrix.
        """
        self._annotationGridNode = gridNode
        lm = slicer.app.layoutManager()
        if not lm:
            return

        for viewName in ("Red", "Yellow", "Green"):
            sw = lm.sliceWidget(viewName)
            if not sw:
                continue
            sliceNode = sw.sliceLogic().GetSliceNode()

            tag = sliceNode.AddObserver(
                vtk.vtkCommand.ModifiedEvent,
                lambda caller, event, vn=viewName: self._updateAnnotationsForView(vn)
            )
            self._annotationObservers.append((sliceNode, tag))

            self._updateAnnotationsForView(viewName)

    def _currentKForView(self, viewName: str):
        """
        Estimate current slice index k in the gridNode's IJK.
        Uses slice plane origin (RAS) -> RAS-to-IJK.

        NOTE: In Slicer 5.10, GetSliceToRAS() returns a vtkMatrix4x4 and takes no args.
        """
        if self._annotationGridNode is None:
            return None

        lm = slicer.app.layoutManager()
        if not lm:
            return None
        sw = lm.sliceWidget(viewName)
        if not sw:
            return None

        sliceNode = sw.sliceLogic().GetSliceNode()

        sliceToRAS = sliceNode.GetSliceToRAS()
        originRAS = [
            sliceToRAS.GetElement(0, 3),
            sliceToRAS.GetElement(1, 3),
            sliceToRAS.GetElement(2, 3),
            1.0
        ]

        rasToIJK = vtk.vtkMatrix4x4()
        self._annotationGridNode.GetRASToIJKMatrix(rasToIJK)

        ijk4 = [0.0, 0.0, 0.0, 0.0]
        rasToIJK.MultiplyPoint(originRAS, ijk4)

        if not np.all(np.isfinite(ijk4[:3])):
            return None

        return int(round(float(ijk4[2])))

    def _updateAnnotationsForView(self, viewName: str):
        lm = slicer.app.layoutManager()
        if not lm:
            return
        sw = lm.sliceWidget(viewName)
        if not sw:
            return

        ca = sw.sliceView().cornerAnnotation()

        ca.SetText(vtk.vtkCornerAnnotation.UpperLeft, self._summaryText or "")

        k = self._currentKForView(viewName)
        if k is None or not self._metricsByK:
            ca.SetText(vtk.vtkCornerAnnotation.UpperRight, "")
            sw.sliceView().scheduleRender()
            return

        r = self._metricsByK.get(k)
        if r is None:
            ca.SetText(vtk.vtkCornerAnnotation.UpperRight, f"k={k}: (no IJV+CCA data)")
            sw.sliceView().scheduleRender()
            return

        txt = (
            f"IJV: {r['ijv_csa_mm2']:.1f} mm²   CCA: {r['cca_csa_mm2']:.1f} mm²\n"
            f"IJV-CCA Angle: {r['angle_deg']:.1f}°"
        )
        ca.SetText(vtk.vtkCornerAnnotation.UpperRight, txt)
        sw.sliceView().scheduleRender()

    # -----------------------------
    # Main compute
    # -----------------------------
    def onCompute(self):
        segNode = self.segSelector.currentNode()
        if not segNode:
            slicer.util.errorDisplay("Select a segmentation node first.")
            return

        ijvKey = self.ijvSegmentEdit.text.strip()
        ccaKey = self.ccaSegmentEdit.text.strip()
        if not ijvKey or not ccaKey:
            slicer.util.errorDisplay("Please enter both IJV and CCA segment names (or SegmentIDs).")
            return

        segmentation = segNode.GetSegmentation()
        ids = vtk.vtkStringArray()
        segmentation.GetSegmentIDs(ids)
        if ids.GetNumberOfValues() == 0:
            slicer.util.errorDisplay("No segments in this segmentation.")
            return

        ijvID = self._findSegmentIdByNameOrId(segmentation, ids, ijvKey)
        ccaID = self._findSegmentIdByNameOrId(segmentation, ids, ccaKey)
        if ijvID is None or ccaID is None:
            available = self._listSegments(segmentation, ids)
            msg = "Could not find:\n"
            if ijvID is None:
                msg += f"- IJV segment '{ijvKey}'\n"
            if ccaID is None:
                msg += f"- CCA segment '{ccaKey}'\n"
            msg += "\nAvailable segments:\n- " + "\n- ".join(available)
            slicer.util.errorDisplay(msg)
            return

        # Clear previous observers/annotations for a clean run
        self._clearSliceAnnotationsAndObservers()
        self._metricsByK = {}
        self._summaryText = ""

        refVolume = self.refVolumeSelector.currentNode()
        fallbackToSegGeom = refVolume is None

        tmpLM = None
        try:
            # ---- Geometry (spacing + IJK<->RAS) ----
            if fallbackToSegGeom:
                tmpLM = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "tmp_ijvcca_geom")
                idsTwo = vtk.vtkStringArray()
                idsTwo.InsertNextValue(ijvID)
                idsTwo.InsertNextValue(ccaID)
                slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(segNode, idsTwo, tmpLM, None)

                spacing = np.asarray(tmpLM.GetSpacing(), dtype=float)
                ijkToRas = vtk.vtkMatrix4x4()
                tmpLM.GetIJKToRASMatrix(ijkToRas)

                ijv_mask = slicer.util.arrayFromSegmentBinaryLabelmap(segNode, ijvID, None)
                cca_mask = slicer.util.arrayFromSegmentBinaryLabelmap(segNode, ccaID, None)

                gridNodeForAnnotations = tmpLM
            else:
                spacing = np.asarray(refVolume.GetSpacing(), dtype=float)
                ijkToRas = vtk.vtkMatrix4x4()
                refVolume.GetIJKToRASMatrix(ijkToRas)

                ijv_mask = slicer.util.arrayFromSegmentBinaryLabelmap(segNode, ijvID, refVolume)
                cca_mask = slicer.util.arrayFromSegmentBinaryLabelmap(segNode, ccaID, refVolume)

                gridNodeForAnnotations = refVolume

            # ---- Prepare masks ----
            ijv_mask = IJVCSALogic.largest_component((ijv_mask > 0).astype(np.uint8))
            cca_mask = IJVCSALogic.largest_component((cca_mask > 0).astype(np.uint8))

            # ---- Per-slice metrics ----
            ijv_k, ijv_area, ijv_ijk = IJVCSALogic.compute_axial_cross_sections(ijv_mask, spacing)
            cca_k, cca_area, cca_ijk = IJVCSALogic.compute_axial_cross_sections(cca_mask, spacing)

            if len(ijv_k) == 0:
                slicer.util.infoDisplay("No voxels found in IJV segment after filtering.")
                return
            if len(cca_k) == 0:
                slicer.util.infoDisplay("No voxels found in CCA segment after filtering.")
                return

            ijk_to_ras = self._makeIJKToRASConverter(ijkToRas)

            ijv_map = {int(k): (float(ijv_area[i]), ijv_ijk[i]) for i, k in enumerate(ijv_k)}
            cca_map = {int(k): (float(cca_area[i]), cca_ijk[i]) for i, k in enumerate(cca_k)}
            common_slices = sorted(set(ijv_map.keys()) & set(cca_map.keys()))
            if not common_slices:
                slicer.util.infoDisplay("No axial slices contain BOTH IJV and CCA foreground. Cannot compute angles.")
                return

            side = self.sideCombo.currentText
            ref_xy = np.array([1.0, 0.0], dtype=float) if side == "Left" else np.array([-1.0, 0.0], dtype=float)
            ref_label = "Vector.right" if side == "Left" else "Vector.left"

            rows = []
            for k in common_slices:
                ijv_a, ijv_pijk = ijv_map[k]
                cca_a, cca_pijk = cca_map[k]

                p_ijv = ijk_to_ras(ijv_pijk)
                p_cca = ijk_to_ras(cca_pijk)

                v = p_cca - p_ijv
                ang = self._angleFromRefDeg(np.array([v[0], v[1]], dtype=float), ref_xy)

                rows.append({
                    "slice_k": int(k),

                    "ijv_csa_mm2": float(ijv_a),
                    "ijv_ras_x": float(p_ijv[0]),
                    "ijv_ras_y": float(p_ijv[1]),
                    "ijv_ras_z": float(p_ijv[2]),

                    "cca_csa_mm2": float(cca_a),
                    "cca_ras_x": float(p_cca[0]),
                    "cca_ras_y": float(p_cca[1]),
                    "cca_ras_z": float(p_cca[2]),

                    "angle_deg": float(ang),
                    "angle_reference": ref_label,

                    "spacing_x_mm": float(spacing[0]),
                    "spacing_y_mm": float(spacing[1]),
                    "spacing_z_mm": float(spacing[2]),
                })

            # ---- Table + CSV ----
            full_df = pd.DataFrame(rows)
            tableNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLTableNode",
                f"{segNode.GetName()} IJV+CCA CSA+Angle (Axial)"
            )
            self._dfToTable(tableNode, full_df)

            outDir = Path(self.outputDir.currentPath)
            try:
                outDir.mkdir(parents=True, exist_ok=True)
                csvPath = outDir / f"{segNode.GetName()}_IJV_CCA_CSA_Angle_axial.csv"
                full_df.to_csv(csvPath, index=False)
            except Exception as save_e:
                slicer.util.warningDisplay(f"Failed to save CSV: {save_e}")

            # ---- Build per-k lookup for annotations ----
            self._metricsByK = {int(r["slice_k"]): r for r in rows}

            # ---- Summary stats for upper-left ----
            ijv_vals = np.array([r["ijv_csa_mm2"] for r in rows], dtype=float)
            ang_vals = np.array([r["angle_deg"] for r in rows], dtype=float)

            def fmt_stats(arr):
                arr = arr[np.isfinite(arr)]
                if arr.size == 0:
                    return "Avg: n/a   Min: n/a   Max: n/a"
                return f"Avg: {arr.mean():.1f}   Min: {arr.min():.1f}   Max: {arr.max():.1f}"

            self._summaryText = (
                f"IJV CSA (mm²)  {fmt_stats(ijv_vals)}\n"
                f"Angle (deg)    {fmt_stats(ang_vals)}\n"
                f"Angle ref: {ref_label}"
            )

            # ---- NEW: Create markups at COM and label with CSA ----
            transformID = gridNodeForAnnotations.GetTransformNodeID() if gridNodeForAnnotations else None
            self._createCenterLabelNodes(segNode.GetName(), transformID)
            self._populateCenterLabelNodes(rows)

            # ---- Install slice observers + show annotations ----
            self._installSliceObservers(gridNodeForAnnotations)

            # QA: set slice viewers background to ref volume if provided
            if not fallbackToSegGeom and refVolume is not None:
                lm = slicer.app.layoutManager()
                for viewName in ("Red", "Yellow", "Green"):
                    sw = lm.sliceWidget(viewName).sliceLogic()
                    sw.GetSliceCompositeNode().SetBackgroundVolumeID(refVolume.GetID())

            slicer.util.infoDisplay(
                "IJV/CCA per-slice CSA + angle complete.\n"
                "Scroll slices to see per-slice values in the corner annotation.\n"
                "CSA labels are also shown at each vessel COM as Markups."
            )

        except Exception as e:
            slicer.util.errorDisplay(
                f"Failed to compute IJV/CCA per-slice metrics.\n\n{e}\n\n{traceback.format_exc()}"
            )
        finally:
            if tmpLM is not None:
                # If tmpLM is being used as the annotation grid (fallback-to-seg-geom),
                # keep it alive. Otherwise remove it.
                if self.refVolumeSelector.currentNode() is not None:
                    slicer.mrmlScene.RemoveNode(tmpLM)

    # -----------------------------
    # DataFrame -> Table
    # -----------------------------
    def _dfToTable(self, tableNode, df):
        import numpy as np
        from vtk import vtkFloatArray, vtkIntArray, vtkStringArray

        vtk_table = vtk.vtkTable()

        for col in df.columns:
            series = df[col]
            if pd.api.types.is_integer_dtype(series):
                arr = vtkIntArray()
            elif pd.api.types.is_float_dtype(series):
                arr = vtkFloatArray()
            else:
                arr = vtkStringArray()

            arr.SetName(str(col))

            for v in series.astype(object).tolist():
                if isinstance(arr, vtkIntArray):
                    arr.InsertNextValue(0 if v is None or v == "" else int(v))
                elif isinstance(arr, vtkFloatArray):
                    arr.InsertNextValue(np.nan if v is None or v == "" else float(v))
                else:
                    arr.InsertNextValue("" if v is None else str(v))

            vtk_table.AddColumn(arr)

        tableNode.SetAndObserveTable(vtk_table)
        tableNode.Modified()


# -----------------------------
# Reload hook (kept from original)
# -----------------------------
def onReload(self):
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
