from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from grid_transform.annotation_workbench import AnnotationWorkbenchConfig, AnnotationWorkbenchState
from grid_transform.source_annotation import GRID_CONTROL_REQUIRED_CONTOURS, GRID_CONSTRAINT_GROUPS


APP_TITLE = "Streamlit Annotation Workbench"


def _path_or_none(value: str) -> Path | None:
    stripped = value.strip()
    return Path(stripped) if stripped else None


def _ensure_state() -> None:
    defaults = {
        "workbench": None,
        "visible_contours": {},
        "last_result": None,
        "last_logs": "",
        "selected_contour": None,
        "dataset_root_input": "",
        "saved_annotation_input": "",
        "artspeech_speaker_input": "P7",
        "session_input": "S2",
        "reference_speaker_input": "1640_s10_0829",
        "source_frame_input": 0,
        "target_frame_input": 143020,
        "target_case_input": "2008-003^01-1791/test",
        "vtnl_dir_input": str(Path("VTNL")),
        "output_dir_input": "",
        "output_mode_input": "both",
        "max_frames_input": 20,
        "new_contour_name_input": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _sync_visible_contours(workbench: AnnotationWorkbenchState) -> dict[str, bool]:
    visible = dict(st.session_state.get("visible_contours", {}))
    names = set(workbench.current_contours)
    for name in names:
        visible.setdefault(name, True)
    stale = [name for name in visible if name not in names]
    for name in stale:
        visible.pop(name, None)
    st.session_state["visible_contours"] = visible
    if not st.session_state.get("selected_contour") or st.session_state["selected_contour"] not in names:
        st.session_state["selected_contour"] = sorted(names)[0] if names else None
    return visible


def _build_settings_df(workbench: AnnotationWorkbenchState, visible: dict[str, bool]) -> pd.DataFrame:
    controls = workbench.grid_controls
    rows = []
    for name in sorted(workbench.current_contours):
        rows.append(
            {
                "name": name,
                "visible": bool(visible.get(name, True)),
                "include_in_grid": bool(controls["include_in_grid"].get(name, True)),
                "constraint_group": str(controls["constraint_group"].get(name, "none")),
            }
        )
    return pd.DataFrame(rows)


def _apply_settings_df(workbench: AnnotationWorkbenchState, settings_df: pd.DataFrame) -> None:
    visible: dict[str, bool] = {}
    grid_controls = {
        "include_in_grid": {},
        "constraint_group": {},
    }
    for row in settings_df.to_dict(orient="records"):
        name = str(row["name"])
        visible[name] = bool(row["visible"])
        grid_controls["include_in_grid"][name] = bool(row["include_in_grid"])
        group = str(row["constraint_group"]).strip().lower()
        grid_controls["constraint_group"][name] = group if group in GRID_CONSTRAINT_GROUPS else "none"
    st.session_state["visible_contours"] = visible
    workbench.set_grid_controls(grid_controls)


def _points_df(points) -> pd.DataFrame:
    return pd.DataFrame(points, columns=["x", "y"])


def _dataframe_to_points(df: pd.DataFrame):
    cleaned = df.copy()
    cleaned["x"] = pd.to_numeric(cleaned["x"], errors="coerce")
    cleaned["y"] = pd.to_numeric(cleaned["y"], errors="coerce")
    cleaned = cleaned.dropna(subset=["x", "y"])
    if cleaned.empty:
        raise ValueError("Contour must contain at least one valid (x, y) point.")
    return cleaned[["x", "y"]].to_numpy(dtype=float)


def _default_new_contour(workbench: AnnotationWorkbenchState):
    cx = 0.5 * (workbench.source_shape[1] - 1)
    cy = 0.5 * (workbench.source_shape[0] - 1)
    return [[cx - 12.0, cy], [cx + 12.0, cy]]


class _StreamlitWriter(io.TextIOBase):
    def __init__(self, placeholder) -> None:
        self.placeholder = placeholder
        self.parts: list[str] = []

    def write(self, text: str) -> int:
        if not text:
            return 0
        self.parts.append(text)
        self.placeholder.code("".join(self.parts), language="text")
        return len(text)

    def flush(self) -> None:
        return None


def _config_from_inputs() -> AnnotationWorkbenchConfig:
    source_frame = int(st.session_state["source_frame_input"])
    return AnnotationWorkbenchConfig(
        artspeech_speaker=st.session_state["artspeech_speaker_input"].strip() or "P7",
        session=st.session_state["session_input"].strip() or "S2",
        reference_speaker=st.session_state["reference_speaker_input"].strip() or None,
        source_frame=source_frame if source_frame > 0 else None,
        target_frame=int(st.session_state["target_frame_input"]),
        target_case=st.session_state["target_case_input"].strip() or "2008-003^01-1791/test",
        dataset_root=_path_or_none(st.session_state["dataset_root_input"]),
        vtnl_dir=_path_or_none(st.session_state["vtnl_dir_input"]) or Path("VTNL"),
        output_dir=_path_or_none(st.session_state["output_dir_input"]),
    )


def _sync_input_fields(workbench: AnnotationWorkbenchState) -> None:
    st.session_state["dataset_root_input"] = str(workbench.dataset_root)
    st.session_state["artspeech_speaker_input"] = workbench.config.artspeech_speaker
    st.session_state["session_input"] = workbench.config.session
    st.session_state["reference_speaker_input"] = workbench.config.reference_speaker or ""
    st.session_state["source_frame_input"] = int(workbench.source_frame_index1)
    st.session_state["target_frame_input"] = int(workbench.config.target_frame)
    st.session_state["target_case_input"] = workbench.config.target_case
    st.session_state["vtnl_dir_input"] = str(workbench.config.vtnl_dir)
    st.session_state["output_dir_input"] = str(workbench.output_dir)


def _load_saved_annotation() -> None:
    annotation_path = _path_or_none(st.session_state["saved_annotation_input"])
    if annotation_path is None:
        raise ValueError("Please enter a saved annotation JSON path.")
    workbench = AnnotationWorkbenchState.from_saved_annotation(
        annotation_path,
        dataset_root=_path_or_none(st.session_state["dataset_root_input"]),
        vtnl_dir=_path_or_none(st.session_state["vtnl_dir_input"]) or Path("VTNL"),
        output_dir=_path_or_none(st.session_state["output_dir_input"]),
    )
    st.session_state["workbench"] = workbench
    st.session_state["last_result"] = None
    st.session_state["last_logs"] = ""
    _sync_input_fields(workbench)
    _sync_visible_contours(workbench)


def _initialize_from_projection() -> None:
    workbench = AnnotationWorkbenchState.from_projection(_config_from_inputs())
    st.session_state["workbench"] = workbench
    st.session_state["last_result"] = None
    st.session_state["last_logs"] = ""
    _sync_input_fields(workbench)
    _sync_visible_contours(workbench)


def _run_save(render_video: bool) -> None:
    workbench: AnnotationWorkbenchState = st.session_state["workbench"]
    output_mode = st.session_state["output_mode_input"]
    max_frames = int(st.session_state["max_frames_input"])
    log_placeholder = st.empty()
    writer = _StreamlitWriter(log_placeholder)
    with redirect_stdout(writer):
        result = workbench.save_bundle(
            render_video=render_video,
            max_output_frames=max_frames,
            output_mode=output_mode,
        )
    st.session_state["last_result"] = result
    st.session_state["last_logs"] = "".join(writer.parts)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    _ensure_state()

    st.title(APP_TITLE)
    st.caption("Local workbench for source annotation editing, grid preview, and session video rendering.")

    inputs_tab, contours_tab, preview_tab, render_tab = st.tabs(["Inputs", "Contours", "Preview", "Render"])

    with inputs_tab:
        st.subheader("Common Inputs")
        left, right = st.columns(2)
        with left:
            st.text_input("Dataset root", key="dataset_root_input", help="Optional explicit ArtSpeech dataset root.")
            st.text_input("VTNL dir", key="vtnl_dir_input", help="Folder containing VTNL images and ROI zips.")
            st.text_input("Output dir", key="output_dir_input", help="Optional explicit save directory.")
        with right:
            st.number_input("Target frame", min_value=1, step=1, key="target_frame_input")
            st.text_input("Target case", key="target_case_input")
            st.selectbox("Output mode", options=["both", "warped", "review"], key="output_mode_input")
            st.number_input("Max frames", min_value=0, step=1, key="max_frames_input")

        st.divider()
        st.subheader("Load Saved Annotation")
        st.text_input("Saved annotation JSON path", key="saved_annotation_input")
        if st.button("Load saved annotation", use_container_width=True):
            try:
                _load_saved_annotation()
                st.success("Loaded saved annotation.")
            except Exception as exc:
                st.exception(exc)

        st.divider()
        st.subheader("Initialize from Projection")
        left, right = st.columns(2)
        with left:
            st.text_input("ArtSpeech speaker", key="artspeech_speaker_input")
            st.text_input("Session", key="session_input")
            st.text_input("Reference speaker", key="reference_speaker_input")
        with right:
            st.number_input("Source frame override (0 = auto)", min_value=0, step=1, key="source_frame_input")
        if st.button("Initialize from VTNL projection", use_container_width=True):
            try:
                _initialize_from_projection()
                st.success("Initialized from projection.")
            except Exception as exc:
                st.exception(exc)

        workbench = st.session_state.get("workbench")
        if workbench is not None:
            st.info(
                f"Loaded: {workbench.config.artspeech_speaker}/{workbench.config.session} "
                f"frame {workbench.source_frame_index1} -> target {workbench.config.target_frame}"
            )

    with contours_tab:
        workbench = st.session_state.get("workbench")
        if workbench is None:
            st.info("Load a saved annotation or initialize from projection first.")
        else:
            visible = _sync_visible_contours(workbench)
            st.caption("Required core contours are always forced back to include_in_grid=True when applied.")

            settings_df = _build_settings_df(workbench, visible)
            edited_settings_df = st.data_editor(
                settings_df,
                num_rows="fixed",
                use_container_width=True,
                hide_index=True,
                column_config={
                    "name": st.column_config.TextColumn("name", disabled=True),
                    "visible": st.column_config.CheckboxColumn("visible"),
                    "include_in_grid": st.column_config.CheckboxColumn("include_in_grid"),
                    "constraint_group": st.column_config.SelectboxColumn(
                        "constraint_group",
                        options=list(GRID_CONSTRAINT_GROUPS),
                    ),
                },
                key="contour_settings_editor",
            )
            if st.button("Apply contour settings", use_container_width=True):
                try:
                    _apply_settings_df(workbench, edited_settings_df)
                    st.success("Contour settings updated.")
                    st.rerun()
                except Exception as exc:
                    st.exception(exc)

            st.divider()
            contour_names = sorted(workbench.current_contours)
            st.selectbox("Contour", options=contour_names, key="selected_contour")
            selected = st.session_state.get("selected_contour")

            create_col, delete_col = st.columns(2)
            with create_col:
                st.text_input("New contour name", key="new_contour_name_input")
                if st.button("Create contour", use_container_width=True):
                    try:
                        new_name = st.session_state["new_contour_name_input"].strip()
                        if not new_name:
                            raise ValueError("Contour name cannot be empty.")
                        if new_name in workbench.current_contours:
                            raise ValueError(f"Contour {new_name!r} already exists.")
                        updated = {
                            name: points.copy()
                            for name, points in workbench.current_contours.items()
                        }
                        updated[new_name] = _default_new_contour(workbench)
                        grid_controls = {
                            "include_in_grid": dict(workbench.grid_controls["include_in_grid"]),
                            "constraint_group": dict(workbench.grid_controls["constraint_group"]),
                        }
                        grid_controls["include_in_grid"][new_name] = True
                        grid_controls["constraint_group"][new_name] = "none"
                        workbench.set_contours(updated, grid_controls=grid_controls)
                        st.session_state["selected_contour"] = new_name
                        _sync_visible_contours(workbench)
                        st.success(f"Created contour {new_name}.")
                        st.rerun()
                    except Exception as exc:
                        st.exception(exc)
            with delete_col:
                delete_disabled = selected in GRID_CONTROL_REQUIRED_CONTOURS if selected else True
                if st.button("Delete contour", use_container_width=True, disabled=delete_disabled):
                    try:
                        if selected is None:
                            raise ValueError("No contour selected.")
                        updated = {
                            name: points.copy()
                            for name, points in workbench.current_contours.items()
                            if name != selected
                        }
                        grid_controls = {
                            "include_in_grid": {
                                name: value
                                for name, value in workbench.grid_controls["include_in_grid"].items()
                                if name != selected
                            },
                            "constraint_group": {
                                name: value
                                for name, value in workbench.grid_controls["constraint_group"].items()
                                if name != selected
                            },
                        }
                        workbench.set_contours(updated, grid_controls=grid_controls)
                        _sync_visible_contours(workbench)
                        st.success(f"Deleted contour {selected}.")
                        st.rerun()
                    except Exception as exc:
                        st.exception(exc)

            if selected:
                edited_points_df = st.data_editor(
                    _points_df(workbench.current_contours[selected]),
                    num_rows="dynamic",
                    use_container_width=True,
                    hide_index=True,
                    key=f"points_editor_{selected}",
                )
                if st.button("Apply point edits", use_container_width=True):
                    try:
                        updated = {
                            name: points.copy()
                            for name, points in workbench.current_contours.items()
                        }
                        updated[selected] = _dataframe_to_points(edited_points_df)
                        workbench.set_contours(updated, grid_controls=workbench.grid_controls)
                        st.success(f"Updated contour {selected}.")
                        st.rerun()
                    except Exception as exc:
                        st.exception(exc)

    with preview_tab:
        workbench = st.session_state.get("workbench")
        if workbench is None:
            st.info("Load a saved annotation or initialize from projection first.")
        else:
            visible = _sync_visible_contours(workbench)
            preview_images = workbench.preview_images(
                visible_contours={name for name, is_visible in visible.items() if is_visible}
            )
            top_left, top_right = st.columns(2)
            bottom_left, bottom_right = st.columns(2)
            top_left.image(preview_images["source"], caption="Source annotation", use_container_width=True)
            top_right.image(preview_images["source_grid"], caption="Source grid", use_container_width=True)
            bottom_left.image(preview_images["target"], caption="Target grid", use_container_width=True)
            bottom_right.image(preview_images["warped"], caption="Warped preview", use_container_width=True)

            warnings = list(getattr(workbench.source_grid, "warnings", []))
            hard_errors = list(getattr(workbench.source_grid, "hard_errors", []))
            st.text_area(
                "Grid diagnostics",
                value="\n".join(["hard_errors:"] + hard_errors + ["", "warnings:"] + warnings) if (hard_errors or warnings) else "No grid warnings.",
                height=140,
            )

    with render_tab:
        workbench = st.session_state.get("workbench")
        if workbench is None:
            st.info("Load a saved annotation or initialize from projection first.")
        else:
            save_col, render_col = st.columns(2)
            with save_col:
                if st.button("Save annotation bundle", use_container_width=True):
                    try:
                        _run_save(render_video=False)
                        st.success("Saved annotation bundle.")
                    except Exception as exc:
                        st.exception(exc)
            with render_col:
                if st.button("Save + render session video", use_container_width=True):
                    try:
                        _run_save(render_video=True)
                        st.success("Saved annotation and rendered session outputs.")
                    except Exception as exc:
                        st.exception(exc)

            result = st.session_state.get("last_result")
            if result:
                st.subheader("Latest Result")
                st.json(result)

                annotation_json = Path(result["edited_annotation_json"])
                if annotation_json.is_file():
                    st.download_button(
                        "Download annotation JSON",
                        data=annotation_json.read_text(encoding="utf-8"),
                        file_name=annotation_json.name,
                        mime="application/json",
                    )

                sequence_summary = result.get("sequence_warp_summary")
                if isinstance(sequence_summary, dict):
                    warped_video_path = sequence_summary.get("warped_video_path")
                    review_video_path = sequence_summary.get("review_video_path")
                    if warped_video_path and Path(warped_video_path).is_file():
                        st.video(warped_video_path)
                    if review_video_path and Path(review_video_path).is_file():
                        st.video(review_video_path)

            if st.session_state.get("last_logs"):
                st.subheader("Render Logs")
                st.code(st.session_state["last_logs"], language="text")


if __name__ == "__main__":
    main()
