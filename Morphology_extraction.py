#!/usr/bin/env python3
"""
Morphological Feature Extraction and Analysis Pipeline
========================================================
Author: Morphology Analysis Pipeline
Date: March 2026
"""


import os
import math
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime



# ============================================================================
# CONFIGURATION
# ============================================================================


class Config:
    INPUT_PKL      = Path('/mnt/lustre/home/claassen/clala950/DC-TSeg/Data/ETH/july2025/Train_tes_split_smaller_model_normal/Test_data_cyto2_finetuned_clean_400mbar_SINGLE_unet_preds_best.pkl')
    OUTPUT_ROOT    = INPUT_PKL.parent / 'morphology_analysis_outputs_best400'
    FINAL_DB_OUT   = INPUT_PKL.parent / 'Test_data_cyto2_finetuned_clean_400mbar_SINGLE_unet_preds_best_morphology.pkl'
    REPORT_FILE    = OUTPUT_ROOT / 'morphology_results.txt'
    CELL_STATS_TXT = INPUT_PKL.parent / 'cell_stats400.txt'
    AREA_SCALE     = 1/4.0
    MODEL_KEYS     = ('Ground_truth', 'Unet_preds')
    SAVE_PLOTS     = False   # <-- Set to False to skip all plot generation



# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def log_message(message, file_handle=None, print_console=True):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}"
    if print_console:
        print(message)
    if file_handle:
        file_handle.write(log_line + "\n")
        file_handle.flush()



def _to_list_of_arrays(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [np.asarray(im) for im in x]
    arr = np.asarray(x)
    if arr.ndim == 2 or arr.ndim == 3:
        return [arr]
    if arr.ndim == 4:
        return [arr[i] for i in range(arr.shape[0])]
    if arr.ndim == 3 and arr.shape[0] > 1 and (arr.shape[-1] != 3 and arr.shape[-1] != 1):
        return [arr[i] for i in range(arr.shape[0])]
    raise ValueError(f"Unsupported array shape for coercion: {arr.shape}")



# ============================================================================
# MORPHOLOGICAL FEATURE EXTRACTION
# ============================================================================


def contour_features(contour, area_scale=1/4.0):
    raw_area  = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)


    hull           = cv2.convexHull(contour)
    hull_area      = cv2.contourArea(hull)
    hull_perimeter = cv2.arcLength(hull, True)


    area          = raw_area * area_scale
    hull_area_val = hull_area * area_scale


    if perimeter > 0 and raw_area > 0:
        deformation = 1 - 2 * math.sqrt(math.pi * raw_area) / perimeter
    else:
        deformation = 0.0


    x, y, w, h  = cv2.boundingRect(contour)
    length       = max(w, h)
    height       = min(w, h)
    aspect_ratio = (length / height) if height > 0 else 0.0


    circularity = (4.0 * math.pi * raw_area / (perimeter * perimeter)) if perimeter > 0 else 0.0
    solidity    = (raw_area / hull_area)       if hull_area > 0 else 0.0
    porosity    = (hull_area / raw_area)       if raw_area  > 0 else 0.0
    convexity   = (hull_perimeter / perimeter) if perimeter > 0 else 0.0
    extent      = (raw_area / (w * h))         if (w > 0 and h > 0) else 0.0
    eq_diameter = math.sqrt(4.0 * area / math.pi) if area > 0 else 0.0


    rect = cv2.minAreaRect(contour)
    (rect_w, rect_h) = rect[1]
    feret_max_px = float(max(rect_w, rect_h))
    feret_min_px = float(min(rect_w, rect_h))


    inertia_ratio   = 0.0
    eccentricity    = 0.0
    orientation_deg = 0.0


    if len(contour) >= 5:
        try:
            (cx, cy), (MA, ma), angle = cv2.fitEllipse(contour)
            major = max(MA, ma)
            minor = min(MA, ma)
            if minor > 0:
                inertia_ratio = (major / minor)
                eccentricity  = math.sqrt(max(0.0, 1.0 - (minor * minor) / (major * major)))
            orientation_deg = float(angle)
        except cv2.error:
            m = cv2.moments(contour)
            if m['m00'] != 0:
                cov  = np.array([[m['mu20']/m['m00'], m['mu11']/m['m00']],
                                  [m['mu11']/m['m00'], m['mu02']/m['m00']]], dtype=float)
                vals, vecs = np.linalg.eig(cov)
                vals = np.sort(vals)[::-1]
                if len(vals) == 2 and vals[1] > 0:
                    inertia_ratio = math.sqrt(vals[0] / vals[1])
                v = vecs[:, np.argmax(vals)]
                orientation_deg = math.degrees(math.atan2(v[1], v[0]))
    else:
        m = cv2.moments(contour)
        if m['m00'] != 0:
            cov  = np.array([[m['mu20']/m['m00'], m['mu11']/m['m00']],
                              [m['mu11']/m['m00'], m['mu02']/m['m00']]], dtype=float)
            vals, vecs = np.linalg.eig(cov)
            vals = np.sort(vals)[::-1]
            if len(vals) == 2 and vals[1] > 0:
                inertia_ratio = math.sqrt(vals[0] / vals[1])
            v = vecs[:, np.argmax(vals)]
            orientation_deg = math.degrees(math.atan2(v[1], v[0]))


    return {
        'Area': area,
        'HullArea': hull_area_val,
        'HullPerimeter_px': hull_perimeter,
        'Perimeter_px': perimeter,
        'Length_px': length,
        'Height_px': height,
        'AspectRatio': aspect_ratio,
        'Deformation': deformation,
        'Circularity': circularity,
        'Solidity': solidity,
        'Porosity': porosity,
        'Convexity': convexity,
        'Extent': extent,
        'EquivalentDiameter': eq_diameter,
        'InertiaRatio': inertia_ratio,
        'Eccentricity': eccentricity,
        'Orientation_deg': orientation_deg,
        'FeretMax_px': feret_max_px,
        'FeretMin_px': feret_min_px,
    }



# ============================================================================
# OBJECT SEPARATION
# ============================================================================


def separate_mask_into_objects(mask_array):
    mask = np.asarray(mask_array, dtype=np.int32)
    uniq = np.unique(mask)
    uniq = uniq[uniq > 0]
    objects = []


    if len(uniq) == 0:
        return objects


    if len(uniq) == 1 and uniq[0] == 1:
        binmask = (mask == 1).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for i, c in enumerate(contours, start=1):
            obj_mask = np.zeros_like(mask, dtype=np.int32)
            cv2.drawContours(obj_mask, [c], 0, i, -1)
            objects.append((i, obj_mask, c))
    else:
        for lab in uniq:
            labmask = (lab == mask).astype(np.uint8) * 255
            contours, _ = cv2.findContours(labmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                continue
            c = max(contours, key=cv2.contourArea)
            obj_mask = np.zeros_like(mask, dtype=np.int32)
            cv2.drawContours(obj_mask, [c], 0, int(lab), -1)
            objects.append((int(lab), obj_mask, c))


    return objects



# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================


def postprocess_for_new_data(db_in,
                             model_keys=('Ground_truth', 'Unet'),
                             area_scale=1/4.0,
                             output_dir=None,
                             report_handle=None,
                             verbose=True):
    os.makedirs(output_dir, exist_ok=True)
    stats_report  = defaultdict(lambda: defaultdict(dict))
    coverage_data = []


    for pid, pdata in db_in.items():
        if verbose:
            log_message(f"\n{'='*70}\nProcessing patient: {pid}\n{'='*70}", report_handle)


        images = _to_list_of_arrays(pdata.get('image'))
        if not images:
            log_message(f"[{pid}] No 'image' found — skipping.", report_handle)
            continue


        for model_key in model_keys:
            if model_key not in pdata:
                log_message(f"  Model {model_key}: not present, skipping.", report_handle)
                continue


            masks_list = _to_list_of_arrays(pdata[model_key])
            if not masks_list:
                log_message(f"  Model {model_key}: empty, skipping.", report_handle)
                continue


            N = min(len(images), len(masks_list))
            log_message(f"\n  Model: {model_key}", report_handle)
            log_message(f"  Frames: images={len(images)} masks={len(masks_list)} -> using {N}", report_handle)


            masked_images_list = []
            morphology_rows    = []
            total_objects      = 0


            for img_idx in range(N):
                mask     = np.asarray(masks_list[img_idx])
                orig_img = np.asarray(images[img_idx])
                objects  = separate_mask_into_objects(mask)
                total_objects += len(objects)


                if verbose:
                    print(f"    Image {img_idx}: {len(objects)} objects")


                for obj_label, obj_mask, contour in objects:
                    feats      = contour_features(contour, area_scale=area_scale)
                    masked_img = orig_img * (obj_mask > 0)
                    masked_images_list.append(masked_img)


                    morph_row = {
                        'Mask_ID':        f"{pid}_{model_key}_obj{obj_label}",
                        'Patient_ID':     pid,
                        'Group':          'CLL' if pid.startswith('CLL') else 'Control',
                        'SourceModel':    model_key,
                        'SourceImageIdx': img_idx,
                        'OriginalLabel':  obj_label,
                    }
                    morph_row.update(feats)
                    morphology_rows.append(morph_row)


            pdata[f'{model_key}_masked_images'] = masked_images_list


            if morphology_rows:
                df = pd.DataFrame(morphology_rows)
                pdata[f'{model_key}_morphology'] = df
                csv_path = os.path.join(output_dir, f"{pid}_{model_key}_morphology.csv")
                df.to_csv(csv_path, index=False)
                log_message(f"    Saved morphology CSV: {csv_path}", report_handle)
            else:
                df = pd.DataFrame(columns=['Mask_ID','Patient_ID','Group','SourceModel','SourceImageIdx','OriginalLabel','Area'])
                pdata[f'{model_key}_morphology'] = df


            coverage_data.append({
                'Patient_ID':    pid,
                'Model':         model_key,
                'Frames':        N,
                'Objects_Found': total_objects,
            })


            stats_report[pid][model_key] = {
                'input_frames':              N,
                'total_objects':             total_objects,
                'morphology_features_count': len(morphology_rows),
            }


            log_message(f"    ✓ Objects extracted: {total_objects}", report_handle)


    return stats_report, coverage_data



# ============================================================================
# VISUALIZATION
# ============================================================================


def plot_coverage(coverage_df, output_dir, report_handle=None):
    df = coverage_df.copy()
    df['Objects_per_Frame'] = df['Objects_Found'] / df['Frames']
    df['Group'] = df['Patient_ID'].apply(lambda x: 'CLL' if x.startswith('CLL') else 'Control')


    models       = df['Model'].unique()
    group_colors = {'CLL': '#E07070', 'Control': '#6FA8D6'}


    fig, axes = plt.subplots(1, len(models), figsize=(max(14, len(df['Patient_ID'].unique()) // 2), 5),
                             sharey=True)
    if len(models) == 1:
        axes = [axes]


    for ax, model in zip(axes, models):
        sub    = df[df['Model'] == model].sort_values('Patient_ID')
        colors = [group_colors[g] for g in sub['Group']]


        ax.bar(range(len(sub)), sub['Objects_per_Frame'], color=colors, edgecolor='white', linewidth=0.5)


        for grp, col in group_colors.items():
            mean_val = sub.loc[sub['Group'] == grp, 'Objects_per_Frame'].mean()
            if not np.isnan(mean_val):
                ax.axhline(mean_val, color=col, linestyle='--', linewidth=1.5,
                           label=f'{grp} mean = {mean_val:.2f}')


        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(sub['Patient_ID'], rotation=90, fontsize=7)
        ax.set_title(model, fontsize=12)
        ax.set_xlabel("Patient", fontsize=11)
        ax.set_ylabel("Objects per Frame", fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=9)


    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=group_colors['CLL'],     label='CLL'),
                       Patch(facecolor=group_colors['Control'], label='Control')]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=10)


    plt.suptitle("Objects Detected per Frame", fontsize=14, y=1.02)
    plt.tight_layout()


    out_path = Path(output_dir) / "coverage_objects_per_frame.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


    log_message(f"  ✓ Coverage plot saved: {out_path}", report_handle)
    return out_path



def _scatter_kde_plot(df, title, out_path, feat_x_pair, feat_y_pair):
    x_col, x_label = feat_x_pair
    y_col, y_label = feat_y_pair


    sub = df[[x_col, y_col]].dropna()
    if sub.empty:
        return


    fig, ax = plt.subplots(figsize=(7, 5))


    try:
        sns.kdeplot(x=sub[x_col], y=sub[y_col],
                    ax=ax, fill=True, cmap="YlGnBu", levels=15, zorder=1)
    except Exception:
        pass


    ax.scatter(sub[x_col], sub[y_col], s=6, alpha=0.3, color="indigo", zorder=2)
    ax.set_xlabel(x_label, fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_title(f"{x_label} vs {y_label}", fontsize=14)
    if y_col == "Deformation":
        ax.set_ylim(0, 0.2)
    ax.grid(True, alpha=0.3)


    plt.suptitle(title, fontsize=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()



FEATURE_PAIRS = [
    (("Area", "Area (μm²)"),  ("Deformation", "Deformation")),
    (("Area", "Area (μm²)"),  ("AspectRatio",  "Aspect Ratio")),
]



def generate_all_plots(db, model_keys, output_dir, report_handle=None):
    plot_dir        = Path(output_dir) / "plots"
    per_patient_dir = plot_dir / "per_patient"
    group_dir       = plot_dir / "group_level"


    plot_dir.mkdir(parents=True, exist_ok=True)
    per_patient_dir.mkdir(exist_ok=True)
    group_dir.mkdir(exist_ok=True)


    all_dfs = {mk: [] for mk in model_keys}


    for pid, pdata in db.items():
        group = 'CLL' if pid.startswith('CLL') else 'Control'


        for model_key in model_keys:
            morph_key = f'{model_key}_morphology'
            if morph_key not in pdata:
                continue
            df = pdata[morph_key]
            if df.empty:
                continue


            if 'Group' not in df.columns:
                df = df.copy()
                df['Group'] = group


            all_dfs[model_key].append(df)


            pat_dir = per_patient_dir / pid
            pat_dir.mkdir(exist_ok=True)


            for (x_feat, x_label), (y_feat, y_label) in FEATURE_PAIRS:
                fname = f"{pid}_{model_key}_{x_feat}_vs_{y_feat}.png"
                _scatter_kde_plot(
                    df=df,
                    title=f"{pid} | {model_key}",
                    out_path=pat_dir / fname,
                    feat_x_pair=(x_feat, x_label),
                    feat_y_pair=(y_feat, y_label),
                )


    log_message("  ✓ Per-patient plots done", report_handle)


    for model_key in model_keys:
        if not all_dfs[model_key]:
            continue


        combined = pd.concat(all_dfs[model_key], ignore_index=True)


        for group_name, group_filter in [
            ("CLL",     combined['Group'] == 'CLL'),
            ("Control", combined['Group'] == 'Control'),
            ("All",     pd.Series([True] * len(combined))),
        ]:
            sub = combined[group_filter]
            if sub.empty:
                continue


            for (x_feat, x_label), (y_feat, y_label) in FEATURE_PAIRS:
                fname = f"{group_name}_{model_key}_{x_feat}_vs_{y_feat}.png"
                _scatter_kde_plot(
                    df=sub,
                    title=f"{group_name} Patients | {model_key}",
                    out_path=group_dir / fname,
                    feat_x_pair=(x_feat, x_label),
                    feat_y_pair=(y_feat, y_label),
                )


    log_message("  ✓ Group-level plots (CLL / Control / All) done", report_handle)
    log_message(f"  ✓ All plots saved in: {plot_dir}", report_handle)


    return plot_dir



# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    Config.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


    with open(Config.REPORT_FILE, 'w') as report_file:
        log_message("="*70, report_file)
        log_message("MORPHOLOGICAL FEATURE EXTRACTION REPORT", report_file)
        log_message("="*70, report_file)
        log_message(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", report_file)
        log_message("", report_file)


        log_message("CONFIGURATION:", report_file)
        log_message("-"*70, report_file)
        log_message(f"Input file:       {Config.INPUT_PKL}", report_file)
        log_message(f"Output directory: {Config.OUTPUT_ROOT}", report_file)
        log_message(f"Final database:   {Config.FINAL_DB_OUT}", report_file)
        log_message(f"Area scale:       {Config.AREA_SCALE} (1 pixel = 0.5 μm, 1 pixel² = 0.25 μm²)", report_file)
        log_message(f"Model keys:       {Config.MODEL_KEYS}", report_file)
        log_message(f"Save plots:       {Config.SAVE_PLOTS}", report_file)
        log_message("", report_file)


        log_message("="*70, report_file)
        log_message("LOADING DATA", report_file)
        log_message("="*70, report_file)


        with open(Config.INPUT_PKL, 'rb') as f:
            db = pickle.load(f)


        log_message(f"✓ Loaded database with {len(db)} patients", report_file)
        log_message(f"Patient IDs: {list(db.keys())}", report_file)
        log_message("", report_file)


        log_message("="*70, report_file)
        log_message("MORPHOLOGICAL FEATURE EXTRACTION", report_file)
        log_message("="*70, report_file)


        stats, coverage_data = postprocess_for_new_data(
            db,
            model_keys=Config.MODEL_KEYS,
            area_scale=Config.AREA_SCALE,
            output_dir=str(Config.OUTPUT_ROOT),
            report_handle=report_file,
            verbose=True
        )


        log_message("", report_file)
        log_message("✓ Morphology extraction complete!", report_file)
        log_message("", report_file)


        log_message("="*70, report_file)
        log_message("COVERAGE SUMMARY", report_file)
        log_message("="*70, report_file)


        coverage_df = pd.DataFrame(coverage_data)
        log_message("", report_file)
        log_message(coverage_df.to_string(index=False), report_file)
        log_message("", report_file)


        coverage_csv_path = Config.OUTPUT_ROOT / "coverage_summary.csv"
        coverage_df.to_csv(coverage_csv_path, index=False)
        log_message(f"✓ Coverage table saved to: {coverage_csv_path}", report_file)
        log_message("", report_file)


        # ── Plots (only when SAVE_PLOTS=True) ────────────────────────────────
        if Config.SAVE_PLOTS:
            log_message("="*70, report_file)
            log_message("GENERATING PLOTS", report_file)
            log_message("="*70, report_file)

            plot_coverage(coverage_df, output_dir=str(Config.OUTPUT_ROOT), report_handle=report_file)

            plot_dir = generate_all_plots(
                db=db,
                model_keys=Config.MODEL_KEYS,
                output_dir=str(Config.OUTPUT_ROOT),
                report_handle=report_file,
            )
        else:
            plot_dir = None
            log_message("⚠ SAVE_PLOTS=False — skipping all plot generation.", report_file)
        # ─────────────────────────────────────────────────────────────────────


        log_message("", report_file)
        log_message("="*70, report_file)
        log_message("SAVING RESULTS", report_file)
        log_message("="*70, report_file)


        with open(Config.FINAL_DB_OUT, 'wb') as f:
            pickle.dump(db, f, protocol=pickle.HIGHEST_PROTOCOL)
        log_message(f"✓ Final database saved to: {Config.FINAL_DB_OUT}", report_file)
        log_message("", report_file)


        # ── write cell stats txt for later plotting ──────────────────────────
        total_cells = sum(
            ms['total_objects']
            for ps in stats.values()
            for ms in ps.values()
        )
        with open(Config.CELL_STATS_TXT, 'w') as sf:
            sf.write(f"# input_pkl: {Config.INPUT_PKL}\n")
            sf.write("patient_id\tgroup\tmodel\tn_frames\tn_valid_cells\tcells_per_frame\n")
            for row in coverage_data:
                pid    = row['Patient_ID']
                group  = 'CLL' if pid.startswith('CLL') else 'Control'
                model  = row['Model']
                frames = row['Frames']
                cells  = row['Objects_Found']
                cpf    = cells / frames if frames > 0 else 0.0
                sf.write(f"{pid}\t{group}\t{model}\t{frames}\t{cells}\t{cpf:.4f}\n")
            sf.write(f"\n# total_patients   : {len(stats)}\n")
            sf.write(f"# total_valid_cells: {total_cells}\n")
        log_message(f"✓ Cell stats saved: {Config.CELL_STATS_TXT}", report_file)
        # ────────────────────────────────────────────────────────────────────


        log_message("="*70, report_file)
        log_message("SUMMARY", report_file)
        log_message("="*70, report_file)


        total_patients = len(stats)
        total_objects  = sum(
            model_stats['total_objects']
            for patient_stats in stats.values()
            for model_stats in patient_stats.values()
        )


        log_message(f"  • Patients processed:      {total_patients}", report_file)
        log_message(f"  • Total objects extracted: {total_objects}", report_file)
        log_message(f"  • Morphology CSVs:         {Config.OUTPUT_ROOT}/*.csv", report_file)
        log_message(f"  • Coverage CSV:            {coverage_csv_path}", report_file)
        if Config.SAVE_PLOTS:
            log_message(f"  • Coverage plot:           {Config.OUTPUT_ROOT}/coverage_objects_per_frame.png", report_file)
            log_message(f"  • Plots:                   {plot_dir}/", report_file)
            log_message(f"    - per_patient/  → one folder per patient", report_file)
            log_message(f"    - group_level/  → CLL / Control / All combined", report_file)
        else:
            log_message(f"  • Plots:                   skipped (SAVE_PLOTS=False)", report_file)
        log_message(f"  • Final database:           {Config.FINAL_DB_OUT}", report_file)
        log_message(f"  • Cell stats TXT:           {Config.CELL_STATS_TXT}", report_file)
        log_message(f"  • This report:              {Config.REPORT_FILE}", report_file)
        log_message("", report_file)


        log_message("="*70, report_file)
        log_message("PROCESSING COMPLETE!", report_file)
        log_message(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", report_file)
        log_message("="*70, report_file)


    print(f"\n✓ Report saved to: {Config.REPORT_FILE}")
    print(f"✓ Cell stats saved to: {Config.CELL_STATS_TXT}")
    print("\n" + "="*70)
    print("ALL PROCESSING COMPLETE!")
    print("="*70)



if __name__ == "__main__":
    main()
# #!/usr/bin/env python3
# """
# Morphological Feature Extraction and Analysis Pipeline
# ========================================================
# Author: Morphology Analysis Pipeline
# Date: March 2026
# """

# import os
# import math
# import pickle
# import numpy as np
# import pandas as pd
# from collections import defaultdict
# from pathlib import Path
# import cv2
# import matplotlib.pyplot as plt
# import seaborn as sns
# from datetime import datetime


# # ============================================================================
# # CONFIGURATION
# # ============================================================================

# class Config:
#     INPUT_PKL      = Path('/mnt/lustre/home/claassen/clala950/DC-TSeg/Data/ETH/july2025/Train_tes_split_smaller_model/Test_data_cyto2_finetuned_clean_400mbar_SINGLE_unet_preds.pkl')
#     OUTPUT_ROOT    = INPUT_PKL.parent / 'morphology_analysis_outputs_400mbar'
#     FINAL_DB_OUT   = INPUT_PKL.parent / 'Test_data_cyto2_finetuned_clean_400mbar_SINGLE_unet_preds_morphology_clean_v2.pkl'
#     REPORT_FILE    = OUTPUT_ROOT / 'morphology_results.txt'
#     CELL_STATS_TXT = INPUT_PKL.parent / 'cell_stats_400mbar.txt'  # <-- new
#     AREA_SCALE     = 1/4.0
#     MODEL_KEYS     = ('Ground_truth', 'Unet_preds')


# # ============================================================================
# # UTILITY FUNCTIONS
# # ============================================================================

# def log_message(message, file_handle=None, print_console=True):
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     log_line = f"[{timestamp}] {message}"
#     if print_console:
#         print(message)
#     if file_handle:
#         file_handle.write(log_line + "\n")
#         file_handle.flush()


# def _to_list_of_arrays(x):
#     if x is None:
#         return []
#     if isinstance(x, (list, tuple)):
#         return [np.asarray(im) for im in x]
#     arr = np.asarray(x)
#     if arr.ndim == 2 or arr.ndim == 3:
#         return [arr]
#     if arr.ndim == 4:
#         return [arr[i] for i in range(arr.shape[0])]
#     if arr.ndim == 3 and arr.shape[0] > 1 and (arr.shape[-1] != 3 and arr.shape[-1] != 1):
#         return [arr[i] for i in range(arr.shape[0])]
#     raise ValueError(f"Unsupported array shape for coercion: {arr.shape}")


# # ============================================================================
# # MORPHOLOGICAL FEATURE EXTRACTION
# # ============================================================================

# def contour_features(contour, area_scale=1/4.0):
#     raw_area  = cv2.contourArea(contour)
#     perimeter = cv2.arcLength(contour, True)

#     hull           = cv2.convexHull(contour)
#     hull_area      = cv2.contourArea(hull)
#     hull_perimeter = cv2.arcLength(hull, True)

#     area          = raw_area * area_scale
#     hull_area_val = hull_area * area_scale

#     if perimeter > 0 and raw_area > 0:
#         deformation = 1 - 2 * math.sqrt(math.pi * raw_area) / perimeter
#     else:
#         deformation = 0.0

#     x, y, w, h  = cv2.boundingRect(contour)
#     length       = max(w, h)
#     height       = min(w, h)
#     aspect_ratio = (length / height) if height > 0 else 0.0

#     circularity = (4.0 * math.pi * raw_area / (perimeter * perimeter)) if perimeter > 0 else 0.0
#     solidity    = (raw_area / hull_area)       if hull_area > 0 else 0.0
#     porosity    = (hull_area / raw_area)       if raw_area  > 0 else 0.0
#     convexity   = (hull_perimeter / perimeter) if perimeter > 0 else 0.0
#     extent      = (raw_area / (w * h))         if (w > 0 and h > 0) else 0.0
#     eq_diameter = math.sqrt(4.0 * area / math.pi) if area > 0 else 0.0

#     rect = cv2.minAreaRect(contour)
#     (rect_w, rect_h) = rect[1]
#     feret_max_px = float(max(rect_w, rect_h))
#     feret_min_px = float(min(rect_w, rect_h))

#     inertia_ratio   = 0.0
#     eccentricity    = 0.0
#     orientation_deg = 0.0

#     if len(contour) >= 5:
#         try:
#             (cx, cy), (MA, ma), angle = cv2.fitEllipse(contour)
#             major = max(MA, ma)
#             minor = min(MA, ma)
#             if minor > 0:
#                 inertia_ratio = (major / minor)
#                 eccentricity  = math.sqrt(max(0.0, 1.0 - (minor * minor) / (major * major)))
#             orientation_deg = float(angle)
#         except cv2.error:
#             m = cv2.moments(contour)
#             if m['m00'] != 0:
#                 cov  = np.array([[m['mu20']/m['m00'], m['mu11']/m['m00']],
#                                   [m['mu11']/m['m00'], m['mu02']/m['m00']]], dtype=float)
#                 vals, vecs = np.linalg.eig(cov)
#                 vals = np.sort(vals)[::-1]
#                 if len(vals) == 2 and vals[1] > 0:
#                     inertia_ratio = math.sqrt(vals[0] / vals[1])
#                 v = vecs[:, np.argmax(vals)]
#                 orientation_deg = math.degrees(math.atan2(v[1], v[0]))
#     else:
#         m = cv2.moments(contour)
#         if m['m00'] != 0:
#             cov  = np.array([[m['mu20']/m['m00'], m['mu11']/m['m00']],
#                               [m['mu11']/m['m00'], m['mu02']/m['m00']]], dtype=float)
#             vals, vecs = np.linalg.eig(cov)
#             vals = np.sort(vals)[::-1]
#             if len(vals) == 2 and vals[1] > 0:
#                 inertia_ratio = math.sqrt(vals[0] / vals[1])
#             v = vecs[:, np.argmax(vals)]
#             orientation_deg = math.degrees(math.atan2(v[1], v[0]))

#     return {
#         'Area': area,
#         'HullArea': hull_area_val,
#         'HullPerimeter_px': hull_perimeter,
#         'Perimeter_px': perimeter,
#         'Length_px': length,
#         'Height_px': height,
#         'AspectRatio': aspect_ratio,
#         'Deformation': deformation,
#         'Circularity': circularity,
#         'Solidity': solidity,
#         'Porosity': porosity,
#         'Convexity': convexity,
#         'Extent': extent,
#         'EquivalentDiameter': eq_diameter,
#         'InertiaRatio': inertia_ratio,
#         'Eccentricity': eccentricity,
#         'Orientation_deg': orientation_deg,
#         'FeretMax_px': feret_max_px,
#         'FeretMin_px': feret_min_px,
#     }


# # ============================================================================
# # OBJECT SEPARATION
# # ============================================================================

# def separate_mask_into_objects(mask_array):
#     mask = np.asarray(mask_array, dtype=np.int32)
#     uniq = np.unique(mask)
#     uniq = uniq[uniq > 0]
#     objects = []

#     if len(uniq) == 0:
#         return objects

#     if len(uniq) == 1 and uniq[0] == 1:
#         binmask = (mask == 1).astype(np.uint8) * 255
#         contours, _ = cv2.findContours(binmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#         for i, c in enumerate(contours, start=1):
#             obj_mask = np.zeros_like(mask, dtype=np.int32)
#             cv2.drawContours(obj_mask, [c], 0, i, -1)
#             objects.append((i, obj_mask, c))
#     else:
#         for lab in uniq:
#             labmask = (lab == mask).astype(np.uint8) * 255
#             contours, _ = cv2.findContours(labmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#             if not contours:
#                 continue
#             c = max(contours, key=cv2.contourArea)
#             obj_mask = np.zeros_like(mask, dtype=np.int32)
#             cv2.drawContours(obj_mask, [c], 0, int(lab), -1)
#             objects.append((int(lab), obj_mask, c))

#     return objects


# # ============================================================================
# # MAIN PROCESSING PIPELINE
# # ============================================================================

# def postprocess_for_new_data(db_in,
#                              model_keys=('Ground_truth', 'Unet'),
#                              area_scale=1/4.0,
#                              output_dir=None,
#                              report_handle=None,
#                              verbose=True):
#     os.makedirs(output_dir, exist_ok=True)
#     stats_report  = defaultdict(lambda: defaultdict(dict))
#     coverage_data = []

#     for pid, pdata in db_in.items():
#         if verbose:
#             log_message(f"\n{'='*70}\nProcessing patient: {pid}\n{'='*70}", report_handle)

#         images = _to_list_of_arrays(pdata.get('image'))
#         if not images:
#             log_message(f"[{pid}] No 'image' found — skipping.", report_handle)
#             continue

#         for model_key in model_keys:
#             if model_key not in pdata:
#                 log_message(f"  Model {model_key}: not present, skipping.", report_handle)
#                 continue

#             masks_list = _to_list_of_arrays(pdata[model_key])
#             if not masks_list:
#                 log_message(f"  Model {model_key}: empty, skipping.", report_handle)
#                 continue

#             N = min(len(images), len(masks_list))
#             log_message(f"\n  Model: {model_key}", report_handle)
#             log_message(f"  Frames: images={len(images)} masks={len(masks_list)} -> using {N}", report_handle)

#             masked_images_list = []
#             morphology_rows    = []
#             total_objects      = 0

#             for img_idx in range(N):
#                 mask     = np.asarray(masks_list[img_idx])
#                 orig_img = np.asarray(images[img_idx])
#                 objects  = separate_mask_into_objects(mask)
#                 total_objects += len(objects)

#                 if verbose:
#                     print(f"    Image {img_idx}: {len(objects)} objects")

#                 for obj_label, obj_mask, contour in objects:
#                     feats      = contour_features(contour, area_scale=area_scale)
#                     masked_img = orig_img * (obj_mask > 0)
#                     masked_images_list.append(masked_img)

#                     morph_row = {
#                         'Mask_ID':        f"{pid}_{model_key}_obj{obj_label}",
#                         'Patient_ID':     pid,
#                         'Group':          'CLL' if pid.startswith('CLL') else 'Control',
#                         'SourceModel':    model_key,
#                         'SourceImageIdx': img_idx,
#                         'OriginalLabel':  obj_label,
#                     }
#                     morph_row.update(feats)
#                     morphology_rows.append(morph_row)

#             pdata[f'{model_key}_masked_images'] = masked_images_list

#             if morphology_rows:
#                 df = pd.DataFrame(morphology_rows)
#                 pdata[f'{model_key}_morphology'] = df
#                 csv_path = os.path.join(output_dir, f"{pid}_{model_key}_morphology.csv")
#                 df.to_csv(csv_path, index=False)
#                 log_message(f"    Saved morphology CSV: {csv_path}", report_handle)
#             else:
#                 df = pd.DataFrame(columns=['Mask_ID','Patient_ID','Group','SourceModel','SourceImageIdx','OriginalLabel','Area'])
#                 pdata[f'{model_key}_morphology'] = df

#             coverage_data.append({
#                 'Patient_ID':    pid,
#                 'Model':         model_key,
#                 'Frames':        N,
#                 'Objects_Found': total_objects,
#             })

#             stats_report[pid][model_key] = {
#                 'input_frames':              N,
#                 'total_objects':             total_objects,
#                 'morphology_features_count': len(morphology_rows),
#             }

#             log_message(f"    ✓ Objects extracted: {total_objects}", report_handle)

#     return stats_report, coverage_data


# # ============================================================================
# # VISUALIZATION
# # ============================================================================

# def plot_coverage(coverage_df, output_dir, report_handle=None):
#     df = coverage_df.copy()
#     df['Objects_per_Frame'] = df['Objects_Found'] / df['Frames']
#     df['Group'] = df['Patient_ID'].apply(lambda x: 'CLL' if x.startswith('CLL') else 'Control')

#     models       = df['Model'].unique()
#     group_colors = {'CLL': '#E07070', 'Control': '#6FA8D6'}

#     fig, axes = plt.subplots(1, len(models), figsize=(max(14, len(df['Patient_ID'].unique()) // 2), 5),
#                              sharey=True)
#     if len(models) == 1:
#         axes = [axes]

#     for ax, model in zip(axes, models):
#         sub    = df[df['Model'] == model].sort_values('Patient_ID')
#         colors = [group_colors[g] for g in sub['Group']]

#         ax.bar(range(len(sub)), sub['Objects_per_Frame'], color=colors, edgecolor='white', linewidth=0.5)

#         for grp, col in group_colors.items():
#             mean_val = sub.loc[sub['Group'] == grp, 'Objects_per_Frame'].mean()
#             if not np.isnan(mean_val):
#                 ax.axhline(mean_val, color=col, linestyle='--', linewidth=1.5,
#                            label=f'{grp} mean = {mean_val:.2f}')

#         ax.set_xticks(range(len(sub)))
#         ax.set_xticklabels(sub['Patient_ID'], rotation=90, fontsize=7)
#         ax.set_title(model, fontsize=12)
#         ax.set_xlabel("Patient", fontsize=11)
#         ax.set_ylabel("Objects per Frame", fontsize=11)
#         ax.grid(axis='y', alpha=0.3)
#         ax.legend(fontsize=9)

#     from matplotlib.patches import Patch
#     legend_elements = [Patch(facecolor=group_colors['CLL'],     label='CLL'),
#                        Patch(facecolor=group_colors['Control'], label='Control')]
#     fig.legend(handles=legend_elements, loc='upper right', fontsize=10)

#     plt.suptitle("Objects Detected per Frame", fontsize=14, y=1.02)
#     plt.tight_layout()

#     out_path = Path(output_dir) / "coverage_objects_per_frame.png"
#     plt.savefig(out_path, dpi=150, bbox_inches="tight")
#     plt.close()

#     log_message(f"  ✓ Coverage plot saved: {out_path}", report_handle)
#     return out_path


# def _scatter_kde_plot(df, title, out_path, feat_x_pair, feat_y_pair):
#     x_col, x_label = feat_x_pair
#     y_col, y_label = feat_y_pair

#     sub = df[[x_col, y_col]].dropna()
#     if sub.empty:
#         return

#     fig, ax = plt.subplots(figsize=(7, 5))

#     try:
#         sns.kdeplot(x=sub[x_col], y=sub[y_col],
#                     ax=ax, fill=True, cmap="YlGnBu", levels=15, zorder=1)
#     except Exception:
#         pass

#     ax.scatter(sub[x_col], sub[y_col], s=6, alpha=0.3, color="indigo", zorder=2)
#     ax.set_xlabel(x_label, fontsize=13)
#     ax.set_ylabel(y_label, fontsize=13)
#     ax.set_title(f"{x_label} vs {y_label}", fontsize=14)
#     if y_col == "Deformation":
#         ax.set_ylim(0, 0.2)
#     ax.grid(True, alpha=0.3)

#     plt.suptitle(title, fontsize=15)
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=150, bbox_inches="tight")
#     plt.close()


# FEATURE_PAIRS = [
#     (("Area", "Area (μm²)"),  ("Deformation", "Deformation")),
#     (("Area", "Area (μm²)"),  ("AspectRatio",  "Aspect Ratio")),
# ]


# def generate_all_plots(db, model_keys, output_dir, report_handle=None):
#     plot_dir        = Path(output_dir) / "plots"
#     per_patient_dir = plot_dir / "per_patient"
#     group_dir       = plot_dir / "group_level"

#     plot_dir.mkdir(parents=True, exist_ok=True)
#     per_patient_dir.mkdir(exist_ok=True)
#     group_dir.mkdir(exist_ok=True)

#     all_dfs = {mk: [] for mk in model_keys}

#     for pid, pdata in db.items():
#         group = 'CLL' if pid.startswith('CLL') else 'Control'

#         for model_key in model_keys:
#             morph_key = f'{model_key}_morphology'
#             if morph_key not in pdata:
#                 continue
#             df = pdata[morph_key]
#             if df.empty:
#                 continue

#             if 'Group' not in df.columns:
#                 df = df.copy()
#                 df['Group'] = group

#             all_dfs[model_key].append(df)

#             pat_dir = per_patient_dir / pid
#             pat_dir.mkdir(exist_ok=True)

#             for (x_feat, x_label), (y_feat, y_label) in FEATURE_PAIRS:
#                 fname = f"{pid}_{model_key}_{x_feat}_vs_{y_feat}.png"
#                 _scatter_kde_plot(
#                     df=df,
#                     title=f"{pid} | {model_key}",
#                     out_path=pat_dir / fname,
#                     feat_x_pair=(x_feat, x_label),
#                     feat_y_pair=(y_feat, y_label),
#                 )

#     log_message("  ✓ Per-patient plots done", report_handle)

#     for model_key in model_keys:
#         if not all_dfs[model_key]:
#             continue

#         combined = pd.concat(all_dfs[model_key], ignore_index=True)

#         for group_name, group_filter in [
#             ("CLL",     combined['Group'] == 'CLL'),
#             ("Control", combined['Group'] == 'Control'),
#             ("All",     pd.Series([True] * len(combined))),
#         ]:
#             sub = combined[group_filter]
#             if sub.empty:
#                 continue

#             for (x_feat, x_label), (y_feat, y_label) in FEATURE_PAIRS:
#                 fname = f"{group_name}_{model_key}_{x_feat}_vs_{y_feat}.png"
#                 _scatter_kde_plot(
#                     df=sub,
#                     title=f"{group_name} Patients | {model_key}",
#                     out_path=group_dir / fname,
#                     feat_x_pair=(x_feat, x_label),
#                     feat_y_pair=(y_feat, y_label),
#                 )

#     log_message("  ✓ Group-level plots (CLL / Control / All) done", report_handle)
#     log_message(f"  ✓ All plots saved in: {plot_dir}", report_handle)

#     return plot_dir


# # ============================================================================
# # MAIN EXECUTION
# # ============================================================================

# def main():
#     Config.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

#     with open(Config.REPORT_FILE, 'w') as report_file:
#         log_message("="*70, report_file)
#         log_message("MORPHOLOGICAL FEATURE EXTRACTION REPORT", report_file)
#         log_message("="*70, report_file)
#         log_message(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", report_file)
#         log_message("", report_file)

#         log_message("CONFIGURATION:", report_file)
#         log_message("-"*70, report_file)
#         log_message(f"Input file:       {Config.INPUT_PKL}", report_file)
#         log_message(f"Output directory: {Config.OUTPUT_ROOT}", report_file)
#         log_message(f"Final database:   {Config.FINAL_DB_OUT}", report_file)
#         log_message(f"Area scale:       {Config.AREA_SCALE} (1 pixel = 0.5 μm, 1 pixel² = 0.25 μm²)", report_file)
#         log_message(f"Model keys:       {Config.MODEL_KEYS}", report_file)
#         log_message("", report_file)

#         log_message("="*70, report_file)
#         log_message("LOADING DATA", report_file)
#         log_message("="*70, report_file)

#         with open(Config.INPUT_PKL, 'rb') as f:
#             db = pickle.load(f)

#         log_message(f"✓ Loaded database with {len(db)} patients", report_file)
#         log_message(f"Patient IDs: {list(db.keys())}", report_file)
#         log_message("", report_file)

#         log_message("="*70, report_file)
#         log_message("MORPHOLOGICAL FEATURE EXTRACTION", report_file)
#         log_message("="*70, report_file)

#         stats, coverage_data = postprocess_for_new_data(
#             db,
#             model_keys=Config.MODEL_KEYS,
#             area_scale=Config.AREA_SCALE,
#             output_dir=str(Config.OUTPUT_ROOT),
#             report_handle=report_file,
#             verbose=True
#         )

#         log_message("", report_file)
#         log_message("✓ Morphology extraction complete!", report_file)
#         log_message("", report_file)

#         log_message("="*70, report_file)
#         log_message("COVERAGE SUMMARY", report_file)
#         log_message("="*70, report_file)

#         coverage_df = pd.DataFrame(coverage_data)
#         log_message("", report_file)
#         log_message(coverage_df.to_string(index=False), report_file)
#         log_message("", report_file)

#         coverage_csv_path = Config.OUTPUT_ROOT / "coverage_summary.csv"
#         coverage_df.to_csv(coverage_csv_path, index=False)
#         log_message(f"✓ Coverage table saved to: {coverage_csv_path}", report_file)
#         log_message("", report_file)

#         log_message("="*70, report_file)
#         log_message("GENERATING PLOTS", report_file)
#         log_message("="*70, report_file)

#         plot_coverage(coverage_df, output_dir=str(Config.OUTPUT_ROOT), report_handle=report_file)

#         plot_dir = generate_all_plots(
#             db=db,
#             model_keys=Config.MODEL_KEYS,
#             output_dir=str(Config.OUTPUT_ROOT),
#             report_handle=report_file,
#         )

#         log_message("", report_file)
#         log_message("="*70, report_file)
#         log_message("SAVING RESULTS", report_file)
#         log_message("="*70, report_file)

#         with open(Config.FINAL_DB_OUT, 'wb') as f:
#             pickle.dump(db, f, protocol=pickle.HIGHEST_PROTOCOL)
#         log_message(f"✓ Final database saved to: {Config.FINAL_DB_OUT}", report_file)
#         log_message("", report_file)

#         # ── write cell stats txt for later plotting ──────────────────────────
#         total_cells = sum(
#             ms['total_objects']
#             for ps in stats.values()
#             for ms in ps.values()
#         )
#         with open(Config.CELL_STATS_TXT, 'w') as sf:
#             sf.write(f"# input_pkl: {Config.INPUT_PKL}\n")
#             sf.write("patient_id\tgroup\tmodel\tn_frames\tn_valid_cells\tcells_per_frame\n")
#             for row in coverage_data:
#                 pid    = row['Patient_ID']
#                 group  = 'CLL' if pid.startswith('CLL') else 'Control'
#                 model  = row['Model']
#                 frames = row['Frames']
#                 cells  = row['Objects_Found']
#                 cpf    = cells / frames if frames > 0 else 0.0
#                 sf.write(f"{pid}\t{group}\t{model}\t{frames}\t{cells}\t{cpf:.4f}\n")
#             sf.write(f"\n# total_patients   : {len(stats)}\n")
#             sf.write(f"# total_valid_cells: {total_cells}\n")
#         log_message(f"✓ Cell stats saved: {Config.CELL_STATS_TXT}", report_file)
#         # ────────────────────────────────────────────────────────────────────

#         log_message("="*70, report_file)
#         log_message("SUMMARY", report_file)
#         log_message("="*70, report_file)

#         total_patients = len(stats)
#         total_objects  = sum(
#             model_stats['total_objects']
#             for patient_stats in stats.values()
#             for model_stats in patient_stats.values()
#         )

#         log_message(f"  • Patients processed:      {total_patients}", report_file)
#         log_message(f"  • Total objects extracted: {total_objects}", report_file)
#         log_message(f"  • Morphology CSVs:         {Config.OUTPUT_ROOT}/*.csv", report_file)
#         log_message(f"  • Coverage plot:           {Config.OUTPUT_ROOT}/coverage_objects_per_frame.png", report_file)
#         log_message(f"  • Plots:                   {plot_dir}/", report_file)
#         log_message(f"    - per_patient/  → one folder per patient", report_file)
#         log_message(f"    - group_level/  → CLL / Control / All combined", report_file)
#         log_message(f"  • Final database:           {Config.FINAL_DB_OUT}", report_file)
#         log_message(f"  • Cell stats TXT:           {Config.CELL_STATS_TXT}", report_file)
#         log_message(f"  • This report:              {Config.REPORT_FILE}", report_file)
#         log_message("", report_file)

#         log_message("="*70, report_file)
#         log_message("PROCESSING COMPLETE!", report_file)
#         log_message(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", report_file)
#         log_message("="*70, report_file)

#     print(f"\n✓ Report saved to: {Config.REPORT_FILE}")
#     print(f"✓ Cell stats saved to: {Config.CELL_STATS_TXT}")
#     print("\n" + "="*70)
#     print("ALL PROCESSING COMPLETE!")
#     print("="*70)


# if __name__ == "__main__":
#     main()


# #!/usr/bin/env python3
# """
# Morphological Feature Extraction and Analysis Pipeline
# ========================================================
# Author: Morphology Analysis Pipeline
# Date: March 2026
# """

# import os
# import math
# import pickle
# import numpy as np
# import pandas as pd
# from collections import defaultdict
# from pathlib import Path
# import cv2
# import matplotlib.pyplot as plt
# import seaborn as sns
# from datetime import datetime


# # ============================================================================
# # CONFIGURATION
# # ============================================================================

# class Config:
#     INPUT_PKL    = Path('/mnt/lustre/home/claassen/clala950/DC-TSeg/Data/Guck2025/Train_Test_Split/Test_data_cyto2_finetuned_clean_SINGLE_unet_preds.pkl')
#     OUTPUT_ROOT  = INPUT_PKL.parent / 'morphology_analysis_outputs'
#     FINAL_DB_OUT = INPUT_PKL.parent / 'Test_data_cyto2_finetuned_clean_SINGLE_unet_preds_morphology.pkl'
#     REPORT_FILE  = OUTPUT_ROOT / 'morphology_results.txt'
#     AREA_SCALE   = 1/4.0
#     MODEL_KEYS   = ('Ground_truth', 'Unet_preds')


# # ============================================================================
# # UTILITY FUNCTIONS
# # ============================================================================

# def log_message(message, file_handle=None, print_console=True):
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     log_line = f"[{timestamp}] {message}"
#     if print_console:
#         print(message)
#     if file_handle:
#         file_handle.write(log_line + "\n")
#         file_handle.flush()


# def _to_list_of_arrays(x):
#     if x is None:
#         return []
#     if isinstance(x, (list, tuple)):
#         return [np.asarray(im) for im in x]
#     arr = np.asarray(x)
#     if arr.ndim == 2 or arr.ndim == 3:
#         return [arr]
#     if arr.ndim == 4:
#         return [arr[i] for i in range(arr.shape[0])]
#     if arr.ndim == 3 and arr.shape[0] > 1 and (arr.shape[-1] != 3 and arr.shape[-1] != 1):
#         return [arr[i] for i in range(arr.shape[0])]
#     raise ValueError(f"Unsupported array shape for coercion: {arr.shape}")


# # ============================================================================
# # MORPHOLOGICAL FEATURE EXTRACTION
# # ============================================================================

# def contour_features(contour, area_scale=1/4.0):
#     raw_area  = cv2.contourArea(contour)
#     perimeter = cv2.arcLength(contour, True)

#     hull           = cv2.convexHull(contour)
#     hull_area      = cv2.contourArea(hull)
#     hull_perimeter = cv2.arcLength(hull, True)

#     area          = raw_area * area_scale
#     hull_area_val = hull_area * area_scale

#     if perimeter > 0 and raw_area > 0:
#         deformation = 1 - 2 * math.sqrt(math.pi * raw_area) / perimeter
#     else:
#         deformation = 0.0

#     x, y, w, h  = cv2.boundingRect(contour)
#     length       = max(w, h)
#     height       = min(w, h)
#     aspect_ratio = (length / height) if height > 0 else 0.0

#     circularity = (4.0 * math.pi * raw_area / (perimeter * perimeter)) if perimeter > 0 else 0.0
#     solidity    = (raw_area / hull_area)       if hull_area > 0 else 0.0
#     porosity    = (hull_area / raw_area)       if raw_area  > 0 else 0.0
#     convexity   = (hull_perimeter / perimeter) if perimeter > 0 else 0.0
#     extent      = (raw_area / (w * h))         if (w > 0 and h > 0) else 0.0
#     eq_diameter = math.sqrt(4.0 * area / math.pi) if area > 0 else 0.0

#     rect = cv2.minAreaRect(contour)
#     (rect_w, rect_h) = rect[1]
#     feret_max_px = float(max(rect_w, rect_h))
#     feret_min_px = float(min(rect_w, rect_h))

#     inertia_ratio   = 0.0
#     eccentricity    = 0.0
#     orientation_deg = 0.0

#     if len(contour) >= 5:
#         try:
#             (cx, cy), (MA, ma), angle = cv2.fitEllipse(contour)
#             major = max(MA, ma)
#             minor = min(MA, ma)
#             if minor > 0:
#                 inertia_ratio = (major / minor)
#                 eccentricity  = math.sqrt(max(0.0, 1.0 - (minor * minor) / (major * major)))
#             orientation_deg = float(angle)
#         except cv2.error:
#             m = cv2.moments(contour)
#             if m['m00'] != 0:
#                 cov  = np.array([[m['mu20']/m['m00'], m['mu11']/m['m00']],
#                                   [m['mu11']/m['m00'], m['mu02']/m['m00']]], dtype=float)
#                 vals, vecs = np.linalg.eig(cov)
#                 vals = np.sort(vals)[::-1]
#                 if len(vals) == 2 and vals[1] > 0:
#                     inertia_ratio = math.sqrt(vals[0] / vals[1])
#                 v = vecs[:, np.argmax(vals)]
#                 orientation_deg = math.degrees(math.atan2(v[1], v[0]))
#     else:
#         m = cv2.moments(contour)
#         if m['m00'] != 0:
#             cov  = np.array([[m['mu20']/m['m00'], m['mu11']/m['m00']],
#                               [m['mu11']/m['m00'], m['mu02']/m['m00']]], dtype=float)
#             vals, vecs = np.linalg.eig(cov)
#             vals = np.sort(vals)[::-1]
#             if len(vals) == 2 and vals[1] > 0:
#                 inertia_ratio = math.sqrt(vals[0] / vals[1])
#             v = vecs[:, np.argmax(vals)]
#             orientation_deg = math.degrees(math.atan2(v[1], v[0]))

#     return {
#         'Area': area,
#         'HullArea': hull_area_val,
#         'HullPerimeter_px': hull_perimeter,
#         'Perimeter_px': perimeter,
#         'Length_px': length,
#         'Height_px': height,
#         'AspectRatio': aspect_ratio,
#         'Deformation': deformation,
#         'Circularity': circularity,
#         'Solidity': solidity,
#         'Porosity': porosity,
#         'Convexity': convexity,
#         'Extent': extent,
#         'EquivalentDiameter': eq_diameter,
#         'InertiaRatio': inertia_ratio,
#         'Eccentricity': eccentricity,
#         'Orientation_deg': orientation_deg,
#         'FeretMax_px': feret_max_px,
#         'FeretMin_px': feret_min_px,
#     }


# # ============================================================================
# # OBJECT SEPARATION
# # ============================================================================

# def separate_mask_into_objects(mask_array):
#     mask = np.asarray(mask_array, dtype=np.int32)
#     uniq = np.unique(mask)
#     uniq = uniq[uniq > 0]
#     objects = []

#     if len(uniq) == 0:
#         return objects

#     if len(uniq) == 1 and uniq[0] == 1:
#         binmask = (mask == 1).astype(np.uint8) * 255
#         contours, _ = cv2.findContours(binmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#         for i, c in enumerate(contours, start=1):
#             obj_mask = np.zeros_like(mask, dtype=np.int32)
#             cv2.drawContours(obj_mask, [c], 0, i, -1)
#             objects.append((i, obj_mask, c))
#     else:
#         for lab in uniq:
#             labmask = (lab == mask).astype(np.uint8) * 255
#             contours, _ = cv2.findContours(labmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#             if not contours:
#                 continue
#             c = max(contours, key=cv2.contourArea)
#             obj_mask = np.zeros_like(mask, dtype=np.int32)
#             cv2.drawContours(obj_mask, [c], 0, int(lab), -1)
#             objects.append((int(lab), obj_mask, c))

#     return objects


# # ============================================================================
# # MAIN PROCESSING PIPELINE
# # ============================================================================

# def postprocess_for_new_data(db_in,
#                              model_keys=('Ground_truth', 'Unet'),
#                              area_scale=1/4.0,
#                              output_dir=None,
#                              report_handle=None,
#                              verbose=True):
#     os.makedirs(output_dir, exist_ok=True)
#     stats_report  = defaultdict(lambda: defaultdict(dict))
#     coverage_data = []

#     for pid, pdata in db_in.items():
#         if verbose:
#             log_message(f"\n{'='*70}\nProcessing patient: {pid}\n{'='*70}", report_handle)

#         images = _to_list_of_arrays(pdata.get('image'))
#         if not images:
#             log_message(f"[{pid}] No 'image' found — skipping.", report_handle)
#             continue

#         for model_key in model_keys:
#             if model_key not in pdata:
#                 log_message(f"  Model {model_key}: not present, skipping.", report_handle)
#                 continue

#             masks_list = _to_list_of_arrays(pdata[model_key])
#             if not masks_list:
#                 log_message(f"  Model {model_key}: empty, skipping.", report_handle)
#                 continue

#             N = min(len(images), len(masks_list))
#             log_message(f"\n  Model: {model_key}", report_handle)
#             log_message(f"  Frames: images={len(images)} masks={len(masks_list)} -> using {N}", report_handle)

#             masked_images_list = []
#             morphology_rows    = []
#             total_objects      = 0

#             for img_idx in range(N):
#                 mask     = np.asarray(masks_list[img_idx])
#                 orig_img = np.asarray(images[img_idx])
#                 objects  = separate_mask_into_objects(mask)
#                 total_objects += len(objects)

#                 if verbose:
#                     print(f"    Image {img_idx}: {len(objects)} objects")

#                 for obj_label, obj_mask, contour in objects:
#                     feats      = contour_features(contour, area_scale=area_scale)
#                     masked_img = orig_img * (obj_mask > 0)
#                     masked_images_list.append(masked_img)

#                     morph_row = {
#                         'Mask_ID':        f"{pid}_{model_key}_obj{obj_label}",
#                         'Patient_ID':     pid,
#                         'Group':          'CLL' if pid.startswith('CLL') else 'Control',
#                         'SourceModel':    model_key,
#                         'SourceImageIdx': img_idx,
#                         'OriginalLabel':  obj_label,
#                     }
#                     morph_row.update(feats)
#                     morphology_rows.append(morph_row)

#             pdata[f'{model_key}_masked_images'] = masked_images_list

#             if morphology_rows:
#                 df = pd.DataFrame(morphology_rows)
#                 pdata[f'{model_key}_morphology'] = df
#                 csv_path = os.path.join(output_dir, f"{pid}_{model_key}_morphology.csv")
#                 df.to_csv(csv_path, index=False)
#                 log_message(f"    Saved morphology CSV: {csv_path}", report_handle)
#             else:
#                 df = pd.DataFrame(columns=['Mask_ID','Patient_ID','Group','SourceModel','SourceImageIdx','OriginalLabel','Area'])
#                 pdata[f'{model_key}_morphology'] = df

#             coverage_data.append({
#                 'Patient_ID':    pid,
#                 'Model':         model_key,
#                 'Frames':        N,
#                 'Objects_Found': total_objects,
#             })

#             stats_report[pid][model_key] = {
#                 'input_frames':              N,
#                 'total_objects':             total_objects,
#                 'morphology_features_count': len(morphology_rows),
#             }

#             log_message(f"    ✓ Objects extracted: {total_objects}", report_handle)

#     return stats_report, coverage_data


# # ============================================================================
# # VISUALIZATION
# # ============================================================================

# def plot_coverage(coverage_df, output_dir, report_handle=None):
#     """
#     Objects per frame (Objects_Found / Frames) for each patient,
#     one panel per model, patients colored by group (CLL vs Control).
#     Saved as coverage_objects_per_frame.png
#     """
#     df = coverage_df.copy()
#     df['Objects_per_Frame'] = df['Objects_Found'] / df['Frames']
#     df['Group'] = df['Patient_ID'].apply(lambda x: 'CLL' if x.startswith('CLL') else 'Control')

#     models     = df['Model'].unique()
#     group_colors = {'CLL': '#E07070', 'Control': '#6FA8D6'}

#     fig, axes = plt.subplots(1, len(models), figsize=(max(14, len(df['Patient_ID'].unique()) // 2), 5),
#                              sharey=True)
#     if len(models) == 1:
#         axes = [axes]

#     for ax, model in zip(axes, models):
#         sub = df[df['Model'] == model].sort_values('Patient_ID')
#         colors = [group_colors[g] for g in sub['Group']]

#         bars = ax.bar(range(len(sub)), sub['Objects_per_Frame'], color=colors, edgecolor='white', linewidth=0.5)

#         # mean lines per group
#         for grp, col in group_colors.items():
#             mean_val = sub.loc[sub['Group'] == grp, 'Objects_per_Frame'].mean()
#             if not np.isnan(mean_val):
#                 ax.axhline(mean_val, color=col, linestyle='--', linewidth=1.5,
#                            label=f'{grp} mean = {mean_val:.2f}')

#         ax.set_xticks(range(len(sub)))
#         ax.set_xticklabels(sub['Patient_ID'], rotation=90, fontsize=7)
#         ax.set_title(model, fontsize=12)
#         ax.set_xlabel("Patient", fontsize=11)
#         ax.set_ylabel("Objects per Frame", fontsize=11)
#         ax.grid(axis='y', alpha=0.3)
#         ax.legend(fontsize=9)

#     # legend patches
#     from matplotlib.patches import Patch
#     legend_elements = [Patch(facecolor=group_colors['CLL'],     label='CLL'),
#                        Patch(facecolor=group_colors['Control'], label='Control')]
#     fig.legend(handles=legend_elements, loc='upper right', fontsize=10)

#     plt.suptitle("Objects Detected per Frame", fontsize=14, y=1.02)
#     plt.tight_layout()

#     out_path = Path(output_dir) / "coverage_objects_per_frame.png"
#     plt.savefig(out_path, dpi=150, bbox_inches="tight")
#     plt.close()

#     log_message(f"  ✓ Coverage plot saved: {out_path}", report_handle)
#     return out_path


# def _scatter_kde_plot(df, title, out_path, feat_x_pair, feat_y_pair):
#     """2D scatter + KDE overlay only."""
#     x_col, x_label = feat_x_pair
#     y_col, y_label = feat_y_pair

#     sub = df[[x_col, y_col]].dropna()
#     if sub.empty:
#         return

#     fig, ax = plt.subplots(figsize=(7, 5))

#     try:
#         sns.kdeplot(x=sub[x_col], y=sub[y_col],
#                     ax=ax, fill=True, cmap="YlGnBu", levels=15, zorder=1)
#     except Exception:
#         pass

#     ax.scatter(sub[x_col], sub[y_col], s=6, alpha=0.3, color="indigo", zorder=2)
#     ax.set_xlabel(x_label, fontsize=13)
#     ax.set_ylabel(y_label, fontsize=13)
#     ax.set_title(f"{x_label} vs {y_label}", fontsize=14)
#     if y_col == "Deformation":
#         ax.set_ylim(0, 0.2)
#     ax.grid(True, alpha=0.3)

#     plt.suptitle(title, fontsize=15)
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=150, bbox_inches="tight")
#     plt.close()


# FEATURE_PAIRS = [
#     (("Area", "Area (μm²)"),  ("Deformation", "Deformation")),
#     (("Area", "Area (μm²)"),  ("AspectRatio",  "Aspect Ratio")),
# ]


# def generate_all_plots(db, model_keys, output_dir, report_handle=None):
#     plot_dir        = Path(output_dir) / "plots"
#     per_patient_dir = plot_dir / "per_patient"
#     group_dir       = plot_dir / "group_level"

#     plot_dir.mkdir(parents=True, exist_ok=True)
#     per_patient_dir.mkdir(exist_ok=True)
#     group_dir.mkdir(exist_ok=True)

#     all_dfs = {mk: [] for mk in model_keys}

#     for pid, pdata in db.items():
#         group = 'CLL' if pid.startswith('CLL') else 'Control'

#         for model_key in model_keys:
#             morph_key = f'{model_key}_morphology'
#             if morph_key not in pdata:
#                 continue
#             df = pdata[morph_key]
#             if df.empty:
#                 continue

#             if 'Group' not in df.columns:
#                 df = df.copy()
#                 df['Group'] = group

#             all_dfs[model_key].append(df)

#             pat_dir = per_patient_dir / pid
#             pat_dir.mkdir(exist_ok=True)

#             for (x_feat, x_label), (y_feat, y_label) in FEATURE_PAIRS:
#                 fname = f"{pid}_{model_key}_{x_feat}_vs_{y_feat}.png"
#                 _scatter_kde_plot(
#                     df=df,
#                     title=f"{pid} | {model_key}",
#                     out_path=pat_dir / fname,
#                     feat_x_pair=(x_feat, x_label),
#                     feat_y_pair=(y_feat, y_label),
#                 )

#     log_message("  ✓ Per-patient plots done", report_handle)

#     for model_key in model_keys:
#         if not all_dfs[model_key]:
#             continue

#         combined = pd.concat(all_dfs[model_key], ignore_index=True)

#         for group_name, group_filter in [
#             ("CLL",     combined['Group'] == 'CLL'),
#             ("Control", combined['Group'] == 'Control'),
#             ("All",     pd.Series([True] * len(combined))),
#         ]:
#             sub = combined[group_filter]
#             if sub.empty:
#                 continue

#             for (x_feat, x_label), (y_feat, y_label) in FEATURE_PAIRS:
#                 fname = f"{group_name}_{model_key}_{x_feat}_vs_{y_feat}.png"
#                 _scatter_kde_plot(
#                     df=sub,
#                     title=f"{group_name} Patients | {model_key}",
#                     out_path=group_dir / fname,
#                     feat_x_pair=(x_feat, x_label),
#                     feat_y_pair=(y_feat, y_label),
#                 )

#     log_message("  ✓ Group-level plots (CLL / Control / All) done", report_handle)
#     log_message(f"  ✓ All plots saved in: {plot_dir}", report_handle)

#     return plot_dir


# # ============================================================================
# # MAIN EXECUTION
# # ============================================================================

# def main():
#     Config.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

#     with open(Config.REPORT_FILE, 'w') as report_file:
#         log_message("="*70, report_file)
#         log_message("MORPHOLOGICAL FEATURE EXTRACTION REPORT", report_file)
#         log_message("="*70, report_file)
#         log_message(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", report_file)
#         log_message("", report_file)

#         log_message("CONFIGURATION:", report_file)
#         log_message("-"*70, report_file)
#         log_message(f"Input file:       {Config.INPUT_PKL}", report_file)
#         log_message(f"Output directory: {Config.OUTPUT_ROOT}", report_file)
#         log_message(f"Final database:   {Config.FINAL_DB_OUT}", report_file)
#         log_message(f"Area scale:       {Config.AREA_SCALE} (1 pixel = 0.5 μm, 1 pixel² = 0.25 μm²)", report_file)
#         log_message(f"Model keys:       {Config.MODEL_KEYS}", report_file)
#         log_message("", report_file)

#         log_message("="*70, report_file)
#         log_message("LOADING DATA", report_file)
#         log_message("="*70, report_file)

#         with open(Config.INPUT_PKL, 'rb') as f:
#             db = pickle.load(f)

#         log_message(f"✓ Loaded database with {len(db)} patients", report_file)
#         log_message(f"Patient IDs: {list(db.keys())}", report_file)
#         log_message("", report_file)

#         log_message("="*70, report_file)
#         log_message("MORPHOLOGICAL FEATURE EXTRACTION", report_file)
#         log_message("="*70, report_file)

#         stats, coverage_data = postprocess_for_new_data(
#             db,
#             model_keys=Config.MODEL_KEYS,
#             area_scale=Config.AREA_SCALE,
#             output_dir=str(Config.OUTPUT_ROOT),
#             report_handle=report_file,
#             verbose=True
#         )

#         log_message("", report_file)
#         log_message("✓ Morphology extraction complete!", report_file)
#         log_message("", report_file)

#         log_message("="*70, report_file)
#         log_message("COVERAGE SUMMARY", report_file)
#         log_message("="*70, report_file)

#         coverage_df = pd.DataFrame(coverage_data)
#         log_message("", report_file)
#         log_message(coverage_df.to_string(index=False), report_file)
#         log_message("", report_file)

#         coverage_csv_path = Config.OUTPUT_ROOT / "coverage_summary.csv"
#         coverage_df.to_csv(coverage_csv_path, index=False)
#         log_message(f"✓ Coverage table saved to: {coverage_csv_path}", report_file)
#         log_message("", report_file)

#         log_message("="*70, report_file)
#         log_message("GENERATING PLOTS", report_file)
#         log_message("="*70, report_file)

#         # Coverage plot
#         plot_coverage(coverage_df, output_dir=str(Config.OUTPUT_ROOT), report_handle=report_file)

#         # Morphology scatter/KDE plots
#         plot_dir = generate_all_plots(
#             db=db,
#             model_keys=Config.MODEL_KEYS,
#             output_dir=str(Config.OUTPUT_ROOT),
#             report_handle=report_file,
#         )

#         log_message("", report_file)
#         log_message("="*70, report_file)
#         log_message("SAVING RESULTS", report_file)
#         log_message("="*70, report_file)

#         with open(Config.FINAL_DB_OUT, 'wb') as f:
#             pickle.dump(db, f, protocol=pickle.HIGHEST_PROTOCOL)
#         log_message(f"✓ Final database saved to: {Config.FINAL_DB_OUT}", report_file)
#         log_message("", report_file)

#         log_message("="*70, report_file)
#         log_message("SUMMARY", report_file)
#         log_message("="*70, report_file)

#         total_patients = len(stats)
#         total_objects  = sum(
#             model_stats['total_objects']
#             for patient_stats in stats.values()
#             for model_stats in patient_stats.values()
#         )

#         log_message(f"  • Patients processed:      {total_patients}", report_file)
#         log_message(f"  • Total objects extracted: {total_objects}", report_file)
#         log_message(f"  • Morphology CSVs:         {Config.OUTPUT_ROOT}/*.csv", report_file)
#         log_message(f"  • Coverage plot:           {Config.OUTPUT_ROOT}/coverage_objects_per_frame.png", report_file)
#         log_message(f"  • Plots:                   {plot_dir}/", report_file)
#         log_message(f"    - per_patient/  → one folder per patient", report_file)
#         log_message(f"    - group_level/  → CLL / Control / All combined", report_file)
#         log_message(f"  • Final database:           {Config.FINAL_DB_OUT}", report_file)
#         log_message(f"  • This report:              {Config.REPORT_FILE}", report_file)
#         log_message("", report_file)

#         log_message("="*70, report_file)
#         log_message("PROCESSING COMPLETE!", report_file)
#         log_message(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", report_file)
#         log_message("="*70, report_file)

#     print(f"\n✓ Report saved to: {Config.REPORT_FILE}")
#     print("\n" + "="*70)
#     print("ALL PROCESSING COMPLETE!")
#     print("="*70)


# if __name__ == "__main__":
#     main()


