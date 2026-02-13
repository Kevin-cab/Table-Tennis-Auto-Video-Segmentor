#!/usr/bin/env python3
"""
video_segmentor_with_fades.py

Full segmentation pipeline that handles fades/dissolves properly:
    - scene detection with multiple detectors
    - refine scenes by detecting fades/dissolves and inserting splits
    - save segments (optional)
    - filter clips (static ads, slow-motion) using existing modules
"""

import os
import glob
import math
import itertools
import shutil
import cv2
import numpy as np
import pandas as pd

from scenedetect import SceneManager, open_video
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.detectors import ContentDetector, ThresholdDetector, AdaptiveDetector, HistogramDetector
from scenedetect.video_splitter import split_video_ffmpeg

# User-provided modules (must exist)
from static_ad_remover import is_static_ad
from slowmo_remover import is_slow_motion, SLOWMO_SHOT_REMOVE_THRESHOLD

# ----------------- DEFAULT CONFIG -----------------
DEFAULT_HISTOGRAM_BINS = 32
DEFAULT_FADE_CORR_THRESHOLD = 0.92
DEFAULT_CORR_SMOOTH_WINDOW = 5
DEFAULT_MIN_FADE_FRAMES = 3
DEFAULT_MAX_FADE_FRAMES = 300
DEFAULT_LUMINANCE_BLACK_THRESH = 30       # Y channel threshold for near-black
DEFAULT_MIN_BLACK_RUN = 3
DEFAULT_INSERT_MARGIN_SECS = 0.03         # don't insert splits within this margin of existing boundaries
# --------------------------------------------------

def refine_scenes_with_fades(video_path, scenes,
                             hist_bins=DEFAULT_HISTOGRAM_BINS,
                             corr_smooth_window=DEFAULT_CORR_SMOOTH_WINDOW,
                             fade_corr_threshold=DEFAULT_FADE_CORR_THRESHOLD,
                             min_fade_frames=DEFAULT_MIN_FADE_FRAMES,
                             max_fade_frames=DEFAULT_MAX_FADE_FRAMES,
                             luminance_black_thresh=DEFAULT_LUMINANCE_BLACK_THRESH,
                             min_black_run=DEFAULT_MIN_BLACK_RUN,
                             insert_margin_secs=DEFAULT_INSERT_MARGIN_SECS):
    """
    Analyze the entire video to find fades/dissolves and return an updated list of scenes.
    Input `scenes` is a list of (start_timecode, end_timecode) as returned by scenedetect.
    Returns a list of (FrameTimecode(start_frame, fps), FrameTimecode(end_frame, fps)).
    """
    # Read video frames, histograms and luminance
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("  [warn] Cannot open video for fade refinement:", video_path)
        return scenes

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames < 2:
        cap.release()
        return scenes

    hist_list = []
    lum_list = []
    # read frames downscaled for speed
    read_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        small = cv2.resize(frame, (320, 240))
        # 3D histogram
        hist = cv2.calcHist([small], [0,1,2], None, [hist_bins, hist_bins, hist_bins],
                            [0,256,0,256,0,256])
        hist = hist.astype('float32')
        cv2.normalize(hist, hist)
        hist_list.append(hist.flatten())
        yuv = cv2.cvtColor(small, cv2.COLOR_BGR2YUV)
        lum_list.append(float(np.mean(yuv[:,:,0])))
        read_idx += 1
    cap.release()

    if len(hist_list) < 2:
        return scenes

    # Pairwise histogram correlation between frame i and i+1
    corrs = []
    for i in range(len(hist_list)-1):
        c = float(cv2.compareHist(hist_list[i], hist_list[i+1], cv2.HISTCMP_CORREL))
        corrs.append(c)
    corrs = np.array(corrs)

    # Smooth correlations to remove jitter
    if corr_smooth_window > 1:
        kernel = np.ones(corr_smooth_window) / corr_smooth_window
        corrs_sm = np.convolve(corrs, kernel, mode='same')
    else:
        corrs_sm = corrs

    # Detect low-correlation runs -> dissolves
    low_mask = corrs_sm < fade_corr_threshold
    runs = []
    for k, g in itertools.groupby(enumerate(low_mask), key=lambda x: x[1]):
        if not k:
            continue
        items = list(g)
        start_idx = items[0][0]
        end_idx = items[-1][0]
        length = end_idx - start_idx + 1
        if length >= min_fade_frames and length <= max_fade_frames:
            runs.append((start_idx, end_idx, length))

    # Detect fade-to/from-black using luminance
    lum_arr = np.array(lum_list)
    black_mask = lum_arr < luminance_black_thresh
    black_runs = []
    if black_mask.any():
        for k, g in itertools.groupby(enumerate(black_mask), key=lambda x: x[1]):
            if not k:
                continue
            items = list(g)
            s = items[0][0]
            e = items[-1][0]
            length = e - s + 1
            if length >= min_black_run:
                black_runs.append((s, e, length))

    proposed_frame_indices = []

    # For dissolves, pick midpoint frame (convert pair-index to frame index: +1)
    for (s, e, length) in runs:
        mid_pair = (s + e) // 2
        mid_frame = mid_pair + 1
        proposed_frame_indices.append(('dissolve', int(mid_frame), length))

    # For black runs, pick darkest frame inside run
    for (s, e, length) in black_runs:
        run_lums = lum_arr[s:(e+1)]
        rel_min = int(np.argmin(run_lums))
        min_frame = s + rel_min
        proposed_frame_indices.append(('black', int(min_frame), length))

    if not proposed_frame_indices:
        return scenes

    # Convert existing scene boundaries to frame indices
    existing_frames = set()
    for st_tc, ed_tc in scenes:
        try:
            # Timecode objects provide get_seconds(); derive frame indices
            s_frame = int(round(st_tc.get_seconds() * fps))
            e_frame = int(round(ed_tc.get_seconds() * fps))
        except Exception:
            # fallback: try FrameTimecode API
            s_frame = int(round(st_tc.get_frames())) if hasattr(st_tc, 'get_frames') else 0
            e_frame = int(round(ed_tc.get_frames())) if hasattr(ed_tc, 'get_frames') else 0
        existing_frames.add(s_frame)
        existing_frames.add(e_frame)

    margin_frames = max(1, int(round(insert_margin_secs * fps)))

    # Accept proposals not too close to existing boundaries
    accepted_frames = []
    for typ, f_idx, length in proposed_frame_indices:
        close = any(abs(f_idx - ex) <= margin_frames for ex in existing_frames)
        if close:
            continue
        if f_idx <= 1 or f_idx >= total_frames - 1:
            continue
        accepted_frames.append((typ, f_idx, length))

    if not accepted_frames:
        return scenes

    # Build final sorted list of frame boundaries merging existing boundaries + proposals
    boundary_frames = set(existing_frames)
    for _, f_idx, _ in accepted_frames:
        boundary_frames.add(f_idx)

    times_sorted = sorted(boundary_frames)

    # Create new scenes as pairs of FrameTimecode objects
    new_scenes = []
    for i in range(len(times_sorted)-1):
        st_f = times_sorted[i]
        ed_f = times_sorted[i+1]
        if ed_f - st_f <= 0:
            continue
        st_tc = FrameTimecode(st_f, fps)
        ed_tc = FrameTimecode(ed_f, fps)
        new_scenes.append((st_tc, ed_tc))

    print(f"  [fade-refine] Proposed splits: {len(accepted_frames)}, inserted new total scenes: {len(new_scenes)}")
    return new_scenes


def segment_video(video_path,
                  output_dir=None,
                  save_clips=False,
                  save_csv=False,
                  adaptive_threshold=3.0,
                  window_width=2,
                  min_content_val=15.0,
                  fade_threshold=12,
                  chi_squared_threshold=1.0,
                  histogram_bins=32,
                  min_scene_len=15,
                  enable_fade_refine=True):
    """
    Segment a single video using SceneManager and optional fade/dissolve refinement.
    Returns dict with segments (list), video_output_dir, csv_path.
    """
    if (save_clips or save_csv) and output_dir is None:
        raise ValueError("output_dir must be provided if save_clips or save_csv=True")
    if (save_clips or save_csv) and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    video_output_dir = None
    csv_path = None
    if save_clips or save_csv:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        video_output_dir = os.path.join(output_dir, f"{base_name}_segments")
        os.makedirs(video_output_dir, exist_ok=True)
        if save_csv:
            csv_path = os.path.join(video_output_dir, "segments.csv")

    # Create detectors: Adaptive + Threshold (fade hits) + Histogram (dissolve)
    adaptive_detector = AdaptiveDetector(
        adaptive_threshold=adaptive_threshold,
        window_width=window_width,
        min_content_val=min_content_val,
        min_scene_len=min_scene_len,
        weights=ContentDetector.Components(delta_hue=1.0, delta_sat=1.0, delta_lum=2.0, delta_edges=1.0)
    )
    fade_detector = ThresholdDetector(threshold=fade_threshold, min_scene_len=min_scene_len)
    histogram_detector = HistogramDetector(threshold=chi_squared_threshold, bins=histogram_bins, min_scene_len=min_scene_len)

    scene_manager = SceneManager()
    scene_manager.add_detector(adaptive_detector)
    scene_manager.add_detector(fade_detector)
    scene_manager.add_detector(histogram_detector)

    # Detect scenes
    video_stream = open_video(video_path)
    scene_manager.detect_scenes(video_stream, show_progress=True)
    scenes = scene_manager.get_scene_list()
    print(f"  [detect] Initial scenes found: {len(scenes)}")

    # Refine using fades/dissolves - insert splits if needed
    if enable_fade_refine:
        try:
            scenes_refined = refine_scenes_with_fades(video_path, scenes,
                                                      hist_bins=histogram_bins,
                                                      corr_smooth_window=5,
                                                      fade_corr_threshold=DEFAULT_FADE_CORR_THRESHOLD,
                                                      min_fade_frames=3,
                                                      max_fade_frames=DEFAULT_MAX_FADE_FRAMES,
                                                      luminance_black_thresh=DEFAULT_LUMINANCE_BLACK_THRESH,
                                                      min_black_run=DEFAULT_MIN_BLACK_RUN,
                                                      insert_margin_secs=DEFAULT_INSERT_MARGIN_SECS)
            scenes = scenes_refined
            print(f"  [refine] Scenes after fade refinement: {len(scenes)}")
        except Exception as e:
            print("  [warn] Fade refinement failed, continuing with original scenes. Error:", e)

    # Convert scenes to simple segment dicts (seconds)
    segments = []
    for i, (start_tc, end_tc) in enumerate(scenes):
        segments.append({
            'scene_id': i,
            'start_sec': float(start_tc.get_seconds()),
            'end_sec': float(end_tc.get_seconds()),
            'duration_sec': float(end_tc.get_seconds() - start_tc.get_seconds())
        })

    # Save clips via ffmpeg segmenter if requested
    if save_clips:
        try:
            split_video_ffmpeg(video_path, scenes, output_dir=video_output_dir, show_progress=True)
            print(f"  [io] Clips saved to {video_output_dir}")
        except Exception as e:
            print("  [error] split_video_ffmpeg failed:", e)

    # Save CSV if requested
    if save_csv and csv_path:
        df = pd.DataFrame(segments)
        df['video_path'] = video_path
        df.to_csv(csv_path, index=False)
        print(f"  [io] CSV saved to {csv_path}")

    print(f"  [done] Detected {len(segments)} segments in {video_path}")
    return {'segments': segments, 'video_output_dir': video_output_dir, 'csv_path': csv_path}


def batch_segment_videos(input_dir,
                         output_dir,
                         video_extensions=('.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv'),
                         recursive=True,
                         **kwargs):
    """Batch process all videos in a directory."""
    if not os.path.exists(input_dir):
        raise ValueError(f"Input directory not found: {input_dir}")

    videos = []
    pattern = '**/*' if recursive else '*'
    for ext in video_extensions:
        videos.extend(glob.glob(os.path.join(input_dir, f"{pattern}{ext}"), recursive=recursive))

    videos = sorted(set(videos))
    print(f"Found {len(videos)} video(s) to process.")

    all_results = {}
    for video_path in videos:
        print(f"\nProcessing: {video_path}")
        result = segment_video(video_path, output_dir=output_dir, **kwargs)
        all_results[video_path] = result

    # If CSVs saved, combine into master CSV
    if kwargs.get('save_csv', False):
        master_rows = []
        for res in all_results.values():
            if res.get('csv_path') and os.path.exists(res['csv_path']):
                master_rows.append(pd.read_csv(res['csv_path']))
        if master_rows:
            master_df = pd.concat(master_rows, ignore_index=True)
            master_csv = os.path.join(output_dir, "all_segments_master.csv")
            master_df.to_csv(master_csv, index=False)
            print(f"\nMaster CSV created: {master_csv}")

    print(f"\nBatch processing complete! Processed {len(videos)} videos.")
    return all_results


# ---------- Filtering function (same as your pipeline; uses imported detectors) ----------
def filter_clips_by_content(clips_dir, enable_filtering=True, sample_every_n=2, min_duration_sec=0.5, save_removed=True):
    """
    Filter out static ads, slow-motion clips, and very short clips from a directory.
    Uses is_static_ad and is_slow_motion from external modules.
    """
    if not enable_filtering:
        return {}

    if not os.path.exists(clips_dir):
        print(f"Clips directory not found: {clips_dir}")
        return {}

    clip_files = sorted([f for f in os.listdir(clips_dir)
                         if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))])

    if not clip_files:
        return {}

    # Create removed_shots folders
    parent_dir = os.path.dirname(clips_dir)
    removed_base = os.path.join(parent_dir, "removed_shots")
    removed_folders = {
        'too_short': os.path.join(removed_base, 'too_short'),
        'static_ads': os.path.join(removed_base, 'static_ads'),
        'slow_motion': os.path.join(removed_base, 'slow_motion')
    }
    if save_removed:
        for folder in removed_folders.values():
            os.makedirs(folder, exist_ok=True)
        print(f"Removed clips will be saved to: {removed_base}")

    filtered_count = 0
    duration_filtered = 0
    ad_filtered = 0
    slowmo_filtered = 0
    filter_results = {}

    for clip_file in clip_files:
        clip_path = os.path.join(clips_dir, clip_file)

        # Duration
        cap = cv2.VideoCapture(clip_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()

        # Filter by duration
        if duration < min_duration_sec:
            filtered_count += 1
            duration_filtered += 1
            if save_removed:
                dest_path = os.path.join(removed_folders['too_short'], clip_file)
                shutil.move(clip_path, dest_path)
                print(f"  ✗ Moved to removed_shots/too_short: {clip_file} - Duration: {duration:.2f}s")
            else:
                os.remove(clip_path)
                print(f"  ✗ Removed: {clip_file} - Duration: {duration:.2f}s")

            filter_results[clip_file] = {'removed': True, 'reason': f'Too short: {duration:.2f}s < {min_duration_sec}s'}
            continue

        # Static ad check (expects signature is_static_ad(path, sample_every_n))
        try:
            is_ad, static_ratio, ad_reason = is_static_ad(clip_path, sample_every_n)
        except TypeError:
            # if function uses no sample arg
            is_ad, static_ratio, ad_reason = is_static_ad(clip_path)

        # Slow-motion check (expects is_slow_motion(path, sample_every_n))
        try:
            is_slowmo, slowmo_ratio, slowmo_reason = is_slow_motion(clip_path, sample_every_n)
        except TypeError:
            is_slowmo, slowmo_ratio, slowmo_reason = is_slow_motion(clip_path)

        # Keep shot if slowmo_ratio not meeting removal threshold (user rule)
        if not is_slowmo and slowmo_ratio > 0.0:
            print(f"  ✓ Kept shot {clip_file}: slowmo_ratio={slowmo_ratio:.2%} < removal_threshold={SLOWMO_SHOT_REMOVE_THRESHOLD:.2f}")

        should_remove = is_ad or is_slowmo

        if should_remove:
            filtered_count += 1
            reason = []
            removal_category = None

            if is_ad:
                ad_filtered += 1
                reason.append(f"Static Ad: {ad_reason}")
                removal_category = 'static_ads'
            if is_slowmo:
                slowmo_filtered += 1
                reason.append(f"Slow-Mo: {slowmo_reason}")
                removal_category = 'slow_motion'

            if save_removed:
                dest_path = os.path.join(removed_folders[removal_category], clip_file)
                shutil.move(clip_path, dest_path)
                print(f"  ✗ Moved to removed_shots/{removal_category}: {clip_file} - {'; '.join(reason)}")
            else:
                os.remove(clip_path)
                print(f"  ✗ Removed: {clip_file} - {'; '.join(reason)}")

            filter_results[clip_file] = {
                'removed': True,
                'reason': '; '.join(reason),
                'filter_decision': 'remove',
                'is_slow_motion': is_slowmo,
                'slowmo_ratio': slowmo_ratio,
                'slowmo_remove_threshold': SLOWMO_SHOT_REMOVE_THRESHOLD,
                'filter_reason': '; '.join(reason)
            }
        else:
            filter_results[clip_file] = {
                'removed': False,
                'reason': 'Valid clip',
                'duration': duration,
                'filter_decision': 'keep',
                'is_slow_motion': is_slowmo,
                'slowmo_ratio': slowmo_ratio,
                'slowmo_remove_threshold': SLOWMO_SHOT_REMOVE_THRESHOLD,
                'filter_reason': f'Kept: ad={is_ad}, slowmo={is_slowmo}'
            }

    print(f"\nFiltering complete:")
    print(f"  Duration filtered (< {min_duration_sec}s): {duration_filtered}")
    print(f"  Static ads removed: {ad_filtered}")
    print(f"  Slow-motion removed: {slowmo_filtered}")
    print(f"  Total removed: {filtered_count}")
    print(f"  Clips kept: {len(clip_files) - filtered_count}")

    if save_removed and filtered_count > 0:
        print(f"\n  Removed clips saved to: {removed_base}")

    return filter_results


def post_process_segments(csv_path, min_duration_sec=10, merge_short=True):
    """Merge short segments into previous scenes and save a filtered CSV."""
    df = pd.read_csv(csv_path)
    if merge_short:
        to_merge = df[df['duration_sec'] < min_duration_sec].index.tolist()
        # Iterate in ascending order of index and merge into previous
        for idx in sorted(to_merge):
            if idx > 0 and idx < len(df):
                df.loc[idx-1, 'end_sec'] = df.loc[idx, 'end_sec']
                df = df.drop(idx)
        df = df.reset_index(drop=True)
        df['scene_id'] = df.index
        df['duration_sec'] = df['end_sec'] - df['start_sec']
    outp = csv_path.replace('.csv', '_filtered.csv')
    df.to_csv(outp, index=False)
    print(f"Filtered CSV saved: {outp}")


# ====================== ONE-CALL USAGE ======================
if __name__ == "__main__":
    result = batch_segment_videos(
        input_dir="video_data/raw_videos",
        output_dir="video_data/video_segments",
        save_clips=True,
        save_csv=True,
        adaptive_threshold=1.35,
        window_width=2,
        min_content_val=15.0,
        fade_threshold=12.0,
        min_scene_len=1,
        chi_squared_threshold=2.5,
        histogram_bins=32,
        enable_fade_refine=True,
        recursive=True
    )

    # Filter static ads and slow-motion, then post-process CSVs
    for res in result.values():
        if res.get('video_output_dir'):
            filter_clips_by_content(
                res['video_output_dir'],
                enable_filtering=True,
                sample_every_n=2,
                min_duration_sec=0.5,
                save_removed=True
            )
        if res.get('csv_path'):
            post_process_segments(res['csv_path'])

    total_segments = sum(len(r['segments']) for r in result.values())
    print(f"\nTotal segments detected across all videos: {total_segments}")
