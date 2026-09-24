"""Compatibility entry point for the maintained EVA HDF5 conversion tool.

Owner: graph integration. The former debug implementation imported a missing
extractor and wrote invented calibration. Raw conversion now delegates to the
canonical data tool; action prestacking and coordinate transforms belong in
the loader's data YAML. Keep this alias until callers migrate.
"""

import warnings
from pathlib import Path


def convert_hdf5_to_zarr(
    hdf5_path,
    zarr_episode_path,
    arm="both",
    *,
    fps=30,
    task_name="",
    task_description="",
    save_mp4=False,
    chunk_timesteps=100,
    **legacy_options,
):
    if legacy_options:
        raise ValueError(
            "The obsolete debug converter options are unsupported: "
            f"{sorted(legacy_options)}. Configure trajectory/frame transforms in "
            "data YAML; use the maintained EVA converter for raw episode storage."
        )
    warnings.warn(
        "Use egomimic.scripts.eva_process.eva_to_zarr.convert_episode; alias owner: graph integration",
        FutureWarning,
        stacklevel=2,
    )
    from egomimic.scripts.eva_process.eva_to_zarr import convert_episode

    path = Path(zarr_episode_path)
    if path.suffix != ".zarr":
        raise ValueError("Output episode must have a .zarr suffix")
    converted, _ = convert_episode(
        Path(hdf5_path),
        path.parent,
        path.stem,
        arm,
        fps,
        task_name=task_name,
        task_description=task_description,
        save_mp4=save_mp4,
        chunk_timesteps=chunk_timesteps,
    )
    return converted


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--hdf5-path", type=Path)
    group.add_argument("--hdf5-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=["left", "right", "both"], default="both")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--task-name", default="")
    parser.add_argument("--task-description", default="")
    parser.add_argument("--chunk-timesteps", type=int, default=100)
    args = parser.parse_args()
    paths = [args.hdf5_path] if args.hdf5_path else sorted(args.hdf5_dir.glob("*.hdf5"))
    if not paths:
        parser.error("No input HDF5 episodes found")
    for source in paths:
        print(
            convert_hdf5_to_zarr(
                source,
                args.output_dir / (source.stem + ".zarr"),
                args.arm,
                fps=args.fps,
                task_name=args.task_name,
                task_description=args.task_description,
                chunk_timesteps=args.chunk_timesteps,
            )
        )


if __name__ == "__main__":
    main()
