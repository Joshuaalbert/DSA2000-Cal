import os
import subprocess


def main():
    if not os.path.exists("/dsa/data/parset.yaml"):
        raise ValueError("Parset file must be present in /dsa/data/parset.yaml.")
    if not os.path.exists("/dsa/data/skymodel.bbs"):
        raise ValueError("Sky model must be present in /dsa/data/skymodel.bbs.")

    print("Converting sky model to Tigger.")
    completed_process = subprocess.run(
        [
            "tigger-convert",
            "-t", "BBS",
            "-o", "Tigger",
            "/dsa/data/skymodel.bbs",  # the input skymodel
            "bright_sky_model.lsm.html"
        ]
    )
    if completed_process.returncode != 0:
        raise RuntimeError(f"tigger-convert failed with return code {completed_process.returncode}")

    print("Running goquartical.")

    completed_process = subprocess.run(
        [
            "goquartical",
            f"/dsa/data/parset.yaml"
        ]
    )

    if completed_process.returncode != 0:
        raise RuntimeError(f"command failed with return code {completed_process.returncode}")


if __name__ == '__main__':
    main()
