# Standalone Calibration

There are two directories defined in the `.env`:

1. `DATA_DIR_HOST` - this is where the data is stored, and is mapped to `/dsa/run` in the container.
2. `RUN_DIR_HOST` - this is where the output of the calibration is stored, and is mapped to `/dsa/data` in the
   container.

Inside the `DATA_DIR_HOST` you must place the following files:

1. `parset.yaml` - this is the parset file that defines the calibration run. Within this you must reference data dir
   as `/dsa/data` and run dir as `/dsa/run`.
2. `skymodel.bbs` - this is the sky model that is used for the calibration in BBS format. It will be converted to Tigger
   format and stored in `bright_sky_model.lsm.html`.
3. `visibilities.ms` - this is the measurement set that is used for the calibration. You can call it whatever you like.
   The `parset.yaml` references this name.

This creates a `calibration` folder in the `RUN_DIR_HOST` that contains the solutions and logs after the run.

To run the calibration, you adjust the `services_order` array in `run.sh` and run:

```bash
./run.sh
```

TODO: make run.sh take service as argument.
