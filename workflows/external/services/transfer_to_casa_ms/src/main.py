import os

from dsa2000_cal.adapter.to_casa_ms import transfer_to_casa
from src.dsa2000_cal.measurement_sets import MeasurementSet


def main(input_ms, output_ms):
    measurement_set = MeasurementSet(input_ms)
    transfer_to_casa(measurement_set, output_ms)


if __name__ == '__main__':
    input_ms = os.environ.get('INPUT_MEASUREMENT_SET')
    output_ms = os.environ.get('OUTPUT_MEASUREMENT_SET')
    if not os.path.exists(input_ms):
        raise FileNotFoundError(f"Input Measurement Set {input_ms} does not exist")
    main(input_ms, output_ms)
