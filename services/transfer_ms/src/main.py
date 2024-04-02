import os

from dsa2000_cal.assets.content_registry import fill_registries

fill_registries()

from dsa2000_cal.adapter.create_casa_ms import transfer_visibilities
from dsa2000_cal.measurement_sets.measurement_set import MeasurementSet


def main(input_ms, output_ms):
    measurement_set = MeasurementSet(input_ms)
    transfer_visibilities(measurement_set, output_ms)


if __name__ == '__main__':
    input_ms = os.environ.get('INPUT_MEASUREMENT_SET')
    output_ms = os.environ.get('OUTPUT_MEASUREMENT_SET')
    if not os.path.exists(input_ms):
        raise FileNotFoundError(f"Input Measurement Set {input_ms} does not exist")
    main(input_ms, output_ms)
