import os

from dsa2000_cal.adapter.from_casa_ms import transfer_from_casa


def main(ms_folder: str, casa_ms: str):
    ms = transfer_from_casa(
        ms_folder=ms_folder,
        casa_ms=casa_ms
    )
    print(f"Created {ms}")


if __name__ == '__main__':
    casa_ms = os.environ.get('INPUT_CASA_MS')
    output_ms = os.environ.get('OUTPUT_MEASUREMENT_SET')
    if not os.path.exists(casa_ms):
        raise FileNotFoundError(f"Input Measurement Set {casa_ms} does not exist")
    main(ms_folder=output_ms, casa_ms=casa_ms)
