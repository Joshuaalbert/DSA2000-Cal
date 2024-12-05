from dsa2000_casa.adapter.from_casa_ms import transfer_from_casa


def _test_transfer_from_casa():
    casa_file = '~/data/forward_modelling/data_dir/lwa01.ms'
    ms_folder = '~/data/forward_modelling/data_dir/lwa01_ms'
    ms = transfer_from_casa(
        ms_folder=ms_folder,
        casa_ms=casa_file,
        convention='engineering'  # Or else UVW coordinates are very wrong.
    )


