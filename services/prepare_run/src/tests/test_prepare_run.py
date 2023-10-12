from dsa2000_cal.utils import build_example
from services.prepare_run.src.main import main, PrepareRunConfig


def test_run_config():
    run_config = build_example(PrepareRunConfig)

    with open('../../example_run_config.json', 'w') as f:
        f.write(run_config.json(indent=2))

    with open('../../schema.json', 'w') as f:
        f.write(run_config.schema_json(indent=2))


def test_main():
    main(build_example(PrepareRunConfig))
