from dsa2000_cal.models.run_config import PrepareRunConfig
from dsa2000_cal.utils import build_example


def test_run_config():
    run_config = build_example(PrepareRunConfig)

    with open('../src/tests/prepare_run_config.json', 'w') as f:
        f.write(run_config.json(indent=2))

    with open('../src/tests/schema.json', 'w') as f:
        f.write(run_config.schema_json(indent=2))
