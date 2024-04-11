import numpy as np
from astropy import units as au

from dsa2000_cal.source_models.wsclean_util import parse_wsclean_source_line, parse_and_process_wsclean_source_line


def test_correct_parsing():
    line = "s5c502,GAUSSIAN,-00:36:28.234,58.50.46.396,5.02217367702533,[8.16220121137914,-27.0241410999176,-27.1300759829061,13.2798319227506],false,57322692.8710938,70.6446013509285,70.6446013509285,0"
    result = parse_wsclean_source_line(line)
    expected = {
        'Name': 's5c502',
        'Type': 'GAUSSIAN',
        'Ra': '-00:36:28.234',
        'Dec': '58.50.46.396',
        'I': 5.02217367702533,
        'SpectralIndex': [8.16220121137914, -27.0241410999176, -27.1300759829061, 13.2798319227506],
        'LogarithmicSI': False,
        'ReferenceFrequency': 57322692.8710938,
        'MajorAxis': 70.6446013509285,
        'MinorAxis': 70.6446013509285,
        'Orientation': 0
    }
    assert result == expected


def test_missing_values():
    line = "s0c0,POINT,-00:37:22.645,58.30.45.773,0.0155566877299659,[-0.0165136382322324,0.104932421771313,0.104196786226242,-0.751510783409754],false,57322692.8710938,,,"
    result = parse_wsclean_source_line(line)
    expected = {
        'Name': 's0c0',
        'Type': 'POINT',
        'Ra': '-00:37:22.645',
        'Dec': '58.30.45.773',
        'I': 0.0155566877299659,
        'SpectralIndex': [-0.0165136382322324, 0.104932421771313, 0.104196786226242, -0.751510783409754],
        'LogarithmicSI': False,
        'ReferenceFrequency': 57322692.8710938,
        'MajorAxis': None,
        'MinorAxis': None,
        'Orientation': None
    }
    assert result == expected


def test_type_conversion():
    line = "s1c1,POINT,00:00:00,00.00.00.000,1,[1.0,2.0,3.0],true,1000000,10,20,30"
    result = parse_wsclean_source_line(line)
    assert isinstance(result['I'], float)
    assert isinstance(result['SpectralIndex'], list)
    assert all(isinstance(si, float) for si in result['SpectralIndex'])
    assert isinstance(result['LogarithmicSI'], bool)
    assert isinstance(result['ReferenceFrequency'], float)
    assert isinstance(result['MajorAxis'], float)
    assert isinstance(result['MinorAxis'], float)
    assert isinstance(result['Orientation'], float)


def test_variable_length_spectral_index():
    line = "s1c1,POINT,00:00:00,00.00.00.000,1,[],true,1000000,10,20,30"
    result = parse_wsclean_source_line(line)
    assert result['SpectralIndex'] == []

    line = "s1c1,POINT,00:00:00,00.00.00.000,1,[1,2,3,4,5,6],true,1000000,10,20,30"
    result = parse_wsclean_source_line(line)
    assert result['SpectralIndex'] == [1, 2, 3, 4, 5, 6]


def test_parse_and_process_wsclean_source_line():
    line = "s1c1,GAUSSIAN,00:00:00,00.00.00.000,1,[1,2,3],true,1000000,10,20,30"
    freqs = au.Quantity([700e6, 800e6], 'Hz')
    wsclean_source = parse_and_process_wsclean_source_line(line, freqs=freqs)
    assert wsclean_source.type_ == 'GAUSSIAN'
    assert wsclean_source.direction.ra.deg == 0
    assert wsclean_source.direction.dec.deg == 0
    assert np.all(wsclean_source.spectrum > 0)
    assert len(wsclean_source.spectrum) == 2
    assert wsclean_source.major.unit.is_equivalent(au.rad)
    assert wsclean_source.minor.unit.is_equivalent(au.rad)
    assert wsclean_source.theta.unit.is_equivalent(au.rad)


def test_empty_input():
    line = ""
    result = parse_wsclean_source_line(line)
    assert result is None
