import os

from astropy import units as au

from dsa2000_common.common.astropy_utils import extract_itrs_coords


def test_extract_itrs_coords():
    with open('test_array.txt', 'w') as f:
        f.write('#X Y Z dish_diam station mount\n'
                '-1601614.0612 -5042001.67655 3554652.4556 25 vla-00 ALT-AZ\n'
                '-1602592.82353 -5042055.01342 3554140.65277 25 vla-01 ALT-AZ\n'
                '-1604008.70191 -5042135.83581 3553403.66677 25 vla-02 ALT-AZ\n'
                '-1605808.59818 -5042230.07046 3552459.16736 25 vla-03 ALT-AZ')
    stations, antenna_coords = extract_itrs_coords('test_array.txt')
    assert len(stations) == 4
    assert len(antenna_coords) == 4
    assert antenna_coords[0].x == -1601614.0612 * au.m
    assert antenna_coords[0].y == -5042001.67655 * au.m
    assert antenna_coords[0].z == 3554652.4556 * au.m
    assert stations == ['vla-00', 'vla-01', 'vla-02', 'vla-03']
    os.remove('test_array.txt')
