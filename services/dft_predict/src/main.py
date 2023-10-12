from africanus.dft import dask


# Creates an empty Measurement Set using simms and then does a DFT prediction of visibilities for given sky model.


def main():
    visibilities = dask.im_to_vis(
        image=...,  # [source, chan, corr]
        uvw=...,  # [row, 3]
        lm=...,  # [source, 2]
        frequency=...,  # [chan]
        convention='casa',
    )  # [row, chan, corr]


if __name__ == '__main__':
    main()
