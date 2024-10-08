## RIME Conventions

There are two conventions for the RIME supported by this codebase. They are referred to as the `physical`
and `engineering` conventions, respectively. The default is the `physical` convention. Be mindful of which you desire
for interfacing with other tools.

### Physical Convention

With `u,v,w` in units of wavelength, the visibilities due to sources outside our solar system are given by the following
equation:

```
V(u,v,w) = ∫∫ I(l,m)/n(l,m) exp{-2πi(ul+vm+w(n(l,m)-1))} dl dm
```

where `n(l,m) = sqrt(1 - l^2 - m^2)`. The "dirty" image is given by the adjoint of the above equation:

```
I(l,m)/n(l,m) = ∫∫∫ V(u,v,w) exp{2πi(ul+vm+w(n-1))} du dv dw
```

### Engineering Convention

This convention uses the opposite sign in the exponent of the Fourier Transform, that is, the visibilities due to
sources outside our solar system are given by the following equation:

```
V(u,v,w) = ∫∫ I(l,m)/n(l,m) exp{2πi(ul+vm+w(n(l,m)-1))} dl dm
```

the "dirty" image is given by the adjoint of the above equation:

```
I(l,m)/n(l,m) = ∫∫∫ V(u,v,w) exp{-2πi(ul+vm+w(n-1))} du dv dw
```

### Changing Conventions

Internally, to go between conventions we negate the `uvw` coordinates for the `engineering` convention. As per an
excerpt from [CASA Memo 229](https://casa.nrao.edu/Memos/229.html): _"Note that the choice of baseline
direction and UVW definition (W towards source direction; V in plane through source and system's pole; U in direction of
increasing longitude coordinate) also determines the sign of the phase of the recorded data."_

Thus, we state clearly here:
**When the UVW are defined as `antenna2 - antenna1`, the `physical` convention must be used. When the UVW are defined as
`antenna1 - antenna2`, the `engineering` convention must be used.**

Different choices have been historically made as to the direction of baselines, in the memo [Convention for UVW
calculations in CASA](https://drive.google.com/file/d/1a-eUwNrfnYjaUQTjJDfOjJCa8ZaSzZcn/view) the choice
is `antenna1 - antenna2`, whereas
in [CASA Memo 229](https://casa.nrao.edu/Memos/229.html) it's `antenna2 - antenna1`.

