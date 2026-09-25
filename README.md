# Vector AD experiments

This repository, which was created for a presentation at
[EuroAD](https://cambridge-iccs.github.io/euroad29/index.html), contains
benchmark programs for showing the impact of vector AD. The goal is to study the
practical speedup provided by vector AD, and whether parallelisation changes the
picture.

The programs are written in [Futhark](https://futhark-lang.org) and can be
compiled to sequential CPU, multicore CPU, and GPU code.

Each program contains code for computing an objective function and the full
Jacobian, using both non-vector and vector AD. Some programs provide both
reverse and forward mode (to illustrate the difference), while others provide
only the most appropriate mode.

`ba` and `ht` have no expected results due to file size restrictions; the other
three are fully validated.

## List of benchmarks

* [ba](ba.fut) from [GradBench][] (originally [ADBench][]), which uses reverse
  mode.

* [ht](ht.fut) from [GradBench][] (originally [ADBench][], as `hand`), which
  uses forward mode.

* [greeks](greeks.fut) computes the sensitivities of a Black-Scholes European
  call price to each of its five inputs, for a portfolio of options. Provides
  both forward and reverse mode.

* [reaction-network](reaction-network.fut) computes parameter sensitivities of a
  chemical reaction network. Provides both forward and reverse mode.

* [batch-reactor](batch-reactor.fut) computes parameter sensitivities of a
  non-isothermal batch reactor. Provides both forward and reverse mode.

[GradBench]: https://github.com/gradbench/gradbench
[ADBench]: https://github.com/microsoft/ADBench

## Running

First download the dependencies:

```
$ futhark pkg sync
```

Then do one of these:

```
$ futhark bench --backend=c *.fut
$ futhark bench --backend=multicore *.fut
$ futhark bench --backend=hip *.fut
$ futhark bench --backend=cuda *.fut
$ futhark bench --backend=opencl *.fut
```

Add `--json results.json` to the end of any of these commands to produce the raw
measurement results in a machine-readable format.

To improve GPU performance, you may need to run the auto-tuner. In particular
`ht.fut` benefits tremendously from autotuning when using a GPU backend:

```
$ futhark autotune --backend=hip ht.fut
```

After doing the above, you will need to pass `--no-tuning` when not using a GPU
backend, as the tuning file is backend-specific.
