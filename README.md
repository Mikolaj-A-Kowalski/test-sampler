[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![build](https://github.com/Mikolaj-A-Kowalski/test-sampler/actions/workflows/build.yml/badge.svg)](https://github.com/Mikolaj-A-Kowalski/test-sampler/actions/workflows/build.yml)
![crates.io](https://img.shields.io/crates/v/test-sampler.svg)


# test-sampler: Unit Test Tool for Sampling Algorithms

The package is intended to provide utilities to unit test sampling algorithms.

In particular, when developing Monte Carlo particle transport codes, one needs
to write procedures to sample a large number of complex distributions that
describe different physical interactions (various types of scattering, fissions etc.).
The models are usually well described by differential cross-sections
$`\frac{d\sigma}{dE'd\Omega}(E)`$ for which explicit expressions exist.
However, in practice getting the normalisation to convert cross-section
into the probability density function (pdf) may require tricky numerical integration.
Thus, in short, getting the shape of pdf is often easy, getting the pdf or cumulative
distribution function (cdf) is not.

This package provides two components to enable unit testing of sampling algorithms.
- An `universal sampler` which can draw samples from an arbitrary shape function that
  describes the continuous 1D distribution on a bounded support.
- A selection of two-sample statistical tests to verify that two samples were
  drawn from the same distribution.

Thus one can use the `universal sampler` to (inefficiently) generate reference
set of samples from a shape function and compare them against the 'production'
algorithm.

# Compile docs
The package is using [Katex](https://katex.org/) to render equations as the
result to compile documentation locally extra flags need to be provided to `cargo`:
```bash
export RUSTDOCFLAGS="--html-in-header <path-to-repo>/katex-header.html"
```
To compile docs from the root of the repositry run::
```bash
RUSTDOCFLAGS="--html-in-header katex-header.html" cargo doc --no-deps
```
