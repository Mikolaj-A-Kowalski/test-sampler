//! Provides One- and Two-Sample Kolmogorov-Smirnov test implementation
//!
use std::f64::consts::PI;
use std::iter::Iterator;
use thiserror::Error;

///
/// Error that can be raise by a statistical test
#[derive(Debug, Error)]
pub enum TestError {
    /// Case when a type to test supports only a partial order and some of the entries are not
    /// present in the order sequence (e.g. NaN in floats)
    #[error("Collection contains values that cannot be placed in an order sequence (e.g. NaNfor floats)")]
    ContainsNotSortableValues,
}

/// Empirical cumulative distribution function
///
/// Is calculated by sorting the list of samples and represents step-like
/// approximation to cdf of underling distribution.
/// The value of ecdf for some `x ∈ [xᵢ, xᵢ₊₁)` is `i/N` where `N` is the total
/// number of samples.
///
/// One can iterate over the ordered samples of ecdf and their associated values
/// as below:
/// ```
/// # use universal_sampler::ks_tests::{Ecdf, TestError};
/// # fn main() -> Result<(), TestError> {
/// let ecdf = Ecdf::new(vec![0.1, 0.0, 0.7 ,0.2])?;
///
/// for (ecdf_value, s) in &ecdf {
///     println!("{s} {ecdf_value}")
/// }
/// # Ok(())}
/// ```
///
#[derive(Debug, Clone)]
pub struct Ecdf<T>
where
    T: PartialOrd + Copy,
{
    samples: Vec<T>,
}

impl<T> Ecdf<T>
where
    T: PartialOrd + Copy,
{
    /// Create a new instance from unordered vector of samples
    ///
    pub fn new(mut samples: Vec<T>) -> Result<Self, TestError> {
        if !all_comparable(&samples) {
            return Err(TestError::ContainsNotSortableValues);
        }
        samples.sort_by(|a, b| a.partial_cmp(b).expect("Not comparable values in the grid"));
        Ok(Self { samples })
    }
}

/// Iterator over ecdf
///
/// Iterates over pairs (ecdf_value, sample) obtained from [Ecdf].
///
#[derive(Debug, Clone)]
pub struct EcdfIterator<'a, T>
where
    T: PartialOrd + Copy,
{
    ecdf: &'a Ecdf<T>,
    idx: usize,
    n: usize,
}

impl<'a, T> EcdfIterator<'a, T>
where
    T: PartialOrd + Copy,
{
    fn new(ecdf: &'a Ecdf<T>) -> Self {
        Self {
            ecdf,
            idx: 0,
            n: ecdf.samples.len(),
        }
    }
}

impl<'a, T> Iterator for EcdfIterator<'a, T>
where
    T: PartialOrd + Copy,
{
    type Item = (f64, T);
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(x) = self.ecdf.samples.get(self.idx) {
            self.idx += 1;
            Some((self.idx as f64 / self.n as f64, *x))
        } else {
            None
        }
    }
}

impl<'a, T> IntoIterator for &'a Ecdf<T>
where
    T: PartialOrd + Copy,
{
    type Item = (f64, T);
    type IntoIter = EcdfIterator<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter::new(self)
    }
}

///
/// Check that all elements in the grid are in the partial order
///
/// Note that e.g. f64 may contain NaNs which are not comparable to other numbers.
///
fn all_comparable<T>(grid: &[T]) -> bool
where
    T: PartialOrd,
{
    return grid.windows(2).all(|x| x[0].partial_cmp(&x[1]).is_some());
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KSResult {
    stat: f64,
    p: f64,
    n: usize,
}

impl KSResult {
    /// Compute complement of Kolmogorov-Smirnov Cumulative Distribution Function
    ///
    /// Computes: `Q(z) = 1 - CDF(z)`
    ///
    /// Implementation is based on the power series definitions from
    /// "Numerical Recipes" by Press et al. (2007)
    /// We select the same selection criteria for each of the two series and keep
    /// the same number of terms
    ///
    /// # Panics
    /// If the value of the statistic is outside the support
    ///
    fn complement_ks_cdf(z: f64) -> f64 {
        if z < 0.0 {
            panic!("Value of test statistic outside the support");
        } else if z == 0.0 {
            1.0
        } else if z < 1.18 {
            let factor = f64::sqrt(2.0 * PI) / z;
            let term = f64::exp(-PI * PI / 8. / (z * z));
            1.0 - factor * (term + term.powi(9) + term.powi(25) + term.powi(49))
        } else {
            let term = f64::exp(-2.0 * z * z);
            2.0 * (term - term.powi(4) + term.powi(9))
        }
    }

    ///
    /// Create a new instance of the KS test result
    ///
    /// #Args
    /// - `stat` - Value of the test statistic
    /// - `n` - effective sample size
    ///
    fn new(stat: f64, n: usize) -> Self {
        let sqrt_n = f64::sqrt(n as f64);
        let arg = sqrt_n + 0.12 + 0.11 / sqrt_n;
        let p = Self::complement_ks_cdf(arg * stat);
        Self { stat, n, p }
    }

    /// Get the p-value of the test
    ///
    /// Probability of observing the data under the assumption that null hypothesis holds
    /// In this case probability that that
    pub fn p_value(&self) -> f64 {
        self.p
    }
}

///
/// Perform one sample Kolmogorov-Smirnov statistical test
///
pub fn ks1_test<T>(cdf: impl Fn(&T) -> f64, samples: Vec<T>) -> Result<KSResult, TestError>
where
    T: PartialOrd + Copy,
{
    let n = samples.len();
    let ecdf = Ecdf::new(samples)?;

    let mut stat = 0.0;

    for (ecdf_value, v) in &ecdf {
        let new_stat = (cdf(&v) - ecdf_value).abs();
        if new_stat > stat {
            stat = new_stat;
        }
    }

    Ok(KSResult::new(stat, n))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ks1_test() {
        let samples = [0.3, 0.2, 0.25, 0.1, 0.9, 0.6];
        let lhs = KSResult::new(4.0 / 6.0 - 0.3, 6);
        let rhs = ks1_test(|x| *x, samples.into()).unwrap();
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn test_ks1_error() {
        let samples = [0.3, 0.2, 0.25, 0.1, 0.9, 0.6, f64::NAN];
        assert!(
            ks1_test(|x| *x, samples.into()).is_err(),
            "Failed to detect nan in the list"
        )
    }

    #[test]
    fn test_ks_cdf_complement() {
        // Approximate reference points obtained from SciPy
        let test_points = [
            (0.0, 1.0),
            (1.0, 2.6999967168e-01),
            (2.0, 6.7092525578e-04),
            (3.0, 3.045996e-08),
            (100.0, 0.0),
        ];

        for (x, val) in test_points {
            approx::assert_relative_eq!(val, KSResult::complement_ks_cdf(x), max_relative = 1.0e-7);
        }
    }

    #[test]
    #[should_panic]
    fn test_ks_cdf_complement_invalid_range() {
        KSResult::complement_ks_cdf(-2.0);
    }
}
