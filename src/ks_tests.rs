//! Provides One- and Two-Sample Kolmogorov-Smirnov test implementation
//!
use std::f64::consts::PI;
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

///
/// Compute the value of Kolmogorov-Smirnov one-sample test statistic
///
/// # Panics
/// If `samples` contain any pair of non -comparable values (e.g. contains a NaN for floats)
///
fn ks1_compute_statistic<T>(cdf: impl Fn(&T) -> f64, samples: Vec<T>) -> f64
where
    T: PartialOrd + Copy,
{
    let mut samples = samples;
    let n = samples.len() as f64;
    let mut stat = 0.0;

    samples.sort_by(|a, b| a.partial_cmp(b).expect("Failed comparison"));

    for (i, s) in samples.iter().enumerate() {
        let ecdf = (i + 1) as f64 / n;
        let new_stat = (cdf(s) - ecdf).abs();
        if new_stat > stat {
            stat = new_stat;
        }
    }
    stat
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
    if samples
        .windows(2)
        .any(|x| x[0].partial_cmp(&x[1]).is_none())
    {
        return Err(TestError::ContainsNotSortableValues);
    }
    let stat = ks1_compute_statistic(cdf, samples);
    Ok(KSResult::new(stat, n))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ks1_statistic() {
        let samples = [0.3, 0.2, 0.25, 0.1, 0.9, 0.6];
        let res = ks1_compute_statistic(|x| *x, samples.into());
        assert_eq!(4.0 / 6.0 - 0.3, res);
    }

    #[test]
    #[should_panic]
    fn test_ks1_noncomparable() {
        let samples = [0.3, 0.2, 0.25, 0.1, 0.9, f64::NAN];
        let _ = ks1_compute_statistic(|x| *x, samples.into());
    }

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
