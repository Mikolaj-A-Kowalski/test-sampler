//! Provides One- and Two-Sample Kolmogorov-Smirnov test implementation
//!
use std::f64::consts::PI;
use std::iter::{Iterator, Peekable};
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

    /// Get the value of ecdf at `val`
    ///
    pub fn get(&self, val: T) -> f64 {
        let idx = self.samples.partition_point(|x| *x <= val);
        idx as f64 / self.samples.len() as f64
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
    n: f64,
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
    fn new(stat: f64, n: f64) -> Self {
        let sqrt_n = f64::sqrt(n);
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

    Ok(KSResult::new(stat, n as f64))
}

///
/// Builds a Brownian bridge of two [Ecdf]s
///
/// The `top` ecdf gives positive contributions, `bottom` negative.
/// We iterate through both sorted sets of samples in-order and depending
/// whether a sample from the `top` or `bottom` ecdf was taken we nudge the
/// sum up or down respectively.
///
/// We skip over the repeated values. Thus for samples:
///   [1, 2, 2, 3]
///   [2]
/// We get the sequence:
///   [0.25, -0.25, 0.0]
///
struct EcdfBridge<'a, T>
where
    T: PartialOrd + Copy,
{
    top: Peekable<std::slice::Iter<'a, T>>,
    bottom: Peekable<std::slice::Iter<'a, T>>,
    delta_top: f64,
    delta_bottom: f64,
    sum: f64,
}

impl<'a, T> EcdfBridge<'a, T>
where
    T: PartialOrd + Copy,
{
    pub fn new(ecdf_top: &'a Ecdf<T>, ecdf_bottom: &'a Ecdf<T>) -> Self {
        Self {
            top: ecdf_top.samples.iter().peekable(),
            bottom: ecdf_bottom.samples.iter().peekable(),
            delta_top: 1.0 / ecdf_top.samples.len() as f64,
            delta_bottom: 1.0 / ecdf_bottom.samples.len() as f64,
            sum: 0.0,
        }
    }

    /// Advance the top ecdf
    ///
    /// Needs to loop over repeated values.
    ///
    /// # Panics
    /// If it is called when `top` is already empty
    ///
    fn advance_top(&mut self) -> f64 {
        loop {
            let v = self.top.next().unwrap();
            self.sum += self.delta_top;

            let next = self.top.peek();

            if next.is_none() || *next.unwrap() != v {
                break;
            }
        }
        self.sum
    }

    /// Advance the bottom ecdf
    ///
    /// Needs to loop over repeated values.
    ///
    /// # Panics
    /// If it is called when `bottom` is already empty
    ///
    fn advance_bottom(&mut self) -> f64 {
        loop {
            let v = self.bottom.next().unwrap();
            self.sum -= self.delta_bottom;

            let next = self.bottom.peek();

            if next.is_none() || *next.unwrap() != v {
                break;
            }
        }
        self.sum
    }
}

impl<'a, T> Iterator for EcdfBridge<'a, T>
where
    T: PartialOrd + Copy,
{
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        let top = self.top.peek();
        let bottom = self.bottom.peek();

        match (top, bottom) {
            (Some(t), Some(b)) => {
                if t < b {
                    Some(self.advance_top())
                } else if t == b {
                    // We need to be careful if same element is present in both `top` and `bottom`.
                    // In that case we need to advance both iterators
                    self.advance_top();
                    Some(self.advance_bottom())
                } else {
                    Some(self.advance_bottom())
                }
            }
            (None, Some(_)) => Some(self.advance_bottom()),
            (Some(_), None) => Some(self.advance_top()),
            (None, None) => None,
        }
    }
}

///
/// Preform 2-sample Kolmogorov-Smirnov test
///
pub fn ks2_test<T>(sample1: Vec<T>, sample2: Vec<T>) -> Result<KSResult, TestError>
where
    T: PartialOrd + Copy,
{
    let n1 = sample1.len() as f64;
    let n2 = sample2.len() as f64;

    let ecdf1 = Ecdf::new(sample1)?;
    let ecdf2 = Ecdf::new(sample2)?;

    // Compute statistic
    let stat = EcdfBridge::new(&ecdf1, &ecdf2)
        .map(|v| v.abs())
        .max_by(|a, b| a.partial_cmp(b).expect("PartialOrd comparison failed"))
        .expect("Empty iterator?!");

    Ok(KSResult::new(stat, n1 * n2 / (n1 + n2)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ecdf() {
        let samples = [0.3, 0.1, 0.5, 0.7];
        let ecdf = Ecdf::new(samples.into()).unwrap();

        assert_eq!(0.0, ecdf.get(-2.0));

        assert_eq!(0.25, ecdf.get(0.1));
        assert_eq!(0.25, ecdf.get(0.2));

        assert_eq!(0.75, ecdf.get(0.5));
        assert_eq!(0.75, ecdf.get(0.6));

        assert_eq!(1.0, ecdf.get(0.7));
        assert_eq!(1.0, ecdf.get(0.71));
    }

    #[test]
    fn test_ks1_test() {
        let samples = [0.3, 0.2, 0.25, 0.1, 0.9, 0.6];
        let lhs = KSResult::new(4.0 / 6.0 - 0.3, 6.0);
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
    fn test_ks2_test() {
        let samples1 = [0.3, 0.2, 0.25, 0.1, 0.9, 0.6];
        let samples2 = [0.1, 0.8, 0.34, 0.09, 0.12, 0.81];

        let rhs = ks2_test(samples1.into(), samples2.into()).unwrap();
        let lhs = KSResult::new(1.0 / 3.0, 3.0);
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn test_ks2_test_repeated_samples() {
        let samples1 = [1, 2, 2, 3];
        let samples2 = [2];

        let rhs = ks2_test(samples1.into(), samples2.into()).unwrap();
        let lhs = KSResult::new(0.25, 0.8);
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn test_ecdf_bridge() {
        let cases = [
            (vec![1, 2, 2, 3], vec![2], vec![0.25, -0.25, 0.0]),
            (vec![2], vec![1, 2, 2, 3], vec![-0.25, 0.25, 0.0]),
            (vec![1, 2, 2, 3], vec![1, 2, 2, 3], vec![0.0, 0.0, 0.0]),
            (vec![1, 2, 3, 4], vec![3, 3], vec![0.25, 0.5, -0.25, 0.0]),
            (vec![2], vec![1, 2, 2, 2], vec![-0.25, 0.0]),
        ];

        for (s1, s2, reference) in cases {
            let ecdf1 = Ecdf::new(s1).unwrap();
            let ecdf2 = Ecdf::new(s2).unwrap();

            let bridge = EcdfBridge::new(&ecdf1, &ecdf2).collect::<Vec<_>>();

            assert_eq!(bridge, reference);
        }
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
