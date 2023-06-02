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

///
/// Result of a statistical test
///
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TestResult {
    pub stat: f64,
    p: f64,
    n: f64,
}

impl TestResult {
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
    /// # Arguments
    /// - `stat` - Value of the test statistic
    /// - `n` - effective sample size
    ///
    fn new_ks(stat: f64, n: f64) -> Self {
        let sqrt_n = f64::sqrt(n);
        let arg = sqrt_n + 0.12 + 0.11 / sqrt_n;
        let p = Self::complement_ks_cdf(arg * stat);
        Self { stat, n, p }
    }

    ///
    /// Create a new instance of the result of Kuiper test result
    ///
    /// # Arguments
    /// - `stat` - Value of the test statistic
    /// - `n` - effective sample size
    ///
    fn new_kuiper(stat: f64, n: f64) -> Self {
        let z = n.sqrt() * stat;
        let mut p = 0.0;

        // In case the z parameter is very small the sum may become unstable.
        // Set p to 1.0 in that case (which is a decent approximation)
        if z < KuiperTerms::Z_MIN {
            return Self { stat, n, p: 1.0 };
        }

        // Limit the maximum number of terms to 200
        for term in KuiperTerms::new(z, n).take(200) {
            let p_old = p;
            p += term;
            if f64::abs(p / p_old - 1.0) < 1e-7 {
                break;
            }
        }
        Self { stat, n, p }
    }

    ///
    /// Create a new instance of the result of Anderson-Darling two-sample test
    ///
    /// # Arguments
    /// - `stat` - Value of the test statistic
    /// - `n` - effective sample size
    ///
    /// We use the approximation of the CDF by (Lewis 1961)
    ///
    /// $$ F(z) = 1 - \sqrt{ 1 - 4 \exp(-(z+1)) } $$
    ///
    /// Or if the value under square root is negative: $F(z)= 1$
    /// This approximation should be overly conservative for the small samples.
    ///
    fn new_ad(stat: f64, n: f64) -> Self {
        let arg = 1.0 - 4.0 * f64::exp(-(stat + 1.0));

        let p = if arg > 0.0 {
            1.0 - arg.sqrt()
        } else {
            1.0
        };
        Self { stat, n, p }
    }

    /// Get the p-value of the test
    ///
    /// Probability of observing the data under the assumption that null hypothesis holds
    ///
    pub fn p_value(&self) -> f64 {
        self.p
    }
}

///
/// Iterator over the terms of Kuiper asymptotic formula for distribution of the test statistic.
///
/// The series is derived in the original paper (Kuiper 1960) and its sum should give
/// the complement of cumulative distribution function for large sample sizes.
///
/// However for the small threshold parameter z the summation can become unstable, thus we
/// set the limit for the z for which the terms can be evaluated.
///
/// N.H. Kuiper (1960), Tests Concerning Random Points on a Circle, Mathematical Statistics.
///
struct KuiperTerms {
    z: f64,
    n: f64,
    i: usize,
}

impl KuiperTerms {
    /// Stability threshold, below the summation may become unstable
    pub const Z_MIN: f64 = 0.1;

    /// Create new sequence of Kuiper terms
    ///
    /// # Arguments
    /// - `z` : value of the threshold parameter P( √n * V > z) where `V` is the Kuiper statistic.
    /// - `n` : effective population of the sample
    ///
    /// # Panics
    /// If `z <= Self::Z_MIN` or `n` is not positive.
    ///
    fn new(z: f64, n: f64) -> Self {
        if z <= Self::Z_MIN {
            panic!("Value of the test statistic is too small {z} for the series to converge");
        } else if n <= 0.0 {
            panic!("Effective population {n} must be +ve value");
        }
        Self { z, n, i: 0 }
    }
}

impl Iterator for KuiperTerms {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        self.i += 1;
        let i = self.i as f64;
        let zi_sq = (self.z * i).powi(2);
        let exp_term = f64::exp(-2. * zi_sq);
        let factor = 8.0 * self.z / (3. * self.n.sqrt());
        let term = (2. * (4. * zi_sq - 1.) + factor * i.powi(2) * (4. * zi_sq - 3.)) * exp_term;
        Some(term)
    }
}

///
/// Perform one sample Kolmogorov-Smirnov statistical test
///
pub fn ks1_test<T>(cdf: impl Fn(&T) -> f64, samples: Vec<T>) -> Result<TestResult, TestError>
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

    Ok(TestResult::new_ks(stat, n as f64))
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
    count: usize,
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
            count: 0,
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
            self.count += 1;
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
            self.count += 1;
            self.sum -= self.delta_bottom;

            let next = self.bottom.peek();

            if next.is_none() || *next.unwrap() != v {
                break;
            }
        }
        self.sum
    }

    /// Enumerate the samples of the bridge
    ///
    /// Iterates over `(i, val)` where `i` is the count of all samples. Like
    /// [EcdfBridge] iterator skips over repeated samples, but they are included
    /// in the count. Thus the increments of `i` may be larger than `1`.
    ///
    pub fn enumerate_samples(self) -> EnumeratedEcdfBridge<'a, T> {
        EnumeratedEcdfBridge { bridge: self }
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
/// Returned from [EcdfBridge::enumerate_samples]
///
/// Wraps around [EcdfBridge] and allows to enumerate the samples, but
/// unlike the ordinary [std::iter::Iterator::enumerate] accounts for the
/// 'skips' (non 1 increment) due to the repeated samples.
///
struct EnumeratedEcdfBridge<'a, T>
where
    T: PartialOrd + Copy,
{
    bridge: EcdfBridge<'a, T>,
}

impl<'a, T> Iterator for EnumeratedEcdfBridge<'a, T>
where
    T: PartialOrd + Copy,
{
    type Item = (usize, f64);

    fn next(&mut self) -> Option<Self::Item> {
        match self.bridge.next() {
            Some(v) => Some((self.bridge.count - 1, v)),
            None => None,
        }
    }
}

///
/// Preform 2-sample Kolmogorov-Smirnov test
///
pub fn ks2_test<T>(sample1: Vec<T>, sample2: Vec<T>) -> Result<TestResult, TestError>
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

    Ok(TestResult::new_ks(stat, n1 * n2 / (n1 + n2)))
}

///
/// Preform 2-sample Kuiper's test
///
pub fn kuiper2_test<T>(sample1: Vec<T>, sample2: Vec<T>) -> Result<TestResult, TestError>
where
    T: PartialOrd + Copy,
{
    let n1 = sample1.len() as f64;
    let n2 = sample2.len() as f64;

    let ecdf1 = Ecdf::new(sample1)?;
    let ecdf2 = Ecdf::new(sample2)?;

    let mut minimum = 0.0;
    let mut maximum = 0.0;

    for v in EcdfBridge::new(&ecdf1, &ecdf2) {
        if v < minimum {
            minimum = v
        } else if v > maximum {
            maximum = v
        }
    }
    let stat = minimum.abs() + maximum.abs();

    Ok(TestResult::new_kuiper(stat, n1 * n2 / (n1 + n2)))
}

///
/// Perform Anderson-Darling two sample test
///
/// Implementation is based on the (Pettitt 1976) paper. As the result can be
/// used only for continuous distributions where the duplicate values in the
/// samples have vanishingly small probability to occur.
///
/// Note that the (Pettitt 1976) seems to have slightly different behaviour
/// from the $ A^2\_{kN} $ statistic of (Scholz 1987) if the duplicate samples
/// are present.
///
/// The p-value of the test is obtained from the large sample approximation to
/// the distribution of the Anderson-Darling statistic provided by (Lewis 1961):
///
/// $$ F(z) = 1 - \sqrt{ 1 - 4 \exp(-(z+1)) } $$
///
/// Or if the value under square root is negative: $F(z)= 1$
/// This approximation should be conservative for smaller sample sizes (true
/// type I error frequency will be smaller than expected)
///
/// # References
/// - Pettitt, A. N. (1976). A Two-Sample Anderson--Darling Rank Statistic. Biometrika, 63(1),
///   161–168. <https://doi.org/10.2307/2335097>
/// - Scholz, F. W., & Stephens, M. A. (1987). K-Sample Anderson-Darling Tests.
///   Journal of the American Statistical Association, 82(399), 918–924.
///   <https://doi.org/10.2307/2288805>
/// - Lewis, P. A. W. (1961). Distribution of the Anderson-Darling Statistic.
///   Ann. Math. Statist., 32(4), 1118-1124.
///
pub fn ad2_test<T>(sample1: Vec<T>, sample2: Vec<T>) -> Result<TestResult, TestError>
where
    T: PartialOrd + Copy,
{
    let n1 = sample1.len() as f64;
    let n2 = sample2.len() as f64;
    let tot_n = sample1.len() + sample2.len();

    let ecdf1 = Ecdf::new(sample1)?;
    let ecdf2 = Ecdf::new(sample2)?;

    let mut stat = 0.0;

    for (idx, v) in EcdfBridge::new(&ecdf1, &ecdf2).enumerate_samples() {
        let denum = (idx + 1) * (tot_n - idx - 1);

        // We need to exclude last entry which will be a NaN [0./0.]
        if denum != 0 {
            stat += v.powi(2) / denum as f64
        }
    }

    // Scale by the population
    stat *= n1 * n2;

    Ok(TestResult::new_ad(stat, tot_n as f64))
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
        let lhs = TestResult::new_ks(4.0 / 6.0 - 0.3, 6.0);
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
        let lhs = TestResult::new_ks(1.0 / 3.0, 3.0);
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn test_ks2_test_repeated_samples() {
        let samples1 = [1, 2, 2, 3];
        let samples2 = [2];

        let rhs = ks2_test(samples1.into(), samples2.into()).unwrap();
        let lhs = TestResult::new_ks(0.25, 0.8);
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn test_kuiper() {
        let sample1 = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let sample2 = vec![11, 2, 3, 4, 5, 6, 7, 8, 9, 10];

        let rhs = kuiper2_test(sample1, sample2).unwrap();
        let lhs = TestResult::new_kuiper(0.1, 5.0);

        println!("{rhs:#?}");
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn test_kuiper_same_sample() {
        let sample1 = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

        let rhs = kuiper2_test(sample1.clone(), sample1.clone()).unwrap();
        let lhs = TestResult::new_kuiper(0.0, 5.0);

        println!("{rhs:#?}");
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn test_kuiper_2() {
        let sample1 = vec![0.07, 0.74, 0.20, 0.55, 0.33, 0.98, 0.32, 0.36, 0.86, 0.43];
        let sample2 = vec![0.73, 0.10, 0.49, 0.18, 0.87, 0.25, 0.80, 0.54, 0.90, 0.06];

        let rhs = kuiper2_test(sample1, sample2).unwrap();
        let lhs = TestResult::new_kuiper(0.4, 5.0);

        println!("{rhs:#?}");
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn test_ad() {
        let sample1 = vec![0.07, 0.74, 0.20, 0.55, 0.33, 0.98, 0.32, 0.36, 0.86, 0.43];
        let sample2 = vec![0.73, 0.10, 0.49, 0.18, 0.87, 0.25, 0.80, 0.54, 0.90, 0.06];

        let rhs = ad2_test(sample1, sample2).unwrap();

        // Reference value taken from SciPy
        // (Internal `_anderson_ksamp_right` was used to get statistic before normalisation)
        let lhs = TestResult::new_ad(0.3634446006350032, 20.0);

        println!("{rhs:#?}");
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
    fn test_ecdf_enumerated_bridge() {
        let sample1 = vec![1, 2, 2, 3, 3];
        let sample2 = vec![0, 2, 4];

        let ecdf1 = Ecdf::new(sample1).unwrap();
        let ecdf2 = Ecdf::new(sample2).unwrap();

        let ref_count: Vec<usize> = vec![0, 1, 4, 6, 7];

        let count = EcdfBridge::new(&ecdf1, &ecdf2)
            .enumerate_samples()
            .map(|(i, _)| i)
            .collect::<Vec<usize>>();

        assert_eq!(ref_count, count);
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
            approx::assert_relative_eq!(
                val,
                TestResult::complement_ks_cdf(x),
                max_relative = 1.0e-7
            );
        }
    }

    #[test]
    #[should_panic]
    fn test_ks_cdf_complement_invalid_range() {
        TestResult::complement_ks_cdf(-2.0);
    }
}
