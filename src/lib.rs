//! Sample 1D distributions given functional shape
//!
//! This crates implements a struct that allows drawing samples from any
//! smooth probability density function shape.
//!
//! It is intended to be used as a reference distribution for 2-sample
//! Kolmogorov-Smirnov test to test correctness of more efficient sampling
//! algorithms.
//!
use is_sorted::IsSorted;
use thiserror::Error;
pub mod ks_tests;

#[derive(Error, Debug)]
pub enum SetupError {
    #[error("Values in a grid are not sorted")]
    UnsortedGrid,
    #[error("Negative values present in probability density function")]
    NegativePdf,
    #[error("Lengths of arrays to form a table are different")]
    LengthMismatch,
}

///
/// Histogram distribution used as a topping distribution for rejection scheme
///
#[derive(Debug)]
pub struct HistogramDistribution {
    x: Vec<f64>,
    pdf: Vec<f64>,
    cdf: Vec<f64>,
}

///
/// Search sorted grid of values and find the lower bound
///
/// Returns `i` such that `grid[i] <= val < grid[i + 1]`
///
fn search_sorted<T>(grid: &[T], val: T) -> Option<usize>
where
    T: PartialOrd,
{
    let first = grid.first().unwrap();
    let last = grid.last().unwrap();

    if !(first..last).contains(&&val) {
        return None;
    }

    match grid.binary_search_by(|k| k.partial_cmp(&val).unwrap()) {
        Ok(j) => Some(j),
        Err(j) => Some(j - 1),
    }
}

/// Calculate non-normalised cdf from a histogram probability density
///
/// # Panics
/// if x and pdf do not match in length
///
fn histogram_cdf(x: &[f64], pdf: &[f64]) -> Vec<f64> {
    if x.len() != pdf.len() {
        panic! {"Length mismatch"}
    }

    let mut cdf = vec![0.0];
    cdf.reserve(x.len());

    let dx_iter = x.windows(2).map(|w| w[1] - w[0]);

    for (dx, p) in std::iter::zip(dx_iter, pdf.iter()) {
        // We know CDF is never empty
        let top = cdf.last().unwrap();
        cdf.push(top + *p * dx)
    }
    cdf
}

impl HistogramDistribution {
    pub fn new(x: Vec<f64>, pdf: Vec<f64>) -> Result<Self, SetupError> {
        // Check preconditions
        if !IsSorted::is_sorted(&mut x.iter()) {
            return Err(SetupError::UnsortedGrid);
        } else if pdf.iter().any(|v| *v < 0.0) {
            return Err(SetupError::NegativePdf);
        } else if x.len() != pdf.len() {
            return Err(SetupError::LengthMismatch);
        }

        // Calculate CDF and return
        let cdf = histogram_cdf(&x, &pdf);
        Ok(Self { x, pdf, cdf })
    }
}

/// Draws samples from the histogram distribution
///
///
impl rand::distributions::Distribution<f64> for HistogramDistribution {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        // We know cdf is not empty
        let val = rng.gen_range(0.0..*self.cdf.last().unwrap());
        let idx = search_sorted(&self.cdf, val).unwrap();

        let x0 = self.x[idx];
        let p0 = self.pdf[idx];
        let c0 = self.cdf[idx];
        (val - c0) / p0 + x0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx;
    use rand::{Rng, SeedableRng};

    #[test]
    fn test_histogram_calculation() {
        let x = vec![-1.0, 0.5, 3.0];
        let pdf = vec![0.1, 0.2, 0.3];
        let cdf = vec![0.0, 0.15, 0.65];

        let cdf_calc = histogram_cdf(&x, &pdf);

        for (v_ref, v_calc) in std::iter::zip(cdf, cdf_calc) {
            approx::assert_relative_eq!(v_ref, v_calc);
        }
    }

    #[test]
    #[should_panic]
    fn test_invalid_histogram_calculation() {
        let _ = histogram_cdf(&[0.0, 1.0, 3.0], &[0.1, 0.1]);
    }

    #[test]
    fn test_histogram_distribution() {
        let x = vec![-1.0, 0.5, 3.0];
        let pdf = vec![0.1, 0.2, 0.3];
        let _ = HistogramDistribution::new(x, pdf).unwrap();
    }

    #[test]
    fn test_search_of_sorted_grid() {
        let grid = [-0.5, 1.0, 2.0, 6.0];

        // Out of range
        assert_eq!(None, search_sorted(&grid, -0.6));
        assert_eq!(None, search_sorted(&grid, 6.0));
        assert_eq!(None, search_sorted(&grid, 9.0));

        // In range
        assert_eq!(Some(0), search_sorted(&grid, -0.5));
        assert_eq!(Some(0), search_sorted(&grid, 0.0));

        assert_eq!(Some(1), search_sorted(&grid, 1.0));
        assert_eq!(Some(1), search_sorted(&grid, 1.5));

        assert_eq!(Some(2), search_sorted(&grid, 5.0));
    }

    #[test]
    fn test_histogram_pdf_sampling() {
        let support = [-1.0, 0.5, 3.0, 4.0];
        let pdf = [0.1, 0.2, 0.35, 0.35];
        let cdf = [0.0, 0.15, 0.65, 1.0];
        let n_samples = 10000;
        let mut rng = rand::rngs::StdRng::seed_from_u64(87674);

        let dist = HistogramDistribution::new(support.into(), pdf.into()).unwrap();

        let samples = (0..n_samples)
            .map(|_| rng.sample(&dist))
            .collect::<Vec<_>>();

        let ks_res = ks_tests::ks1_test(
            |x| {
                let idx = search_sorted(&support, *x).unwrap();
                let x0 = support[idx];
                let x1 = support[idx + 1];
                let c0 = cdf[idx];
                let c1 = cdf[idx + 1];
                (x - x0) / (x1 - x0) * (c1 - c0) + c0
            },
            samples,
        )
        .unwrap();

        // Print the test results in case of a failure
        println!("{:?}", ks_res);
        assert!(ks_res.p_value() > 0.01)
    }

    #[test]
    fn test_histogram_distribution_errors() {
        let x = vec![-1.0, 0.5, 3.0];
        let pdf = vec![0.1, 0.2, 0.3];

        assert!(
            HistogramDistribution::new(vec![1.0, 0.5, 3.0], pdf.clone()).is_err(),
            "Failed to detect non-sorted grid"
        );
        assert!(
            HistogramDistribution::new(x.clone(), vec![0.1, -0.2, 0.3]).is_err(),
            "Failed to detect -ve pdf"
        );
        assert!(
            HistogramDistribution::new(x.clone(), vec![0.1, 0.3]).is_err(),
            "Failed to detect length mistmatch"
        );
    }
}
