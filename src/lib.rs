//! Contains tools to perform unit testing of sampling algorithms.
//!
//! It has been developed particularly to help with the development of Monte
//! Carlo particle transport codes, where a large number of various sampling
//! procedures is required to stochastically simulate physical interactions of
//! radiation with matter.
//!
//! In general models are described with differential cross-sections
//! $\frac{d\sigma}{dE'd\Omega}(E)$, which provide with the shape of probability
//! density function. In general normalisation is difficult to get without
//! complex integration.
//!
//! For that reason this package is composed of two parts:
//! - [FunctionSampler] which allows to (inefficiently) draw samples from
//!   a non-normalised pdf shape function
//! - Suite of statistical tests [crate::stat_tests], which allow to verify that
//!   samples from a tested distribution match the one generated with [FunctionSampler]
//!
//! Thus to verify sampling one needs to:
//! - Verify shape function with deterministic unit tests
//! - Compare sampling procedure against reference values from [FunctionSampler]
//!   using statistical tests
//!
//! Note that as a result of statistical uncertainty and variable *power* of
//! statistical tests for different defects and sample populations the sampling
//! unit tests cannot ever provide with the same level of certainty as the
//! deterministic one. Also the appropriate number of samples and type(s) of
//! the test(s) will depend on a particular application.
//!
//! # Example
//!
//! Let us verify simple inversion sampling of $f(x) = x$ on $\[0;1\]$,
//! for which we can generate samples with $\hat{f} = \sqrt{r}$ where $r$ is
//! uniformly distributed random number.
//!
//! ```
//! use test_sampler::FunctionSampler;
//! use rand::prelude::*;
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//!
//! // Seed rngs
//! let mut rng1 = StdRng::seed_from_u64(87674);
//! let mut rng2 = StdRng::seed_from_u64(87674);
//!
//! // Draw reference samples from the FunctionSampler
//! let support = 0.0..1.0;
//! let num_bins = 30;
//!
//! let reference_dist = FunctionSampler::new(|x| x, support, num_bins)?;
//! let s_ref : Vec<f64> = rng1.sample_iter(&reference_dist).take(100).collect();
//!
//! // Samples to test
//! let s : Vec<f64> = (0..100).map(|_| rng2.gen()).map(|r: f64| r.sqrt()).collect();
//!
//! // Perform tests
//! // Vectors of samples will be moved inside the procedures
//! // It is necessary since the samples must be sorted (and mutated)
//! let ks_res = test_sampler::stat_tests::ks2_test(s_ref.clone(), s.clone())?;
//! let kup_res = test_sampler::stat_tests::kuiper2_test(s_ref.clone(), s.clone())?;
//! let ad_res = test_sampler::stat_tests::ad2_test(s_ref, s)?;
//!
//! // Check test results
//! assert!(ks_res.p_value() > 0.05);
//! assert!(kup_res.p_value() > 0.05);
//! assert!(ad_res.p_value() > 0.05);
//!
//! # Ok(()) }
//! ```
//!
use argmin::core::CostFunction;
use is_sorted::IsSorted;
use std::ops::Range;
use thiserror::Error;

pub mod stat_tests;

/// Error raised when the setup of a sampling distribution has failed
///
#[derive(Error, Debug)]
pub enum SetupError {
    /// Grid for tabulated values si not sorted
    #[error("Values in a grid are not sorted")]
    UnsortedGrid,
    /// Negative entries were found in probability density function (pdf)
    #[error("Negative values present in probability density function")]
    NegativePdf,
    /// Length of vectors that form a table is not the same
    #[error("Lengths of arrays to form a table are different")]
    LengthMismatch,
    /// Wraps errors from failed optimisation by [argmin]
    #[error("Optimisation with argmin has failed")]
    OptimisationError(#[from] argmin::core::Error),
    /// Number of bins to construct the tabulated data (`0`) is lower then required  (`1`)
    #[error("Insufficient number of bins: {0}, must have at least {1}")]
    InsufficientBins(usize, usize),
    /// Was given empty range to represent a mon-empty interval
    #[error("Got empty range where non-empty is required")]
    EmptyRange,
}

/// Distribution described by non-normalised histogram
///
/// The distribution is given as a table of `x` and `pdf` which follows histogram
/// interpolation. For `x ∈ [xᵢ₊₁; xᵢ]` probability if `pdf(x) = pdfᵢ`. The
/// cumulative distribution function becomes piece-wise linear which makes
/// sampling form the table quite easy.
///
/// To support better approximation of different pdfs, the grid is not
/// equal-spaced in general. Hence binary search is needed to find correct bin.
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
/// Local function required to search the [HistogramDistribution]
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
    /// Create a new instance of the histogram distribution
    ///
    /// The cumulative distribution function will be calculated.
    ///
    /// Condition `x.len() == pdf.len() > 1`, must be met.
    /// Thus the last value in the `pdf` vector will be ignored.
    ///
    pub fn new(x: Vec<f64>, pdf: Vec<f64>) -> Result<Self, SetupError> {
        if !IsSorted::is_sorted(&mut x.iter()) {
            return Err(SetupError::UnsortedGrid);
        } else if pdf.iter().any(|v| *v < 0.0) {
            return Err(SetupError::NegativePdf);
        } else if x.len() != pdf.len() {
            return Err(SetupError::LengthMismatch);
        } else if x.len() <= 1 {
            return Err(SetupError::InsufficientBins(x.len(), 2));
        }

        let cdf = histogram_cdf(&x, &pdf);
        Ok(Self { x, pdf, cdf })
    }

    /// Sample a value from the histogram and return the value of non-normalised probability
    ///
    /// # Result
    /// Tuple `(s, p)` where `s` is the sample and `p` probability in the bin
    ///
    /// We need a way to sample while returning probability value in the bin as well
    /// to implement rejection sampling scheme without repeating a binary search of the grid.
    ///
    pub fn sample_with_value<RNG>(&self, rng: &mut RNG) -> (f64, f64)
    where
        RNG: rand::Rng + ?Sized,
    {
        // We know cdf is not empty
        let val = rng.gen_range(0.0..*self.cdf.last().unwrap());
        let idx = search_sorted(&self.cdf, val).unwrap();

        let x0 = self.x[idx];
        let p0 = self.pdf[idx];
        let c0 = self.cdf[idx];
        ((val - c0) / p0 + x0, p0)
    }
}

/// Draws samples from the histogram distribution
///
///
impl rand::distributions::Distribution<f64> for HistogramDistribution {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        self.sample_with_value(rng).0
    }
}

///
/// Wrap a function to maximise
///
/// We need a separate struct to use [argmin] library, because we need to implement
/// some Traits to use it as optimisation problem with other `argmin` components.
///
/// Since by convention, objective are minimised, we also need to take the
/// negative of `function` as the cost.
///
struct FlipSign<T: Fn(f64) -> f64> {
    pub function: T,
}

impl<T> CostFunction for FlipSign<T>
where
    T: Fn(f64) -> f64,
{
    type Param = f64;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        Ok(-(self.function)(*param))
    }
}

///
/// Creates linearly spaced grid between `start` and `end` of size `n`
///
/// ```
/// # use test_sampler::linspace;
/// assert_eq!(vec![1.0, 2.0, 3.0], linspace(1.0, 3.0, 3));
/// assert_eq!(vec![3.0, 2.0, 1.0], linspace(3.0, 1.0, 3));
/// assert_eq!(vec![3.0, 3.0, 3.0], linspace(3.0, 3.0, 3));
/// ```
///
/// # Panics
/// If number of points `n` is 0 or 1
///
pub fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    if n < 2 {
        panic! {"Grid cannot have {n} values. At least 2 are required."}
    }

    let delta = (end - start) / (n - 1) as f64;
    (0..n).map(|i| start + delta * i as f64).collect()
}

/// Distribution described by a non-negative shape function on an interval
///
/// Allows sampling from a generic distribution described by some shape function $f(x)$.
/// The function does not need to be normalised i.e. $ \int f(s) d(s) \ne 1 $ in general
///
/// It is intended to be used as a reference distribution for verification of
/// more efficient sampling algorithms.
///
/// # Sampling procedure
///
/// To generate the samples a relatively expensive setup step is required. The user
/// provides an interval $[x_0, x_1]$ which is a support of the shape function $f(x)$.
/// The support is then subdivided into number of bins. In each a local maximum
/// is found by numerical means. This allows to create a histogram approximation
/// of the probability distribution, that 'tops' the actual distribution. Hence,
/// we may draw samples from the histogram and use rejection scheme to obtain
/// the distribution described by $f(x)$.
///
/// Since the numerical maximisation is associated with some tolerance, a safety
/// factor of 5% is applied on the local maxima to ensure that the supremum criterion
/// required for rejection sampling is met.
///
/// Note that for very sharp functions (large $\frac{df}{dx} $) the safety factor may be
/// insufficient. Also a general check if  $f(x) \ge 0 ~\forall~ x \in [x_0; x_1]$
/// is met is not feasible. Hence sampling may **panic** if either of the conditions occurs.
///
/// # Usage
///
/// The distribution is integrated with [rand] package and can be used to construct
/// a sampling iterator as follows:
/// ```
/// # use test_sampler::FunctionSampler;
/// use rand::{self, Rng};
///
/// # fn main() -> Result<(), test_sampler::SetupError> {
/// let dist = FunctionSampler::new(|x| -x*x + x, 0.0..1.0, 30)?;
/// let samples = rand::thread_rng().sample_iter(&dist).take(10);
/// # Ok(())
/// # }
/// ```
///
pub struct FunctionSampler<T: Fn(f64) -> f64> {
    function: T,
    hist: HistogramDistribution,
}

impl<T> FunctionSampler<T>
where
    T: Fn(f64) -> f64,
{
    /// Safety factor to increase bin maxima to protect against error due
    /// to tolerance of numerical optimisation
    pub const SAFETY_FACTOR: f64 = 1.05;

    /// Helper function to find maximum in a given bin
    ///
    fn maximise(function: &T, start: f64, end: f64) -> Result<f64, SetupError> {
        let problem = FlipSign { function };
        let solver = argmin::solver::brent::BrentOpt::new(start, end);
        let res = argmin::core::Executor::new(problem, solver).run()?;

        Ok(-res.state().cost)
    }

    /// New sampler from components
    ///
    /// # Arguments
    /// - `function` - A function `Fn(f64) -> f64` that is non-negative on `range`
    /// - `range` - Support of the probability distribution with the shape of `function`
    /// - `bins` - Number of bins to construct topping histogram (at least 1)
    ///
    pub fn new(function: T, range: Range<f64>, bins: usize) -> Result<Self, SetupError> {
        if bins == 0 {
            return Err(SetupError::InsufficientBins(bins, 1));
        } else if range.is_empty() {
            return Err(SetupError::EmptyRange);
        }

        // Create subdivision grid
        let grid = linspace(range.start, range.end, bins + 1);

        // Get maxima in each window
        // We need the vector of maxima to match
        let mut maxima = grid
            .windows(2)
            .map(|x| Self::maximise(&function, x[0], x[1]).map(|x| Self::SAFETY_FACTOR * x))
            .collect::<Result<Vec<f64>, SetupError>>()?;
        maxima.push(*maxima.last().unwrap());

        // Construct topping distribution
        let hist = HistogramDistribution::new(grid, maxima)?;

        // Profit
        Ok(Self { function, hist })
    }
}

/// Draws samples from the FunctionSampler
///
/// # Panics
/// Sampling may panic if the shape function has  negative values or upper bound
/// is not is not fulfilled. This may happen if gradient of the shape function
/// is strong and the safety factor on the tolerance of optimisation was
/// insufficient
///
impl<T> rand::distributions::Distribution<f64> for FunctionSampler<T>
where
    T: Fn(f64) -> f64,
{
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        // Draw sample from the histogram
        loop {
            let (sample, p_top) = self.hist.sample_with_value(rng);
            let p_val = (self.function)(sample);
            if p_top < p_val {
                panic!("Upper bound {p_top} is lower than {p_val} at {sample}");
            } else if p_val < 0.0 {
                panic!("Negative value {p_val} at {sample}")
            }
            if p_val / p_top > rng.gen() {
                return sample;
            }
        }
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

        let ks_res = stat_tests::ks1_test(
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
        assert!(
            HistogramDistribution::new(vec![0.1], vec![0.1]).is_err(),
            "Failed to detect too short vectors"
        );
    }

    #[test]
    fn test_function_sampler_sampling() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(87674);
        let dist = FunctionSampler::new(|x| -x * x + x, 0.0..1.0, 30).unwrap();
        let n_samples = 10000;

        let samples = (0..n_samples)
            .map(|_| rng.sample(&dist))
            .collect::<Vec<_>>();

        let ks_res = stat_tests::ks1_test(|x| 3.0 * x * x - 2.0 * x * x * x, samples).unwrap();
        // Print the test results in case of a failure
        println!("{:?}", ks_res);
        assert!(ks_res.p_value() > 0.01)
    }

    #[test]
    fn test_function_sampler_setup_errors() {
        assert!(
            FunctionSampler::new(|x| -x * x + x, 1.0..0.0, 30).is_err(),
            "Failed to detect empty range"
        );
        assert!(
            FunctionSampler::new(|x| -x * x + x, 0.0..1.0, 0).is_err(),
            "Failed to detect insufficient number of bins"
        );
        assert!(
            FunctionSampler::new(|x| -x * x + x - 0.2, 0.0..1.0, 30).is_err(),
            "Failed to detect negative maxima in thr bins"
        );
    }

    #[test]
    #[should_panic]
    fn test_function_sampler_negative_pdf() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(87674);

        // We select only single bin so that negative pdf is hidden from the histogram
        let dist = FunctionSampler::new(|x| -x * x + x - 0.1, 0.0..1.0, 1).unwrap();

        // Will panic on sampling
        let _samples = (0..100).map(|_| rng.sample(&dist)).collect::<Vec<_>>();
    }
}
