//! Internal utility functions, types, and data structures.

use crate::GeoNum;
use geo_types::{Coord, CoordFloat};
use num_traits::{Float, FromPrimitive};

/// Partition a mutable slice in-place so that it contains all elements for
/// which `predicate(e)` is `true`, followed by all elements for which
/// `predicate(e)` is `false`. Returns sub-slices to all predicated and
/// non-predicated elements, respectively.
///
/// https://github.com/llogiq/partition/blob/master/src/lib.rs
pub fn partition_slice<T, P>(data: &mut [T], predicate: P) -> (&mut [T], &mut [T])
where
    P: Fn(&T) -> bool,
{
    let len = data.len();
    if len == 0 {
        return (&mut [], &mut []);
    }
    let (mut l, mut r) = (0, len - 1);
    loop {
        while l < len && predicate(&data[l]) {
            l += 1;
        }
        while r > 0 && !predicate(&data[r]) {
            r -= 1;
        }
        if l >= r {
            return data.split_at_mut(l);
        }
        data.swap(l, r);
    }
}

pub enum EitherIter<I1, I2> {
    A(I1),
    B(I2),
}

impl<I1, I2> ExactSizeIterator for EitherIter<I1, I2>
where
    I1: ExactSizeIterator,
    I2: ExactSizeIterator<Item = I1::Item>,
{
    #[inline]
    fn len(&self) -> usize {
        match self {
            EitherIter::A(i1) => i1.len(),
            EitherIter::B(i2) => i2.len(),
        }
    }
}

impl<T, I1, I2> Iterator for EitherIter<I1, I2>
where
    I1: Iterator<Item = T>,
    I2: Iterator<Item = T>,
{
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            EitherIter::A(iter) => iter.next(),
            EitherIter::B(iter) => iter.next(),
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            EitherIter::A(iter) => iter.size_hint(),
            EitherIter::B(iter) => iter.size_hint(),
        }
    }
}

// The Rust standard library has `max` for `Ord`, but not for `PartialOrd`
pub fn partial_max<T: PartialOrd>(a: T, b: T) -> T {
    if a > b { a } else { b }
}

// The Rust standard library has `min` for `Ord`, but not for `PartialOrd`
pub fn partial_min<T: PartialOrd>(a: T, b: T) -> T {
    if a < b { a } else { b }
}

use std::cmp::Ordering;

/// Compare two coordinates lexicographically: first by the
/// x coordinate, and break ties with the y coordinate.
#[inline]
pub fn lex_cmp<T: GeoNum>(p: &Coord<T>, q: &Coord<T>) -> Ordering {
    p.x.total_cmp(&q.x).then(p.y.total_cmp(&q.y))
}

/// Compute index of the least point in slice. Comparison is
/// done using [`lex_cmp`].
///
/// Should only be called on a non-empty slice with no `nan`
/// coordinates.
pub fn least_index<T: GeoNum>(pts: &[Coord<T>]) -> usize {
    pts.iter()
        .enumerate()
        .min_by(|(_, p), (_, q)| lex_cmp(p, q))
        .unwrap()
        .0
}

/// Normalize a longitude to coordinate to ensure it's within [-180,180]
pub fn normalize_longitude<T: CoordFloat + FromPrimitive>(coord: T) -> T {
    let one_eighty = T::from(180.0f64).unwrap();
    let three_sixty = T::from(360.0f64).unwrap();
    let five_forty = T::from(540.0f64).unwrap();

    ((coord + five_forty) % three_sixty) - one_eighty
}

/// `floor(log2(|value|))`, computed exactly from the float's representation. `value` must be
/// finite and non-zero.
pub(crate) fn binary_exponent<F: Float>(value: F) -> i32 {
    debug_assert!(value.is_finite() && value != F::zero());
    let (mantissa, exponent, _) = value.integer_decode();
    exponent as i32 + (63 - mantissa.leading_zeros() as i32)
}

/// The largest binary exponent a finite `F` can carry: `1023` for `f64`, `127` for `f32`.
pub(crate) fn max_binary_exponent<F: Float>() -> i32 {
    binary_exponent(F::max_value())
}

/// Returns a scaling factor to scale the exponent of `max_abs` to `target_exponent`, clamped
/// by ±`max_binary_exponent` to ensure it is representable.
///
/// This is the shared primitive behind the overflow/underflow retries in
/// [`RobustKernel::orient2d`](crate::kernels::RobustKernel) and
/// [`line_intersection`](crate::line_intersection::line_intersection).
/// used for fixing scaling in non-degenerate, just extreme, geometries.
///
/// Relies on division and multiplication of floats by two being precise. This is a valid assumption
/// because multiplying a float by a power of two just changes the exponent.
///
/// This is best effort as we can only return a float. The extreme case is the smallest subnormal,
/// where `power_of_two_scale(f64::from_bits(1), 500)` returns `2^1023` and lands `max_abs` at
/// `2^-51` — 551 exponents short of the target. This still guarantees the preciseness of the
/// required results.
///
/// `max_abs` must be finite and positive.
pub(crate) fn power_of_two_scale<F: Float>(max_abs: F, target_exponent: i32) -> F {
    debug_assert!(max_abs.is_finite() && max_abs > F::zero());
    let max_exponent = max_binary_exponent::<F>();
    let shift = (target_exponent - binary_exponent(max_abs)).clamp(-max_exponent, max_exponent);

    // Repeated multiplication by two (or one half) is exact at every step, so the result is
    // exactly `2^shift` without relying on how `powi` happens to be implemented.
    let two = F::one() + F::one();
    let step = if shift < 0 { F::one() / two } else { two };
    (0..shift.unsigned_abs()).fold(F::one(), |scale, _| scale * step)
}

#[cfg(test)]
mod test {
    use super::*;
    use hegel::TestCase;
    use hegel::generators as gs;

    /// Valid for `e` in `[-1074, 1023]`.
    fn exact_pow2(e: i32) -> f64 {
        if e >= -1022 {
            f64::from_bits(((e + 1023) as u64) << 52)
        } else {
            f64::from_bits(1u64 << (e + 1074))
        }
    }

    #[hegel::test]
    fn test_binary_exponent_is_exact(tc: TestCase) {
        let val = tc.draw(
            gs::floats::<f64>()
                .min_value(0.0)
                .exclude_min(true)
                .allow_infinity(false),
        );
        let e = binary_exponent(val);
        assert!((-1074..=1023).contains(&e), "{val:e} e={e}");
        assert!(exact_pow2(e) <= val, "{val:e} e={e}");
        // `2^1024` is not representable, so the upper bound is vacuous at the top.
        assert!(e == 1023 || val < exact_pow2(e + 1), "{val:e} e={e}");
    }

    #[hegel::test]
    fn test_binary_exponent_round_trips_powers_of_two(tc: TestCase) {
        let e = tc.draw(gs::integers::<i32>().min_value(-1074).max_value(1023));
        assert_eq!(binary_exponent(exact_pow2(e)), e);
        assert_eq!(binary_exponent(-exact_pow2(e)), e);
    }

    #[test]
    fn test_max_binary_exponent() {
        assert_eq!(max_binary_exponent::<f64>(), 1023);
        assert_eq!(max_binary_exponent::<f32>(), 127);
    }

    #[test]
    fn power_of_two_scale_is_exact_and_capped() {
        assert_eq!(power_of_two_scale(2f64.powi(1023), 500), 2f64.powi(-523));
        assert_eq!(power_of_two_scale(2f64.powi(-143), 500), 2f64.powi(643));
        assert_eq!(power_of_two_scale(3.0f64, 0), 0.5);
        // Subnormal inputs get the largest representable power of two.
        assert_eq!(power_of_two_scale(f64::from_bits(1), 500), 2f64.powi(1023));
        assert_eq!(power_of_two_scale(f32::from_bits(1), 38), 2f32.powi(127));
    }

    use super::{partial_max, partial_min};

    #[test]
    fn test_partial_max() {
        assert_eq!(5, partial_max(5, 4));
        assert_eq!(5, partial_max(5, 5));
    }

    #[test]
    fn test_partial_min() {
        assert_eq!(4, partial_min(5, 4));
        assert_eq!(4, partial_min(4, 4));
    }
}

/// Generators and helpers shared by the property-based tests that run under `hegel`.
#[cfg(test)]
pub(crate) mod property_tests {
    use crate::{Coord, Point, Rect, Validation};
    use hegel::TestCase;

    hegel::derive_generator!(CoordGenerator for Coord {
        x: f64,
        y: f64,
    });

    /// Draws a rect from two arbitrary corners, rejecting the test case if it is not valid
    /// (for example, if a coordinate is NaN). Degenerate rects (zero width or height) are valid
    /// and deliberately kept: they are where the predicates are most likely to disagree.
    pub(crate) fn draw_valid_rect(tc: &TestCase) -> Rect<f64> {
        let rect = Rect::new(
            tc.draw(CoordGenerator::new()),
            tc.draw(CoordGenerator::new()),
        );
        tc.assume(rect.check_validation().is_ok());
        rect
    }

    /// Draws an arbitrary point, rejecting the test case if it is not valid.
    pub(crate) fn draw_valid_point(tc: &TestCase) -> Point<f64> {
        let point = Point::from(tc.draw(CoordGenerator::new()));
        tc.assume(point.check_validation().is_ok());
        point
    }
}
