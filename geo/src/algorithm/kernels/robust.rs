use super::{CoordNum, Kernel, Orientation};
use crate::Coord;

use crate::utils::power_of_two_scale;
use num_traits::{Float, NumCast};

/// Robust kernel that uses [fast robust
/// predicates](//www.cs.cmu.edu/~quake/robust.html) to
/// provide robust floating point predicates. Should only be
/// used with types that can _always_ be casted to `f64`
/// _without loss in precision_.
#[derive(Default, Debug)]
pub struct RobustKernel;

impl<T> Kernel<T> for RobustKernel
where
    T: CoordNum + Float,
{
    fn orient2d(p: Coord<T>, q: Coord<T>, r: Coord<T>) -> Orientation {
        let p = to_robust_coord(p);
        let q = to_robust_coord(q);
        let r = to_robust_coord(r);

        let mut orientation = robust::orient2d(p, q, r);

        // `robust::orient2d` is exact as long as none of its intermediate
        // values overflow or underflow. We check if there is a possibility here
        // and use a rescaled version if so (to avoid the {under, over} flow).
        // An overflow would lead to a Collinear result as NaN is neither pos
        // or neg, and an underflow may flush to zero.
        if !orientation.is_finite() || (orientation == 0.0 && may_have_underflowed(p, q, r)) {
            orientation = rescaled_orient2d(p, q, r).unwrap_or(orientation);
        }

        if orientation < 0. {
            Orientation::Clockwise
        } else if orientation > 0. {
            Orientation::CounterClockwise
        } else {
            Orientation::Collinear
        }
    }
}

fn to_robust_coord<T: CoordNum + Float>(c: Coord<T>) -> robust::Coord<f64> {
    robust::Coord {
        x: <f64 as NumCast>::from(c.x).unwrap(),
        y: <f64 as NumCast>::from(c.y).unwrap(),
    }
}

/// Whether a zero result from `robust::orient2d` could be an artifact of
/// underflow rather than a genuine collinearity.
///
/// Underflow is the only way `orient2d` loses information. It represents the
/// determinant exactly, as a sum of `f64` components rather than one rounded
/// number, and a component too small to be a normal `f64` flushes to zero and
/// drops out of that sum — so a determinant carried entirely by such
/// components reads as zero. The two products below are the largest
/// components, and every other one is within about `2^-159` of them (the two
/// coordinate subtractions and the product each shed one `f64` mantissa).
/// Bounding those products therefore bounds the whole sum: at or above
/// `2^-800` its smallest component is still `2^-959` and normal, so nothing
/// was dropped and the zero is exact.
fn may_have_underflowed(
    p: robust::Coord<f64>,
    q: robust::Coord<f64>,
    r: robust::Coord<f64>,
) -> bool {
    // ~2^-800: far above where correction terms could go subnormal, far below
    // anything ordinary coordinates produce, so the retry stays off the hot path.
    const UNDERFLOW_SUSPECT_BOUND: f64 = 1.5e-241; // ~2^-800

    let left = ((p.x - r.x) * (q.y - r.y)).abs();
    let right = ((p.y - r.y) * (q.x - r.x)).abs();
    left < UNDERFLOW_SUSPECT_BOUND && right < UNDERFLOW_SUSPECT_BOUND
}

/// Evaluates `orient2d` on inputs rescaled so that the predicate can neither
/// overflow nor underflow. Returns `None` if the inputs are not finite.
///
/// We have to scale each axis independently as a triple can mix ordinates near
/// `f64::MAX` on one axis with ordinates around `1e-170` on the other.
/// This is valid due to the properties of the determinant. Scaling `x` by
/// `a > 0` and `y` by `b > 0` scales the determinant by `ab`.
///
/// Since `ab > 0` the sign is unchanged, which is all `orient2d` reports. See
/// [`crate::utils::power_of_two_scale`] for the exactness argument this shares
/// with `line_intersection`'s retry, and
/// `orient2d_survives_mixed_magnitudes_across_axes` for the counterexample
/// that forced per-axis scales.
fn rescaled_orient2d(
    p: robust::Coord<f64>,
    q: robust::Coord<f64>,
    r: robust::Coord<f64>,
) -> Option<f64> {
    let sx = axis_scale([p.x, q.x, r.x])?;
    let sy = axis_scale([p.y, q.y, r.y])?;
    let scale = |c: robust::Coord<f64>| robust::Coord {
        x: c.x * sx,
        y: c.y * sy,
    };
    Some(robust::orient2d(scale(p), scale(q), scale(r)))
}

/// Binary exponent the largest ordinate of each axis is brought to before
/// re-evaluating.
///
/// With every ordinate below `2^501`, the differences inside `orient2d` stay
/// below `2^502` and their products below `2^1004`, clear of `f64::MAX`
/// (`~2^1024`). This also balances against underflow (> ~2^-511).
const RESCALE_TARGET_EXPONENT: i32 = 500;

/// Returns a power of two that brings the largest of `ordinates` to about
/// `2^RESCALE_TARGET_EXPONENT`: `None` if any ordinate is not finite, `1` if
/// all of them are zero (nothing to scale).
fn axis_scale(ordinates: [f64; 3]) -> Option<f64> {
    if ordinates.iter().any(|ordinate| !ordinate.is_finite()) {
        return None;
    }
    let max_abs = ordinates
        .iter()
        .fold(0.0f64, |max, ordinate| max.max(ordinate.abs()));
    if max_abs == 0.0 {
        return Some(1.0);
    }
    Some(power_of_two_scale(max_abs, RESCALE_TARGET_EXPONENT))
}
