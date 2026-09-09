use super::{LineIntersection, LineIntersector};
use crate::{GeoFloat, Line};

/// A robust version of [LineIntersector](traits.LineIntersector).
#[derive(Clone)]
pub(crate) struct RobustLineIntersector;

impl RobustLineIntersector {
    pub fn new() -> RobustLineIntersector {
        RobustLineIntersector
    }
}

impl<F: GeoFloat> LineIntersector<F> for RobustLineIntersector {
    fn compute_intersection(&mut self, p: Line<F>, q: Line<F>) -> Option<LineIntersection<F>> {
        crate::line_intersection::line_intersection(p, q)
    }
}
