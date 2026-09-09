use crate::{Coord, GeoFloat, Line};
use std::cmp::Ordering;

/// Represents a point on an edge which intersects with another edge.
///
/// The intersection may either be a single point, or a line segment (in which case this point is
/// the start of the line segment) The intersection point must be precise.
///
/// Intersections are kept in a `BTreeSet` per edge, ordered by their position along the edge:
/// first by the segment they fall on, then by where they fall within that segment.
#[derive(Debug, Clone)]
pub(crate) struct EdgeIntersection<F: GeoFloat> {
    coord: Coord<F>,
    segment_index: usize,
    /// Whether `coord` is the start vertex of segment `segment_index`.
    is_at_segment_start: bool,
    /// Position of `coord` along its segment, as a lexicographic key that increases from the
    /// segment's start to its end: `[major, minor]`, where `major` is the ordinate along the
    /// axis in which the segment extends further, and each ordinate is multiplied by `-1` when
    /// the segment runs in the negative direction along that axis.
    /// 
    /// For example, a segment running from (10, 3) -> (2, 5) would have positions of the form
    /// \x y -> [-x, y].
    position: [F; 2],
}

impl<F: GeoFloat> EdgeIntersection<F> {
    /// `segment` must be the segment `segment_index` of the edge this intersection lies on.
    pub fn new(coord: Coord<F>, segment_index: usize, segment: Line<F>) -> EdgeIntersection<F> {
        let dx = segment.end.x - segment.start.x;
        let dy = segment.end.y - segment.start.y;

        // `+1` for a zero delta (including `-0.0`) — the sign only has to be consistent for a
        // given segment.
        let direction = |delta: F| {
            if delta < F::zero() {
                -F::one()
            } else {
                F::one()
            }
        };
        let x = coord.x * direction(dx);
        let y = coord.y * direction(dy);

        // On a tie the y axis is the major axis, matching JTS's `computeEdgeDistance`.
        let position = if dx.abs() > dy.abs() { [x, y] } else { [y, x] };

        EdgeIntersection {
            coord,
            segment_index,
            is_at_segment_start: coord == segment.start,
            position,
        }
    }

    pub fn coordinate(&self) -> Coord<F> {
        self.coord
    }

    pub fn segment_index(&self) -> usize {
        self.segment_index
    }

    /// Whether this intersection lies exactly on the start vertex of its segment.
    pub fn is_at_segment_start(&self) -> bool {
        self.is_at_segment_start
    }
}

impl<F: GeoFloat> PartialEq for EdgeIntersection<F> {
    fn eq(&self, other: &EdgeIntersection<F>) -> bool {
        // Defined via `cmp` so that equality can never disagree with the ordering. Within one
        // segment, `position` is the coordinate up to sign flips, so this is coordinate equality.
        self.cmp(other) == Ordering::Equal
    }
}

impl<F: GeoFloat> Eq for EdgeIntersection<F> {}

impl<F: GeoFloat> PartialOrd for EdgeIntersection<F> {
    fn partial_cmp(&self, other: &EdgeIntersection<F>) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<F: GeoFloat> Ord for EdgeIntersection<F> {
    /// A reimplementation of `cmp` based on the position field of the `EdgeIntersection`.
    /// If segment index is equal, then we first compare with the major axis, and then the
    /// minor axis.
    fn cmp(&self, other: &EdgeIntersection<F>) -> Ordering {
        // `BTreeSet` requires a total order, but we're comparing floats, so we require non-NaN
        // coordinates for valid results. `partial_cmp` (rather than `total_cmp`) keeps
        // `-0.0 == 0.0`, consistent with coordinate equality elsewhere in the graph.
        debug_assert!(
            self.position
                .iter()
                .chain(other.position.iter())
                .all(|ordinate| !ordinate.is_nan())
        );
        self.segment_index
            .cmp(&other.segment_index)
            .then_with(|| {
                self.position[0]
                    .partial_cmp(&other.position[0])
                    .unwrap_or(Ordering::Equal)
            })
            .then_with(|| {
                self.position[1]
                    .partial_cmp(&other.position[1])
                    .unwrap_or(Ordering::Equal)
            })
    }
}

impl<F: GeoFloat> EdgeIntersection<F> {}
