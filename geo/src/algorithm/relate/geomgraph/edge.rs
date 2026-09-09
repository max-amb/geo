use super::LineIntersection;
use super::{Dimensions, Direction, EdgeIntersection, IntersectionMatrix, Label};
use crate::{Coord, GeoFloat, Line};

use std::collections::BTreeSet;

/// An `Edge` represents a one dimensional line in a geometry.
///
/// This is based on [JTS's `Edge` as of 1.18.1](https://github.com/locationtech/jts/blob/jts-1.18.1/modules/core/src/main/java/org/locationtech/jts/geomgraph/Edge.java)
#[derive(Debug, PartialEq, Clone)]
pub(crate) struct Edge<F: GeoFloat> {
    /// `coordinates` of the line geometry
    coords: Vec<Coord<F>>,

    /// an edge is "isolated" if no other edge touches it
    is_isolated: bool,

    /// other edges that this edge intersects with
    edge_intersections: BTreeSet<EdgeIntersection<F>>,

    /// where the line's topological classification to the two geometries is recorded
    label: Label,
}

impl<F: GeoFloat> Edge<F> {
    /// Create a new Edge.
    ///
    /// - `coords` a *non-empty* Vec of Coords
    /// - `label` an appropriately dimensioned topology label for the Edge. See [`TopologyPosition`]
    ///   for details
    pub(crate) fn new(mut coords: Vec<Coord<F>>, label: Label) -> Edge<F> {
        assert!(!coords.is_empty(), "Can't add empty edge");
        // Once set, `edge.coords` never changes length.
        coords.shrink_to_fit();
        Edge {
            coords,
            label,
            is_isolated: true,
            edge_intersections: BTreeSet::new(),
        }
    }

    pub(crate) fn label(&self) -> &Label {
        &self.label
    }

    pub(crate) fn label_mut(&mut self) -> &mut Label {
        &mut self.label
    }

    /// When comparing two prepared geometries, we cache each geometry's topology graph. Depending
    /// on the order of the operation - `a.relate(b)` vs `b.relate(a)` - we may need to swap the
    /// label.
    pub fn swap_label_args(&mut self) {
        self.label.swap_args()
    }

    pub fn coords(&self) -> &[Coord<F>] {
        &self.coords
    }

    /// The segment starting at vertex `index`.
    ///
    /// Protects against `add_intersection` and `add_edge_intersection_list_endpoints`
    /// using a segment that doesn't exist. This can happen when the intersection is 
    /// at the end of the final segment. We create a zero-length final segment in this case.
    /// This is ok because if this segment is used, only one coordinate would ever be found.
    pub fn segment(&self, index: usize) -> Line<F> {
        let last = self.coords.len() - 1;
        Line::new(self.coords[index], self.coords[(index + 1).min(last)])
    }

    pub fn is_isolated(&self) -> bool {
        self.is_isolated
    }

    pub fn mark_as_unisolated(&mut self) {
        self.is_isolated = false;
    }

    pub fn edge_intersections(&self) -> &BTreeSet<EdgeIntersection<F>> {
        &self.edge_intersections
    }

    pub fn edge_intersections_mut(&mut self) -> &mut BTreeSet<EdgeIntersection<F>> {
        &mut self.edge_intersections
    }

    pub fn add_edge_intersection_list_endpoints(&mut self) {
        let max_segment_index = self.coords().len() - 1;
        let first_coord = self.coords()[0];
        let max_coord = self.coords()[max_segment_index];
        let first_segment = self.segment(0);
        let last_segment = self.segment(max_segment_index);
        self.edge_intersections_mut()
            .insert(EdgeIntersection::new(first_coord, 0, first_segment));
        self.edge_intersections_mut().insert(EdgeIntersection::new(
            max_coord,
            max_segment_index,
            last_segment,
        ));
    }

    pub fn is_closed(&self) -> bool {
        self.coords().first() == self.coords().last()
    }

    /// Adds EdgeIntersections for one or both intersections found for a segment of an edge to the
    /// edge intersection list.
    pub fn add_intersections(&mut self, intersection: LineIntersection<F>, segment_index: usize) {
        match intersection {
            LineIntersection::SinglePoint { intersection, .. } => {
                self.add_intersection(intersection, segment_index);
            }
            LineIntersection::Collinear { intersection } => {
                self.add_intersection(intersection.start, segment_index);
                self.add_intersection(intersection.end, segment_index);
            }
        }
    }

    /// Add an EdgeIntersection for `intersection`.
    ///
    /// An intersection that falls exactly on a vertex of the edge is normalized to use the higher
    /// of the two possible `segment_index`
    pub fn add_intersection(&mut self, intersection_coord: Coord<F>, segment_index: usize) {
        let mut normalized_segment_index = segment_index;

        let next_segment_index = normalized_segment_index + 1;

        if next_segment_index < self.coords.len() {
            let next_coord = self.coords[next_segment_index];
            if intersection_coord == next_coord {
                normalized_segment_index = next_segment_index;
            }
        }
        // The position key is relative to the segment the intersection is filed under, so it
        // must be computed against the normalized segment.
        let segment = self.segment(normalized_segment_index);
        self.edge_intersections.insert(EdgeIntersection::new(
            intersection_coord,
            normalized_segment_index,
            segment,
        ));
    }

    /// Update the IM with the contribution for this component.
    ///
    /// A component only contributes if it has a labelling for both parent geometries
    pub fn update_intersection_matrix(label: &Label, intersection_matrix: &mut IntersectionMatrix) {
        intersection_matrix.set_at_least_if_in_both(
            label.position(0, Direction::On),
            label.position(1, Direction::On),
            Dimensions::OneDimensional,
        );

        if label.is_area() {
            intersection_matrix.set_at_least_if_in_both(
                label.position(0, Direction::Left),
                label.position(1, Direction::Left),
                Dimensions::TwoDimensional,
            );
            intersection_matrix.set_at_least_if_in_both(
                label.position(0, Direction::Right),
                label.position(1, Direction::Right),
                Dimensions::TwoDimensional,
            );
        }
    }
}
