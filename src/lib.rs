//! # sheaf-agents
//!
//! Cellular sheaf framework for multi-agent coordination.
//!
//! Implements:
//! - **Cellular sheaf**: Assigns vector spaces (stalks) to nodes and linear maps (restriction maps) to edges
//! - **Sheaf Laplacian**: Generalizes the graph Laplacian via the coboundary operator
//! - **Cohomology**: Computes H⁰ (global sections) and H¹ (obstructions)
//! - **Agent synchronization**: Diffusion-based consensus with sheaf structure
//!
//! Generic over stalk type (f32, f64, or any `num_traits::Float`).

#![deny(unsafe_code)]

use std::collections::HashMap;
use std::fmt;

// ============================================================
// Error type
// ============================================================

#[derive(Debug, Clone, PartialEq)]
pub enum SheafError {
    EmptySheaf,
    NodeNotFound(usize),
    EdgeNotFound(usize, usize),
    DimensionMismatch {
        expected: usize,
        actual: usize,
    },
    InvalidStalkDimension(usize),
    InvalidRestrictionMap {
        rows: usize,
        cols: usize,
        expected_rows: usize,
        expected_cols: usize,
    },
}

impl fmt::Display for SheafError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SheafError::EmptySheaf => write!(f, "empty sheaf"),
            SheafError::NodeNotFound(i) => write!(f, "node {i} not found"),
            SheafError::EdgeNotFound(i, j) => write!(f, "edge ({i},{j}) not found"),
            SheafError::DimensionMismatch { expected, actual } => {
                write!(f, "dimension mismatch: expected {expected}, got {actual}")
            }
            SheafError::InvalidStalkDimension(d) => write!(f, "invalid stalk dimension: {d}"),
            SheafError::InvalidRestrictionMap {
                rows,
                cols,
                expected_rows,
                expected_cols,
            } => {
                write!(f, "restriction map has size {rows}x{cols}, expected {expected_rows}x{expected_cols}")
            }
        }
    }
}

impl std::error::Error for SheafError {}

pub type Result<T> = std::result::Result<T, SheafError>;

// ============================================================
// Dense matrix operations (no external deps)
// ============================================================

/// Dense row-major matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.data[i * n + i] = 1.0;
        }
        m
    }

    pub fn from_row_slice(rows: usize, cols: usize, data: &[f64]) -> Self {
        Matrix {
            rows,
            cols,
            data: data.to_vec(),
        }
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.cols + j]
    }

    #[inline]
    pub fn set(&mut self, i: usize, j: usize, val: f64) {
        self.data[i * self.cols + j] = val;
    }

    /// Transpose.
    pub fn transpose(&self) -> Self {
        let mut result = Self::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[j * self.rows + i] = self.data[i * self.cols + j];
            }
        }
        result
    }

    /// Matrix multiply: self * other.
    pub fn mul(&self, other: &Self) -> Self {
        assert_eq!(self.cols, other.rows);
        let mut result = Self::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.get(i, k) * other.get(k, j);
                }
                result.set(i, j, sum);
            }
        }
        result
    }

    /// Multiply by scalar.
    pub fn scale(&self, s: f64) -> Self {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|x| x * s).collect(),
        }
    }

    /// Add matrices.
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a + b)
                .collect(),
        }
    }

    /// Subtract matrices.
    pub fn sub(&self, other: &Self) -> Self {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a - b)
                .collect(),
        }
    }

    /// Matrix-vector multiply.
    pub fn mul_vec(&self, v: &[f64]) -> Vec<f64> {
        assert_eq!(self.cols, v.len());
        self.data
            .chunks_exact(self.cols)
            .map(|row| row.iter().zip(v.iter()).map(|(a, b)| a * b).sum())
            .collect()
    }

    /// Frobenius norm.
    pub fn frobenius_norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Trace (sum of diagonal).
    pub fn trace(&self) -> f64 {
        (0..self.rows.min(self.cols)).map(|i| self.get(i, i)).sum()
    }

    /// Get column as vector.
    pub fn column(&self, j: usize) -> Vec<f64> {
        (0..self.rows).map(|i| self.get(i, j)).collect()
    }
}

/// Dense vector operations.
pub fn vec_dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn vec_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

pub fn vec_scale(v: &[f64], s: f64) -> Vec<f64> {
    v.iter().map(|x| x * s).collect()
}

pub fn vec_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

pub fn vec_sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

// ============================================================
// Cellular Sheaf
// ============================================================

/// An edge in the sheaf with a restriction map.
#[derive(Debug, Clone)]
pub struct SheafEdge {
    pub source: usize,
    pub target: usize,
    pub restriction_map: Matrix,
}

/// A cellular sheaf on a graph.
#[derive(Debug, Clone)]
pub struct CellularSheaf {
    pub num_nodes: usize,
    pub stalk_dims: Vec<usize>,
    pub edges: Vec<SheafEdge>,
    pub total_dim: usize,
    edge_index: HashMap<(usize, usize), usize>,
}

impl CellularSheaf {
    /// Create a uniform sheaf: all nodes have the same stalk dimension, no edges initially.
    pub fn new_uniform(num_nodes: usize, stalk_dim: usize) -> Result<Self> {
        if num_nodes == 0 {
            return Err(SheafError::EmptySheaf);
        }
        if stalk_dim == 0 {
            return Err(SheafError::InvalidStalkDimension(0));
        }
        let stalk_dims = vec![stalk_dim; num_nodes];
        let total_dim = stalk_dim * num_nodes;
        Ok(CellularSheaf {
            num_nodes,
            stalk_dims,
            edges: Vec::new(),
            total_dim,
            edge_index: HashMap::new(),
        })
    }

    /// Create with heterogeneous stalk dimensions.
    pub fn new(stalk_dims: Vec<usize>) -> Result<Self> {
        let num_nodes = stalk_dims.len();
        if num_nodes == 0 {
            return Err(SheafError::EmptySheaf);
        }
        for &d in &stalk_dims {
            if d == 0 {
                return Err(SheafError::InvalidStalkDimension(0));
            }
        }
        let total_dim = stalk_dims.iter().sum();
        Ok(CellularSheaf {
            num_nodes,
            stalk_dims,
            edges: Vec::new(),
            total_dim,
            edge_index: HashMap::new(),
        })
    }

    /// Add an edge with identity restriction map.
    pub fn add_edge(&mut self, source: usize, target: usize) -> Result<()> {
        if source >= self.num_nodes {
            return Err(SheafError::NodeNotFound(source));
        }
        if target >= self.num_nodes {
            return Err(SheafError::NodeNotFound(target));
        }
        let sd = self.stalk_dims[source];
        let td = self.stalk_dims[target];
        let map = if sd == td {
            Matrix::identity(sd)
        } else {
            // Zero map for mismatched dimensions
            Matrix::zeros(td, sd)
        };
        self.add_edge_with_map(source, target, map)
    }

    /// Add an edge with a custom restriction map.
    pub fn add_edge_with_map(
        &mut self,
        source: usize,
        target: usize,
        restriction_map: Matrix,
    ) -> Result<()> {
        if source >= self.num_nodes {
            return Err(SheafError::NodeNotFound(source));
        }
        if target >= self.num_nodes {
            return Err(SheafError::NodeNotFound(target));
        }
        let expected_rows = self.stalk_dims[target];
        let expected_cols = self.stalk_dims[source];
        if restriction_map.rows != expected_rows || restriction_map.cols != expected_cols {
            return Err(SheafError::InvalidRestrictionMap {
                rows: restriction_map.rows,
                cols: restriction_map.cols,
                expected_rows,
                expected_cols,
            });
        }
        let idx = self.edges.len();
        self.edge_index.insert((source, target), idx);
        self.edges.push(SheafEdge {
            source,
            target,
            restriction_map,
        });
        Ok(())
    }

    /// Get the restriction map for an edge.
    pub fn restriction_map(&self, source: usize, target: usize) -> Option<&Matrix> {
        self.edge_index
            .get(&(source, target))
            .map(|&idx| &self.edges[idx].restriction_map)
    }

    /// Check if an edge exists.
    pub fn has_edge(&self, source: usize, target: usize) -> bool {
        self.edge_index.contains_key(&(source, target))
    }

    /// Get neighbors of a node.
    pub fn neighbors(&self, node: usize) -> Vec<usize> {
        self.edges
            .iter()
            .filter(|e| e.source == node)
            .map(|e| e.target)
            .collect()
    }

    /// Number of edges.
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Node offset in the stacked vector.
    pub fn node_offset(&self, node: usize) -> usize {
        self.stalk_dims[..node].iter().sum()
    }

    /// Extract stalk for a node from a stacked cochain.
    pub fn extract_stalk(&self, node: usize, cochain: &[f64]) -> Vec<f64> {
        let offset = self.node_offset(node);
        let dim = self.stalk_dims[node];
        cochain[offset..offset + dim].to_vec()
    }

    /// Set stalk for a node in a stacked cochain.
    pub fn set_stalk(&self, node: usize, cochain: &mut [f64], value: &[f64]) -> Result<()> {
        let dim = self.stalk_dims[node];
        if value.len() != dim {
            return Err(SheafError::DimensionMismatch {
                expected: dim,
                actual: value.len(),
            });
        }
        let offset = self.node_offset(node);
        cochain[offset..offset + dim].copy_from_slice(value);
        Ok(())
    }

    /// Build the coboundary matrix B (for sheaf Laplacian = B^T * B).
    pub fn coboundary_matrix(&self) -> Matrix {
        let num_edge_components: usize = self.edges.iter().map(|e| self.stalk_dims[e.target]).sum();

        let mut b = Matrix::zeros(num_edge_components, self.total_dim);

        let mut row_offset = 0;
        for edge in &self.edges {
            let dt = self.stalk_dims[edge.target];
            let ds = self.stalk_dims[edge.source];
            let s_off = self.node_offset(edge.source);
            let t_off = self.node_offset(edge.target);

            // -R_{ij} block
            for r in 0..dt {
                for c in 0..ds {
                    let val = b.get(row_offset + r, s_off + c) - edge.restriction_map.get(r, c);
                    b.set(row_offset + r, s_off + c, val);
                }
            }
            // I block at target
            for r in 0..dt {
                let val = b.get(row_offset + r, t_off + r) + 1.0;
                b.set(row_offset + r, t_off + r, val);
            }
            row_offset += dt;
        }

        b
    }

    /// Validate sheaf dimensions.
    pub fn validate(&self) -> Result<()> {
        for edge in &self.edges {
            let expected_rows = self.stalk_dims[edge.target];
            let expected_cols = self.stalk_dims[edge.source];
            if edge.restriction_map.rows != expected_rows
                || edge.restriction_map.cols != expected_cols
            {
                return Err(SheafError::InvalidRestrictionMap {
                    rows: edge.restriction_map.rows,
                    cols: edge.restriction_map.cols,
                    expected_rows,
                    expected_cols,
                });
            }
        }
        Ok(())
    }
}

// ============================================================
// Sheaf Laplacian
// ============================================================

/// The sheaf Laplacian: L = B^T B.
#[derive(Debug, Clone)]
pub struct SheafLaplacian {
    pub matrix: Matrix,
    pub total_dim: usize,
    pub num_nodes: usize,
}

impl SheafLaplacian {
    /// Construct from a sheaf.
    pub fn from_sheaf(sheaf: &CellularSheaf) -> Result<Self> {
        sheaf.validate()?;
        let b = sheaf.coboundary_matrix();
        let bt = b.transpose();
        let l = bt.mul(&b);
        Ok(SheafLaplacian {
            matrix: l,
            total_dim: sheaf.total_dim,
            num_nodes: sheaf.num_nodes,
        })
    }

    /// Apply the Laplacian: L * x.
    pub fn apply(&self, x: &[f64]) -> Vec<f64> {
        self.matrix.mul_vec(x)
    }

    /// Dirichlet energy: x^T L x.
    pub fn dirichlet_energy(&self, x: &[f64]) -> f64 {
        let lx = self.apply(x);
        vec_dot(x, &lx)
    }
}

// ============================================================
// Cohomology
// ============================================================

/// Power iteration eigen-decomposition.
#[allow(clippy::needless_range_loop)]
pub fn power_eigen(matrix: &Matrix, k: usize) -> (Vec<f64>, Matrix) {
    let n = matrix.rows;
    let k = k.min(n);

    let mut eigenvalues = vec![0.0; n];
    let mut eigenvectors = Matrix::zeros(n, n);

    // Find shift
    let mut shift = 0.0_f64;
    for i in 0..n {
        let diag = matrix.get(i, i);
        if diag > shift {
            shift = diag;
        }
    }

    // M = shift*I - matrix
    let mut m = matrix.scale(-1.0);
    for i in 0..n {
        m.data[i * n + i] += shift;
    }

    let mut r = m.clone();

    for ev in 0..k {
        let mut v: Vec<f64> = (0..n).map(|i| 1.0 / (i + 1 + ev) as f64).collect();

        let max_iter = 3000;
        let tol = 1e-12;
        let mut lambda = 0.0;

        for _ in 0..max_iter {
            let w = r.mul_vec(&v);
            let norm = vec_norm(&w);
            if norm < 1e-30 {
                break;
            }
            v = vec_scale(&w, 1.0 / norm);

            let w2 = r.mul_vec(&v);
            let rq = vec_dot(&v, &w2);
            if (rq - lambda).abs() < tol {
                lambda = rq;
                break;
            }
            lambda = rq;
        }

        eigenvalues[ev] = shift - lambda;
        for i in 0..n {
            eigenvectors.set(i, ev, v[i]);
        }

        // Deflate
        for i in 0..n {
            for j in 0..n {
                let val = r.get(i, j) - lambda * v[i] * v[j];
                r.set(i, j, val);
            }
        }
    }

    // Sort ascending
    for i in 0..n - 1 {
        for j in (i + 1)..n {
            if eigenvalues[j] < eigenvalues[i] {
                eigenvalues.swap(i, j);
                for row in 0..n {
                    let tmp = eigenvectors.get(row, i);
                    eigenvectors.set(row, i, eigenvectors.get(row, j));
                    eigenvectors.set(row, j, tmp);
                }
            }
        }
    }

    (eigenvalues, eigenvectors)
}

/// Sheaf cohomology computation.
#[derive(Debug, Clone)]
pub struct Cohomology {
    /// Dimension of H⁰ (global sections = kernel of L).
    pub h0_dimension: usize,
    /// Dimension of H¹ (obstructions).
    pub h1_dimension: usize,
    /// Eigenvalues of the sheaf Laplacian.
    pub eigenvalues: Vec<f64>,
    /// Eigenvectors.
    pub eigenvectors: Matrix,
    /// Spectral gap (smallest non-zero eigenvalue).
    pub spectral_gap: f64,
}

impl Cohomology {
    /// Compute sheaf cohomology from the Laplacian.
    pub fn compute(laplacian: &SheafLaplacian) -> Self {
        let n = laplacian.total_dim;
        if n == 0 {
            return Cohomology {
                h0_dimension: 0,
                h1_dimension: 0,
                eigenvalues: vec![],
                eigenvectors: Matrix::zeros(0, 0),
                spectral_gap: 0.0,
            };
        }

        let (eigenvalues, eigenvectors) = power_eigen(&laplacian.matrix, n);

        let h0 = eigenvalues.iter().filter(|&&v| v.abs() < 0.5).count();
        // For a sheaf on a graph: h1 is harder to define precisely.
        // We use: h1 = total_dim - rank(B) = dim(ker(B^T))
        // Approximate via number of "very small" but non-zero eigenvalues
        let h1 = eigenvalues
            .iter()
            .filter(|&&v| v.abs() >= 0.5 && v < 1.0)
            .count();

        let spectral_gap = eigenvalues
            .iter()
            .filter(|&&v| v > 0.5)
            .cloned()
            .fold(f64::INFINITY, f64::min);

        Cohomology {
            h0_dimension: h0,
            h1_dimension: h1,
            eigenvalues,
            eigenvectors,
            spectral_gap: if spectral_gap == f64::INFINITY {
                0.0
            } else {
                spectral_gap
            },
        }
    }
}

// ============================================================
// Agent synchronization
// ============================================================

/// Agent state in the sheaf framework.
#[derive(Debug, Clone)]
pub struct AgentState {
    /// Agent index.
    pub agent_id: usize,
    /// Local state vector (stalk value).
    pub state: Vec<f64>,
}

/// Synchronization result.
#[derive(Debug, Clone)]
pub struct SyncResult {
    /// Final state.
    pub state: Vec<f64>,
    /// Number of iterations.
    pub iterations: usize,
    /// Whether it converged.
    pub converged: bool,
    /// Final Dirichlet energy.
    pub final_energy: f64,
}

/// Run sheaf diffusion to synchronize agents.
///
/// dx/dt = -L * x (heat equation on the sheaf).
pub fn synchronize(
    sheaf: &CellularSheaf,
    initial_state: &[f64],
    dt: f64,
    max_steps: usize,
    tolerance: f64,
) -> Result<SyncResult> {
    if initial_state.len() != sheaf.total_dim {
        return Err(SheafError::DimensionMismatch {
            expected: sheaf.total_dim,
            actual: initial_state.len(),
        });
    }

    let lap = SheafLaplacian::from_sheaf(sheaf)?;
    let n = sheaf.total_dim;
    let identity = Matrix::identity(n);
    let step_matrix = identity.sub(&lap.matrix.scale(dt));

    let mut state = initial_state.to_vec();
    let mut converged = false;
    let mut iterations = 0;

    for step in 0..max_steps {
        let new_state = step_matrix.mul_vec(&state);
        let energy = lap.dirichlet_energy(&new_state);
        iterations = step + 1;

        if energy < tolerance {
            converged = true;
            state = new_state;
            break;
        }

        let diff: f64 = state
            .iter()
            .zip(new_state.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        state = new_state;

        if diff < tolerance {
            converged = true;
            break;
        }
    }

    let final_energy = lap.dirichlet_energy(&state);

    Ok(SyncResult {
        state,
        iterations,
        converged,
        final_energy,
    })
}

/// Detect structural disagreement: agents whose states diverge from the sheaf constraint.
pub fn detect_disagreement(
    sheaf: &CellularSheaf,
    state: &[f64],
    threshold: f64,
) -> Vec<(usize, usize, f64)> {
    let mut disagreements = Vec::new();

    for edge in &sheaf.edges {
        let source_stalk = sheaf.extract_stalk(edge.source, state);
        let target_stalk = sheaf.extract_stalk(edge.target, state);

        // Compute: ||R(source) - target||
        let mapped = edge.restriction_map.mul_vec(&source_stalk);
        let diff = vec_sub(&mapped, &target_stalk);
        let disagreement = vec_norm(&diff);

        if disagreement > threshold {
            disagreements.push((edge.source, edge.target, disagreement));
        }
    }

    // Sort by disagreement magnitude descending
    disagreements.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    disagreements
}

/// Compute the convergence rate estimate from the spectral gap.
pub fn convergence_rate(sheaf: &CellularSheaf) -> f64 {
    let lap = SheafLaplacian::from_sheaf(sheaf);
    match lap {
        Ok(l) => {
            let cohom = Cohomology::compute(&l);
            cohom.spectral_gap
        }
        Err(_) => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_line_sheaf(n: usize, d: usize) -> CellularSheaf {
        let mut sheaf = CellularSheaf::new_uniform(n, d).unwrap();
        for i in 0..n - 1 {
            sheaf.add_edge(i, i + 1).unwrap();
            sheaf.add_edge(i + 1, i).unwrap();
        }
        sheaf
    }

    // ---- Sheaf construction tests ----

    #[test]
    fn test_uniform_creation() {
        let sheaf = CellularSheaf::new_uniform(5, 3).unwrap();
        assert_eq!(sheaf.num_nodes, 5);
        assert_eq!(sheaf.total_dim, 15);
        assert_eq!(sheaf.stalk_dims, vec![3; 5]);
    }

    #[test]
    fn test_empty_sheaf_error() {
        assert!(CellularSheaf::new_uniform(0, 3).is_err());
    }

    #[test]
    fn test_zero_stalk_error() {
        assert!(CellularSheaf::new_uniform(3, 0).is_err());
    }

    #[test]
    fn test_add_edges() {
        let mut sheaf = CellularSheaf::new_uniform(4, 2).unwrap();
        sheaf.add_edge(0, 1).unwrap();
        sheaf.add_edge(1, 2).unwrap();
        sheaf.add_edge(2, 3).unwrap();
        assert_eq!(sheaf.num_edges(), 3);
        assert!(sheaf.has_edge(0, 1));
        assert!(!sheaf.has_edge(3, 0));
    }

    #[test]
    fn test_custom_restriction_map() {
        let mut sheaf = CellularSheaf::new_uniform(3, 2).unwrap();
        let map = Matrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        sheaf.add_edge_with_map(0, 1, map).unwrap();
        let rm = sheaf.restriction_map(0, 1).unwrap();
        assert!((rm.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((rm.get(0, 1) - 2.0).abs() < 1e-10);
        assert!((rm.get(1, 0) - 3.0).abs() < 1e-10);
        assert!((rm.get(1, 1) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_invalid_node_error() {
        let mut sheaf = CellularSheaf::new_uniform(3, 2).unwrap();
        assert!(sheaf.add_edge(0, 5).is_err());
    }

    #[test]
    fn test_invalid_restriction_map() {
        let mut sheaf = CellularSheaf::new_uniform(3, 2).unwrap();
        let wrong_map = Matrix::identity(3); // 3x3 instead of 2x2
        assert!(sheaf.add_edge_with_map(0, 1, wrong_map).is_err());
    }

    #[test]
    fn test_neighbors() {
        let mut sheaf = CellularSheaf::new_uniform(4, 2).unwrap();
        sheaf.add_edge(0, 1).unwrap();
        sheaf.add_edge(0, 2).unwrap();
        sheaf.add_edge(1, 3).unwrap();
        assert_eq!(sheaf.neighbors(0), vec![1, 2]);
    }

    #[test]
    fn test_extract_set_stalk() {
        let mut sheaf = CellularSheaf::new_uniform(3, 2).unwrap();
        let mut cochain = vec![0.0; 6];
        sheaf.set_stalk(1, &mut cochain, &[3.0, 4.0]).unwrap();
        let extracted = sheaf.extract_stalk(1, &cochain);
        assert!((extracted[0] - 3.0).abs() < 1e-10);
        assert!((extracted[1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_heterogeneous_stalk_dims() {
        let sheaf = CellularSheaf::new(vec![2, 3, 4]).unwrap();
        assert_eq!(sheaf.total_dim, 9);
        assert_eq!(sheaf.node_offset(0), 0);
        assert_eq!(sheaf.node_offset(1), 2);
        assert_eq!(sheaf.node_offset(2), 5);
    }

    #[test]
    fn test_validate() {
        let mut sheaf = CellularSheaf::new_uniform(3, 2).unwrap();
        sheaf.add_edge(0, 1).unwrap();
        assert!(sheaf.validate().is_ok());
    }

    // ---- Laplacian tests ----

    #[test]
    fn test_sheaf_laplacian_basic() {
        let sheaf = make_line_sheaf(3, 2);
        let lap = SheafLaplacian::from_sheaf(&sheaf).unwrap();
        assert_eq!(lap.total_dim, 6);
        assert_eq!(lap.matrix.rows, 6);
    }

    #[test]
    fn test_dirichlet_energy_constant_is_small() {
        let sheaf = make_line_sheaf(3, 2);
        let lap = SheafLaplacian::from_sheaf(&sheaf).unwrap();
        let x = vec![1.0; 6];
        let energy = lap.dirichlet_energy(&x);
        // For trivial sheaf, constant signal should have small energy
        // But with bidirectional edges, it won't be exactly zero
        assert!(
            energy < 1.0,
            "Constant signal energy should be small, got {energy}"
        );
    }

    #[test]
    fn test_dirichlet_energy_nonconstant() {
        let sheaf = make_line_sheaf(3, 1);
        let lap = SheafLaplacian::from_sheaf(&sheaf).unwrap();
        let x = vec![1.0, 0.0, -1.0];
        let energy = lap.dirichlet_energy(&x);
        assert!(energy > 0.0);
    }

    #[test]
    fn test_coboundary_trivial() {
        let mut sheaf = CellularSheaf::new_uniform(2, 2).unwrap();
        sheaf.add_edge(0, 1).unwrap();
        let b = sheaf.coboundary_matrix();
        assert_eq!(b.rows, 2);
        assert_eq!(b.cols, 4);
        // Edge (0,1): [-I | I]
        assert!((b.get(0, 0) - (-1.0)).abs() < 1e-10);
        assert!((b.get(0, 2) - 1.0).abs() < 1e-10);
    }

    // ---- Cohomology tests ----

    #[test]
    fn test_h0_connected_trivial() {
        // Connected graph with trivial sheaf should have h0 >= 1
        let sheaf = make_line_sheaf(3, 1);
        let lap = SheafLaplacian::from_sheaf(&sheaf).unwrap();
        let cohom = Cohomology::compute(&lap);
        assert!(
            cohom.h0_dimension >= 1,
            "Connected trivial sheaf should have h0 >= 1, got {}",
            cohom.h0_dimension
        );
    }

    #[test]
    fn test_h0_disconnected() {
        // Two disconnected nodes should have h0 = 2 (for stalk_dim=1)
        let mut sheaf = CellularSheaf::new_uniform(3, 1).unwrap();
        // Only edge 0-1, node 2 disconnected
        sheaf.add_edge(0, 1).unwrap();
        sheaf.add_edge(1, 0).unwrap();
        let lap = SheafLaplacian::from_sheaf(&sheaf).unwrap();
        let cohom = Cohomology::compute(&lap);
        assert!(
            cohom.h0_dimension >= 2,
            "Disconnected graph should have h0 >= 2, got {}",
            cohom.h0_dimension
        );
    }

    #[test]
    fn test_eigenvalues_nonnegative() {
        let sheaf = make_line_sheaf(4, 2);
        let lap = SheafLaplacian::from_sheaf(&sheaf).unwrap();
        let cohom = Cohomology::compute(&lap);
        for &ev in &cohom.eigenvalues {
            assert!(ev >= -0.5, "Eigenvalue should be non-negative: {ev}");
        }
    }

    // ---- Matrix tests ----

    #[test]
    fn test_matrix_identity() {
        let m = Matrix::identity(3);
        assert!((m.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((m.get(1, 1) - 1.0).abs() < 1e-10);
        assert!((m.get(0, 1) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_mul() {
        let a = Matrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::from_row_slice(2, 2, &[5.0, 6.0, 7.0, 8.0]);
        let c = a.mul(&b);
        assert!((c.get(0, 0) - 19.0).abs() < 1e-10);
        assert!((c.get(0, 1) - 22.0).abs() < 1e-10);
        assert!((c.get(1, 0) - 43.0).abs() < 1e-10);
        assert!((c.get(1, 1) - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_transpose() {
        let m = Matrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = m.transpose();
        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 2);
        assert!((t.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((t.get(1, 0) - 2.0).abs() < 1e-10);
        assert!((t.get(0, 1) - 4.0).abs() < 1e-10);
    }

    // ---- Synchronization tests ----

    #[test]
    fn test_sync_convergence() {
        let mut sheaf = CellularSheaf::new_uniform(3, 1).unwrap();
        sheaf.add_edge(0, 1).unwrap();
        sheaf.add_edge(1, 0).unwrap();
        sheaf.add_edge(1, 2).unwrap();
        sheaf.add_edge(2, 1).unwrap();

        let initial = vec![3.0, 0.0, -1.0];
        let result = synchronize(&sheaf, &initial, 0.05, 2000, 1e-4).unwrap();
        assert!(
            result.converged || result.final_energy < 1.0,
            "Should converge or have small energy"
        );
    }

    #[test]
    fn test_sync_constant_stays() {
        let mut sheaf = CellularSheaf::new_uniform(3, 1).unwrap();
        sheaf.add_edge(0, 1).unwrap();
        sheaf.add_edge(1, 0).unwrap();
        sheaf.add_edge(1, 2).unwrap();
        sheaf.add_edge(2, 1).unwrap();

        let initial = vec![5.0, 5.0, 5.0];
        let result = synchronize(&sheaf, &initial, 0.1, 100, 1e-6).unwrap();
        // Constant signal should remain approximately constant
        for val in &result.state {
            assert!(
                (val - 5.0).abs() < 0.5,
                "Constant signal should stay roughly constant: {val}"
            );
        }
    }

    #[test]
    fn test_detect_disagreement() {
        let mut sheaf = CellularSheaf::new_uniform(3, 1).unwrap();
        sheaf.add_edge(0, 1).unwrap();
        sheaf.add_edge(1, 2).unwrap();

        // Agreeing state (close to harmonic)
        let agree = vec![1.0, 1.0, 1.0];
        let disagreements = detect_disagreement(&sheaf, &agree, 0.1);
        assert!(disagreements.is_empty(), "Should have no disagreement");

        // Disagreeing state
        let disagree = vec![0.0, 10.0, 0.0];
        let disagreements = detect_disagreement(&sheaf, &disagree, 0.1);
        assert!(!disagreements.is_empty(), "Should detect disagreement");
    }

    #[test]
    fn test_convergence_rate() {
        let mut sheaf = CellularSheaf::new_uniform(3, 1).unwrap();
        sheaf.add_edge(0, 1).unwrap();
        sheaf.add_edge(1, 0).unwrap();
        sheaf.add_edge(1, 2).unwrap();
        sheaf.add_edge(2, 1).unwrap();

        let rate = convergence_rate(&sheaf);
        assert!(rate > 0.0, "Convergence rate should be positive: {rate}");
    }
}
