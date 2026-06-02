# sheaf-agents

Cellular sheaf framework for multi-agent coordination via sheaf cohomology in Rust.

## Features

- **Cellular sheaf**: Assign stalks (vector spaces) to nodes and restriction maps to edges
- **Sheaf Laplacian**: Generalized Laplacian via the coboundary operator
- **Cohomology**: H⁰ (global sections) and H¹ (obstructions) computation
- **Agent synchronization**: Diffusion-based consensus with sheaf structure
- **Disagreement detection**: Identify agents whose states violate sheaf constraints

## Usage

```rust
use sheaf_agents::*;

// Create a sheaf on 4 agents, each with 2D state
let mut sheaf = CellularSheaf::new_uniform(4, 2).unwrap();
sheaf.add_edge(0, 1).unwrap();
sheaf.add_edge(1, 2).unwrap();
sheaf.add_edge(2, 3).unwrap();

// Compute sheaf Laplacian
let lap = SheafLaplacian::from_sheaf(&sheaf).unwrap();

// Compute cohomology
let cohom = Cohomology::compute(&lap);
println!("H⁰ dimension: {}", cohom.h0_dimension);

// Synchronize agents
let initial = vec![1.0, 0.0, -1.0, 0.0, 0.5, 0.0, -0.5, 0.0];
let result = synchronize(&sheaf, &initial, 0.1, 1000, 1e-4).unwrap();
```

## Test Count

25 tests.

## License

MIT
