#[cfg(feature = "python-bridge")]
use pyo3::prelude::*;
use std::collections::{HashSet, HashMap};

#[cfg(feature = "python-bridge")]
#[pyfunction]
fn expand_entities_by_hops(
    adj_list: HashMap<String, Vec<String>>,
    initial_entities: Vec<String>,
    hops: usize
) -> PyResult<HashSet<String>> {
    let mut expanded = HashSet::new();
    let mut current_layer = HashSet::new();

    for entity in initial_entities {
        current_layer.insert(entity.clone());
        expanded.insert(entity);
    }

    for _ in 0..hops {
        let mut next_layer = HashSet::new();
        for node in &current_layer {
            if let Some(neighbors) = adj_list.get(node) {
                for neighbor in neighbors {
                    if !expanded.contains(neighbor) {
                        next_layer.insert(neighbor.clone());
                        expanded.insert(neighbor.clone());
                    }
                }
            }
        }
        if next_layer.is_empty() {
            break;
        }
        current_layer = next_layer;
    }

    Ok(expanded)
}

#[cfg(feature = "python-bridge")]
#[pymodule]
fn kryonix_brain_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(expand_entities_by_hops, m)?)?;
    Ok(())
}
