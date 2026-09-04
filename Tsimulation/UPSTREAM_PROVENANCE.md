# Tsimulation import provenance

This tree was imported into `GaTech-RL2/EgoVerse-graph` from
`GaTech-RL2/EgoVerse`, branch `bf/2-sim`, at commit
`a50459a5bff817e80fbca0cd749b772ebe1e3686`.

The integration adds the camera metadata required by the current graph
repository's Zarr schema, preserves the simulator embodiment IDs 15–17, and
uses the graph repository's `get_planar_keymap` API in the end-to-end smoke
test. The physics/environment implementation remains the imported version.
