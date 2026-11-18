//! The 1:1 Earth (`--globe`, docs/EARTH_PLAN.md milestone 1): cube-sphere quadtree terrain from
//! NOAA ETOPO 2022, float64 camera, per-tile BLAS streaming and a ground-to-orbit atmosphere.

pub mod app;
pub mod dem;
pub mod etopo;
pub mod forest;
pub mod landcover;
pub mod render;
pub mod tiles;
