#![feature(test)]
extern crate test;

pub mod ops;
#[cfg(test)] mod ops_tests;
pub mod img;
#[cfg(test)] mod img_tests;
pub mod utils;
#[cfg(test)] mod utils_tests;

pub mod solvers;
#[cfg(test)] mod solvers_tests;
