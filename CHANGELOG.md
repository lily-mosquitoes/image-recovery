# Changelog

### [v0.3.1](///compare/v0.3.0...v0.3.1) (2023-08-07)

#### Features

* make ImageArray clone (d3ae037)

## [v0.3.0](///compare/v0.2.0...v0.3.0) (2023-08-07)

## [v0.2.0](///compare/v0.1.0...v0.2.0) (2023-08-07)

### ⚠ BREAKING CHANGE

* new api for library use


### Features

* implemented new imagearray struct with associated traits as main library
object, changes denoise solver to use the new struct and operations (4c11776)
* **image_array:** implement ImageArray as concrete type over trait
DifferentiableArray (db8462e)
* **differentiable_array:** create new trait DifferentiableArray and implement
for Array generically (46775c8)
* add log crate; make print statement a debug log (2071911)

## v0.1.0 (2022-04-18)
