# Changelog

## [v0.2.0](https://github.com/lily-mosquitoes/image-recovery/compare/v0.1.0...v0.2.0) (2023-08-07)

### ⚠ BREAKING CHANGE

* new api for library use


### Features

* implemented new imagearray struct with associated traits as main library
object, changes denoise solver to use the new struct and operations
([4c11776](https://github.com/lily-mosquitoes/image-recovery/commit/4c117762cdd187f581f1f9da2d0100cefb083327))
* **image_array:** implement ImageArray as concrete type over trait
DifferentiableArray
([db8462e](https://github.com/lily-mosquitoes/image-recovery/commit/db8462e3b25a0ee30be152658fd84b714ae29caa))
* **differentiable_array:** create new trait DifferentiableArray and implement
for Array generically
([46775c8](https://github.com/lily-mosquitoes/image-recovery/commit/46775c8a836273110e0807826b7a2f78744dfe7f))
* add log crate; make print statement a debug log
([2071911](https://github.com/lily-mosquitoes/image-recovery/commit/2071911328cfc10c01b88cab7330a5935cefcb4c))

## v0.1.0 (2022-04-18)
