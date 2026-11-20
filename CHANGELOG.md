# Changelog

All notable changes to HAPI2LIBIS are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project (more or less) follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed
- NO2 table `schneider` (`crs_no2_012.dat`) gave a zero cross-section at
  every wavelength. The file ends with two padding rows of zeros, which broke
  the wavelength interpolation. Padding rows are now removed when a table is
  read, and a table whose wavelengths are not increasing stops the run with
  an error.

### Changed
- Table gases: all cross-section tables are described in one dictionary,
  `TABLE_GASES`, and are read and evaluated by two functions, `load_table()`
  and `table_cross_section()`. These replace `ozone_cross_section()`,
  `nitrogen_oxide_cross_section()`, `oxygen_dimer()` and the separate loading
  code for each gas. Results are unchanged (except `schneider`, see above).
- A gas in `crs_table_gases` without a table now stops the run at start-up
  with a message listing the available gases. Previously the run failed later
  with a `KeyError`, or, with a custom atmosphere containing that gas, the gas
  was left out with only a message.
- The warning that a table does not cover the requested wavelength range is
  printed once at start-up instead of once per layer.
  

## [1.1.0] - 2026-10-16

Bug-fix and robustness release. Configuration files and cross-section
databases (`xsec_db_folder`) from v1.0.0 work without changes.

With the default settings, results are identical to v1.0.0. The only
numerical change is in the bounding-box interpolation (see first item
under *Fixed*), and it only shows with tight `interpolation_error_tolerance`
values.

### Fixed
- Bounding-box interpolation: the box volume used to accept or reject an
  interpolation box that could be negative, so boxes larger than
  `interpolation_error_tolerance` were accepted. The volume now uses
  absolute side lengths, as in Eq. (6) of Kukkurainen et al. (2025).
  This only affects runs with `use_xsec_db: True` and
  `interpolation_method: 'bounding-box'`.
- Writing the output file crashed when only table gases (`crs_table_gases`)
  were used and `selected_gases` was empty.
- An unknown `interpolation_method` is now reported at start-up with a clear
  error. Before, the run failed later with an unrelated error.

### Changed
- HITRAN download failures are no longer silent:
  - a gas with no lines in the requested range is skipped with a warning;
  - on any other failure (e.g. no internet connection) the download is
    retried 3 times, 10 s apart, after which HAPI2LIBIS stops with an error.
    Previously the gas was silently left out and the output file was
    written without it.
- Missing input files now give clear error messages naming the file:
  atmosphere file, auxiliary US-standard profiles, cross-section table files,
  and `kurudz_full.dat` (warning only).
- An altitude mismatch between the atmosphere file and an auxiliary
  US-standard profile now raises a `ValueError` with the file name instead
  of a bare `assert`.
- Settings loading moved to a function, `load_settings()`. A missing or
  invalid configuration file gives a clear error message. The default
  configuration path can be changed with `DEFAULT_CONFIG_PATH`.
- All file paths use `pathlib` for compatibility between operating systems.
  Cross-section database file names are unchanged.
- Warnings about wavelengths outside a table's range now name the gas.
- HAPI updated from 1.2.2.2 to 1.3.0.0. HAPI now uses the TIPS-2025
  partition sums by default (previously TIPS-2021). In the test cases the
  optical depths change by less than 1e-5 (relative) and column
  transmittances by less than 4e-8.
- Code tidy-up: section headers, docstrings and consistent formatting.
  The unused `os` import was removed.


### Removed
- Table id `bogumil_exp` for O3 and NO2. Use `bogumil`.


## [1.0.0] - 2025-10-13

Initial release, described in Kukkurainen et al. (2025), *Geosci. Model Dev.*,
18, 7529–7544, https://doi.org/10.5194/gmd-18-7529-2025.
