# HAPI2LIBIS
Authors: Antti Kukkurainen<sup>1</sup> & Antti Mikkonen<sup>1</sup>\
1: Finnish Meteorological Institute\
Correspondence: antti.kukkurainen@fmi.fi

HAPI2LIBIS enables easy creation of spectral molecular absorption files for the radiative transfer simulator software libRadtran. The up-to-date molecular data comes from the HITRAN (High-Resolution Transmission Molecular Absorption) database.

## Installation
Clone the repo\
```git clone https://github.com/amikko/hapi2libis.git```

Required Python packages can be installed, for example, using `pip install numpy scipy pyyaml netcdf4` or with conda/mamba.

## Usage
HAPI2LIBIS is ran with the following command
```python hapi2libis.py [configfile]```
where [configfile] is optional. If it is not provided, `config.yaml` is used as the configuration file.

### Configuration file
The HAPI2LIBIS configuration file is a YAML-formatted text file containing all the input necessary for the molecular absorption computation.

#### Basic settings

The basic settings contain options that should be changed based on local configuration and simulation needs.

These are currently:

- `libradtranpath`: Location of the libRadtran folder.
- `wl_range_nm`: Desired wavelength range in nanometers.
- `nu_resolution`: HAPI resolution in wavenumbers.
- `atmosphere_id`: Number ID for atmosphere file. Numbers 1-6 refer to atmosphere files in libRadtran. For custom atmosphere, check advanced options.
- `name_suffix`: Name suffix for the created molecular absorption file.
- `selected_gases`: List of gases to use.
- `function_choice`: Broadening function.
- `use_xsec_db`: Use or turn off irregular interpolation.

#### Advanced settings

The advanced settings contain options that normally should not be changed from their default values. It also contains the custom atmosphere loading for advanced users.

These are currently:

- `xsec_db_folder`: Folder to save irregular interpolation files.
- `cl_interp_p`: Controls the number of points used for the interpolation.
- `interpolation_method`: Chosen interpolation method.
- `interpolation_error_tolerance`: The maximum volume of the box allowed for interpolation, or the maximum distance from particular point.
- `hapi_dl_dir`: Folder to save HITRAN data.
- `load_aux_gases_from_us_standard`: Load profiles for CH4, N2O, CO and N2 from auxiliary US-standard files.
- `create_mol_abs_netcdf_file`: Write molecular absorption netcdf file for libradtran.
- `interpolate_to_layer_midpoint`: Interpolate atmospheric gases to layer middle point.
- `boundary_flag`: Force computation edges to selected range.
- `check_wavelength_limits`: Returns maximum range for non-reflectivity calculations if full kurucs solar irradiance file is present.
- `crs_table_gases`: Use gas cross-section data provided by libRadtran tables.
- `custom_atm_dict`: Define custom atmospheric profiles.
