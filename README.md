# ViT-C: Vision Transformer Code in C

This repository contains a C implementation of a compact Vision Transformer (ViT) model targeting ARM processors.

The project is organized to separate the model execution flow, computational kernels, model parameters, utility functions, headers, tests, and reference models used for validation.

## Build Requirements

The project uses a standard `Makefile` and requires:

- `gcc`
- `make`
- standard C math library (`libm`)

## Build

To build both the application and the test executable:

```bash
make
```

or equivalently:

```bash
make all
```

This generates the following binaries:

```text
build/vit_app
build/test_vit
```

## Run the Application

To build only the main ViT application:

```bash
make app
```

Then run:

```bash
./build/vit_app
```

## Run Tests

To build only the test executable:

```bash
make test
```

Then run:

```bash
./build/test_vit
```

## Clean Build Files

To remove generated build artifacts:

```bash
make clean
```

## Repository Structure

```text
vit-c/
├── README.md
├── Makefile
├── include/
│   ├── vit.h
│   ├── vit_types.h
│   ├── vit_kernels.h
│   ├── vit_params.h
│   └── utils.h
├── src/
│   ├── main.c
│   ├── vit.c
│   ├── vit_kernels.c
│   ├── vit_params.c
│   └── utils.c
├── test/
│   └── test_main.c
└── models/
    └── quantized_vit_integer_patched.ipynb
```

## Directory Overview

### `include/`

Contains public header files used by the ViT implementation.

- `vit.h`: top-level interface for the ViT model
- `vit_types.h`: common data types and model-related definitions
- `vit_kernels.h`: declarations for low-level computational kernels
- `vit_params.h`: model parameters and generated constants
- `utils.h`: utility function declarations

### `src/`

Contains the C source files implementing the model, kernels, parameters, and utilities.

- `main.c`: application entry point
- `vit.c`: ViT model execution flow
- `vit_kernels.c`: computational kernel implementations
- `vit_params.c`: model parameters and weights
- `utils.c`: helper and utility functions

### `test/`

Contains the test program for validating the ViT-C implementation.

- `test_main.c`: test entry point

The test infrastructure is currently under development 

### `models/`

Contains the reference model and related model-generation artifacts.

The notebook:

```text
quantized_vit_integer_patched.ipynb
```

defines the golden reference model using Brevitas. This directory is also intended to store datasets, model metadata, and intermediate outputs used to validate the C implementation against the reference model.

## Purpose

The goal of this project is to provide a lightweight C implementation of a quantized Vision Transformer model suitable for execution and evaluation on ARM-based platforms.

The Brevitas model in `models/` acts as the golden reference, while the C implementation in `src/` provides the target implementation for deployment and testing.