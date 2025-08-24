# Warp Fitter

This program fits **warped structures in protoplanetary discs** to model velocity perturbations retrieved from molecular line observations (see [Winter et al. 2025](https://ui.adsabs.harvard.edu/abs/2025arXiv250711669W/abstract)). It was originally designed for the **exoALMA** sample of discs, but it also supports input from any disc analysed with [Discminer](https://github.com/andizq/discminer).

---

## Installation

First, clone this repository and move into the project folder:

```bash
git clone https://github.com/andizq/warpfitter.git
cd warpfitter
```

---

## Typical Usage

### 1. ExoALMA discs with existing models

For an exoALMA target (e.g. MWC 758), you can run directly:

```bash
python warp_fitter.py --warp --grid --plot --target mwc758
```

---

### 2. Other discs (or exoALMA discs with new models)

1. Place the following files inside the **downloaded folder**:
   - A reduced (prototype) **datacube** of the disc.
   - A **`parfile.json`** file with disc parameters.

These are typically generated using:  
   ```bash
   python prepare_data.py
   discminer parfile
   ```
   (see [Discminer example](https://github.com/andizq/discminer/tree/main/example)).

2. Run the warp fitter:
   ```bash
   python warp_fitter.py --warp --grid --plot --parfile --initdiscminer
   ```

- `--parfile` : read parameters from the existing `parfile.json`.  
- `--initdiscminer` : compute moment maps and residuals (only required once per target).  

Subsequent runs for the same disc do not require `--initdiscminer`.  
