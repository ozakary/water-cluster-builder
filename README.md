# xyz-cluster-builder

---

📄 Author: **Ouail Zakary**  
- 📧 Email: [Ouail.Zakary@oulu.fi](mailto:Ouail.Zakary@oulu.fi)  
- 🔗 ORCID: [0000-0002-7793-3306](https://orcid.org/0000-0002-7793-3306)  
- 🌐 Website: [Personal Webpage](https://cc.oulu.fi/~nmrwww/members/Ouail_Zakary.html)  
- 📁 Portfolio: [GitHub Portfolio](https://ozakary.github.io/)

---

Create spherical molecular clusters from periodic MD trajectories by removing the furthest water molecules around a central atom (e.g., Xenon).

## Features

- Unwraps periodic boundary conditions (PBC) with molecular integrity
- Centers clusters around Xenon (or other central atom)
- Identifies water molecules via O-H bonds and H-O-H angles
- Removes furthest waters based on maximum atomic distance
- Maintains charge neutrality (only removes complete water molecules)
- Parallel processing for trajectories
- Reports cluster size statistics

## Installation

```bash
pip install numpy ase scipy joblib tqdm
```

## Usage

```bash
python remove_waters.py input.xyz output.xyz --n-remove 100 --n-jobs 4  --hoh-angle-min 88 --hoh-angle-max 125
```

### Arguments

- `input.xyz` - Input XYZ trajectory with PBC
- `output.xyz` - Output XYZ trajectory without PBC
- `--n-remove` - Number of water molecules to remove (required)
- `--n-jobs` - Number of parallel CPU cores (default: 1)
- `--oh-bond-max` - Maximum O-H bond length in Å (default: 1.2)
- `--hoh-angle-min` - Minimum H-O-H angle in degrees (default: 95)
- `--hoh-angle-max` - Maximum H-O-H angle in degrees (default: 115)

## Example

```bash
# Remove 500 water molecules using 8 cores
python remove_waters.py trajectory.xyz cluster.xyz --n-remove 500 --n-jobs 8
```

## Output

- Unwrapped coordinates centered on Xenon
- Cluster diameter statistics (average, min, max, std)
- Atom count summary

## Use Case

Ideal for preparing molecular clusters for:
- Quantum mechanical calculations (QM/MM)
- NMR parameter calculations
- Guest-host system analysis in porous materials
