# Room simulation and RIR generation (MATLAB)

Scripts used to simulate the rooms and the 19-channel room impulse responses (RIRs) of the training and validation sets.

They run on top of the **SMIR Simulator** by **Julio Carabias** (MIT License), which wraps the [SMIR-Generator](https://github.com/ehabets/SMIR-Generator) and provides helpers such as `readyaml.m`. That toolkit is not included here; copy these files into its folder to use them.

| File | Description |
|---|---|
| `Simulated_rir_generator_v2.m` | RIR generator adapted to the Zylia ZM-1: random rooms, a T60 sweep, 2–7 sources per room at a fixed height and at least 50° apart in azimuth. It writes the RIRs (`rir/roomX_Y.mat`) and the room/RIR info CSVs. |
| `config2.yaml` | Simulation parameters, including the 19 Zylia capsule positions and the 4.9 cm radius. |
| `Pasar_cartesianas_zylia.m` | Converts the Zylia capsule coordinates from the datasheet (Cartesian) to azimuth/inclination for the configuration. |
| `Prueba_generacion.m` | Quick test that generates a few rooms without the parallel pool. |

## Room presets

`config2.yaml` holds one preset at a time; the comments list the values used for each set:

| Parameter | General (train / standard validation) | Small, low reverberation | Large, reverberant |
|---|---|---|---|
| Width × length (m) | 5.5–20 × 5.5–20 | 3–5 × 3–5 | 21–26 × 21–26 |
| Height (m) | 3.5–6 | 1.5–3 | 7–9 |
| Source distance (m) | 1.3–2.5 | 0.8–1.2 | 3–5 |
| T60 (s) | 0.2–1.0 | 0.06–0.15 | 1.1–1.4 |
| Reflection order | 10 | 3 | 15 |

## Usage

```matlab
% from the SMIR Simulator folder, with these files copied into it
Simulated_rir_generator_v2(1, 320, 12)   % first room id, number of rooms, parallel workers
```

Inside the file the function is still declared as `simulated_rir_generator`, so call it by its file name as above. `Prueba_generacion.m` calls `simulated_rir_generator`, which runs the original generator of the toolkit unless this file replaces it.

The mixture CSVs in [`../data/info/`](../data/info) combine these RIRs with excerpts of the audio datasets.
