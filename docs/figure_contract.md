# Figure and dashboard contract

The figures describe logged behavior; they make no advance claim that one controller wins.
The evidence is the terminal training episode CSV, unchanged validation CSV, saved evaluation summaries, scenario hashes, and recorded physical trajectories.
The backend is Python/matplotlib. Figures use a quantitative grid, white background, a restrained palette, editable SVG text, embedded TrueType PDF text, and PNG previews. Microsoft YaHei is used when installed, with Arial/DejaVu Sans fallback and English metric labels.

## Training and validation

- `plots/training_curves.{png,svg,pdf}` shows episode return, normalized weighted delay J, total episode energy in joules, deadline completion fraction, total episode penalty, and full feasibility. Each raw observation is one **completed** episode. In vector training, the horizontal axis is completion order rather than scenario episode ID.
- The solid line is the trailing mean of up to 25 completed episodes. Its first point is the first actual value; no leading zeroes or future observations are inserted. Raw values remain faintly visible. The fraction of fully feasible episodes in the trailing window is descriptive.
- `plots/training_energy.*` separates cumulative energy per UAV and shows the per-UAV energy budget when it exists in the saved configuration. It never compares a sum across UAVs against a per-UAV budget.
- `plots/validation_curves.*` uses training environment steps on the x-axis, and uses only logged validation records. It does not interpolate nonexistent evaluations.
- Episode penalty is read from `penalty_sum` when present; otherwise it follows the unchanged reward identity `-episode_return - frames * weighted_delay_mean`. It is the aggregate weighted penalty, not a newly defined optimization objective.
- A final CSV append without a newline is ignored until complete. Missing metrics remain unavailable. Baselines have no learning history and are marked “not applicable.” Existing CSV logs are never rewritten by reporting.

## Best, last and all-group evaluations

`evaluation_best` and `evaluation_last` stay separate; requesting one never substitutes the other. Baselines use `evaluation`. Before writing a batch comparison, every group declared in `experiment_manifest.json` must have a completed evaluation and identical environment configuration plus recorded evaluation episode IDs, scenario hashes and frame hashes. Missing results and mismatched protocols reject the comparison instead of silently dropping methods.

`comparison/best` and `comparison/last` contain an all-group metric figure, a full trajectory grid when trajectories are recorded, a machine-readable `summary.csv`, and provenance metadata. All scalar metrics that exist in the source are preserved in the table, including per-UAV energies. Missing metrics are blank/N/A. Numbers are means across paired evaluation episodes. The current protocol records one training seed per group, so the plots contain no error bars, confidence intervals, significance tests or generalization claims across training seeds.

`comparison/{best,last}/training_comparison.*` independently overlays the eight actual learning histories in six panels: episode return, J, episode energy, completion, aggregate penalty and full feasibility. Faint raw values and trailing 25-episode means remain visible. The horizontal axis is completed training episodes in completion order for every method; recorded episode IDs, start/end steps, frame counts and scenario hashes must agree over shared history prefixes. Saved training environment configurations must also match. Live histories can differ in length, without padded or fabricated tails. The separate legend strip wraps long names; hybrid curves additionally use dashes. Baselines are listed as excluded rather than receiving invented learning curves. `render_training_comparison` can refresh this figure before end-of-training evaluations exist.

## Physical paths

`plots/trajectory_best.*` and `plots/trajectory_last.*` show the first logged evaluation episode. The path contains exactly the recorded pre-step `uav_position` at q0 followed by each frame's recorded `uav_xy_m` or `next_uav_position`, giving H+1 points for H frames. The same rule applies to users. Missing q0 is an error; no guessed initial placement is inserted. Every UAV has a distinct color, start circle, end cross and direction arrows where it moves. All users are shown in gray. Plot limits come from the evaluation's `environment_config.area_m`, in metres, with equal x/y scale. A stationary UAV can have overlapping start/end markers; both recorded endpoints remain in the metadata.

Endpoint markers remain visible when a recorded position lies exactly on a map boundary. Batch trajectory titles wrap within each panel and put the evaluation episode on a separate line.

## Live view and review

The dashboard binds only `127.0.0.1`, reads existing files, and polls every five seconds. It does not train, evaluate, plot, execute source files, list directories, or mutate data. HTTP responses disable caching. Paths containing parent traversal, Windows separators/drives, NULs, and links resolving outside the experiment root are rejected. Only explicit data/image/document suffixes are served.

For each smoke run, inspect the PNG previews for legible axes, unclipped labels and paths, correct start/end locations, and baseline applicability. Check that SVG contains editable text and that every exported figure links to its actual source data through report metadata. The tests additionally reject unpaired groups, partial CSV writes and dashboard path traversal. Full scientific claims require independently repeated training runs beyond this single-seed contract.
