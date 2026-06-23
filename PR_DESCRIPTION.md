# Stop button fix, recording job names, and episode browser improvements

## Summary

1. **Stop teleop reliably when an arm is faulted.** The global `Stop()` raises if any robot fails to stop, which previously aborted before the operational pairs were halted — leaving them teleoperating. `TeleopService.stop()` now stops each pair individually via `StopWithIdx(idx)` so a faulted pair can't block the rest, and always marks the loop stopped (per the TDK contract, recovery requires `Init()` + `Start()` regardless).

2. **Job name for episode recordings.** Added a "Job name" text box (default `job_0`) above the Start button in the Episode Recording panel. Episodes sharing a job name are saved together under `episodes/<job_name>/`. The name is sanitized into a safe single path segment server-side, and the box is locked while an episode is actively recording or awaiting save/discard.

3. **Job-grouped episode listing.** `list_episode_datasets` and the path browser recurse one level into job folders, tagging each episode with its job. The episode picker shows episodes grouped under job headings with job badges; episodes left in the old flat layout still appear (untagged) for backward compatibility.

4. **Scroll for long episode lists.** The browser modal is now capped to the viewport height and its item list scrolls, so a long episode list no longer pushes the footer (and Load button) off-screen.

5. **Sort episodes by time created.** Added a toolbar toggle in the episode browser to order episodes ascending/descending by creation time (default newest-first). Sorting preserves job grouping and re-renders without a server round-trip; the backend now returns a `created` timestamp per entry.

## Testing

- `pytest` — 114 passed.
- Added coverage for fault-resilient stop, job-name sanitization/save-grouping, job-grouped listing, and `created`-time annotation.
