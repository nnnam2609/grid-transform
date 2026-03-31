# ArtSpeech Speaker And Session Mapping

Validated on `2026-03-28` against:

- raw segmentation root:
  `C:\Users\nhnguyen\PhD_A2A\iadi\ArtSpeech_Vocal_Tract_Segmentation\ArtSpeech_Vocal_Tract_Segmentation`
- encoded ArtSpeech root:
  `C:\Users\nhnguyen\PhD_A2A\Data\Artspeech_database`
- current VTLN bundle:
  `C:\Users\nhnguyen\PhD_A2A\grid transform\VTLN\iadi_replace_merge`

## Speaker Map

Confirmed speaker map:

| Raw subject | Encoded speaker |
|---|---|
| `1612` | `P1` |
| `1617` | `P2` |
| `1618` | `P3` |
| `1628` | `P4` |
| `1635` | `P5` |
| `1638` | `P6` |
| `1640` | `P7` |
| `1653` | `P8` |
| `1659` | `P9` |
| `1662` | `P10` |

Working note:

- The encoded speaker order `P1..P10` follows the raw-subject order
  `1612, 1617, 1618, 1628, 1635, 1638, 1640, 1653, 1659, 1662`.
- `1775` should be treated as unresolved relative to `P1..P10`.

## Session Rule

Main result:

- The encoded `P*` sessions preserve the chronological order of the retained raw sessions.
- The most reliable key is DICOM `SeriesTime`.
- So the safe rule is:
  sort raw sessions by `SeriesTime`, sort encoded sessions by `SeriesTime`, then match equal times.
- In clean cases this becomes a simple renumbering offset.

Useful shortcuts:

- `1612 -> P1`: raw `S7..S22 -> S1..S16`
- `1618 -> P3`: raw `S7..S22 -> S1..S16`
- `1635 -> P5`: raw `S7..S22 -> S1..S16`
- `1638 -> P6`: raw `S7..S22 -> S1..S16`
- `1659 -> P9`: raw `S7..S22 -> S1..S16`
- `1640 -> P7`: raw `S9..S24 -> S1..S16`
  Example: raw `1640/S10 -> P7/S2`

Order is still preserved, but these speakers have dropped/extra/empty sessions:

- `1617 -> P2`:
  encoded `S1` is extra, raw `S8` is not present in encoded, then raw `S9..S23 -> encoded S2..S16`
- `1628 -> P4`:
  encoded `S1` is extra, raw `S18` and `S23` are missing in encoded, encoded `S16` is empty
- `1653 -> P8`:
  matched order is preserved, but raw session numbering is not a single constant offset
- `1662 -> P10`:
  raw `S20` is empty, encoded `S14` is extra

## VTLN Ids

Important interpretation:

- VTLN names keep the raw speaker id and raw session id.
- So `1640_s10_0654` should be read as raw subject `1640`, raw session `S10`, raw-like frame id `0654`.
- Since raw `1640/S10 -> P7/S2`, the VTLN reference `1640_s10_0654` corresponds to encoded session `P7/S2`.

Frame-id caveat:

- The VTLN session id is reliable for recovering the encoded session.
- The VTLN frame id is not always the exact raw contour frame used in the current merge bundle.
- Current checked shifts from `VTLN/iadi_replace_merge/iadi_merge_notes.txt`:
  - `1612_s10_0654 -> raw 1612/S10/0655`
  - `1617_s10_0822 -> raw 1617/S10/0823`
  - `1618_s10_0757 -> raw 1618/S10/0756`
  - `1628_s10_0943 -> raw 1628/S10/0944`
  - `1635_s10_0898 -> raw 1635/S10/0899`
  - `1638_s10_1167 -> raw 1638/S10/1166`
  - `1659_s10_0654 -> raw 1659/S10/1337`

Practical takeaway:

- Use the speaker map above to recover `P*`.
- Use the raw session number in the VTLN id to recover the encoded session.
- Do not expect the encoded DICOM filename to preserve the raw frame number.

## VTLN Frame To Session Check

I checked each available VTLN frame against the sessions of its mapped encoded
speaker.

Interpretation:

- `expected session` means the encoded session recovered from raw-session lineage
  using the DICOM `SeriesTime` match.
- For the initially ambiguous cases (`1618`, `1635`, `1659`), a more detailed review
  against top candidate sessions brought them back to the lineage session.
- So the current working rule is:
  if the speaker mapping is correct, the VTLN frame should be attached to the
  lineage session below.

| VTLN frame | Speaker | Confirmed encoded session |
|---|---|---|
| `1612_s10_0654` | `P1` | `S4` |
| `1617_s10_0822` | `P2` | `S3` |
| `1618_s10_0757` | `P3` | `S4` |
| `1628_s10_0943` | `P4` | `S4` |
| `1635_s10_0898` | `P5` | `S4` |
| `1638_s10_1167` | `P6` | `S4` |
| `1640_s10_0654` | `P7` | `S2` |
| `1640_s10_0829` | `P7` | `S2` |
| `1653_s10_1729` | `P8` | `S2` |
| `1659_s10_0654` | `P9` | `S4` |

Practical rule:

- If you want provenance, use the lineage session in the table above.
- For the current resolved VTLN frames, that lineage session is also the session to use.

Current status:

- Confirmed lineage-session agreement from current VTLN frames:
  `P1`, `P2`, `P3`, `P4`, `P5`, `P6`, `P7`, `P8`, `P9`
- No checked VTLN frame from the current bundle for `P10` yet.
- `1775` remains outside the confirmed `P1..P10` mapping.
