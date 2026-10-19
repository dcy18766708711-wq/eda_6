# IC pin and ordinary-symbol pin stage

This stage runs after component assembly and before wire/net construction.

## Protocol rules implemented here

- Pin coordinates are measured from the image. They are not produced by a
  library offset, a default distance, or Qwen.
- The local detector follows the Appendix 3 rule: the point is the actual
  terminal/wire interface in the image coordinate system, then converted to
  the contest bottom-left coordinate system.
- For an IC, the library supplies semantic information only. A formal
  `pin_<number>` is emitted only when local OCR visibly supports that number,
  or when a visible OCR pin name maps to exactly one library number.
- Ambiguous names such as `NC` are not mapped to an arbitrary library pin.
  Geometry-only IC candidates remain diagnostics and are never completed in
  library order. `library_order_fallback_used` must remain `false`.
- A physical IC identity is emitted at most once. If OCR sees the same number
  more than once, the strongest local assignment is kept and the other
  candidate remains in diagnostics.
- For ordinary symbols, the PDF automatic-number rule is used only after a
  real local terminal path has been measured: `pin_1`, `pin_2`, ... in
  perimeter order. Empty `pinname` is intentional when no semantic pin name
  is visibly available. A missing geometric terminal is not fabricated.
- Qwen may resolve OCR/model-name ambiguity, but it cannot modify a measured
  pin point or create an invisible pin.

## One-step execution

```powershell
python PCB_analysis/deployment/topology/run_ic_pin_stage.py `
  --image <image.png> `
  --intermediate <yolo_ocr.json> `
  --component-json <components.json> `
  --index PCB_analysis/deployment/ic_library/BoxICLibrary_200cases.index.json `
  --resolver-output <ic_resolution.json> `
  --output <ic_pins.json>
```

The output `diagnostics` records local candidates, OCR assignments, dropped
duplicate identities, search-boundary expansion, and any missing geometry.
