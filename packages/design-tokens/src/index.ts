/**
 * Same values as tokens.css, as TS constants — for canvas/SVG use
 * where CSS custom properties are awkward (React Flow edges, D3 scales).
 */
export const palette = {
  paper: "#F2F1EC",
  ink: "#161616",
  muted: "#747474",
  line: "#C7C6C0",
  panel: "#EBEAE5",
  signal: "#E85A3F",
} as const;

export type PaletteColor = keyof typeof palette;

export const fonts = {
  editorial: "var(--font-editorial)",
  interface: "var(--font-interface)",
  machine: "var(--font-machine)",
} as const;
