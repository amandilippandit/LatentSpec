import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        accent: {
          50: "#f5f3ff",
          500: "#7c3aed",
          600: "#6d28d9",
          700: "#5b21b6",
        },
        ok: "#16a34a",
        warn: "#ca8a04",
        crit: "#dc2626",
      },
      fontFamily: {
        sans: ['"Inter"', "system-ui", "sans-serif"],
        mono: ['"JetBrains Mono"', "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
};

export default config;
