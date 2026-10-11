/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        bg: "#070B12",
        card: "#0E1420",
        "inner-tile": "#060A12",
        border: "#1B2536",
        text: "#E6EBF5",
        muted: "#7C8AA3",
        brandGreen: "#34D399",
        brandSky: "#38BDF8",
        brandViolet: "#C084FC",
        brandAmber: "#FBBF24",
        alertRed: "#F43F5E",
      },
      fontFamily: {
        mono: ['"JetBrains Mono"', "ui-monospace", "SFMono-Regular", "monospace"],
        sans: ['"Plus Jakarta Sans"', "ui-sans-serif", "system-ui", "sans-serif"],
      },
      borderRadius: {
        card: "18px",
      },
      boxShadow: {
        glowGreen: "0 0 25px -4px rgba(52, 211, 153, 0.28)",
        glowRed: "0 0 25px -4px rgba(244, 63, 94, 0.32)",
        glowViolet: "0 0 25px -4px rgba(192, 132, 252, 0.28)",
      },
    },
  },
  plugins: [],
}
