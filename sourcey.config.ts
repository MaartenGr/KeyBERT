import { defineConfig, mkdocs } from "sourcey";

export default defineConfig({
  name: "KeyBERT",
  siteUrl: "https://maartengr.github.io",
  baseUrl: "/KeyBERT",
  prettyUrls: false,
  repo: "https://github.com/MaartenGr/KeyBERT",
  editBranch: "master",
  editBasePath: "docs",
  theme: {
    preset: "default",
    colors: {
      primary: "#111827",
      light: "#374151",
      dark: "#030712",
    },
  },
  navigation: {
    tabs: [
      {
        tab: "Documentation",
        slug: "",
        source: mkdocs("./mkdocs.yml"),
      },
    ],
  },
});
