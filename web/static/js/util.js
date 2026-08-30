// Shared helpers. Loaded first; every later panel script depends on these two.
// Plain globals rather than ES modules: no bundler, no import map, no build
// step (AGENTS.md §7), and load order in index.html is the whole dependency graph.

const $ = id => document.getElementById(id);

// Escapes text interpolated into innerHTML. Audit payloads, router reasons and
// model output all pass through here — none of them are trusted markup.
const esc = s => String(s).replace(/[<&]/g, c => c === '<' ? '&lt;' : '&amp;');
