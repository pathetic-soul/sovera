// Ambient cursor light + load choreography (AGENTS.md §2.1: vendored, no CDN).
//
// Cost control matters here: §17 records that GPU pressure on this machine does
// not raise, it silently runs 20-25x slower, and an 8B model is resident during
// the demo. So this writes two CSS custom properties at most once per frame and
// nothing else — the paint is one composited radial-gradient layer, and there is
// no timer, no loop and no work at all while the pointer is still.

(function () {
  var root = document.documentElement;
  var reduced = window.matchMedia('(prefers-reduced-motion: reduce)');

  // Entrance choreography, one authored moment on first paint.
  requestAnimationFrame(function () { document.body.classList.add('lit'); });

  // The resting heartbeat layer. Injected rather than authored into index.html
  // so the markup stays about the product and this stays purely presentational.
  if (!reduced.matches) {
    var drift = document.createElement('div');
    drift.className = 'ambient-drift';
    drift.setAttribute('aria-hidden', 'true');
    document.body.appendChild(drift);
  }

  // Faceted background field. Two CSS-border triangles per cell, alternate rows
  // offset by half a cell so the facets interlock.
  //
  // CAP is the one thing added to the reference construction: an uncapped grid
  // on a 4K display builds a couple of thousand elements, and this sits behind
  // a page that also runs an agent loop. 900 cells covers 2560x1440 at this
  // size and costs one static paint.
  var TRI = 52, CAP = 900, facets;

  function buildFacets() {
    if (reduced.matches) return;
    if (!facets) {
      facets = document.createElement('div');
      facets.className = 'facets';
      facets.setAttribute('aria-hidden', 'true');
      document.body.appendChild(facets);
    }
    var cell = TRI * 2 + 2;
    var cols = Math.ceil(window.innerWidth / cell) + 1;
    var rows = Math.ceil(window.innerHeight / (1.733 * TRI)) + 1;
    if (cols * rows > CAP) rows = Math.max(1, Math.floor(CAP / cols));

    facets.style.setProperty('--tri', TRI + 'px');
    facets.style.setProperty('--columns', cols);
    var frag = document.createDocumentFragment();       // one reflow, not cols*rows
    for (var y = 0; y < rows; y++) {
      for (var x = 0; x < cols; x++) {
        var i = document.createElement('i');
        if (y % 2 === 0) i.className = 'off';
        frag.appendChild(i);
      }
    }
    facets.replaceChildren(frag);
  }

  buildFacets();

  var resizeTimer;
  window.addEventListener('resize', function () {
    clearTimeout(resizeTimer);                          // rebuilding per resize event is a jank factory
    resizeTimer = setTimeout(buildFacets, 180);
  });

  if (reduced.matches) return;

  var x = 0, y = 0, queued = false;

  function paint() {
    queued = false;
    root.style.setProperty('--mx', x + 'px');
    root.style.setProperty('--my', y + 'px');
  }

  window.addEventListener('pointermove', function (e) {
    x = e.clientX; y = e.clientY;
    if (!queued) { queued = true; requestAnimationFrame(paint); }
  }, { passive: true });

  // Pointer leaving the window parks the light instead of freezing it mid-edge.
  window.addEventListener('pointerleave', function () {
    root.style.setProperty('--mx', '50vw');
    root.style.setProperty('--my', '40vh');
  }, { passive: true });
})();
