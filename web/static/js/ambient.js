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

  // ---- Trailing cursor light ------------------------------------------------
  // Three points chase the pointer at different rates, so the light reads as a
  // trail rather than a dot welded to the cursor. This is the CSS-native stand-in
  // for the threejs-components "tubes" cursor: that one needs three.js from a
  // CDN (§2.1 forbids it, and test_no_remote_es_module_import fails the build)
  // and runs WebGL continuously on the same 6 GB card that holds an 8B model
  // (§17: GPU pressure here does not raise, it runs 20-25x slower).
  //
  // Cost is bounded two ways: the rAF loop STOPS once the points have settled,
  // so a still pointer costs nothing, and the paint is three radial gradients on
  // one already-composited layer.
  var PT = [
    { x: 0, y: 0, ease: 0.22 },
    { x: 0, y: 0, ease: 0.12 },
    { x: 0, y: 0, ease: 0.07 }
  ];
  var tx = window.innerWidth / 2, ty = window.innerHeight * 0.4, running = false;

  PT.forEach(function (p) { p.x = tx; p.y = ty; });

  function frame() {
    var moved = false;
    for (var i = 0; i < PT.length; i++) {
      var p = PT[i];
      p.x += (tx - p.x) * p.ease;
      p.y += (ty - p.y) * p.ease;
      if (Math.abs(tx - p.x) > 0.5 || Math.abs(ty - p.y) > 0.5) moved = true;
      root.style.setProperty('--x' + i, p.x.toFixed(1) + 'px');
      root.style.setProperty('--y' + i, p.y.toFixed(1) + 'px');
    }
    // Settled: stop the loop entirely rather than idling a rAF forever.
    if (moved) requestAnimationFrame(frame); else running = false;
  }

  function kick() { if (!running) { running = true; requestAnimationFrame(frame); } }

  window.addEventListener('pointermove', function (e) {
    tx = e.clientX; ty = e.clientY;
    root.style.setProperty('--mx', tx + 'px');
    root.style.setProperty('--my', ty + 'px');
    kick();
  }, { passive: true });

  // Pointer leaving the window parks the light instead of freezing it mid-edge.
  window.addEventListener('pointerleave', function () {
    tx = window.innerWidth / 2; ty = window.innerHeight * 0.4;
    root.style.setProperty('--mx', '50vw');
    root.style.setProperty('--my', '40vh');
    kick();
  }, { passive: true });

  // Click cycles the palette, as in the reference. The palettes are this
  // console's own semantic hues rather than random colours: a security panel
  // that flashes arbitrary pink on click stops looking like an instrument, and
  // random hex would collide with the green/amber/red the UI uses for state.
  var PALETTES = [
    ['rgba(121,192,255,.13)', 'rgba(63,185,80,.10)',  'rgba(88,140,255,.09)'],
    ['rgba(63,185,80,.13)',   'rgba(121,192,255,.09)','rgba(140,200,255,.08)'],
    ['rgba(210,153,34,.11)',  'rgba(121,192,255,.10)','rgba(63,185,80,.08)'],
    ['rgba(160,120,255,.12)', 'rgba(121,192,255,.10)','rgba(63,185,80,.08)']
  ];
  var pal = 0;
  function applyPalette(n) {
    PALETTES[n].forEach(function (c, i) { root.style.setProperty('--c' + i, c); });
  }
  applyPalette(0);

  document.addEventListener('click', function (e) {
    // Never steal a click that was meant for the app.
    if (e.target.closest('button, a, input, textarea, select, label')) return;
    pal = (pal + 1) % PALETTES.length;
    applyPalette(pal);
  }, { passive: true });
})();
