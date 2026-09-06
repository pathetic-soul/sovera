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
