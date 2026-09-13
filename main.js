// Page behaviour: navigation, scroll reveals, the suggestion box, and the
// three 3D scenes, which are created when they come near the viewport and
// only render while they are on screen.
import { createHero } from './hero.js';
import { createEditor, createRedesign } from './editor.js';

// Where suggestions go. With no endpoint, the form opens the visitor's mail
// app addressed to CONTACT with the suggestion filled in. To collect them
// without a mail app, point SUGGEST_ENDPOINT at a form service that forwards
// to CONTACT (a Web3Forms or Formspree endpoint, or Netlify Forms) and the
// form posts JSON {suggestion, email} there instead.
const CONTACT = 'contact@oasis-spaces.info';
const SUGGEST_ENDPOINT = '';

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;

// ------------------------------------------------------------------ nav
const nav = $('#nav');
const onScroll = () => nav.classList.toggle('is-solid', scrollY > 24);
addEventListener('scroll', onScroll, { passive: true });
onScroll();
const toggle = $('#nav-toggle');
toggle.addEventListener('click', () => {
  const open = nav.classList.toggle('is-open');
  toggle.setAttribute('aria-expanded', String(open));
  toggle.setAttribute('aria-label', open ? 'Close menu' : 'Open menu');
});
$$('#nav-links a').forEach((a) => a.addEventListener('click', () => {
  nav.classList.remove('is-open');
  toggle.setAttribute('aria-expanded', 'false');
}));

// -------------------------------------------------------------- reveals
const pending = new Set($$('.reveal, .reveal-lines'));
function reveal(el) { el.classList.add('is-in'); pending.delete(el); io.unobserve(el); }
const io = new IntersectionObserver((entries) => {
  for (const e of entries) if (e.isIntersecting) reveal(e.target);
}, { rootMargin: '0px 0px -10% 0px', threshold: 0.05 });
pending.forEach((el) => io.observe(el));
// Belt and braces: some browsers only report intersections after the first
// scroll, so also check positions ourselves on load and while scrolling.
function revealVisible() {
  const limit = innerHeight * 0.9;
  for (const el of [...pending]) {
    const r = el.getBoundingClientRect();
    if (r.top < limit && r.bottom > 0) reveal(el);
  }
}
if (reduced) pending.forEach(reveal);
else {
  revealVisible();
  addEventListener('load', revealVisible);
  addEventListener('scroll', revealVisible, { passive: true });
  addEventListener('resize', revealVisible);
}

// ---------------------------------------------------------------- form
const form = $('#suggest-form');
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  if (!form.reportValidity()) return;
  const data = Object.fromEntries(new FormData(form));
  const button = $('button[type=submit]', form);
  button.disabled = true;
  try {
    if (SUGGEST_ENDPOINT) {
      const res = await fetch(SUGGEST_ENDPOINT, {
        method: 'POST', headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify({ ...data, page: location.href }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      $('#form-done p:last-child').textContent = 'Your suggestion is on its way to us.';
    } else {
      const body = data.suggestion + (data.email ? `\n\nReply to: ${data.email}` : '');
      location.href = `mailto:${CONTACT}?subject=${encodeURIComponent('Suggestion for Oasis-spaces')}`
        + `&body=${encodeURIComponent(body)}`;
    }
    $$('.field, .form__actions', form).forEach((el) => { el.hidden = true; });
    $('#form-done').hidden = false;
  } catch (err) {
    button.disabled = false;
    $('#form-note').textContent = `That did not go through. Please write to ${CONTACT} instead.`;
  }
});
$('#copy-address')?.addEventListener('click', async (e) => {
  try {
    await navigator.clipboard.writeText(CONTACT);
    e.target.textContent = 'Copied';
    setTimeout(() => { e.target.textContent = 'Copy address'; }, 1800);
  } catch { e.target.textContent = CONTACT; }
});

// --------------------------------------------------------------- scenes
// One animation loop; each scene updates only while its stage is visible.
const scenes = [];
let last = performance.now();
function loop(now) {
  const dt = Math.min(0.05, (now - last) / 1000);
  last = now;
  // (a background tab already pauses requestAnimationFrame)
  for (const s of scenes) if (s.visible) s.scene.update(dt, now);
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);

function mount(stageId, create) {
  const stage = document.getElementById(stageId);
  if (!stage) return;
  const canvas = $('canvas', stage);
  const entry = { visible: false, scene: null };
  let failed = false;
  function ensure() {
    if (entry.scene || failed) return;
    try {
      entry.scene = create(canvas, stage);
      scenes.push(entry);
    } catch (err) {
      failed = true;
      console.error(`${stageId}: 3D preview unavailable`, err);
      stage.classList.add('stage--fallback');
      const note = document.createElement('p');
      note.className = 'stage__fallback';
      note.textContent = 'The 3D preview needs WebGL, which this browser has turned off.';
      stage.append(note);
      seen.disconnect();
    }
  }
  const seen = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting) ensure();
      entry.visible = e.isIntersecting && !failed;
    }
  }, { rootMargin: '200px 0px' });
  seen.observe(stage);
  // the same fallback as the reveals, for browsers that stay quiet until a scroll
  const check = () => {
    const r = stage.getBoundingClientRect();
    const near = r.bottom > -200 && r.top < innerHeight + 200;
    if (near) ensure();
    entry.visible = near && !failed;
  };
  check();
  addEventListener('load', check);
  addEventListener('scroll', check, { passive: true });
}

mount('hero-stage', (canvas) => createHero(canvas, {
  status: $('#hero-status'), replay: $('#hero-replay'),
}));

mount('editor-stage', (canvas, stage) => {
  const tb = $('#editor-toolbar');
  return createEditor(canvas, {
    hint: $('#editor-hint'), label: $('#editor-label'),
    add: $('[data-act=add]', tb), rotate: $('[data-act=rotate]', tb), smaller: $('[data-act=smaller]', tb),
    larger: $('[data-act=larger]', tb), remove: $('[data-act=remove]', tb), undo: $('[data-act=undo]', tb),
    addMenu: $('#editor-add-menu'), needSelection: $$('[data-needs-selection]', tb),
  });
});

mount('redesign-stage', (canvas, stage) => createRedesign(canvas, {
  compare: $$('.compare button', stage), chips: $$('#redesign-styles .chip'),
  title: $('#suggest-title'), text: $('#suggest-text'), list: $('#suggest-list'),
}));
