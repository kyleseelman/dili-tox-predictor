(() => {
  "use strict";

  const BASE = "learning_websites-main/";

  const CATEGORIES = {
    ai:        { label: "AI & ML",   color: "#8b5cf6" },
    science:   { label: "Science",   color: "#22d3ee" },
    history:   { label: "History",   color: "#f59e0b" },
    languages: { label: "Languages", color: "#f472b6" },
  };

  const PAGES = [
    {
      id: "latent-space",
      path: "latent-space.html",
      cat: "ai",
      title: "Latent Space",
      en: "An interactive guide to modern AI",
      kicker: "AI · Visual explainer",
      glyph: "∇",
      c1: "#8b5cf6", c2: "#22d3ee", bg: "linear-gradient(135deg,#1a0b3d,#07070d 60%,#0b2a3a)",
      desc: "How models actually work: tokens, embeddings, attention and the forward pass, then scaling, diffusion, assistants and retrieval. Lots of visuals, very little hand-waving.",
      tags: ["Tokens", "Embeddings", "Attention", "Diffusion", "RAG"],
      inside: ["Models don't read words. They read tokens.", "Meaning is a direction", "Every token looks at every other token", "It's all just guessing the next token", "Inside a forward pass", "Huge model. Small bill.", "Sculpting images out of noise", "From autocomplete to assistant", "Bigger is predictably better", "Give the model an open book"],
    },
    {
      id: "post-training",
      path: "radical-numerics-prep/post-training.html",
      cat: "ai",
      title: "Post-Training",
      en: "From basics to advanced",
      kicker: "AI · Deep dive",
      glyph: "RL",
      c1: "#3b82f6", c2: "#10b981", bg: "linear-gradient(135deg,#0c2a55,#0b1220 55%,#0b3b2e)",
      desc: "The full post-training stack, from SFT and reward models to RLHF/PPO, DPO, rejection sampling, RLAIF and RLVR/GRPO.",
      tags: ["SFT", "RLHF", "DPO", "GRPO", "Reward models"],
      inside: ["The pipeline and where the stages sit", "SFT: supervised fine-tuning", "Reward modeling", "RLHF / PPO", "DPO and preference optimization", "Rejection sampling / Best-of-N", "RLAIF & synthetic feedback", "RLVR / GRPO: verifiable rewards"],
    },
    {
      id: "radical-numerics",
      path: "radical-numerics-prep/index.html",
      cat: "ai",
      title: "Radical Numerics",
      en: "Technical brief: genomic foundation models",
      kicker: "AI × Bio · Research brief",
      glyph: "RN",
      c1: "#1f5fa8", c2: "#1a7355", bg: "linear-gradient(135deg,#123a6b,#0a1424 55%,#0f3d2f)",
      desc: "A research brief on the team behind Hyena, Evo and StripedHyena: Omnii-1, genome benchmarks, applications, and how post-training and evaluation fit in.",
      tags: ["Hyena / Evo", "Genomics", "Benchmarks", "Evaluation"],
      inside: ["Research lineage: the architectures this team invented", "Omnii-1: what it actually is", "Genome benchmarks: know these numbers", "The eight applications, in depth", "Post-training: the role, in depth", "Evaluation", "Competitive landscape"],
    },
    {
      id: "bio-ai",
      path: "bio-ai.html",
      cat: "science",
      title: "The Living Code",
      en: "AI in biology & medicine",
      kicker: "Science · AI × Biology",
      glyph: "ATCG",
      c1: "#14b8a6", c2: "#6366f1", bg: "linear-gradient(135deg,#0b3a3a,#0d1530 55%,#2a1f5c)",
      desc: "From four letters of DNA to designed proteins and simulated cells: where machine learning is changing biology and medicine.",
      tags: ["DNA", "Protein folding", "Protein design", "Virtual cells", "Drug discovery"],
      inside: ["Four letters, twenty amino acids", "From a string to a shape", "Learning biology by filling in blanks", "Sculpting proteins out of noise", "Reading the 98% that isn't genes", "Simulating a cell before touching a pipette", "Where AI already ships", "Drug discovery: nine failures in ten", "The model zoo", "What the models can't do (yet)"],
    },
    {
      id: "physics",
      path: "physics-pages/physics.html",
      cat: "science",
      title: "How the Physical World Works",
      en: "From foundations to the frontier",
      kicker: "Science · Full course",
      glyph: "ψ",
      c1: "#60a5fa", c2: "#a78bfa", bg: "radial-gradient(circle at 30% 30%,#1b2a6b,#050816 70%)",
      desc: "A self-contained interactive physics course: units and Newton, least action, thermodynamics, electromagnetism, relativity, quantum mechanics, QFT and the open problems.",
      tags: ["Mechanics", "Relativity", "Quantum", "QFT", "Simulations"],
      inside: ["Prologue: what physics is", "The math toolkit", "Classical & analytical mechanics", "Waves", "Thermodynamics & statistical mechanics", "Electromagnetism & optics", "Special & general relativity", "Quantum mechanics", "Nuclear & particle physics", "Quantum field theory: an intuition", "Frontiers & open problems", "Practice & mastery"],
    },
    {
      id: "rome",
      path: "rome.html",
      cat: "history",
      title: "SPQR",
      en: "The Roman Empire: rise, conquest, reign & fall",
      kicker: "History · 753 BC – 1453 AD",
      glyph: "SPQR",
      c1: "#b91c1c", c2: "#d4a24c", bg: "linear-gradient(135deg,#5a1010,#22180e 60%,#6b4a14)",
      desc: "Two thousand years of Rome, from the wolf and the twins through the Republic and Augustus to the empire's peak and its long fall.",
      tags: ["Republic", "Caesar", "Augustus", "Legions", "The Fall"],
      inside: ["Two thousand years in one map", "A wolf, two twins, and seven hills", "Res publica: the public thing", "From Latium to Mare Nostrum", "The Republic devours itself", "Augustus and the Caesars", "The legion: a machine of 5,000 men", "Roads, water and concrete", "Four rulers, one new faith", "The fall of the West", "The Rome that didn't fall"],
    },
    {
      id: "ancient-greece",
      path: "greece-pages/ancient-greece.html",
      cat: "history",
      title: "Hellas",
      en: "A museum-quality history of Ancient Greece",
      kicker: "History · Bronze Age → today",
      glyph: "Ω",
      c1: "#c2410c", c2: "#1f5b8c", bg: "linear-gradient(135deg,#7a2e12,#2b2119 55%,#123b5c)",
      desc: "Geography, Minoans and Mycenaeans, the poleis, the Persian and Peloponnesian Wars, Alexander, the Hellenistic world, Rome, and an epilogue to modern Greece. Includes maps and timelines.",
      tags: ["Minoans", "Athens & Sparta", "Alexander", "Maps", "Timeline"],
      inside: ["A civilization of islands and arguments", "The wine-dark sea: geography", "Nine thousand years at a glance", "Palaces of bulls and lords of gold", "The Dark Age and the return of the Muses", "The Age of Experiment", "The Persian Wars", "Alexander and the Hellenistic world"],
    },
    {
      id: "japan-history",
      path: "japan-pages/japan-history.html",
      cat: "history",
      title: "日本",
      en: "Japan: a history from myth to Reiwa",
      kicker: "History · Jōmon → 2026",
      glyph: "日本",
      c1: "#dc2626", c2: "#1f3a5f", bg: "linear-gradient(135deg,#7a1a0c,#1f1b17 55%,#1f3a5f)",
      desc: "From the Paleolithic and the founding myths to the Reiwa era, covering culture, religion, the arts, language and society.",
      tags: ["Heian", "Shoguns", "Tokugawa", "Meiji", "Reiwa"],
      inside: ["An archipelago of reinvention", "Prehistory & the Age of the Gods", "Kofun, Asuka, Nara & Heian", "Kamakura Shogunate & Kenmu Restoration", "Ashikaga Shogunate & the Warring States", "Azuchi–Momoyama: the three unifiers", "Tokugawa: 265 years of peace", "Bakumatsu & the Meiji Restoration", "Occupation & economic miracle", "Reiwa: Japan today", "Shinto, Buddhism & the sacred"],
    },
    {
      id: "learn-japanese",
      path: "japan-pages/learn-japanese.html",
      cat: "languages",
      title: "はじめての日本語",
      en: "Learn Japanese from zero",
      kicker: "Languages · Beginner",
      glyph: "あ",
      c1: "#f472b6", c2: "#8fb573", bg: "linear-gradient(135deg,#d9577a,#2f3f73 70%)",
      desc: "Hiragana, katakana, a mnemonic for every sound, stroke order, quizzes with spaced repetition, plus grammar, kanji and phrases.",
      tags: ["Hiragana", "Katakana", "Mnemonics", "Quizzes", "Kanji"],
      inside: ["Three scripts, one sentence", "The gojūon chart, with a story for every sound", "Write it the way your hand will remember", "Every kana was once a kanji", "Look-alikes, untangled", "Quiz & flashcards with a memory", "Five vowels, steady rhythm", "Verb last. Particles do the heavy lifting.", "58 starter kanji, built from pictures", "Keigo: the ladder of respect", "From zero to JLPT N1"],
    },
    {
      id: "learn-greek",
      path: "greece-pages/learn-greek.html",
      cat: "languages",
      title: "Μάθε Ελληνικά",
      en: "Learn Greek from zero",
      kicker: "Languages · Beginner",
      glyph: "Αα",
      c1: "#2a7ab8", c2: "#c9a227", bg: "linear-gradient(135deg,#0d3b66,#1f5fa8 50%,#6b7a3a)",
      desc: "The alphabet, sounds, grammar, phrases, quizzes and numbers in Modern Greek, plus a look at Ancient and Koine Greek.",
      tags: ["Alphabet", "Pronunciation", "Phrases", "Numbers", "Koine"],
      inside: ["3,400 years of writing Greek", "24 letters, 24 pictures", "Letters that will try to trick you", "From Phoenician to Latin & Cyrillic", "Letter pairs, stress & pronunciation", "Quiz & flashcards", "The essentials, gently", "Counting in Greek"],
    },
    {
      id: "language-transfer-greek",
      path: "greece-pages/language-transfer-greek-1-15.html",
      cat: "languages",
      title: "Complete Greek, Tracks 1–15",
      en: "A Language Transfer study companion",
      kicker: "Languages · Study guide",
      glyph: "1–15",
      c1: "#6b7a3a", c2: "#2a7ab8", bg: "linear-gradient(135deg,#3c4a1c,#13283d 60%,#1f5fa8)",
      desc: "A study companion for Language Transfer's free Complete Greek course, with vocabulary, pronunciation, pattern notes, sentence-building drills and flashcards.",
      tags: ["Vocabulary", "Verb forms", "Flashcards", "Drills"],
      inside: ["How to use this guide", "The course's sound guide", "Master vocabulary: every word so far", "Verb forms seen so far", "Flashcards", "Sentence-builder drill"],
    },
    {
      id: "advanced-spanish",
      path: "spanish-pages/advanced-spanish.html",
      cat: "languages",
      title: "El español con matices",
      en: "Advanced Spanish, B2 → C2",
      kicker: "Languages · Advanced",
      glyph: "ñ",
      c1: "#e8a317", c2: "#c2562f", bg: "linear-gradient(135deg,#9c3f1f,#2a1f1a 55%,#9e1b32)",
      desc: "The hard parts of Spanish: subjunctive, sequence of tenses, conditionals, periphrases, every use of «se», relative clauses, connectors and precise vocabulary, with spaced-repetition practice.",
      tags: ["Subjuntivo", "Condicionales", "«se»", "Connectors", "SRS"],
      inside: ["Diagnóstico y hoja de ruta", "Dominar el subjuntivo", "Condicionales sin miedo", "El sistema verbal completo", "Perífrasis verbales", "Pronombres y el universo de «se»", "Oraciones de relativo", "Construir oraciones complejas"],
    },
  ];

  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const href = (p) => BASE + p.path;

  const ICON_EYE = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12Z"/><circle cx="12" cy="12" r="3"/></svg>';
  const ICON_ARROW = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 6l6 6-6 6"/></svg>';

  /* ---------------- Tabs ---------------- */
  const tabs = $$(".tab");
  const indicator = $(".tab-indicator");
  const views = $$(".view");
  let currentTab = null;

  function moveIndicator() {
    const active = tabs.find((t) => t.dataset.tab === currentTab);
    if (!active) return;
    indicator.style.width = active.offsetWidth + "px";
    indicator.style.transform = `translateX(${active.offsetLeft}px)`;
  }

  function showTab(name, push = true) {
    if (!views.some((v) => v.dataset.view === name)) name = "home";
    if (name === currentTab) return;
    currentTab = name;
    tabs.forEach((t) => t.setAttribute("aria-selected", String(t.dataset.tab === name)));
    views.forEach((v) => v.classList.toggle("active", v.dataset.view === name));
    moveIndicator();
    if (push) history.pushState(null, "", "#" + name);
    window.scrollTo({ top: 0, behavior: "instant" in window ? "instant" : "auto" });
    observeReveals();
    if (name === "learning") onLibraryShown();
  }

  tabs.forEach((t) => t.addEventListener("click", () => showTab(t.dataset.tab)));
  document.addEventListener("click", (e) => {
    const link = e.target.closest("[data-tab-link]");
    if (!link) return;
    e.preventDefault();
    showTab(link.dataset.tabLink);
  });
  $$(".tile-lib").forEach((t) => t.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); showTab("learning"); }
  }));
  window.addEventListener("popstate", () => showTab(location.hash.slice(1) || "home", false));
  window.addEventListener("resize", moveIndicator);

  /* ---------------- Reveal on scroll ---------------- */
  const revealIO = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting) { e.target.classList.add("in"); revealIO.unobserve(e.target); }
    });
  }, { threshold: 0.12 });
  function observeReveals() {
    $$(".view.active .reveal:not(.in)").forEach((el, i) => {
      el.style.transitionDelay = Math.min(i * 50, 300) + "ms";
      revealIO.observe(el);
    });
  }

  /* ---------------- Spotlight hover on tiles ---------------- */
  document.addEventListener("pointermove", (e) => {
    const tile = e.target.closest(".tile");
    if (!tile) return;
    const r = tile.getBoundingClientRect();
    tile.style.setProperty("--mx", e.clientX - r.left + "px");
    tile.style.setProperty("--my", e.clientY - r.top + "px");
  });

  /* ---------------- Home decorations ---------------- */
  const rotWords = ["3D & 4D generation", "content understanding", "agentic systems", "human-centered NLP", "people, not benchmarks"];
  (function rotator() {
    const el = $("#rotator-text");
    if (!el || reduceMotion) return;
    let w = 0, i = rotWords[0].length, deleting = true;
    const tick = () => {
      const word = rotWords[w];
      if (deleting) {
        i--;
        if (i <= 0) { deleting = false; w = (w + 1) % rotWords.length; }
      } else {
        i++;
        if (i >= rotWords[w].length) { deleting = true; el.textContent = rotWords[w]; return setTimeout(tick, 2200); }
      }
      el.textContent = (deleting ? word : rotWords[w]).slice(0, Math.max(i, 0));
      setTimeout(tick, deleting ? 35 : 65);
    };
    setTimeout(tick, 2400);
  })();

  const gradOf = (p) => `linear-gradient(120deg, ${p.c1}, ${p.c2})`;

  $("#marquee").innerHTML = [...PAGES, ...PAGES].map((p) =>
    `<a class="marquee-item" href="#learning" data-tab-link="learning" style="--g:${gradOf(p)}"><b>${esc(p.glyph)}</b>${esc(p.title)}</a>`
  ).join("");

  const orbitIcons = [
    // Bohr atom
    `<svg viewBox="0 0 32 32" aria-hidden="true"><ellipse cx="16" cy="16" rx="13" ry="5.2" fill="none" stroke="#22d3ee" stroke-width="1.8" transform="rotate(60 16 16)"/><ellipse cx="16" cy="16" rx="13" ry="5.2" fill="none" stroke="#a78bfa" stroke-width="1.8" transform="rotate(-60 16 16)"/><ellipse cx="16" cy="16" rx="13" ry="5.2" fill="none" stroke="#f472b6" stroke-width="1.8"/><circle cx="16" cy="16" r="2.4" fill="#fbbf24"/></svg>`,
    // DNA helix
    `<svg viewBox="0 0 32 32" aria-hidden="true"><path d="M11 3c0 5 10 5 10 10S11 18 11 23s10 5 10 6" fill="none" stroke="#22d3ee" stroke-width="1.8" stroke-linecap="round"/><path d="M21 3c0 5-10 5-10 10s10 5 10 10-10 5-10 6" fill="none" stroke="#f472b6" stroke-width="1.8" stroke-linecap="round"/><path d="M12.2 8.2h7.6M11 16h10M12.2 23.8h7.6" stroke="#c4b5fd" stroke-width="1.6" stroke-linecap="round"/></svg>`,
    // Benzene ring
    `<svg viewBox="0 0 32 32" aria-hidden="true"><polygon points="16,4 26.4,10 26.4,22 16,28 5.6,22 5.6,10" fill="none" stroke="#5eead4" stroke-width="1.8" stroke-linejoin="round"/><polygon points="16,9 22,12.5 22,19.5 16,23 10,19.5 10,12.5" fill="none" stroke="#5eead4" stroke-width="1.5" stroke-linejoin="round"/></svg>`,
    // Sine wave
    `<svg viewBox="0 0 32 32" aria-hidden="true"><path d="M2 16c3-7 5-7 7 0s4 7 7 0 4-7 7 0 4 7 7 0" fill="none" stroke="#60a5fa" stroke-width="2" stroke-linecap="round"/></svg>`,
    // Cell
    `<svg viewBox="0 0 32 32" aria-hidden="true"><circle cx="16" cy="16" r="12" fill="none" stroke="#86efac" stroke-width="1.8"/><circle cx="16" cy="15" r="4.2" fill="none" stroke="#f472b6" stroke-width="1.7"/><circle cx="16" cy="15" r="1.6" fill="#f472b6"/><ellipse cx="24" cy="22" rx="3.2" ry="1.6" fill="none" stroke="#fbbf24" stroke-width="1.5" transform="rotate(-30 24 22)"/></svg>`,
  ];
  $("#orbit").innerHTML = orbitIcons.map((svg, i) => {
    const a = (i / orbitIcons.length) * Math.PI * 2 - Math.PI / 2;
    const x = Math.cos(a) * 50, y = Math.sin(a) * 50;
    return `<span style="left:${50 + x}%;top:${50 + y}%">${svg}</span>`;
  }).join("");

  $("#lib-stack").innerHTML = ["latent-space", "learn-japanese", "rome", "physics"].map((id) => {
    const p = PAGES.find((x) => x.id === id);
    return `<span style="--g:${p.bg}">${esc(p.glyph)}</span>`;
  }).join("");

  $("#stat-pages").textContent = PAGES.length;
  $("#tab-count").textContent = PAGES.length;
  $("#year").textContent = new Date().getFullYear();

  /* ---------------- Library ---------------- */
  const lib = $("#lib");
  const state = { cat: "all", q: "", layout: "gallery", center: 0 };

  const counts = PAGES.reduce((m, p) => ((m[p.cat] = (m[p.cat] || 0) + 1), m), {});
  $("#filters").innerHTML =
    `<button class="filter on" data-cat="all">All <span class="n">${PAGES.length}</span></button>` +
    Object.entries(CATEGORIES).map(([k, c]) =>
      `<button class="filter" data-cat="${k}" style="--c:${c.color}"><span class="dot"></span>${c.label} <span class="n">${counts[k] || 0}</span></button>`
    ).join("");

  lib.innerHTML = PAGES.map((p, i) => `
    <article class="card" data-id="${p.id}" style="--i:${i};--c1:${p.c1};--c2:${p.c2};--poster-bg:${p.bg}" tabindex="0" aria-label="${esc(p.title)}: ${esc(p.en)}">
      <div class="preview">
        <div class="poster">${esc(p.glyph)}</div>
        <span class="preview-badge">${CATEGORIES[p.cat].label}</span>
        <span class="preview-hint">Click for full preview</span>
      </div>
      <div class="card-body">
        <div class="card-kicker">${esc(p.kicker)}</div>
        <h3>${esc(p.title)}</h3>
        <p class="card-en">${esc(p.en)}</p>
        <p class="card-desc">${esc(p.desc)}</p>
        <div class="tags">${p.tags.map((t) => `<span class="tag">${esc(t)}</span>`).join("")}</div>
        <div class="card-actions">
          <button class="a peek" type="button">${ICON_EYE} Preview</button>
          <a class="a go" href="${href(p)}">Open ${ICON_ARROW}</a>
        </div>
      </div>
    </article>`).join("");

  const cards = $$(".card", lib);
  const cardById = (id) => cards.find((c) => c.dataset.id === id);
  const pageOf = (card) => PAGES.find((p) => p.id === card.dataset.id);

  // Live thumbnails: a real 1280×800 iframe of each page, scaled to fit the card.
  const sizeObserver = new ResizeObserver((entries) => {
    entries.forEach((e) => e.target.style.setProperty("--s", e.contentRect.width / 1280));
  });
  let loadQueue = Promise.resolve();
  function loadPreview(card) {
    if (card.dataset.loaded) return;
    card.dataset.loaded = "1";
    const p = pageOf(card);
    const preview = $(".preview", card);
    loadQueue = loadQueue.then(() => new Promise((resolve) => {
      const f = document.createElement("iframe");
      f.title = `${p.title} preview`;
      f.tabIndex = -1;
      f.setAttribute("aria-hidden", "true");
      f.setAttribute("scrolling", "no");
      const done = () => { preview.classList.add("loaded"); resolve(); };
      f.addEventListener("load", done, { once: true });
      setTimeout(resolve, 2500);
      f.src = href(p);
      preview.appendChild(f);
    }));
  }
  cards.forEach((c) => sizeObserver.observe($(".preview", c)));

  const previewIO = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (!e.isIntersecting) return;
      const card = e.target;
      if (state.layout === "carousel" && !card.classList.contains("near")) return;
      loadPreview(card);
    });
  }, { rootMargin: "200px" });

  let libraryShownOnce = false;
  function onLibraryShown() {
    if (libraryShownOnce) { layout(); return; }
    libraryShownOnce = true;
    cards.forEach((c) => previewIO.observe(c));
    layout();
  }

  // Hovering a card auto-scrolls the live page inside it so you can "flip through" it.
  function startFlip(card) {
    const f = $(".preview iframe", card);
    if (!f || reduceMotion) return;
    let win;
    try { win = f.contentWindow; void win.scrollY; } catch { return; }
    cancelAnimationFrame(card._raf);
    const step = () => {
      try {
        const max = win.document.documentElement.scrollHeight - win.innerHeight;
        if (win.scrollY < max - 2) win.scrollTo(0, win.scrollY + 9);
      } catch { return; }
      card._raf = requestAnimationFrame(step);
    };
    card._flipDelay = setTimeout(() => (card._raf = requestAnimationFrame(step)), 350);
  }
  function stopFlip(card) {
    clearTimeout(card._flipDelay);
    cancelAnimationFrame(card._raf);
    const f = $(".preview iframe", card);
    try { f && f.contentWindow.scrollTo({ top: 0, behavior: "smooth" }); } catch {}
  }

  cards.forEach((card) => {
    card.addEventListener("mouseenter", () => {
      if (state.layout === "carousel" && !card.classList.contains("center")) return;
      startFlip(card);
    });
    card.addEventListener("mouseleave", () => {
      stopFlip(card);
      card.style.removeProperty("--rx");
      card.style.removeProperty("--ry");
    });
    card.addEventListener("pointermove", (e) => {
      if (state.layout !== "gallery") return;
      const r = card.getBoundingClientRect();
      const x = (e.clientX - r.left) / r.width - 0.5;
      const y = (e.clientY - r.top) / r.height - 0.5;
      card.style.setProperty("--ry", x * 6 + "deg");
      card.style.setProperty("--rx", -y * 6 + "deg");
    });
    card.addEventListener("click", (e) => {
      if (e.target.closest("a.go")) return;
      if (state.layout === "carousel" && !card.classList.contains("center")) {
        state.center = visibleCards().indexOf(card);
        layout();
        return;
      }
      openQuickLook(card.dataset.id);
    });
    card.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && e.target === card) openQuickLook(card.dataset.id);
    });
  });

  function matches(p) {
    if (state.cat !== "all" && p.cat !== state.cat) return false;
    if (!state.q) return true;
    const hay = [p.title, p.en, p.kicker, p.desc, CATEGORIES[p.cat].label, ...p.tags, ...p.inside].join(" ").toLowerCase();
    return state.q.split(/\s+/).every((w) => hay.includes(w));
  }
  const visibleCards = () => cards.filter((c) => !c.hidden);

  function applyFilter() {
    let n = 0;
    cards.forEach((c) => {
      const show = matches(pageOf(c));
      c.hidden = !show;
      if (show) c.style.setProperty("--i", n++);
    });
    $("#empty").classList.toggle("show", n === 0);
    state.center = Math.min(state.center, Math.max(n - 1, 0));
    layout();
  }

  function layout() {
    const car = state.layout === "carousel";
    lib.classList.toggle("carousel", car);
    lib.classList.toggle("gallery", !car);
    const vis = visibleCards();
    $("#carousel-nav").classList.toggle("show", car && vis.length > 0);

    if (!car) {
      cards.forEach((c) => {
        c.style.transform = "";
        c.style.zIndex = "";
        c.classList.remove("center", "near", "far");
      });
      return;
    }

    const step = Math.min(window.innerWidth * 0.22, 300);
    vis.forEach((c, i) => {
      const o = i - state.center;
      const a = Math.abs(o);
      c.classList.toggle("center", o === 0);
      c.classList.toggle("near", a <= 2);
      c.classList.toggle("far", a > 3);
      c.style.zIndex = String(100 - a);
      c.style.transform =
        `translateX(${o * step}px) translateZ(${-a * 160}px) rotateY(${Math.max(-1, Math.min(1, o)) * -32}deg) scale(${o === 0 ? 1 : 0.9})`;
      if (a <= 1) loadPreview(c);
    });

    const dots = $("#car-dots");
    dots.innerHTML = vis.map((c, i) =>
      `<i class="${i === state.center ? "on" : ""}" data-i="${i}" style="--g:${gradOf(pageOf(c))}" title="${esc(pageOf(c).title)}"></i>`
    ).join("");
  }

  function moveCarousel(d) {
    const n = visibleCards().length;
    if (!n) return;
    stopFlip(visibleCards()[state.center]);
    state.center = (state.center + d + n) % n;
    layout();
  }

  $("#car-prev").addEventListener("click", () => moveCarousel(-1));
  $("#car-next").addEventListener("click", () => moveCarousel(1));
  $("#car-dots").addEventListener("click", (e) => {
    const d = e.target.closest("i");
    if (!d) return;
    state.center = +d.dataset.i;
    layout();
  });

  $("#filters").addEventListener("click", (e) => {
    const b = e.target.closest(".filter");
    if (!b) return;
    $$(".filter").forEach((f) => f.classList.toggle("on", f === b));
    state.cat = b.dataset.cat;
    state.center = 0;
    applyFilter();
  });

  const search = $("#search");
  search.addEventListener("input", () => {
    state.q = search.value.trim().toLowerCase();
    state.center = 0;
    applyFilter();
  });

  $$(".seg button").forEach((b) => b.addEventListener("click", () => {
    $$(".seg button").forEach((x) => {
      x.classList.toggle("on", x === b);
      x.setAttribute("aria-pressed", String(x === b));
    });
    state.layout = b.dataset.layout;
    layout();
  }));

  $("#surprise").addEventListener("click", () => {
    const vis = visibleCards().length ? visibleCards() : cards;
    const pick = vis[Math.floor(Math.random() * vis.length)];
    if (state.layout === "carousel") { state.center = vis.indexOf(pick); layout(); }
    else pick.scrollIntoView({ behavior: "smooth", block: "center" });
    setTimeout(() => openQuickLook(pick.dataset.id), state.layout === "carousel" ? 650 : 450);
  });

  window.addEventListener("resize", () => state.layout === "carousel" && layout());

  /* ---------------- Quick look ---------------- */
  const ql = $("#ql");
  const qlFrame = $("#ql-frame");
  const qlIframe = $("#ql-iframe");
  let qlId = null;

  function openQuickLook(id) {
    const p = PAGES.find((x) => x.id === id);
    if (!p) return;
    qlId = id;
    const list = visibleCards().length ? visibleCards() : cards;
    const idx = list.indexOf(cardById(id));

    ql.style.setProperty("--c1", p.c1);
    ql.style.setProperty("--c2", p.c2);
    ql.style.setProperty("--poster-bg", p.bg);
    $("#ql-counter").textContent = `${String(idx + 1).padStart(2, "0")} / ${String(list.length).padStart(2, "0")}`;
    $("#ql-kicker").textContent = p.kicker;
    $("#ql-title").textContent = p.title;
    $("#ql-en").textContent = p.en;
    $("#ql-desc").textContent = p.desc;
    $("#ql-tags").innerHTML = p.tags.map((t) => `<span class="tag">${esc(t)}</span>`).join("");
    $("#ql-inside").innerHTML = p.inside.map((s) => `<li>${esc(s)}</li>`).join("");
    $("#ql-open").href = href(p);
    $("#ql-newtab").href = href(p);
    $("#ql-url").textContent = location.host ? `${location.host}/${href(p)}` : href(p);
    $("#ql-loading").textContent = p.glyph;

    qlFrame.classList.remove("loaded");
    qlIframe.onload = () => qlFrame.classList.add("loaded");
    qlIframe.src = href(p);

    if (!ql.open) ql.showModal();
  }

  function stepQuickLook(d) {
    const list = visibleCards().length ? visibleCards() : cards;
    const i = list.indexOf(cardById(qlId));
    const next = list[(i + d + list.length) % list.length];
    if (state.layout === "carousel") { state.center = list.indexOf(next); layout(); }
    openQuickLook(next.dataset.id);
  }

  $("#ql-close").addEventListener("click", () => ql.close());
  $("#ql-prev").addEventListener("click", () => stepQuickLook(-1));
  $("#ql-next").addEventListener("click", () => stepQuickLook(1));
  ql.addEventListener("click", (e) => { if (e.target === ql) ql.close(); });
  ql.addEventListener("close", () => { qlIframe.src = "about:blank"; qlId = null; });

  /* ---------------- Keyboard ---------------- */
  document.addEventListener("keydown", (e) => {
    const typing = /input|textarea/i.test(document.activeElement.tagName);
    if (ql.open) {
      if (e.key === "ArrowRight") stepQuickLook(1);
      if (e.key === "ArrowLeft") stepQuickLook(-1);
      return;
    }
    if (typing) {
      if (e.key === "Escape") search.blur();
      return;
    }
    if (e.key === "/" && currentTab === "learning") { e.preventDefault(); search.focus(); return; }
    if (currentTab === "learning" && state.layout === "carousel") {
      if (e.key === "ArrowRight") moveCarousel(1);
      if (e.key === "ArrowLeft") moveCarousel(-1);
      if (e.key === "Enter" && !e.target.closest(".card")) {
        const c = visibleCards()[state.center];
        c && openQuickLook(c.dataset.id);
      }
    }
  });

  /* ---------------- Ambient particle field ---------------- */
  (function field() {
    const cv = $("#field");
    const ctx = cv.getContext("2d");
    let w, h, dpr, pts = [];
    const mouse = { x: -1e4, y: -1e4 };
    const colors = ["139,92,246", "34,211,238", "244,114,182"];

    function resize() {
      dpr = Math.min(devicePixelRatio || 1, 2);
      w = cv.width = innerWidth * dpr;
      h = cv.height = innerHeight * dpr;
      const n = Math.round(Math.min(90, (innerWidth * innerHeight) / 16000));
      pts = Array.from({ length: n }, () => ({
        x: Math.random() * w,
        y: Math.random() * h,
        vx: (Math.random() - 0.5) * 0.25 * dpr,
        vy: (Math.random() - 0.5) * 0.25 * dpr,
        r: (Math.random() * 1.4 + 0.4) * dpr,
        c: colors[(Math.random() * colors.length) | 0],
      }));
    }

    function frame() {
      ctx.clearRect(0, 0, w, h);
      const link = 130 * dpr;
      for (let i = 0; i < pts.length; i++) {
        const p = pts[i];
        const dx = mouse.x - p.x, dy = mouse.y - p.y;
        const md = Math.hypot(dx, dy);
        if (md < 180 * dpr) { p.vx += (dx / md) * 0.012; p.vy += (dy / md) * 0.012; }
        p.vx *= 0.995; p.vy *= 0.995;
        p.x += p.vx; p.y += p.vy;
        if (p.x < 0 || p.x > w) p.vx *= -1;
        if (p.y < 0 || p.y > h) p.vy *= -1;
        for (let j = i + 1; j < pts.length; j++) {
          const q = pts[j];
          const d = Math.hypot(p.x - q.x, p.y - q.y);
          if (d < link) {
            ctx.strokeStyle = `rgba(${p.c},${(1 - d / link) * 0.22})`;
            ctx.lineWidth = dpr * 0.6;
            ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y); ctx.stroke();
          }
        }
        ctx.fillStyle = `rgba(${p.c},.8)`;
        ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2); ctx.fill();
      }
      if (!document.hidden && !reduceMotion) requestAnimationFrame(frame);
    }

    resize();
    addEventListener("resize", resize);
    addEventListener("pointermove", (e) => { mouse.x = e.clientX * dpr; mouse.y = e.clientY * dpr; });
    document.addEventListener("visibilitychange", () => !document.hidden && !reduceMotion && requestAnimationFrame(frame));
    requestAnimationFrame(frame);
  })();

  /* ---------------- Init ---------------- */
  showTab(location.hash.slice(1) || "home", false);
  requestAnimationFrame(moveIndicator);
  document.fonts && document.fonts.ready.then(moveIndicator);
})();
