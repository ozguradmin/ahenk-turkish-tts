// Load the demo page in a real Chrome / Edge (fresh profile, so a cold first visit), wait until the model is ready,
// speak a few sentences through the page's own button and print load time and real-time factors as JSON.
//
//   node tools/web/browser_bench.mjs --deps <dir with node_modules/puppeteer-core> --url http://localhost:8000/
//        [--browser "C:/Program Files/Google/Chrome/Application/chrome.exe"] [--headless new|false] [--repeat 1]
//
// Serve site/ first (python tools/web/serve.py). Without --isolate the page relies on coi-serviceworker.js for
// cross-origin isolation, exactly as on GitHub Pages; the first load registers it and reloads once.

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';

const a = { browser: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: 'new', repeat: '1' };
const v = process.argv.slice(2);
for (let i = 0; i < v.length; i += 2) a[v[i].replace(/^--/, '')] = v[i + 1];
if (!a.deps || !a.url) throw new Error('--deps and --url are required');
const entry = createRequire(path.resolve(a.deps, 'package.json')).resolve('puppeteer-core');
const mod = await import(pathToFileURL(entry).href);
const puppeteer = mod.default ?? mod;

const SH = 'Bu ürünün garantisi var mı?';
const MD = "Randevunuz 14.10.2026 saat 09:30'da, 3. katta.";
const LG = 'Karşılaştırma isteyenlere eski ve yeni paketi yan yana anlatırız; farkı müşteri kendisi görür. Akşam olduğunda kısa bir değerlendirme yaparız: hangi sorular arttı, hangi açıklama iyi çalıştı, nerede zorlandık?';
const ITEMS = [[SH, 16], [MD, 16], [LG, 16], [SH, 16], [MD, 16], [LG, 16], [LG, 8], [LG, 4]];

const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'ahenk-bench-'));
const browser = await puppeteer.launch({
  executablePath: a.browser, headless: a.headless === 'false' ? false : a.headless, userDataDir: profile,
  args: ['--no-first-run', '--no-default-browser-check', '--enable-unsafe-webgpu', '--autoplay-policy=no-user-gesture-required'],
});
const results = [];
try {
  for (let r = 0; r < Number(a.repeat); r++) {
    const page = await browser.newPage();
    const t0 = Date.now();
    await page.goto(a.url, { waitUntil: 'load' });
    await page.waitForFunction(() => window.__ahenk?.info || document.getElementById('load')?.classList.contains('error'),
      { timeout: 180000, polling: 200 });
    const readyMs = Date.now() - t0;
    const out = await page.evaluate(async (items) => {
      if (!window.__ahenk) return { error: document.getElementById('load-text').textContent };
      for (const [t, steps] of items) {
        const n = window.__ahenk.runs.length;
        document.getElementById('text').value = t;
        document.getElementById('steps').value = String(steps);
        document.getElementById('go').click();
        await new Promise((res) => {
          const iv = setInterval(() => {
            if (window.__ahenk.runs.length > n || document.getElementById('status').classList.contains('error')) {
              clearInterval(iv);
              res();
            }
          }, 20);
        });
      }
      return {
        info: window.__ahenk.info, isolated: crossOriginIsolated, status: document.getElementById('status').textContent,
        runs: window.__ahenk.runs.map((x) => ({ steps: x.steps, seconds: x.seconds, ms: Math.round(x.ms), rtf: +x.rtf.toFixed(3) })),
      };
    }, ITEMS);
    results.push({ visit: r + 1, readyMs, ...out, userAgent: await browser.userAgent() });
    await page.close();
  }
} finally {
  await browser.close();
  fs.rmSync(profile, { recursive: true, force: true });
}
console.log(JSON.stringify(results, null, 1));
