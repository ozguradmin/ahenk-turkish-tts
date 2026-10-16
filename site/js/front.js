// The text front-end in Pyodide: trnorm (src/trnorm, copied to site/py/trnorm) and site/py/ahenk_front.py run as
// the same Python they are in the repo, so the spoken form is the one PyTorch's pipeline sees.

export const PY_FILES = [
  'ahenk_front.py',
  'trnorm/__init__.py',
  'trnorm/casing.py',
  'trnorm/itn.py',
  'trnorm/lexicon.py',
  'trnorm/normalizer.py',
  'trnorm/numbers.py',
];

// pyodide: a loaded Pyodide; readFile(path) -> Promise<string> (the file under site/py/); vocab: manifest.vocab.
export async function setupFront(pyodide, readFile, vocab) {
  const root = '/home/pyodide/ahenk';
  pyodide.FS.mkdirTree(`${root}/trnorm`);
  const texts = await Promise.all(PY_FILES.map((p) => readFile(p)));
  PY_FILES.forEach((p, i) => pyodide.FS.writeFile(`${root}/${p}`, texts[i], { encoding: 'utf8' }));
  pyodide.runPython(`import sys\nsys.path.insert(0, ${JSON.stringify(root)})\nimport ahenk_front`);
  const mod = pyodide.pyimport('ahenk_front');
  const pyVocab = pyodide.toPy(vocab);
  const prepare = (text, speed) => JSON.parse(mod.prepare(text, speed, pyVocab));
  const normalize = (text) => mod.normalize(text);
  return { prepare, normalize };
}
