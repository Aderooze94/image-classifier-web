// --- éléments UI
const statusEl  = document.getElementById('status');
const inputEl   = document.getElementById('fileInput');
const imgEl     = document.getElementById('preview');
const resultsEl = document.getElementById('results');

let model = null;

// Petit helper pour afficher les étapes à l'écran
function setStatus(t) { statusEl.textContent = t; }

// Initialise TF + backend + modèle
(async function init() {
  try {
    setStatus('Initialisation TensorFlow…');
    await tf.ready();

    // Essaye WebGL d’abord (plus rapide). Si échec, bascule CPU.
    try {
      await tf.setBackend('webgl');
      await tf.ready();
      setStatus('Backend: WebGL');
    } catch {
      await tf.setBackend('cpu');
      await tf.ready();
      setStatus('Backend: CPU');
    }

    // (facultatif) warm-up rapide
    tf.tidy(() => tf.zeros([1, 224, 224, 3]).add(1).abs().dispose());

    setStatus(s => s + ' — chargement MobileNet…');
    model = await mobilenet.load(); // v2 par défaut
    setStatus('Modèle prêt. Prends une photo 👇');

  } catch (e) {
    console.error(e);
    setStatus('Erreur init TF/Modèle. Regarde la console si possible.');
  }
})();

// Attend que <img> soit affichable (dimensions non nulles)
function waitImage(img) {
  return new Promise((resolve, reject) => {
    if (img.complete && img.naturalWidth > 0) return resolve();
    img.onload  = () => resolve();
    img.onerror = (e) => reject(e);
  });
}

inputEl.addEventListener('change', async (e) => {
  const file = e.target.files && e.target.files[0];
  if (!file) return;

  resultsEl.textContent = '';
  setStatus('Préparation de l’image…');

  // Aperçu via blob URL
  const url = URL.createObjectURL(file);
  imgEl.src = url;
  try {
    await waitImage(imgEl);
  } catch (err) {
    console.error('Chargement image', err);
    setStatus('Impossible de charger l’image.');
    return;
  }

  if (!model) { setStatus('Modèle non prêt.'); return; }

  try {
    setStatus('Classification en cours…');
    // Top-3
    const preds = await model.classify(imgEl);
    if (!preds || preds.length === 0) {
      resultsEl.textContent = 'Aucune prédiction.';
    } else {
      resultsEl.textContent = preds.slice(0, 3)
        .map(p => `${p.className} (${p.probability.toFixed(3)})`)
        .join('\n');
    }
    setStatus('Terminé.');
  } catch (err) {
    console.error(err);
    setStatus('Erreur pendant l’analyse.');
    resultsEl.textContent = String(err);
  }
  const dbg = document.getElementById('debug');
function log(...args){ console.log(...args); if(dbg){ dbg.textContent += args.join(' ') + '\n'; } }

});
