import { readFileSync, existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import assert from 'node:assert/strict';
const root = resolve('public');
const article = readFileSync(resolve(root, 'index.html'), 'utf8');
const fragment = readFileSync(resolve(root, 'basic-html.html'), 'utf8');
assert(article.includes(fragment), 'Homepage must contain the exact Drupal article');
assert.equal(article, readFileSync(resolve(root, 'reading.html'), 'utf8'));
for (const html of [article, fragment]) {
  assert(!/<(?:script|style|link|iframe|form|input|select|button|details|summary|canvas|svg)\b|\s(?:on[a-z]+|style|class)\s*=/i.test(html), 'Article must be fully static with no CSS dependencies or controls');
  assert(html.includes('<strong>TL;DR:</strong>'), 'Missing TL;DR');
  assert(!/Reveal all|Open a choice|close each answer|Choose a proposed coloring|Try a challenge|your settings uncover/i.test(html), 'Interactive directions remain');
  assert(html.includes('3 × 2 × 1 = 6 colorings'), 'Missing static graph solution');
  assert(html.includes('1,024') && html.includes('1.16') && html.includes('95.1%'), 'Study values changed');
  const images = [...html.matchAll(/<img\b[^>]*>/g)].map(m => m[0]);
  assert.equal(images.length, 5, 'Expected five figures');
  for (const image of images) assert(/alt="[^"]+"/.test(image), 'Missing figure alt text');
  for (const name of ['01-graph-six-colorings', '02-same-accuracy', '03-measured-concentration', '04-feedback', '05-sampling']) {
    assert(html.includes(`src="figures/${name}.png"`), `Missing figure: ${name}`);
  }
}
for (const name of ['index.html', 'basic-html.html', 'reading.html', 'games.html', 'puzzle.html', 'sampler.html']) {
  const page = readFileSync(resolve(root, name), 'utf8');
  for (const [, url] of page.matchAll(/(?:src|href)="([^"]+)"/g)) {
    if (/^(?:https?:|data:|#)/.test(url)) continue;
    const path = url.split(/[?#]/)[0];
    assert(existsSync(resolve(dirname(resolve(root, name)), path)), `Missing ${url} in ${name}`);
  }
}
assert(readFileSync(resolve(root, 'paper.pdf')).subarray(0, 5).equals(Buffer.from('%PDF-')), 'Missing paper PDF');
for (const archive of ['drupal_static_package.zip', 'drupal_static_figures.zip']) {
  assert(readFileSync(resolve(root, archive)).subarray(0, 2).equals(Buffer.from('PK')), `Missing archive: ${archive}`);
}
console.log('Static blog validated: exact Drupal article, TL;DR, one worked puzzle, five figures, no CSS or interactions, all local assets present.');
