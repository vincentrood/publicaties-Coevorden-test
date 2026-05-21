/**
 * process-woo.mjs
 *
 * Verwerkt Woo-dossiers (markdown-bestanden met PDF-tekst) naar:
 *   - Een korte samenvatting (max 2 zinnen)
 *   - Een gesorteerde, ontdubbelde tijdlijn van milestones
 *
 * Verbeteringen t.o.v. vorige versie:
 *   ✓ Dubbele alreadyDone-check verwijderd
 *   ✓ JSON.parse in try/catch met leesbare foutmelding
 *   ✓ writeAtomic gebruikt dezelfde map als doelbestand (cross-device veilig)
 *   ✓ MIN_YEAR geïnjecteerd in systeem-prompt (niet hardcoded)
 *   ✓ Chunkknippen op woordgrens
 *   ✓ scoreBlock() als aparte, testbare functie
 *   ✓ Legacy ai_hash comment toegevoegd
 */

import fs from "fs";
import path from "path";
import crypto from "crypto";
import matter from "gray-matter";
import OpenAI from "openai";
import { globSync } from "glob";

/* ─────────────────────────────────────────────
   CONFIG — pas hier aan
───────────────────────────────────────────── */
const MODEL              = "gpt-4o-mini";
const MAX_TOKENS_CHUNK   = 6_000;   // Max tokens per chunk naar AI
const MAX_TOKENS_SUMMARY = 30_000;  // Harde bovengrens samenvatting-input
const MAX_CHUNKS         = 4;       // Max chunks per bestand voor milestones
const CHUNK_PAUSE_MS     = 1_500;   // Pauze tussen chunks (ms)
const FILE_PAUSE_MS      = 1_000;   // Pauze tussen bestanden (ms)
const MIN_YEAR           = 2020;    // Milestones vóór dit jaar worden genegeerd

/* ─────────────────────────────────────────────
   CLI-argumenten
───────────────────────────────────────────── */
const args    = process.argv.slice(2);
const DRY_RUN = args.includes("--dry-run");
const FOLDER  = (args.find(a => a.startsWith("--folder=")) ?? "").replace("--folder=", "") || "2024";
const GLOB    = `docs/${FOLDER}/**/*.md`;

if (DRY_RUN) console.log("🔍 DRY-RUN modus — er worden geen bestanden aangepast.\n");

/* ─────────────────────────────────────────────
   OPENAI CLIENT
───────────────────────────────────────────── */
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

/* ─────────────────────────────────────────────
   UTIL
───────────────────────────────────────────── */

/** Ruwe schatting: 1 token ≈ 4 tekens */
const estimateTokens = (text) => Math.ceil(text.length / 4);

/** MD5-hash van de markdown-inhoud — gebruikt als change-detectie */
const contentHash = (text) => crypto.createHash("md5").update(text).digest("hex");

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/**
 * Normaliseert een datumstring naar YYYY-MM-DD.
 * Pakt ook "2024-1-5", "01/03/2024", "3 jan 2024" op.
 * Geeft null terug als parsen mislukt.
 */
function normalizeDate(raw) {
  if (!raw) return null;
  const s = String(raw).trim();

  // Al correct: YYYY-MM-DD
  if (/^\d{4}-\d{2}-\d{2}$/.test(s)) return s;

  // YYYY-M-D of YYYY-MM-D
  const isoLoose = s.match(/^(\d{4})-(\d{1,2})-(\d{1,2})$/);
  if (isoLoose) {
    const [, y, m, d] = isoLoose;
    return `${y}-${m.padStart(2, "0")}-${d.padStart(2, "0")}`;
  }

  // DD/MM/YYYY of DD-MM-YYYY
  const dmy = s.match(/^(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{4})$/);
  if (dmy) {
    const [, d, m, y] = dmy;
    return `${y}-${m.padStart(2, "0")}-${d.padStart(2, "0")}`;
  }

  // "3 januari 2024" / "03 jan. 2024"
  const maanden = {
    jan: 1, feb: 2, maa: 3, mar: 3, apr: 4, mei: 5, may: 5,
    jun: 6, jul: 7, aug: 8, sep: 9, okt: 10, oct: 10, nov: 11, dec: 12,
  };
  const nl = s.match(/^(\d{1,2})\s+([a-zë]+)\.?\s+(\d{4})$/i);
  if (nl) {
    const [, d, mon, y] = nl;
    const m = maanden[mon.toLowerCase().slice(0, 3)];
    if (m) return `${y}-${String(m).padStart(2, "0")}-${d.padStart(2, "0")}`;
  }

  return null;
}

/* ─────────────────────────────────────────────
   RETRY WRAPPER
───────────────────────────────────────────── */
async function withRetry(fn, retries = 5) {
  let lastErr;
  for (let i = 0; i < retries; i++) {
    try {
      return await fn();
    } catch (err) {
      lastErr = err;
      if (err?.status === 429 || err?.message?.includes("Rate limit")) {
        const wait = 2_000 * Math.pow(2, i);
        console.warn(`  ⏳ Rate limit — wacht ${wait / 1000}s (poging ${i + 1}/${retries})`);
        await sleep(wait);
        continue;
      }
      if (err?.status === 400 || err?.message?.includes("maximum context length")) {
        console.warn("  ⚠️  Context te groot, chunk overgeslagen.");
        return null;
      }
      throw err;
    }
  }
  throw lastErr;
}

/* ─────────────────────────────────────────────
   TEKST-SELECTIE
───────────────────────────────────────────── */

/**
 * Scoort één paragraaf op relevantie voor milestone-extractie.
 * Aparte functie zodat de logica los testbaar is.
 */
function scoreBlock(paragraph) {
  const ignore    = /bezwaar en beroep|wettelijk kader|artikel 5\./i;
  const important = /woo-verzoek|besluit|beslissing|verlengingsbesluit|opschorting|verdaging|zienswijze|openbaar|ingebrekestelling/i;
  const noise     = /re:|fwd:|verzonden:|bijlage|groet,|bespreken|afstemmen/i;
  const dateRe    = /\d{1,2}[-/\s](jan|feb|maa|apr|mei|jun|jul|aug|sep|okt|nov|dec|[0-9]{1,2})[-/\s]\d{4}/i;

  let score = 0;
  if (ignore.test(paragraph))    score -= 5;
  if (noise.test(paragraph))     score -= 4;
  if (important.test(paragraph)) score += 6;
  if (dateRe.test(paragraph))    score += 4;
  if (paragraph.length > 80)     score += 1;
  return score;
}

/** Scoort paragrafen op relevantie voor milestone-extractie */
function extractRelevantBlocks(text) {
  return text
    .split(/\n\s*\n/)
    .map((p) => ({ text: p.trim(), score: scoreBlock(p.trim()) }))
    .filter((p) => p.score > 5)
    .sort((a, b) => b.score - a.score)
    .map((p) => p.text);
}

/**
 * Knipt tekst op de laatste woordgrens vóór `maxChars`.
 * Voorkomt dat een woord midden doorbroken wordt.
 */
function splitAtWordBoundary(text, maxChars) {
  if (text.length <= maxChars) return [text, ""];
  const boundary = text.lastIndexOf(" ", maxChars);
  const cut = boundary > 0 ? boundary : maxChars;
  return [text.slice(0, cut), text.slice(cut).trimStart()];
}

/** Bouwt veilige chunks die de token-limiet niet overschrijden */
function buildSafeChunks(blocks) {
  const maxChars = MAX_TOKENS_CHUNK * 4;
  const chunks   = [];
  let current    = [];
  let tokens     = 0;

  for (const block of blocks) {
    const t = estimateTokens(block);

    // Extreem grote blokken opknippen op woordgrens
    if (t > MAX_TOKENS_CHUNK) {
      if (current.length) { chunks.push(current); current = []; tokens = 0; }
      let rem = block;
      while (rem.length > 0) {
        const [head, tail] = splitAtWordBoundary(rem, maxChars);
        chunks.push([head]);
        rem = tail;
      }
      continue;
    }

    if (tokens + t > MAX_TOKENS_CHUNK) {
      if (current.length) chunks.push(current);
      current = [block];
      tokens  = t;
    } else {
      current.push(block);
      tokens += t;
    }
  }
  if (current.length) chunks.push(current);
  return chunks;
}

/** Selecteert de meest informatieve paragrafen voor de samenvatting */
function extractSummaryBlocks(content) {
  let summaryText = content
    .split(/\n\s*\n/)
    .map((p) => {
      let score = 0;
      const l = p.toLowerCase();
      if (/besluit|beslissing|toegekend|afgewezen|verlengd|gegrond|ongegrond/.test(l)) score += 5;
      if (/aanvraag|verzoek|reactie|zienswijze|document|onderzoek|rapport/.test(l))    score += 3;
      if (/college|burgemeester|gemeente|bestuurlijk|afdeling/.test(l))                score += 2;
      if (p.length > 200) score += 1;
      return { text: p.trim(), score };
    })
    .filter((p) => p.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, 30)
    .map((p) => p.text)
    .join("\n\n");

  if (estimateTokens(summaryText) > MAX_TOKENS_SUMMARY) {
    summaryText = summaryText.substring(0, MAX_TOKENS_SUMMARY * 4);
  }
  return summaryText;
}

/* ─────────────────────────────────────────────
   AI CALLS — SAMENVATTING
───────────────────────────────────────────── */
async function fetchSummary(text) {
  if (!text?.trim()) return "";

  const result = await withRetry(async () => {
    const response = await openai.chat.completions.create({
      model: MODEL,
      temperature: 0,
      response_format: { type: "json_object" },
      messages: [
        {
          role: "system",
          content:
            "Je bent een expert in Nederlandse Woo-dossiers. " +
            "Schrijf een neutrale, feitelijke samenvatting van maximaal 2 zinnen. " +
            "Geef uitsluitend valide JSON terug.",
        },
        {
          role: "user",
          content: `Geef STRICT JSON:\n{ "summary": "max 2 zinnen" }\n\nTEKST:\n${text}`,
        },
      ],
    });

    const raw = response.choices[0].message.content.replace(/```json|```/g, "").trim();

    try {
      return JSON.parse(raw);
    } catch (parseErr) {
      throw new Error(`JSON-parse mislukt (samenvatting). Ruwe output:\n${raw}`);
    }
  });

  return result?.summary?.trim() ?? "";
}

/* ─────────────────────────────────────────────
   AI CALLS — MILESTONES
───────────────────────────────────────────── */
async function fetchMilestones(textBlocks) {
  if (!textBlocks?.length) return [];

  const text = Array.isArray(textBlocks) ? textBlocks.join("\n\n") : textBlocks;

  const result = await withRetry(async () => {
    const response = await openai.chat.completions.create({
      model: MODEL,
      temperature: 0,
      response_format: { type: "json_object" },
      messages: [
        {
          role: "system",
          content:
            "Je bent een expert in Nederlandse Woo-dossiers. Je taak is het extraheren van een chronologische tijdlijn van HOOFDMILESTONES.\n\n" +
            "WEL extraheren (voorbeelden van relevante milestones):\n" +
            "- Ontvangst/indiening van het Woo-verzoek\n" +
            "- Besluiten (primair besluit, beslissing op bezwaar)\n" +
            "- Formele correspondentie (zienswijze opgevraagd, verdaging/termijnverlenging)\n" +
            "- Publicatie of openbaarmaking van documenten\n\n" +
            "STRIKT NEGEREN (ruis):\n" +
            "- Dagelijkse e-mailwisselingen tussen ambtenaren\n" +
            "- Agenda-afspraken of interne vergaderdata\n" +
            "- Versienummers van documenten met een datum\n" +
            "- Datums die genoemd worden in de lopende tekst maar geen formele processtap zijn.\n\n" +
            `Gebruik ISO 8601 datums (YYYY-MM-DD). Negeer events vóór ${MIN_YEAR}. Geef uitsluitend valide JSON terug.`,
        },
        {
          role: "user",
          content:
            `Geef STRICT JSON:\n` +
            `{ "milestones": [{ "date": "YYYY-MM-DD", "event": "Korte, zakelijke omschrijving van de formele processtap" }] }\n\n` +
            `Als er geen relevante processtappen in de tekst staan, geef dan een lege array.\n\n` +
            `TEKST:\n${text}`,
        },
      ],
    });

    const raw = response.choices[0].message.content.replace(/```json|```/g, "").trim();

    try {
      return JSON.parse(raw);
    } catch (parseErr) {
      throw new Error(`JSON-parse mislukt (milestones). Ruwe output:\n${raw}`);
    }
  });

  return result?.milestones ?? [];
}

/* ─────────────────────────────────────────────
   MILESTONE CLEANING
───────────────────────────────────────────── */
function cleanMilestones(milestones) {
  const seen = new Set();
  return milestones
    .map((m) => ({
      date:  normalizeDate(m.date),
      event: m.event?.trim(),
    }))
    .filter((m) => m.date && m.event)
    .filter((m) => parseInt(m.date.slice(0, 4)) >= MIN_YEAR)
    .filter((m) => {
      const id = `${m.date}||${m.event.toLowerCase()}`;
      if (seen.has(id)) return false;
      seen.add(id);
      return true;
    })
    .sort((a, b) => a.date.localeCompare(b.date));
}

/* ─────────────────────────────────────────────
   ATOMISCHE BESTANDSWRITE
   Schrijft naar een tmp-bestand in dezelfde map als het doelbestand,
   en hernoemt — crashveilig én cross-device veilig (geen cross-fs rename).
───────────────────────────────────────────── */
function writeAtomic(filePath, content) {
  const tmpPath = path.join(
    path.dirname(filePath),
    `.tmp-${crypto.randomBytes(6).toString("hex")}`
  );
  try {
    fs.writeFileSync(tmpPath, content, "utf8");
    fs.renameSync(tmpPath, filePath);
  } catch (err) {
    // Opruimen bij fout, maar originele fout blijft leidend
    try { fs.unlinkSync(tmpPath); } catch (_) {}
    throw err;
  }
}

/* ─────────────────────────────────────────────
   BESTANDSVERWERKING
───────────────────────────────────────────── */
async function processFile(file, index, total) {
  const label = `[${index}/${total}] ${path.basename(file)}`;

  try {
    const raw               = fs.readFileSync(file, "utf8");
    const { data, content } = matter(raw);
    const hash              = contentHash(content);

    // Skip als de inhouds-hash exact overeenkomt met wat we eerder hebben opgeslagen
    if (data.ai_content_hash === hash) {
      console.log(`⏭  Skip (ongewijzigd): ${label}`);
      return;
    }

    console.log(`🔄 Verwerken: ${label}`);

    if (DRY_RUN) {
      console.log(`   (dry-run — geen wijzigingen)`);
      return;
    }

    /* ── Stap 1: Samenvatting ── */
    const summaryInput = extractSummaryBlocks(content);
    const summary      = await fetchSummary(summaryInput);

    /* ── Stap 2: Milestones per chunk ── */
    const blocks        = extractRelevantBlocks(content);
    const chunks        = buildSafeChunks(blocks).slice(0, MAX_CHUNKS);
    let   allMilestones = [];

    for (const chunk of chunks) {
      await sleep(CHUNK_PAUSE_MS);
      const results = await fetchMilestones(chunk);
      if (results?.length) allMilestones.push(...results);
    }

    const milestones = cleanMilestones(allMilestones);

    /* ── Stap 3: Frontmatter bijwerken & atomisch schrijven ── */
    data.summary         = summary;
    data.milestones      = milestones;
    data.ai_processed_at = new Date().toISOString();
    data.ai_content_hash = hash;

    // Legacy-veld 'ai_hash' werd in een eerdere versie van dit script
    // gebruikt als change-detectie. Vervangen door 'ai_content_hash'.
    delete data.ai_hash;

    writeAtomic(file, matter.stringify(content, data));

    console.log(`✅ Klaar: ${label} (${milestones.length} milestone${milestones.length !== 1 ? "s" : ""})`);

  } catch (err) {
    console.error(`❌ Fout: ${label}\n   ${err.message}`);
  }
}

/* ─────────────────────────────────────────────
   MAIN — één voor één, veilig voor rate limits
───────────────────────────────────────────── */
async function main() {
  const files = globSync(GLOB);

  if (files.length === 0) {
    console.log(`Geen bestanden gevonden voor: ${GLOB}`);
    return;
  }

  console.log(`\n📂 Map       : docs/${FOLDER}`);
  console.log(`📄 Bestanden : ${files.length}`);
  console.log(`🧩 Max chunks: ${MAX_CHUNKS} per bestand`);
  console.log(`⏱  Pauze     : ${CHUNK_PAUSE_MS}ms tussen chunks, ${FILE_PAUSE_MS}ms tussen bestanden`);
  if (DRY_RUN) console.log(`🔍 Modus     : dry-run`);
  console.log();

  for (let i = 0; i < files.length; i++) {
    await processFile(files[i], i + 1, files.length);
    if (i < files.length - 1) await sleep(FILE_PAUSE_MS);
  }

  console.log("\n✔ Gereed.");
}

main();
