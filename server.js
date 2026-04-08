// server/server.js
// Simple Express server that:
// - Predicts emotion from text using Hugging Face Inference API (optional, requires HUGGINGFACE_API_KEY).
// - Falls back to a keyword-based predictor if HF key is not present.
// - Returns a matched song and resolves a real YouTube video id by calling YouTube Data API (optional, requires YOUTUBE_API_KEY).
//
// Start: `node server/server.js` (after npm install)

import express from "express";
import fetch from "node-fetch";
import cors from "cors";
import dotenv from "dotenv";
import path from "path";
import { fileURLToPath } from "url";

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(cors());
app.use(express.json());

// Config
const HF_API_KEY = process.env.HUGGINGFACE_API_KEY || "";
const YT_API_KEY = process.env.YOUTUBE_API_KEY || "";
const HF_MODEL = process.env.HF_MODEL || "j-hartmann/emotion-english-distilroberta-base";
const PORT = process.env.PORT || 3000;

// A curated mapping of mood keys to descriptive names and representative songs.
// Song entries are title + artist (we will search YouTube for these).
const SONG_CATALOG = {
  happy: {
    name: "Happy / Joyful",
    songs: [
      { title: "Gallan Goodiyaan", artist: "Dil Dhadakne Do" },
      { title: "Badtameez Dil", artist: "Yeh Jawaani Hai Deewani" },
      { title: "Senorita", artist: "Zindagi Na Milegi Dobara" }
    ]
  },
  sad: {
    name: "Sad / Melancholic",
    songs: [
      { title: "Channa Mereya", artist: "Ae Dil Hai Mushkil" },
      { title: "Tadap Tadap Ke", artist: "Hum Dil De Chuke Sanam" },
      { title: "Agar Tum Saath Ho", artist: "Tamasha" }
    ]
  },
  romantic: {
    name: "Romantic / Love",
    songs: [
      { title: "Tum Hi Ho", artist: "Aashiqui 2" },
      { title: "Raabta", artist: "Agent Vinod" },
      { title: "Pehla Nasha", artist: "Jo Jeeta Wohi Sikandar" }
    ]
  },
  energetic: {
    name: "Energetic / Pumped",
    songs: [
      { title: "Kar Gayi Chull", artist: "Kapoor & Sons" },
      { title: "Malhari", artist: "Bajirao Mastani" },
      { title: "Kala Chashma", artist: "Baar Baar Dekho" }
    ]
  },
  calm: {
    name: "Calm / Relaxed",
    songs: [
      { title: "Kahin Toh Hogi", artist: "Jaane Tu... Ya Jaane Na" },
      { title: "Tum Se Hi (soft)", artist: "Jab We Met" },
      { title: "Kun Faya Kun", artist: "Rockstar" }
    ]
  },
  nostalgic: {
    name: "Nostalgic / Reminiscent",
    songs: [
      { title: "Lag Ja Gale", artist: "Woh Kaun Thi" },
      { title: "Tere Bina Zindagi Se", artist: "Aandhi" },
      { title: "Yaad Kiya Dil Ne", artist: "Patanga" }
    ]
  },
  angry: {
    name: "Angry / Frustrated",
    songs: [
      { title: "Dhoom Again", artist: "Dhoom 2" },
      { title: "Bala", artist: "Housefull 4" },
      { title: "Munni Badnaam Hui", artist: "Dabangg" }
    ]
  }
};

// Keyword fallback (if HF not configured)
const KEYWORDS = {
  happy: ['happy','joy','joyful','cheerful','excited','glad','amazing','good','positive','blessed'],
  sad: ['sad','lonely','heartbroken','depressed','cry','tears','sorrow','hurt','missing','lost'],
  romantic: ['love','romance','crush','in love','longing','miss you','loving','affection'],
  energetic: ['energetic','pumped','hyped','dance','party','upbeat','hyped','excited','groove'],
  calm: ['calm','relaxed','peaceful','chill','tranquil','soothing','serene','relax'],
  nostalgic: ['nostalgic','memories','remember','past','old days','youth','retro'],
  angry: ['angry','mad','furious','annoyed','hate','rage','frustrated','irritated','upset','pissed']
};

// Helpers
function tokenize(text) {
  return text.toLowerCase().replace(/[^\w\s]/g, ' ').split(/\s+/).filter(Boolean);
}

function keywordPredict(text) {
  const tokens = tokenize(text);
  const scores = {};
  for (const k of Object.keys(KEYWORDS)) {
    scores[k] = 0;
    for (const kw of KEYWORDS[k]) {
      for (const t of tokens) {
        if (t.includes(kw)) scores[k] += 1;
      }
    }
  }
  // fallback heuristic
  const any = Object.values(scores).some(v => v > 0);
  if (!any) {
    // simple sentiment-ish words
    const positive = ['good','great','awesome','fantastic','happy','joy','love'];
    const negative = ['bad','sad','lonely','depressed','angry','hate'];
    let p=0,n=0;
    for (const t of tokens) {
      if (positive.includes(t)) p++;
      if (negative.includes(t)) n++;
    }
    if (p>n) scores['happy'] += p || 1;
    else if (n>p) scores['sad'] += n || 1;
    else scores['calm'] += 1;
  }
  let best = Object.keys(scores)[0];
  for (const k of Object.keys(scores)) {
    if (scores[k] > scores[best]) best = k;
  }
  return { mood: best, scores, matched: [] };
}

// Map HF labels to our mood keys
const HF_LABEL_MAP = {
  joy: 'happy',
  love: 'romantic',
  sadness: 'sad',
  anger: 'angry',
  fear: 'angry',      // map fear to angry/frustrated bucket
  surprise: 'energetic',
  neutral: 'calm'
};

async function predictWithHF(text) {
  if (!HF_API_KEY) throw new Error('Hugging Face API key not configured');
  const url = `https://api-inference.huggingface.co/models/${HF_MODEL}`;
  const res = await fetch(url, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${HF_API_KEY}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ inputs: text })
  });
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`HF API error: ${res.status} ${txt}`);
  }
  const data = await res.json();
  // HF returns array of {label,score} for many classification models
  if (!Array.isArray(data) || data.length === 0) throw new Error('Unexpected HF response');
  const best = data.reduce((a,b)=> a.score>b.score? a:b);
  const label = best.label.toLowerCase();
  const mapped = HF_LABEL_MAP[label] || 'calm';
  return { mood: mapped, label: label, scores: data };
}

// Search YouTube for the given query (title + artist) and return videoId or null
async function searchYouTubeVideoId(q) {
  if (!YT_API_KEY) return null;
  const params = new URLSearchParams({
    key: YT_API_KEY,
    part: 'snippet',
    q,
    type: 'video',
    maxResults: '1',
    safeSearch: 'none'
  });
  const url = `https://www.googleapis.com/youtube/v3/search?${params.toString()}`;
  const res = await fetch(url);
  if (!res.ok) {
    const t = await res.text();
    console.warn('YouTube API error', res.status, t);
    return null;
  }
  const data = await res.json();
  if (!data.items || data.items.length === 0) return null;
  return data.items[0].id.videoId || null;
}

// Routes

// Simple health
app.get("/api/health", (req,res) => {
  res.json({ ok: true, hf: !!HF_API_KEY, yt: !!YT_API_KEY });
});

// Predict mood from text
app.post("/api/predict", async (req,res) => {
  try {
    const text = (req.body && req.body.text) ? String(req.body.text) : "";
    if (!text || text.trim().length === 0) return res.status(400).json({ error: "No text provided" });

    let result;
    if (HF_API_KEY) {
      try {
        const hf = await predictWithHF(text);
        result = {
          source: "huggingface",
          mood: hf.mood,
          label: hf.label,
          scores: hf.scores
        };
      } catch (err) {
        console.warn("HF error, falling back to keyword:", err.message);
        const kw = keywordPredict(text);
        result = { source: "keyword-fallback", mood: kw.mood, scores: kw.scores };
      }
    } else {
      const kw = keywordPredict(text);
      result = { source: "keyword", mood: kw.mood, scores: kw.scores };
    }

    res.json(result);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message || 'server error' });
  }
});

// Get a song for a mood (returns one song with resolved youtube videoId if possible)
app.get("/api/song", async (req,res) => {
  try {
    const mood = String(req.query.mood || "calm");
    if (!SONG_CATALOG[mood]) return res.status(400).json({ error: "Unknown mood" });

    // pick a random song entry from the catalog
    const list = SONG_CATALOG[mood].songs;
    const idx = Math.floor(Math.random() * list.length);
    const chosen = list[idx];
    const query = `${chosen.title} ${chosen.artist} full song`;

    let videoId = null;
    if (YT_API_KEY) {
      videoId = await searchYouTubeVideoId(query);
    }

    res.json({
      mood,
      moodName: SONG_CATALOG[mood].name,
      title: chosen.title,
      artist: chosen.artist,
      youtubeId: videoId // may be null if YT key not set
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message || 'server error' });
  }
});

// Serve static frontend (optional)
app.use(express.static(path.join(__dirname, "../frontend")));
app.get("/", (req,res) => {
  res.sendFile(path.join(__dirname, "../frontend/index.html"));
});

app.listen(PORT, () => {
  console.log(`Server listening on http://localhost:${PORT} (HF=${!!HF_API_KEY}, YT=${!!YT_API_KEY})`);
});