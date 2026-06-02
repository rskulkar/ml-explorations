# CCA-F Mock Exam — Vercel Deployment

## What this is
A mock exam for the Claude Certified Architect Foundations (CCA-F) certification.
Generates fresh 25-question exams using Claude, with anti-pattern analysis and Markdown export.

---

## Deploy in 5 steps

### 1. Install prerequisites
```bash
# Node.js 18+ required
node -v

# Install Vercel CLI
npm install -g vercel
```

### 2. Clone / copy this project
```bash
# If you have the zip, unzip it. Then:
cd cca-exam
npm install
```

### 3. Add your Anthropic API key to Vercel
```bash
vercel login          # sign in or create a free Vercel account
vercel env add ANTHROPIC_API_KEY
# When prompted: paste your key, select all environments (Production, Preview, Development)
```

Your API key is stored encrypted in Vercel — it never appears in your code or the browser.

### 4. Deploy
```bash
vercel --prod
```

Vercel will:
- Build the React frontend
- Deploy the `api/generate.js` edge function
- Give you a URL like `https://cca-exam-xyz.vercel.app`

Share that URL with your friends. Done.

### 5. (Optional) Custom domain
```bash
vercel domains add yourdomain.com
```

---

## Rate limiting
The proxy allows **3 exam generations per IP per hour** to protect your account.
To change the limit, edit `api/generate.js`:
```js
const LIMIT = 3;           // max generations
const WINDOW_MS = 60 * 60 * 1000;  // per hour
```

For stricter persistent rate limiting (survives cold starts), upgrade to Vercel KV
and replace the in-memory Map with KV reads/writes.

---

## Local development
```bash
# Install Vercel CLI if not done
npm install -g vercel

# Run locally (edge functions + frontend together)
vercel dev

# App runs at http://localhost:3000
# The proxy at http://localhost:3000/api/generate
```

For local dev, create a `.env.local` file:
```
ANTHROPIC_API_KEY=sk-ant-...
```

---

## Project structure
```
cca-exam/
├── api/
│   └── generate.js     # Edge function — holds API key, proxies to Anthropic
├── src/
│   ├── main.jsx        # React entry point
│   └── App.jsx         # Full exam app — calls /api/generate (not Anthropic directly)
├── public/             # Static assets (empty, add favicon here if desired)
├── index.html          # HTML shell
├── vite.config.js      # Vite build config
├── vercel.json         # Vercel routing config
├── package.json
└── README.md
```

---

## Costs
- **Vercel**: Free tier covers this comfortably (Hobby plan, no credit card needed)
- **Anthropic**: Each exam generation = 5 API calls to claude-haiku-4-5, ~2500 tokens each
  = roughly $0.003–0.005 per full exam generation at current Haiku pricing
